"""Diffusion-model converters: W4A8 codebook and W6A8 (comfy-kitchen
AsymW4A8Int8Layout), replacing the old ratio-based "W4A8/W6A8 + INT8 Mixed"
nodes. No int8 ratio / ranking: every eligible layer gets the chosen format.

Per-layer rules
  exclusion list (MODEL_TYPE_EXCLUSIONS)        -> kept at source precision
  2D float weight, K % 256 == 0 and K % group   -> chosen grouped format
  other 2D float weight                         -> int8 ConvRot with the
                                                   largest group (256/64/16)
                                                   dividing K (rule-based
                                                   fallback, as Kijai's
                                                   checkpoints do)
  everything else                               -> copied unchanged
  prune_donor (MiniMax H3)                      -> curve-form adaln tensors
                                                   + config merged from donor

Output: Kijai-style single '_quantization_metadata' header block, streamed
to disk tensor-by-tensor (RAM ~ one tensor; needs ~2x output size of free
disk while writing).

Error measurement (report only):
  weights            per layer: Rel-RMSE, SNR, Outlier %, per-channel MSE
                     (MSE/MAE/max kept as local-check values in the JSON)
  weights+timesteps  + a short sampling run with the connected calib_*
                     inputs. Every model call runs twice — reference and
                     fake-quantized (decoded weights + int8 ConvRot
                     activations) — on the same input (teacher forcing).
                     Tracks the prediction-noise variance per timestep,
                     per-layer local error per step, and the diagonal
                     Hessian used to weight each layer's weight distance.
"""

import json
import logging
import os

import torch
import torch.nn.functional as F

import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_detection

from .int8_unet_loader import MODEL_TYPE_EXCLUSIONS
from .grouped_quant import (
    FORMAT_W4A8, FORMAT_W6A8, GROUPED_BITS,
    kitchen_support, grouped_spec, int8_spec,
    SourceReader, StreamingSafetensorsWriter,
    print_plan, execute_plan, map_modules, FakeQuantSession,
    summarize_local, log_summary, write_report,
)


_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _flat(t):
    if torch.is_tensor(t):
        return t.float().flatten()
    if isinstance(t, (list, tuple)):
        return torch.cat([_flat(x) for x in t])
    raise TypeError(f"unsupported model output type {type(t)}")


def _build_plan(reader, donor, model_type, fmt, format_name, group, label):
    """Plan entries for every tensor of the (normalized, donor-merged) model."""
    excluded = MODEL_TYPE_EXCLUSIONS.get(model_type, [])
    metadata = dict(reader.metadata)

    if any(k.endswith(".comfy_quant") for k in reader.keys) or "_quantization_metadata" in metadata:
        raise RuntimeError(f"{label}: source already contains quantized layers — "
                           "select the original bf16/fp16 checkpoint.")

    # Native key names (mirror of the loader's prefix strip).
    pairs = [(k, k) for k in reader.keys]
    try:
        prefix = comfy.model_detection.unet_prefix_from_state_dict(reader.meta_state_dict())
        if prefix:
            stripped = [(k[len(prefix):], k) for k in reader.keys if k.startswith(prefix)]
            if stripped:
                pairs = stripped
    except Exception as ex:
        logging.warning(f"{label}: prefix detection failed ({ex}); keeping key names as-is.")

    sources = [(out_key, reader, src_key, False) for out_key, src_key in pairs]

    if donor is not None:
        if not any("adaln_t_table" in k for k in donor.keys):
            raise RuntimeError(f"{label}: prune_donor has no 'adaln_t_table' — select a *pruned* checkpoint.")
        before = len(sources)
        sources = [s for s in sources if not any(p in s[0] for p in ("adaln_proj", "time_embedder"))]
        dropped = before - len(sources)
        taken = set()
        for k in donor.keys:
            if any(p in k for p in ("adaln_proj", "adaln_t_table", "time_embedder")):
                taken.add(k)
                if k.endswith(".weight"):
                    for side in (".comfy_quant", ".weight_scale"):
                        sk = k[:-len(".weight")] + side
                        if sk in donor.keys:
                            taken.add(sk)
        sources += [(k, donor, k, True) for k in sorted(taken)]
        metadata = {**metadata, **dict(donor.metadata)}
        metadata.pop("_quantization_metadata", None)
        table_key = next((k for k in taken if "adaln_t_table" in k), None)
        if "config" in metadata and table_key is not None:
            try:
                conf_json = json.loads(metadata["config"])
                tr = conf_json.get("transformer", {})
                tshape = donor.shape(table_key)
                tr["adaln_curve_grid"] = int(tshape[0])
                tr["time_embed_dim"] = int(tshape[1])
                for stale in ("timestep_input_dim", "time_embed_hidden_size"):
                    tr.pop(stale, None)
                conf_json["transformer"] = tr
                metadata["config"] = json.dumps(conf_json)
            except Exception:
                metadata.pop("config", None)
        logging.info(f"{label}: pruned-donor merge — dropped {dropped}, took {len(taken)} tensors "
                     f"+ metadata from the donor.")

    plan, n_fallback, fallback_examples = [], 0, []
    for out_key, rdr, src_key, from_donor in sources:
        shape = rdr.shape(src_key)
        dtype = rdr.dtype(src_key)
        e = {"out_key": out_key, "reader": rdr, "src_key": src_key, "shape": shape,
             "itemsize": rdr.itemsize(src_key), "is_weight": out_key.endswith(".weight") and len(shape) >= 2,
             "action": "copy", "spec": None}
        is_lin = (out_key.endswith(".weight") and len(shape) == 2 and dtype in _FLOAT_DTYPES
                  and shape[0] > 1 and shape[1] > 1)
        if from_donor or not is_lin:
            plan.append(e)
            continue
        if any(ex in out_key for ex in excluded):
            e["action"] = "bf16"
            plan.append(e)
            continue
        n, k = int(shape[0]), int(shape[1])
        e["out_base"] = out_key[:-len(".weight")]
        if k % 256 == 0 and k % int(group) == 0:
            e["action"] = "grouped"
            e["spec"] = grouped_spec(fmt, format_name, k, n, group)
        else:
            e["action"] = "int8"
            e["spec"] = int8_spec(k, n)
            n_fallback += 1
            if len(fallback_examples) < 4:
                fallback_examples.append(f"{e['out_base']} (K={k}, cg={e['spec']['cg']})")
        plan.append(e)
    if n_fallback:
        logging.info(f"{label}: {n_fallback} layers with K not divisible by 256/group use the "
                     f"int8 ConvRot fallback, e.g. {fallback_examples}")
    return plan, metadata


def _track_timesteps(session, specs2d, calib_model, positive, negative, latent,
                     steps, cfg, seed, label):
    """Teacher-forced sampling: every model call runs reference + fake-quant on
    the same input; returns per-call noise-variance records and the sigma map."""
    import comfy.sample
    dm = calib_model.model.diffusion_model
    mapping = map_modules(dm, specs2d, want="linear")
    hooked = session.install_linear(mapping)
    logging.info(f"{label}: timestep tracking — {hooked}/{len(specs2d)} quantized layers matched "
                 f"in calib_model")
    if hooked == 0:
        session.remove()
        raise RuntimeError("no calib_model layers matched the converted file — connect the SAME "
                           "model loaded in high precision")
    records, sig_index = [], {}

    def wrapper(apply_model, args):
        x, t, c = args["input"], args["timestep"], args["c"]
        session.quant = False
        session.step = None
        ref = apply_model(x, t, **c)
        sig = round(float(t.flatten()[0]), 6)
        idx = sig_index.setdefault(sig, len(sig_index))
        session.step = idx
        session.quant = True
        try:
            q = apply_model(x, t, **c)
        finally:
            session.quant = False
        with torch.no_grad():
            r = _flat(ref)
            e = _flat(q) - r
            vr = r.var().clamp(min=1e-30)
            ve = e.var()
            records.append({
                "step": idx, "sigma": sig,
                "noise_var": float(ve), "ref_var": float(vr),
                "nsr": float(ve / vr),
                "snr_db": float(10.0 * torch.log10(vr / ve.clamp(min=1e-30))),
                "rel_rmse": float((e.pow(2).mean() / r.pow(2).mean().clamp(min=1e-30)).sqrt()),
                "cos": float(F.cosine_similarity(r + e, r, dim=0)),
            })
        return ref

    m = calib_model.clone()
    m.set_model_unet_function_wrapper(wrapper)
    try:
        latent_img = latent["samples"]
        if hasattr(comfy.sample, "fix_empty_latent_channels"):
            latent_img = comfy.sample.fix_empty_latent_channels(m, latent_img)
        noise = comfy.sample.prepare_noise(latent_img, seed)
        comfy.sample.sample(m, noise, steps, cfg, "euler", "simple", positive, negative,
                            latent_img, denoise=1.0, seed=seed, disable_pbar=True)
    finally:
        session.remove()
    return records, {v: k for k, v in sig_index.items()}


def _log_timesteps(label, records):
    by_step = {}
    for r in records:
        by_step.setdefault(r["step"], []).append(r)
    rows = []
    logging.info(f"{label}: time-step noise variance (quantized vs reference prediction, "
                 f"teacher-forced)")
    logging.info("  step     sigma     noise_var        NSR     SNR dB    rel-RMSE     cos")
    for step in sorted(by_step):
        rs = by_step[step]
        n = len(rs)
        row = {"step": step, "sigma": rs[0]["sigma"], "calls": n}
        for key in ("noise_var", "ref_var", "nsr", "snr_db", "rel_rmse", "cos"):
            row[key] = sum(r[key] for r in rs) / n
        rows.append(row)
        logging.info(f"  {step:4d}  {row['sigma']:8.4f}  {row['noise_var']:12.4e}  {row['nsr']:9.3e}  "
                     f"{row['snr_db']:7.2f}  {row['rel_rmse']:10.4f}  {row['cos']:.6f}")
    if len(rows) >= 2:
        n = len(rows)
        xm = (n - 1) / 2.0
        ym = sum(r["nsr"] for r in rows) / n
        den = sum((i - xm) ** 2 for i in range(n)) or 1.0
        slope = sum((i - xm) * (r["nsr"] - ym) for i, r in enumerate(rows)) / den
        logging.info(f"{label}: NSR trend {slope:+.3e} per step "
                     f"({'rising toward low noise' if slope > 0 else 'falling toward low noise'})")
    return rows


class _DiffusionGroupedBase:
    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    BITS = 4

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def _inputs(cls, group_choices, group_default, extra, default_format, default_prefix):
        req = {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "High-precision (bf16/fp16) source model."}),
            "model_type": (list(MODEL_TYPE_EXCLUSIONS.keys()), {"tooltip": "Sensitive layers kept at source precision."}),
            "prune_donor": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "MiniMax H3 pruned conversion: existing *_pruned_* checkpoint supplying the adaln curve tensors + config metadata; the source's full-width modulation/time_embedder are dropped."}),
            "group_size": (group_choices, {"default": group_default}),
            "scale_dtype": (["fp8_e4m3", "fp32"], {"default": "fp8_e4m3", "tooltip": "Group-scale storage. fp8 = kitchen default and required for the fastest fused GEMM."}),
        }
        req.update(extra)
        req.update({
            "convrot_groupsize": (["256"], {"default": "256", "tooltip": "Fixed at 256 for the grouped format (kitchen fused kernels). Layers with K not divisible by 256 use int8 ConvRot with group 64/16."}),
            "format_name": ("STRING", {"default": default_format, "tooltip": "Per-layer 'format' string in _quantization_metadata (must match ComfyUI's QUANT_ALGOS)."}),
            "measure_error": (["weights", "off", "weights+timesteps"], {"default": "weights", "tooltip": "weights: per-layer weight error (no extra inputs). weights+timesteps: also a short teacher-forced sampling run with the calib_* inputs — per-step noise variance, per-layer per-step error, Hessian-weighted distances."}),
            "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Print the per-layer plan and projected size, write nothing."}),
            "filename_prefix": ("STRING", {"default": default_prefix}),
            "calib_steps": ("INT", {"default": 4, "min": 1, "max": 50, "tooltip": "Sampling steps for timestep tracking."}),
            "calib_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "calib_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
        })
        return {"required": req, "optional": {
            "calib_model": ("MODEL", {"tooltip": "For weights+timesteps: the SAME checkpoint loaded in high precision."}),
            "calib_positive": ("CONDITIONING",),
            "calib_negative": ("CONDITIONING",),
            "calib_latent": ("LATENT",),
        }}

    def _run(self, unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
             scale_search, format_name, measure_error, dry_run, filename_prefix,
             calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
             calib_negative, calib_latent):
        bits = self.BITS
        fmt = FORMAT_W4A8 if bits == 4 else FORMAT_W6A8
        label = "W4A8 Codebook" if bits == 4 else "W6A8"
        group = int(group_size)
        if bits == 6 and (group < 16 or group % 16 != 0):
            raise RuntimeError(f"{label}: group_size must be a multiple of 16, got {group}.")
        s_dtype = torch.float8_e4m3fn if scale_dtype == "fp8_e4m3" else torch.float32
        device = comfy.model_management.get_torch_device()

        reader = SourceReader(folder_paths.get_full_path("diffusion_models", unet_name))
        donor = (SourceReader(folder_paths.get_full_path("diffusion_models", prune_donor))
                 if prune_donor and prune_donor != "None" else None)
        plan, metadata = _build_plan(reader, donor, model_type, fmt, format_name, group, label)

        src_bytes = sum(reader.itemsize(k) * int(torch.Size(reader.shape(k)).numel()) for k in reader.keys)
        print_plan(plan, label, 1 if s_dtype != torch.float32 else 4, src_bytes)
        if dry_run:
            logging.info(f"{label}: dry run — nothing written.")
            return {}

        kitchen = kitchen_support(bits)
        want_ts = measure_error == "weights+timesteps"
        if want_ts and any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
            logging.warning(f"{label}: weights+timesteps needs calib_model/positive/negative/latent — "
                            "falling back to weight error only.")
            want_ts = False

        full_folder, filename, counter, _sub, _pref = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)
        out_path = os.path.join(full_folder, f"{filename}_{counter:05}_.safetensors")
        writer = StreamingSafetensorsWriter(out_path)
        try:
            qmeta, records, specs, decoder_ok = execute_plan(
                plan, writer, device, label, conf_style="header", kitchen=kitchen,
                grouped_opts={"scale_dtype": s_dtype, "codebook": codebook, "scale_search": scale_search},
                measure_weights=measure_error != "off")
            metadata["_quantization_metadata"] = json.dumps({"layers": dict(sorted(qmeta.items()))})
            writer.finalize(metadata)
        except BaseException:
            writer.abort()
            raise
        logging.info(f"{label}: wrote {out_path}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        report = {"converter": label, "source": unet_name, "model_type": model_type,
                  "format": format_name, "group_size": group, "scale_dtype": scale_dtype,
                  "codebook": bool(codebook) if bits == 4 else None,
                  "scale_search": bool(scale_search) if bits == 6 else None,
                  "prune_donor": prune_donor if donor is not None else None,
                  "measure_error": measure_error, "layers": records}

        if want_ts:
            try:
                session = FakeQuantSession(out_path, specs, decoder_ok)
                specs2d = {b: s for b, s in specs.items() if len(s["shape"]) == 2}
                ts_records, step_sigma = _track_timesteps(
                    session, specs2d, calib_model, calib_positive, calib_negative, calib_latent,
                    calib_steps, calib_cfg, calib_seed, label)
                report["timesteps"] = _log_timesteps(label, ts_records)
                for base, s in summarize_local(session.local).items():
                    if s.get("worst_step") is not None:
                        s["worst_sigma"] = step_sigma.get(s["worst_step"])
                    records.setdefault(base, {})["activation"] = s
                for base, h in session.hessian_metrics(device).items():
                    records.setdefault(base, {})["hessian"] = h
            except Exception as ex:
                logging.warning(f"{label}: timestep tracking failed: {ex}")

        if measure_error != "off":
            log_summary(label, records, key="weight")
            if any(r.get("activation") for r in records.values()):
                log_summary(label, records, key="activation")
            write_report(out_path, report)
        return {}


class W4A8CodebookSave(_DiffusionGroupedBase):
    """bf16/fp16 diffusion model -> W4A8 codebook (asym_w4a8_int8)."""
    BITS = 4
    DESCRIPTION = ("Convert a diffusion model to W4A8 codebook (comfy-kitchen AsymW4A8Int8Layout): "
                   "4-bit storage at ~int8 speed. Kijai-style metadata, streamed to disk.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(["16", "32", "64"], "16", {
            "codebook": ("BOOLEAN", {"default": True, "tooltip": "Per-tensor Lloyd-Max 16-level codebook (recommended)."}),
        }, FORMAT_W4A8, "int4_models/W4A8")

    def convert(self, unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
                convrot_groupsize, format_name, measure_error, dry_run, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_seed=0, calib_model=None,
                calib_positive=None, calib_negative=None, calib_latent=None):
        return self._run(unet_name, model_type, prune_donor, group_size, scale_dtype, codebook,
                         True, format_name, measure_error, dry_run, filename_prefix,
                         calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
                         calib_negative, calib_latent)


class W6A8Save(_DiffusionGroupedBase):
    """bf16/fp16 diffusion model -> W6A8 (w6a8_int8)."""
    BITS = 6
    DESCRIPTION = ("Convert a diffusion model to W6A8 (comfy-kitchen PR #191): ~3x lower weight "
                   "error than W4A8 at the same speed, ~0.78 bytes/param. Streamed to disk.")

    @classmethod
    def INPUT_TYPES(cls):
        return cls._inputs(["32", "16", "64"], "32", {
            "scale_search": ("BOOLEAN", {"default": True, "tooltip": "Kitchen's grid-aware group-scale search (better quality, slower conversion, no runtime cost)."}),
        }, FORMAT_W6A8, "int4_models/W6A8")

    def convert(self, unet_name, model_type, prune_donor, group_size, scale_dtype, scale_search,
                convrot_groupsize, format_name, measure_error, dry_run, filename_prefix,
                calib_steps=4, calib_cfg=1.0, calib_seed=0, calib_model=None,
                calib_positive=None, calib_negative=None, calib_latent=None):
        return self._run(unet_name, model_type, prune_donor, group_size, scale_dtype, False,
                         scale_search, format_name, measure_error, dry_run, filename_prefix,
                         calib_steps, calib_cfg, calib_seed, calib_model, calib_positive,
                         calib_negative, calib_latent)

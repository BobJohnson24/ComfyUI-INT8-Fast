"""INT W4A4 + 20% 8-bit mixed conversion (ConvRot, arXiv:2512.03673).

Converts a high-precision diffusion model into ComfyUI's NATIVE quantized
checkpoint format so the result loads with the stock "Load Diffusion Model"
node — no custom nodes required at inference time:

- ~80% of linear layers  -> ``convrot_w4a4``  (ComfyUI PR #14859, comfy-kitchen
  TensorCoreConvRotW4A4Layout: regular-Hadamard rotation + group-64 INT4
  weights, INT4/INT8 tensor-core matmul via ``linear_dtype``)
- ~20% of linear layers  -> ``int8_tensorwise`` (ComfyUI PR #14636, per-row
  INT8 weights, optional ConvRot) — the paper's mixed-precision strategy
  (Appendix 10) that restores fine detail lost under full INT4.
- Model-specific sensitive layers (embedders, modulation/adaLN, final layers)
  stay at the source precision, same lists as UNetLoaderINTW8A8.

Layer selection for the 8-bit subset ("selection" input):
  structural  - first/last transformer blocks and attention-out / MLP-down
                projections get the INT8 budget, ends-inward (default; mirrors
                the paper's "empirically selected" strategy, no calibration)
  calibrated  - short sampling pass over the connected calib_* inputs with
                forward hooks measuring the INT4 quantization error of the
                *rotated activations* per layer; combined with weight error.
                Most accurate: W4A4 error is dominated by activations, which
                the weight-only proxy cannot see.
  sensitivity - weight-only INT4 quantization error after rotation
  random      - random subset (paper's Table 2 caption)

Per-layer metadata is written as ``<layer>.comfy_quant`` uint8-JSON tensors,
exactly matching ``comfy.ops._quantized_weight_state_dict``:
  W4A4: {"format": "convrot_w4a4", "convrot_groupsize": N[, "linear_dtype": "int8"]}
        (quant_group_size is fixed to 64 by the format and never serialized)
  INT8: {"format": "int8_tensorwise", "convrot": bool[, "convrot_groupsize": N]}
"""

import json
import logging
import os
import re

import torch

import folder_paths
import comfy.utils
import comfy.model_management
import comfy.model_detection

from .convrot import build_hadamard, rotate_weight, rotate_activation
from .int8_quant import quantize_int8_axiswise
from .int8_unet_loader import MODEL_TYPE_EXCLUSIONS

# Fixed by ComfyUI's convrot_w4a4 format (PR #14859): int4 weights are
# group-quantized with a group size of 64 and this value is intentionally
# NOT serialized into comfy_quant.
W4A4_QUANT_GROUP_SIZE = 64

# Power-of-4 rotation sizes accepted by the regular Hadamard construction,
# largest first, used as fallbacks when in_features is not divisible by the
# requested groupsize.
_ROT_SIZES = (1024, 256, 64, 16)


def _native_w4a4_support():
    """Return (QuantizedTensor, True) if this ComfyUI ships the convrot_w4a4
    kitchen layout (PR #14859 / comfy-kitchen >= 0.2.17), else (None, False)."""
    try:
        from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS
        if "convrot_w4a4" in QUANT_ALGOS:
            return QuantizedTensor, True
    except Exception:
        pass
    return None, False


def _pick_rot_groupsize(in_features, requested):
    """Largest power-of-4 groupsize <= requested that divides in_features."""
    for size in _ROT_SIZES:
        if size <= requested and in_features % size == 0:
            return size
    return None


def _is_linear_weight(key, tensor):
    return (
        key.endswith(".weight")
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tensor.is_floating_point()
        and tensor.shape[0] > 1
        and tensor.shape[1] > 1
    )


@torch.no_grad()
def _w4a4_sensitivity(weight, rot_groupsize, device):
    """Relative INT4 group-quantization error of a rotated weight.

    Cheap weight-only proxy for layer sensitivity: rotate with the regular
    Hadamard (as convrot_w4a4 does), symmetric group-64 INT4 quantize,
    return ||W_q - W|| / ||W||. Higher = more damaged by INT4.
    """
    w = weight.to(device=device, dtype=torch.float32, non_blocking=True)
    H = build_hadamard(rot_groupsize, device=w.device, dtype=w.dtype)
    w = rotate_weight(w, H, group_size=rot_groupsize)

    out_f, in_f = w.shape
    g = w.view(out_f, in_f // W4A4_QUANT_GROUP_SIZE, W4A4_QUANT_GROUP_SIZE)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    deq = (g / scale).round_().clamp_(-8, 7).mul_(scale)
    err = (deq - g).pow_(2).sum()
    denom = g.pow(2).sum().clamp(min=1e-12)
    return (err / denom).sqrt().item()


# ---------------------------------------------------------------------------
# Structural selection
# ---------------------------------------------------------------------------

# Matches "<blocks_name>.<index>." anywhere in a key, e.g. "layers.0.",
# "double_blocks.18.", "transformer_blocks.46.".
_BLOCK_RE = re.compile(r'(?:^|\.)([A-Za-z_]+)\.(\d+)\.')

# Substrings identifying attention output projections and MLP down projections
# — the linears that sum many channels, where int4 activation outliers hurt
# most. Covers z-image/Lumina (attention.out, feed_forward.w2), Flux/Chroma/
# Krea (attn.proj, mlp.2, single-block linear2), Qwen/SDXL-style (to_out,
# ff.net.2), Wan (self_attn.o, cross_attn.o, ffn.2).
_OUT_PROJ_MARKERS = (
    'attention.out', 'attn.proj', 'attn.to_out', 'to_out.', 'attn.o.',
    'self_attn.o', 'cross_attn.o', 'feed_forward.w2', 'mlp.fc2', 'mlp.2.',
    'linear2.', 'ff.net.2', 'ffn.2', 'proj_mlp',
)


def _structural_scores(candidates):
    """Heuristic INT8-priority score per layer, no calibration needed.

    Layers in the first/last transformer blocks touch the least-redundant
    representations, so int4 error there is not averaged away downstream;
    out/down projections accumulate the channel sums where rotated-activation
    outliers concentrate. Score = ends-inward block proximity + out-proj bonus.
    """
    group_max = {}
    infos = {}
    for key, _ in candidates:
        m = _BLOCK_RE.search(key)
        if m:
            grp, idx = m.group(1), int(m.group(2))
            group_max[grp] = max(group_max.get(grp, 0), idx)
            infos[key] = (grp, idx)
        else:
            infos[key] = None

    scores = {}
    for key, _ in candidates:
        info = infos[key]
        if info is None:
            dist = 1  # un-indexed linear: mid priority
        else:
            grp, idx = info
            dist = min(idx, group_max[grp] - idx)  # blocks from nearest end
        is_out = any(mk in key for mk in _OUT_PROJ_MARKERS)
        scores[key] = 2.0 / (1.0 + dist) + (1.0 if is_out else 0.0)
    return scores


# ---------------------------------------------------------------------------
# Activation calibration
# ---------------------------------------------------------------------------

def _strip_model_prefix(name):
    for p in ("model.diffusion_model.", "diffusion_model.", "model."):
        if name.startswith(p):
            return name[len(p):]
    return name


@torch.no_grad()
def _rotated_act_int4_relerr(x, rot_gs):
    """Relative INT4 group-64 quantization error of a rotated activation."""
    C = x.shape[-1]
    x2 = x.reshape(-1, C)
    if x2.shape[0] > 2048:  # cap cost per hook call
        x2 = x2[:2048]
    x2 = x2.float()
    H = build_hadamard(rot_gs, device=x2.device, dtype=x2.dtype)
    xr = rotate_activation(x2, H, group_size=rot_gs)
    g = xr.view(x2.shape[0], C // W4A4_QUANT_GROUP_SIZE, W4A4_QUANT_GROUP_SIZE)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    deq = (g / scale).round_().clamp_(-8, 7).mul_(scale)
    num = (deq - g).pow_(2).sum()
    den = g.pow(2).sum().clamp(min=1e-12)
    return (num / den).sqrt().item()


def _calibrate_activation_errors(model, positive, negative, latent, steps, cfg,
                                 seed, candidate_map):
    """Run a short euler sampling pass with forward-pre-hooks on every
    candidate linear, recording the mean INT4 quantization error of its
    rotated input activations. Returns {sd_key: mean_rel_err}."""
    import comfy.sample

    stats = {}   # key -> [sum, count]
    handles = []

    def make_hook(key, rot_gs):
        def hook(module, args):
            x = args[0] if args else None
            if not isinstance(x, torch.Tensor) or x.shape[-1] % rot_gs != 0:
                return
            try:
                rel = _rotated_act_int4_relerr(x, rot_gs)
            except Exception:
                return
            s = stats.setdefault(key, [0.0, 0])
            s[0] += rel
            s[1] += 1
        return hook

    hooked = 0
    for name, module in model.model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = _strip_model_prefix(name) + ".weight"
        rot_gs = candidate_map.get(key)
        if rot_gs is None:
            continue
        handles.append(module.register_forward_pre_hook(make_hook(key, rot_gs)))
        hooked += 1

    if hooked == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(
            "INT4 Mixed Save: calibration model layers do not match the selected "
            "checkpoint (0 hooks placed). Connect the SAME model file, loaded in "
            "high precision with the stock 'Load Diffusion Model' node."
        )

    logging.info(f"INT4 Mixed Save: calibrating {hooked} layers over {steps} steps...")
    try:
        latent_img = latent["samples"]
        if hasattr(comfy.sample, "fix_empty_latent_channels"):
            latent_img = comfy.sample.fix_empty_latent_channels(model, latent_img)
        noise = comfy.sample.prepare_noise(latent_img, seed)
        comfy.sample.sample(
            model, noise, steps, cfg, "euler", "simple",
            positive, negative, latent_img,
            denoise=1.0, seed=seed, disable_pbar=True,
        )
    finally:
        for h in handles:
            h.remove()

    return {k: s / max(1, n) for k, (s, n) in stats.items()}


class INT4W4A4MixedSave:
    """Offline converter: bf16/fp16 diffusion model -> native ComfyUI
    convrot_w4a4 + int8_tensorwise mixed checkpoint."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "High-precision (bf16/fp16/fp32) source model to convert."}),
                "model_type": (list(MODEL_TYPE_EXCLUSIONS.keys()), {"tooltip": "Selects which sensitive layers (embedders, modulation, final layers) are kept at source precision."}),
                "int8_ratio": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of quantizable linear layers kept at INT8 (W8A8) instead of INT4. 0.20 reproduces the paper's 'INT W4A4 +20% 8bit Mixed' setting."}),
                "selection": (["structural", "calibrated", "sensitivity", "random"], {"default": "structural", "tooltip": "How the INT8 subset is chosen. 'structural': first/last blocks + attention-out/MLP-down projections, ends-inward (recommended, no calibration). 'calibrated': short sampling pass measuring per-layer INT4 activation error — requires the calib_* inputs. 'sensitivity': weight-only INT4 error. 'random': paper's random subset."}),
                "linear_dtype": (["int4", "int8"], {"default": "int4", "tooltip": "Matmul dtype for the W4A4 layers. int4 uses int4 tensor cores where supported (fastest); int8 does the matrix multiplication in int8 — activations are quantized to 8 bits instead of 4, much higher quality at the same 4-bit storage."}),
                "int8_mm_ratio": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Middle tier: fraction of layers that stay 4-bit storage but run the matmul in int8 (linear_dtype=int8). Applied to the next-most-sensitive layers after the full-INT8 subset. Ignored when linear_dtype is already int8."}),
                "convrot_groupsize": (["256", "1024", "64"], {"default": "256", "tooltip": "Regular Hadamard rotation group size (power of 4). Falls back per-layer to the largest divisor of in_features."}),
                "int8_convrot": ("BOOLEAN", {"default": True, "tooltip": "Also apply ConvRot rotation to the INT8 layers (recommended)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Seed for 'random' selection and calibration noise."}),
                "filename_prefix": ("STRING", {"default": "int4_models/INT4_W4A4_Mixed"}),
                "calib_steps": ("INT", {"default": 4, "min": 1, "max": 50, "tooltip": "Sampling steps for 'calibrated' selection. 4 covers high and low noise levels."}),
                "calib_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "CFG for the calibration pass. Use your normal value for the model."}),
            },
            "optional": {
                "calib_model": ("MODEL", {"tooltip": "For 'calibrated' selection: the SAME checkpoint loaded in high precision with the stock 'Load Diffusion Model' node."}),
                "calib_positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the calibration pass. Use a representative prompt."}),
                "calib_negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the calibration pass."}),
                "calib_latent": ("LATENT", {"tooltip": "Latent (e.g. Empty Latent Image at your usual resolution) for the calibration pass."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "loaders"
    DESCRIPTION = "Convert a model to INT W4A4 + INT8 mixed precision (ConvRot, arXiv:2512.03673) in ComfyUI's native quantized format."

    def convert(self, unet_name, model_type, int8_ratio, selection, linear_dtype,
                convrot_groupsize, int8_convrot, seed, filename_prefix,
                int8_mm_ratio=0.30, calib_steps=4, calib_cfg=3.0, calib_model=None,
                calib_positive=None, calib_negative=None, calib_latent=None):
        QuantizedTensor, native_ok = _native_w4a4_support()
        if not native_ok:
            raise RuntimeError(
                "INT4 Mixed Save: this ComfyUI does not support the 'convrot_w4a4' "
                "quant format. Update ComfyUI (needs PR #14859 / comfy-kitchen >= 0.2.17)."
            )

        requested_gs = int(convrot_groupsize)
        excluded = MODEL_TYPE_EXCLUSIONS.get(model_type, [])
        device = comfy.model_management.get_torch_device()

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)

        # ---- Pass 0: normalize keys to ComfyUI-native names -------------------
        # The stock loader remaps prefixed/diffusers-style keys at load time.
        # Our sidecar keys (.comfy_quant / .weight_scale) would NOT survive that
        # remap: the int8/int4 weights would load as plain tensors and produce
        # black images. Converting the state dict to native names first makes
        # the loader's remap a no-op. Mirrors comfy.sd.load_diffusion_model_state_dict.
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        if prefix:
            temp_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
            if len(temp_sd) > 0:
                sd = temp_sd
        model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
        if model_config is None:
            model_config = comfy.model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                raise RuntimeError(
                    "INT4 Mixed Save: could not detect the model architecture of "
                    f"'{unet_name}'. The converter needs a checkpoint the stock "
                    "'Load Diffusion Model' node can load."
                )
            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
            if len(sd) > 0:
                logging.warning(
                    f"INT4 Mixed Save: {len(sd)} keys had no native mapping and were "
                    "dropped (the stock loader ignores them at load time too)."
                )
            sd = new_sd
            logging.info(
                "INT4 Mixed Save: converted diffusers-style key names to "
                "ComfyUI-native names before quantization."
            )

        if any(k.endswith(".comfy_quant") for k in sd) or any(
            isinstance(t, torch.Tensor) and t.dtype == torch.int8 for t in sd.values()
        ):
            raise RuntimeError(
                "INT4 Mixed Save: source model already contains quantized layers. "
                "Select the original high-precision (bf16/fp16) checkpoint."
            )

        # ---- Pass 1: collect candidates -------------------------------------
        candidates = []  # (key, rot_groupsize)
        for key, tensor in sd.items():
            if not _is_linear_weight(key, tensor):
                continue
            if any(ex in key for ex in excluded):
                continue
            rot_gs = _pick_rot_groupsize(tensor.shape[1], requested_gs)
            if rot_gs is None or tensor.shape[1] % W4A4_QUANT_GROUP_SIZE != 0:
                continue  # can't rotate/group-quantize -> stays high precision
            candidates.append((key, rot_gs))

        if not candidates:
            raise RuntimeError("INT4 Mixed Save: no quantizable linear layers found.")

        # ---- Pass 2: rank layers most-sensitive-first, slice into tiers ------
        # Tier 1 (worst):        int8_tensorwise            (int8_ratio)
        # Tier 2 (next worst):   convrot_w4a4 + int8 matmul (int8_mm_ratio)
        # Tier 3 (rest):         convrot_w4a4 + int4 matmul
        n_int8 = max(0, min(int(round(int8_ratio * len(candidates))), len(candidates)))

        if selection == "random":
            gen = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(candidates), generator=gen).tolist()
            ranked_keys = [candidates[i][0] for i in order]
        elif selection == "structural":
            struct = _structural_scores(candidates)
            ranked_keys = [k for k, _ in sorted(candidates, key=lambda c: (-struct[c[0]], c[0]))]
        elif selection == "calibrated":
            if any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
                raise RuntimeError(
                    "INT4 Mixed Save: 'calibrated' selection needs calib_model, "
                    "calib_positive, calib_negative and calib_latent connected. "
                    "Load the same checkpoint with the stock 'Load Diffusion Model' "
                    "node and wire it in, or switch selection to 'structural'."
                )
            act_err = _calibrate_activation_errors(
                calib_model, calib_positive, calib_negative, calib_latent,
                calib_steps, calib_cfg, seed, dict(candidates),
            )
            observed = sorted(act_err.values())
            fallback = observed[len(observed) // 2] if observed else 1.0
            scores = []
            for key, rot_gs in candidates:
                w_err = _w4a4_sensitivity(sd[key], rot_gs, device)
                scores.append((act_err.get(key, fallback) * w_err, key))
            scores.sort(reverse=True)
            ranked_keys = [k for _, k in scores]
            worst = ", ".join(ranked_keys[:min(8, max(1, n_int8))])
            logging.info(f"INT4 Mixed Save: calibration ranking (worst first): {worst} ...")
        else:  # sensitivity (weight-only)
            logging.info(f"INT4 Mixed Save: scoring {len(candidates)} layers for INT4 sensitivity...")
            scores = []
            for key, rot_gs in candidates:
                scores.append((_w4a4_sensitivity(sd[key], rot_gs, device), key))
            scores.sort(reverse=True)
            ranked_keys = [k for _, k in scores]

        int8_keys = set(ranked_keys[:n_int8])
        if linear_dtype == "int8":
            int8mm_keys = set(ranked_keys[n_int8:])  # global override: all W4A4 use int8 mm
        else:
            n_mm = max(0, min(int(round(int8_mm_ratio * len(candidates))), len(candidates) - n_int8))
            int8mm_keys = set(ranked_keys[n_int8:n_int8 + n_mm])

        # ---- Pass 3: quantize -------------------------------------------------
        out_sd = {}
        candidate_map = dict(candidates)
        pbar = comfy.utils.ProgressBar(len(candidates))
        n_w4a4 = n_int8_done = n_w4a4_i8mm = 0

        for key, tensor in sd.items():
            if key not in candidate_map:
                out_sd[key] = tensor
                continue

            base = key[: -len(".weight")]
            rot_gs = candidate_map[key]

            if key in int8_keys:
                # --- INT8 tensorwise (per-row), optionally ConvRot-rotated ---
                w = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
                use_convrot = bool(int8_convrot)
                if use_convrot:
                    H = build_hadamard(rot_gs, device=w.device, dtype=w.dtype)
                    w = rotate_weight(w, H, group_size=rot_gs)
                q_weight, q_scale = quantize_int8_axiswise(w, dim=1)
                out_sd[key] = q_weight.cpu()
                out_sd[base + ".weight_scale"] = q_scale.cpu()
                quant_conf = {"format": "int8_tensorwise", "convrot": use_convrot}
                if use_convrot:
                    quant_conf["convrot_groupsize"] = rot_gs
                del w, q_weight, q_scale
                n_int8_done += 1
            else:
                # --- ConvRot W4A4 via ComfyUI's native kitchen layout ---
                w = tensor.to(device=device, non_blocking=True)
                if w.dtype == torch.float32:
                    w = w.to(torch.bfloat16)
                qt = QuantizedTensor.from_float(
                    w,
                    "TensorCoreConvRotW4A4Layout",
                    convrot_groupsize=rot_gs,
                    quant_group_size=W4A4_QUANT_GROUP_SIZE,
                )
                out_sd[key] = qt._qdata.cpu()
                out_sd[base + ".weight_scale"] = qt._params.scale.cpu()
                quant_conf = {"format": "convrot_w4a4", "convrot_groupsize": rot_gs}
                if key in int8mm_keys:
                    quant_conf["linear_dtype"] = "int8"
                    n_w4a4_i8mm += 1
                del w, qt
                n_w4a4 += 1

            out_sd[base + ".comfy_quant"] = torch.tensor(
                list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8
            )
            pbar.update(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Save -------------------------------------------------------------
        out_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        out_metadata["convrot_w4a4_mixed"] = json.dumps({
            "int8_ratio": int8_ratio,
            "int8_mm_ratio": int8_mm_ratio,
            "selection": selection,
            "linear_dtype": linear_dtype,
            "convrot_groupsize": requested_gs,
            "w4a4_int4mm_layers": n_w4a4 - n_w4a4_i8mm,
            "w4a4_int8mm_layers": n_w4a4_i8mm,
            "int8_layers": n_int8_done,
            "paper": "arXiv:2512.03673",
        })
        # Record which layers got which tier so runs are comparable.
        out_metadata["int8_selected_layers"] = json.dumps(sorted(int8_keys))
        out_metadata["int8mm_selected_layers"] = json.dumps(sorted(int8mm_keys))

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        output_checkpoint = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.safetensors"
        )

        comfy.utils.save_torch_file(out_sd, output_checkpoint, metadata=out_metadata)
        logging.info(
            f"INT4 Mixed Save: wrote {output_checkpoint} "
            f"({n_w4a4 - n_w4a4_i8mm} w4a4-int4mm + {n_w4a4_i8mm} w4a4-int8mm + "
            f"{n_int8_done} int8_tensorwise layers, "
            f"{len(candidates) - n_w4a4 - n_int8_done} high-precision). "
            "Load it with the stock 'Load Diffusion Model' node."
        )
        return {}

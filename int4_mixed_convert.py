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


# ---------------------------------------------------------------------------
# calibrated+ : per-layer tier decisions from measured output error
# ---------------------------------------------------------------------------

# Error-threshold presets: (rel_rmse_int4mm, cos_int4mm, rel_rmse_int8mm, cos_int8mm)
# A layer gets the cheapest tier whose simulated output stays within thresholds;
# layers failing both go to full int8_tensorwise. Strictest -> loosest:
# QualityQ > Quality > QualityS ~ BalancedQ > Balanced > BalancedS.
_CALIB_PLUS_PRESETS = {
    "QualityQ":  (0.035, 0.9985, 0.070, 0.9930),
    "Quality":   (0.050, 0.9975, 0.100, 0.9900),
    "QualityS":  (0.070, 0.9960, 0.140, 0.9860),
    "BalancedQ": (0.070, 0.9930, 0.130, 0.9820),
    "Balanced":  (0.100, 0.9900, 0.180, 0.9750),
    "BalancedS": (0.140, 0.9850, 0.250, 0.9650),
}

_CALIB_PLUS_MAX_SIMS = 12   # per-layer cap on simulated forward measurements
_CALIB_PLUS_TOKENS = 256    # activation rows sampled per measurement


@torch.no_grad()
def _output_metrics(y, ref):
    """Quality metrics of a simulated quantized output vs the bf16 reference.
    Mirrors the Metrics.md suite at per-layer scope."""
    err = y - ref
    ref_pow = ref.pow(2).mean().clamp(min=1e-20)
    mse = err.pow(2).mean()
    denom = y.norm().clamp(min=1e-12) * ref.norm().clamp(min=1e-12)
    ch = err.pow(2).mean(dim=0)
    return {
        "mse": mse.item(),
        "mae": err.abs().mean().item(),
        "max": err.abs().max().item(),
        "rel": (mse / ref_pow).sqrt().item(),
        "snr": (10.0 * torch.log10(ref_pow / mse.clamp(min=1e-20))).item(),
        "cos": ((y.flatten() @ ref.flatten()) / denom).item(),
        "var": (y.var() / ref.var().clamp(min=1e-20)).item(),
        "outl": (err.abs() > 0.1 * ref.std()).float().mean().item(),
        "chmax": ch.max().item(),
        "chstd": ch.std().item(),
        # Worst per-channel RMSE relative to the LAYER-wide signal RMS —
        # catches localized channel damage that averages hide. Normalized
        # against the layer scale, NOT per-channel power: near-silent
        # channels would otherwise inflate the ratio and veto every layer
        # (observed on z-image, where the old gate blocked all int4).
        "chrel": (ch / ref_pow).max().sqrt().item(),
    }


@torch.no_grad()
def _tier_sim_metrics(x, w, rot_gs):
    """Simulate all three tiers for one linear on real activations.

    x: (tokens, in) float32; w: (out, in) float32, same device.
    Tiers: int8 (per-row W8 + per-token A8), int8mm (group-64 W4 + A8),
    int4mm (group-64 W4 + group-64 A4). Weight is processed in row chunks to
    bound memory. Returns {"int8": {...}, "int8mm": {...}, "int4mm": {...}}.
    """
    device = x.device
    H = build_hadamard(rot_gs, device=device, dtype=torch.float32) if rot_gs else None
    xr = rotate_activation(x, H, group_size=rot_gs) if H is not None else x

    # Activation quants (per token)
    a_scale8 = xr.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    xa8 = (xr / a_scale8).round_().clamp_(-127, 127).mul_(a_scale8)
    t, c = xr.shape
    gact = xr.view(t, c // W4A4_QUANT_GROUP_SIZE, W4A4_QUANT_GROUP_SIZE)
    a_scale4 = gact.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    xa4 = (gact / a_scale4).round_().clamp_(-8, 7).mul_(a_scale4).view(t, c)

    out_f, in_f = w.shape
    y_ref = torch.empty((t, out_f), dtype=torch.float32, device=device)
    y8 = torch.empty_like(y_ref)
    y48 = torch.empty_like(y_ref)
    y44 = torch.empty_like(y_ref)

    chunk = max(256, (1 << 24) // max(in_f * 4, 1))  # ~16MB fp32 of W rows
    for r0 in range(0, out_f, chunk):
        wc = w[r0:r0 + chunk]
        wr = rotate_weight(wc, H, group_size=rot_gs) if H is not None else wc
        # per-row INT8 weight
        s8 = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        w8 = (wr / s8).round_().clamp_(-127, 127).mul_(s8)
        # group-64 INT4 weight
        gw = wr.view(wr.shape[0], in_f // W4A4_QUANT_GROUP_SIZE, W4A4_QUANT_GROUP_SIZE)
        s4 = gw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        w4 = (gw / s4).round_().clamp_(-8, 7).mul_(s4).view(wr.shape[0], in_f)

        y_ref[:, r0:r0 + chunk] = xr @ wr.T
        y8[:, r0:r0 + chunk] = xa8 @ w8.T
        y48[:, r0:r0 + chunk] = xa8 @ w4.T
        y44[:, r0:r0 + chunk] = xa4 @ w4.T
        del wc, wr, w8, gw, w4

    return {
        "int8": _output_metrics(y8, y_ref),
        "int8mm": _output_metrics(y48, y_ref),
        "int4mm": _output_metrics(y44, y_ref),
    }


def _calibrate_tier_metrics(model, positive, negative, latent, steps, cfg, seed,
                            candidate_map):
    """calibrated+ measurement pass: hooks simulate all three tiers per layer
    on real activations during a short sampling run.

    Returns {sd_key: {"n": int, "tiers": {tier: [metric dicts in call order]}}}.
    """
    import comfy.sample

    stats = {}
    handles = []

    def make_hook(key, rot_gs):
        def hook(module, args):
            x = args[0] if args else None
            if not isinstance(x, torch.Tensor):
                return
            rec = stats.setdefault(
                key, {"n": 0, "tiers": {"int8": [], "int8mm": [], "int4mm": []}})
            if rec["n"] >= _CALIB_PLUS_MAX_SIMS:
                return
            w = getattr(module, "weight", None)
            if not isinstance(w, torch.Tensor) or w.ndim != 2:
                return
            try:
                x2 = x.reshape(-1, x.shape[-1])
                if x2.shape[0] > _CALIB_PLUS_TOKENS:
                    x2 = x2[:_CALIB_PLUS_TOKENS]
                x2 = x2.float()
                wf = w.to(device=x2.device, dtype=torch.float32)
                res = _tier_sim_metrics(x2, wf, rot_gs)
                del wf
            except Exception as e:
                logging.debug(f"INT4 Mixed Save: calibrated+ sim failed for {key}: {e}")
                return
            rec["n"] += 1
            for tname, m in res.items():
                rec["tiers"][tname].append(m)
        return hook

    hooked = 0
    for name, module in model.model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = _strip_model_prefix(name) + ".weight"
        if key not in candidate_map:
            continue
        handles.append(module.register_forward_pre_hook(make_hook(key, candidate_map[key])))
        hooked += 1

    if hooked == 0:
        for h in handles:
            h.remove()
        raise RuntimeError(
            "INT4 Mixed Save: calibrated+ model layers do not match the selected "
            "checkpoint (0 hooks placed). Connect the SAME model file, loaded in "
            "high precision with the stock 'Load Diffusion Model' node."
        )

    logging.info(
        f"INT4 Mixed Save: calibrated+ measuring {hooked} layers x 3 tiers "
        f"over {steps} steps (this is slower than plain 'calibrated')..."
    )
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

    return stats


def _mean_and_drift(mlist):
    """Aggregate a per-call metric list: mean of every metric plus the linear
    drift slope of MSE and cos-sim across call order (per-step trend)."""
    n = len(mlist)
    mean = {k: sum(m[k] for m in mlist) / n for k in mlist[0]}

    def slope(vals):
        if n < 3:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(vals) / n
        den = sum((i - xm) ** 2 for i in range(n)) or 1.0
        return sum((i - xm) * (v - ym) for i, v in enumerate(vals)) / den

    mean["dmse"] = slope([m["mse"] for m in mlist])
    mean["dcos"] = slope([m["cos"] for m in mlist])
    return mean


def _decide_tiers_calibrated_plus(stats, candidates, preset, layer_params):
    """Per-layer tier assignment from measured metrics, cost-aware.

    Gates per tier: Rel-RMSE, cos-sim, worst per-channel relative RMSE
    (2.5x the layer gate — catches localized channel damage), and MSE drift
    (rising error compounds over real step counts -> promote a tier).

    Small-layer polish: layers under ~0.3% of total candidate params whose
    int4 margin is mediocre are promoted to full INT8, capped at 2% of total
    params — near-free quality for almost no VRAM/speed cost.

    Returns (int8_keys, int8mm_keys); every other candidate runs int4 matmul.
    """
    rel44, cos44, rel48, cos48 = _CALIB_PLUS_PRESETS[preset]
    ch44, ch48 = 2.5 * rel44, 2.5 * rel48
    int8_keys, int8mm_keys = set(), set()
    total_params = max(1, sum(layer_params.values()))

    logging.info(
        f"INT4 Mixed Save: calibrated+ preset '{preset}' gates: "
        f"int4mm relRMSE<={rel44} cos>={cos44} chrel<={ch44:.3f} | "
        f"int8mm relRMSE<={rel48} cos>={cos48} chrel<={ch48:.3f}"
    )
    logging.info(
        "INT4 Mixed Save: per-layer metrics below are int4mm | int8mm | int8, "
        "each 'rel/snr/cos' = Rel-RMSE / SNR dB / cosine-similarity vs bf16."
    )

    polish_pool = []  # (int4 rel margin severity, key) for small-layer polish

    for key, _rot in sorted(candidates):
        rec = stats.get(key)
        if not rec or not rec["tiers"]["int4mm"]:
            int8_keys.add(key)
            logging.info(f"calibrated+  int8    {key}  (no activation samples — safe default)")
            continue

        m44 = _mean_and_drift(rec["tiers"]["int4mm"])
        m48 = _mean_and_drift(rec["tiers"]["int8mm"])
        m8 = _mean_and_drift(rec["tiers"]["int8"])
        n = rec["n"]

        # Drift promotion: rising MSE that would grow to a large fraction of
        # the mean over the measured window means the tier degrades over steps.
        drift44 = m44["dmse"] > 0 and m44["dmse"] * n > 0.5 * m44["mse"]
        drift48 = m48["dmse"] > 0 and m48["dmse"] * n > 0.5 * m48["mse"]

        pass44 = (m44["rel"] <= rel44 and m44["cos"] >= cos44
                  and m44["chrel"] <= ch44 and not drift44)
        pass48 = (m48["rel"] <= rel48 and m48["cos"] >= cos48
                  and m48["chrel"] <= ch48 and not drift48)

        if pass44:
            tier = "int4mm"
            # Mediocre margin on a tiny layer -> candidate for INT8 polish.
            if (layer_params.get(key, 0) / total_params < 0.003
                    and m44["rel"] > 0.5 * rel44):
                polish_pool.append((m44["rel"] / rel44, key))
        elif pass48:
            tier = "int8mm"
            int8mm_keys.add(key)
        else:
            tier = "int8"
            int8_keys.add(key)

        note = " [drift-promoted]" if (tier != "int4mm" and drift44) or (tier == "int8" and drift48) else ""
        logging.info(
            f"calibrated+  {tier:7s} {key}  "
            f"rel {m44['rel']:.3f}/{m48['rel']:.3f}/{m8['rel']:.3f}  "
            f"snr {m44['snr']:.1f}/{m48['snr']:.1f}/{m8['snr']:.1f}dB  "
            f"cos {m44['cos']:.4f}/{m48['cos']:.4f}/{m8['cos']:.4f}  "
            f"mae {m44['mae']:.4f}  max {m44['max']:.3f}  outl {m44['outl']*100:.2f}%  "
            f"chrel {m44['chrel']:.3f}  dMSE/step {m44['dmse']:+.2e}  "
            f"dCos/step {m44['dcos']:+.2e}  (n={n}){note}"
        )

    # Small-layer polish: promote the worst-margin tiny layers to INT8 until
    # the 2%-of-params budget is spent.
    polish_pool.sort(reverse=True)
    budget = 0.02 * total_params
    spent = 0
    for _sev, key in polish_pool:
        cost = layer_params.get(key, 0)
        if spent + cost > budget:
            continue
        spent += cost
        int8_keys.add(key)
        int8mm_keys.discard(key)
        logging.info(f"calibrated+  int8    {key}  [small-layer polish: near-free quality]")

    n_int4 = len(candidates) - len(int8_keys) - len(int8mm_keys)
    int8_params = sum(layer_params.get(k, 0) for k in int8_keys)
    logging.info(
        f"INT4 Mixed Save: calibrated+ decisions — {n_int4} int4mm, "
        f"{len(int8mm_keys)} int8mm, {len(int8_keys)} int8 "
        f"(effective int8_ratio {len(int8_keys)/max(1,len(candidates)):.2f} by count, "
        f"{int8_params/total_params:.2f} by params; "
        f"int8mm_ratio {len(int8mm_keys)/max(1,len(candidates)):.2f})"
    )
    return int8_keys, int8mm_keys


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
                "prune_donor": (["None"] + folder_paths.get_filename_list("diffusion_models"), {"default": "None", "tooltip": "MiniMax H3 pruned conversion: select the existing *_pruned_* checkpoint. Its adaln curve table + basis-form adaln weights and metadata are merged into the output, and the source's full-width modulation + time_embedder weights (~13B params) are dropped. Leave on None for every other model."}),
                "int8_ratio": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of quantizable linear layers kept at INT8 (W8A8) instead of INT4. 0.20 reproduces the paper's 'INT W4A4 +20% 8bit Mixed' setting."}),
                "selection": (["structural", "calibrated", "calibrated+", "sensitivity", "random"], {"default": "structural", "tooltip": "How the INT8 subset is chosen. 'structural': first/last blocks + attention-out/MLP-down projections, ends-inward (no calibration). 'calibrated': short sampling pass ranking layers by INT4 activation error, sliced by the ratio widgets. 'calibrated+': measures every layer's actual output error (MSE/Rel-RMSE/SNR/cos-sim vs bf16) under all three tiers and assigns each layer the cheapest acceptable tier per the calibrated_plus_preset — IGNORES int8_ratio, int8_mm_ratio and linear_dtype. Both calibrated modes need the calib_* inputs. 'sensitivity': weight-only INT4 error. 'random': paper's random subset."}),
                "calibrated_plus_preset": (["Balanced", "BalancedQ", "BalancedS", "Quality", "QualityQ", "QualityS"], {"default": "Balanced", "tooltip": "Error-threshold preset for 'calibrated+'. Quality bases are ~2x stricter than Balanced (more INT8, better fidelity). Q variants tighten thresholds (quality-leaning), S variants loosen them (speed/size-leaning). Ignored for other selection modes."}),
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
                prune_donor="None", int8_mm_ratio=0.30, calibrated_plus_preset="Balanced",
                calib_steps=4, calib_cfg=3.0, calib_model=None,
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

        # ---- Pass 0.5: pruned-donor merge (MiniMax H3 style) ------------------
        # The pruned H3 checkpoints re-parameterize AdaLN: the ~13B full-width
        # modulation weights + time_embedder are replaced by a precomputed
        # adaln_t_table curve basis and small basis-form adaln_proj weights.
        # Those never change with quantization, so we drop the full-width
        # tensors from the bf16 source and take the curve tensors + the config
        # metadata straight from the existing pruned checkpoint.
        if prune_donor and prune_donor != "None":
            donor_path = folder_paths.get_full_path("diffusion_models", prune_donor)
            donor_sd, donor_metadata = comfy.utils.load_torch_file(donor_path, return_metadata=True)
            if not any("adaln_t_table" in k for k in donor_sd):
                raise RuntimeError(
                    "INT4 Mixed Save: prune_donor has no 'adaln_t_table' — select a "
                    "*pruned* checkpoint (e.g. minimax_h3_*_pruned_int8_convrot)."
                )
            drop_patterns = ("adaln_proj", "time_embedder")
            copy_patterns = ("adaln_proj", "adaln_t_table", "time_embedder")
            dropped = [k for k in sd if any(p in k for p in drop_patterns)]
            for k in dropped:
                del sd[k]
            copied = 0
            for k in list(donor_sd.keys()):
                if not any(p in k for p in copy_patterns):
                    continue
                sd[k] = donor_sd[k]
                copied += 1
                if k.endswith(".weight"):
                    base = k[:-len(".weight")]
                    for side in (".comfy_quant", ".weight_scale"):
                        if base + side in donor_sd:
                            sd[base + side] = donor_sd[base + side]
                            copied += 1
            del donor_sd
            if isinstance(donor_metadata, dict) and donor_metadata:
                metadata = {**(metadata if isinstance(metadata, dict) else {}), **donor_metadata}

            # CRITICAL: ComfyUI's H3 detection reads adaln_curve_grid and
            # time_embed_dim from the table SHAPE, but a "config" JSON in the
            # metadata OVERRIDES that afterwards. The full bf16 model's config
            # (time_embed_dim 2688) would rebuild full-width adaln against the
            # curve-form tensors -> size mismatch on adaln_t_table at load.
            # Patch the transformer config to the curve form taken from the
            # donor table's actual shape.
            table_key = next((k for k in sd if "adaln_t_table" in k), None)
            if isinstance(metadata, dict) and "config" in metadata and table_key is not None:
                try:
                    conf_json = json.loads(metadata["config"])
                    tr = conf_json.get("transformer", {})
                    tshape = sd[table_key].shape
                    tr["adaln_curve_grid"] = int(tshape[0])
                    tr["time_embed_dim"] = int(tshape[1])
                    for stale in ("timestep_input_dim", "time_embed_hidden_size"):
                        tr.pop(stale, None)
                    conf_json["transformer"] = tr
                    metadata["config"] = json.dumps(conf_json)
                    logging.info(
                        f"INT4 Mixed Save: patched config metadata for curve-form adaln "
                        f"(grid {int(tshape[0])}, basis {int(tshape[1])})."
                    )
                except Exception as e:
                    logging.warning(
                        f"INT4 Mixed Save: could not patch config metadata ({e}); "
                        "removing it so shape-based detection applies instead."
                    )
                    metadata.pop("config", None)
            if not dropped or copied == 0:
                raise RuntimeError(
                    "INT4 Mixed Save: pruned-donor merge matched nothing — the source "
                    f"dropped {len(dropped)} tensors and the donor supplied {copied}. "
                    "Check that source is the full bf16 and donor the pruned checkpoint "
                    "of the SAME variant (fl2va vs ref2va)."
                )
            logging.info(
                f"INT4 Mixed Save: pruned-donor merge — dropped {len(dropped)} full-width "
                f"modulation/time-embedder tensors from the source, copied {copied} curve "
                f"tensors + metadata from '{prune_donor}'."
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

        # ---- Pass 2: assign tiers ---------------------------------------------
        # calibrated+ : per-layer measured decisions (ignores the ratio widgets)
        # other modes : rank most-sensitive-first, slice by the ratio widgets
        #   Tier 1 (worst):        int8_tensorwise            (int8_ratio)
        #   Tier 2 (next worst):   convrot_w4a4 + int8 matmul (int8_mm_ratio)
        #   Tier 3 (rest):         convrot_w4a4 + int4 matmul
        if selection == "calibrated+":
            if any(x is None for x in (calib_model, calib_positive, calib_negative, calib_latent)):
                raise RuntimeError(
                    "INT4 Mixed Save: 'calibrated+' needs calib_model, calib_positive, "
                    "calib_negative and calib_latent connected. Load the same checkpoint "
                    "with the stock 'Load Diffusion Model' node and wire it in."
                )
            logging.info(
                "INT4 Mixed Save: calibrated+ active — int8_ratio, int8_mm_ratio and "
                "linear_dtype widgets are ignored; tiers come from measured error."
            )
            stats = _calibrate_tier_metrics(
                calib_model, calib_positive, calib_negative, calib_latent,
                calib_steps, calib_cfg, seed, dict(candidates),
            )
            layer_params = {key: sd[key].numel() for key, _ in candidates}
            int8_keys, int8mm_keys = _decide_tiers_calibrated_plus(
                stats, candidates, calibrated_plus_preset, layer_params)
        else:
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
            "calibrated_plus_preset": calibrated_plus_preset if selection == "calibrated+" else None,
            "prune_donor": prune_donor if prune_donor != "None" else None,
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

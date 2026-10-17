"""Shared core for the INT8-ConvRot / W4A8-codebook / W6A8 converters.

Used by the diffusion-model nodes (w4a8_codebook_convert.py) and the text-
encoder / LLM node (te_quant_convert.py).

Formats (all ConvRot-rotated, all run with int8 activations at runtime):
  int8_tensorwise  per-row INT8 weights, fp32 scale [N, 1]; ConvRot group
                   256 / 64 / 16 (largest that divides K)
  asym_w4a8_int8   comfy-kitchen AsymW4A8Int8Layout, 4-bit codes + Lloyd-Max
                   codebook, fp8 group scales (kitchen PR #90)
  w6a8_int8        same layout, uniform 6-bit codes (kitchen PR #191)
  Fused MoE expert banks [E, N, K] are quantized as one [E*N, K] matrix and
  carry "num_experts" in their conf (layout verified against Comfy-Org files).

Error measurement (report only — never changes a layer's format):
  primary      Rel-RMSE, SNR (dB), Outlier %, per-channel MSE (max / std,
               plus worst-channel relative RMSE)
  local check  MSE, MAE, max error — kept in the JSON report only
  Hessian      diagonal input second moment E[x_k^2] (rotated basis, the
               basis the runtime quantizes in) collected during calibration;
               weights the weight-error distance so channels the model
               actually drives count proportionally (OBS/GPTQ-style)
"""

import inspect
import json
import logging
import math
import os
import re
import struct

import torch
import torch.nn.functional as F

FORMAT_INT8 = "int8_tensorwise"
FORMAT_W4A8 = "asym_w4a8_int8"
FORMAT_W6A8 = "w6a8_int8"
GROUPED_BITS = {FORMAT_W4A8: 4, FORMAT_W6A8: 6}
INT8_CONVROT_GROUPS = (256, 64, 16)
GROUPED_CONVROT = 256  # kitchen fused kernels support ConvRot 256 only

_CHUNK_ELEMS_INT8 = 1 << 25      # rows per int8 chunk ~ 32M elems (~128 MB fp32)
_CHUNK_ELEMS_GROUPED = 1 << 26   # elems per kitchen quantize call
_METRIC_MAX_ELEMS = 1 << 25      # weight metrics use an even row sample above this
_LOCAL_RECORDS_PER_LAYER = 64


# =============================================================================
# Regular-Hadamard ConvRot (involutory: H is symmetric and orthogonal)
# =============================================================================

_H_CACHE = {}


def _is_pow4(n):
    n = int(n)
    if n < 4:
        return False
    while n % 4 == 0:
        n //= 4
    return n == 1


def hadamard(size, device, dtype=torch.float32):
    """Normalized regular Hadamard (Kronecker powers of the ConvRot H4) —
    same construction as convrot.build_hadamard, with an integer power-of-4
    check (safe for 16 / 64 / 256 / 1024)."""
    key = (int(size), str(device), dtype)
    H = _H_CACHE.get(key)
    if H is None:
        if not _is_pow4(size):
            raise ValueError(f"ConvRot group must be a power of 4, got {size}")
        H4 = torch.tensor([[1, 1, 1, -1],
                           [1, 1, -1, 1],
                           [1, -1, 1, 1],
                           [-1, 1, 1, 1]], dtype=torch.float64)
        H = H4
        while H.shape[0] < size:
            H = torch.kron(H, H4)
        H = (H / math.sqrt(size)).to(device=device, dtype=dtype)
        _H_CACHE[key] = H
    return H


def rotate_rows(w, cg):
    """Group-wise ConvRot along the last dim. Applying it twice is identity,
    so the same call rotates and un-rotates (weights and activations)."""
    if not cg:
        return w
    shp = w.shape
    k = shp[-1]
    dt = w.dtype if w.dtype in (torch.float32, torch.float64) else torch.float32
    H = hadamard(cg, w.device, dt)
    x = w.reshape(-1, k // cg, cg)
    if x.dtype != dt:
        x = x.to(dt)
    return torch.matmul(x, H).reshape(shp)


def pick_convrot_group(k, groups=INT8_CONVROT_GROUPS):
    for g in groups:
        if k % g == 0:
            return g
    return None


# =============================================================================
# comfy-kitchen capability check
# =============================================================================

def kitchen_support(bits):
    """Return (QuantizedTensor, caps) for grouped formats, or raise."""
    try:
        from comfy.quant_ops import QuantizedTensor, get_layout_class
        layout = get_layout_class("AsymW4A8Int8Layout")
    except Exception:
        layout = None
    if layout is None:
        raise RuntimeError(
            "AsymW4A8Int8Layout not found. Update comfy-kitchen (PR #90) and "
            "ComfyUI (>= v0.31.0).")
    try:
        params = inspect.signature(layout.quantize).parameters
    except (TypeError, ValueError):
        params = {}
    caps = {
        "bits": "bits" in params,
        "scale_search": "scale_search" in params,
        "codebook_tensor": "codebook_tensor" in params,
    }
    if bits == 6:
        if not caps["bits"]:
            raise RuntimeError(
                "This comfy-kitchen has no 6-bit support (AsymW4A8Int8Layout.quantize "
                "lacks 'bits'). Update comfy-kitchen to a build with PR #191.")
        try:
            from comfy.quant_ops import QUANT_ALGOS
            if FORMAT_W6A8 not in QUANT_ALGOS:
                logging.warning(
                    "W6A8: this ComfyUI does not register 'w6a8_int8' (ComfyUI PR #16483). "
                    "Conversion works, but LOADING the output needs a newer ComfyUI.")
        except Exception:
            pass
    return QuantizedTensor, caps


def _kitchen_kwargs(bits, group, scale_dtype, codebook, scale_search, caps,
                    cb_tensor=None, use_cb=None):
    kw = {"group_size": int(group), "convrot_groupsize": GROUPED_CONVROT,
          "symmetric": True, "scale_dtype": scale_dtype}
    if bits == 6:
        kw["bits"] = 6
        kw["codebook"] = False
        if caps.get("scale_search"):
            kw["scale_search"] = bool(scale_search)
    else:
        kw["codebook"] = bool(codebook if use_cb is None else use_cb)
        if caps.get("bits"):
            kw["bits"] = 4
        if cb_tensor is not None:
            kw["codebook_tensor"] = cb_tensor
    return kw


# =============================================================================
# Quantizers
# =============================================================================

@torch.no_grad()
def quantize_int8_convrot(w, cg, device):
    """Per-row INT8 (+ConvRot) of a 2D weight or fused [E, N, K] bank.
    Row-chunked on 'device'. Returns {"weight": int8, "weight_scale": f32 [...,1]}."""
    shp = tuple(w.shape)
    k = shp[-1]
    w2 = w.reshape(-1, k)
    n = w2.shape[0]
    q = torch.empty((n, k), dtype=torch.int8)
    s = torch.empty((n, 1), dtype=torch.float32)
    rows = max(1, _CHUNK_ELEMS_INT8 // k)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        wc = rotate_rows(w2[r0:r1].to(device=device, dtype=torch.float32), cg)
        sc = (wc.abs().amax(dim=1, keepdim=True) / 127.0).clamp_(min=1e-30)
        q[r0:r1] = (wc / sc).round_().clamp_(-128, 127).to(torch.int8).cpu()
        s[r0:r1] = sc.cpu()
        del wc, sc
    return {"weight": q.reshape(shp), "weight_scale": s.reshape(*shp[:-1], 1)}


@torch.no_grad()
def quantize_grouped(w, bits, group, scale_dtype, codebook, scale_search, kitchen, device):
    """W4A8 / W6A8 via kitchen's AsymW4A8Int8Layout. 2D or fused [E, N, K].
    Large tensors are row-chunked; a 4-bit codebook is fitted once on an even
    row sample and shared by every chunk (one codebook per tensor, as the
    format requires)."""
    QuantizedTensor, caps = kitchen
    shp = tuple(w.shape)
    k = shp[-1]
    w2 = w.reshape(-1, k)
    n = w2.shape[0]

    def call(chunk, cb_tensor=None, use_cb=None):
        kw = _kitchen_kwargs(bits, group, scale_dtype, codebook, scale_search, caps,
                             cb_tensor=cb_tensor, use_cb=use_cb)
        x = chunk.to(device=device)
        if x.dtype not in (torch.bfloat16, torch.float16):
            x = x.to(torch.bfloat16)
        qt = QuantizedTensor.from_float(x.contiguous(), "AsymW4A8Int8Layout", **kw)
        p = qt._params
        if getattr(p, "correction", None) is not None:
            raise RuntimeError("asymmetric W4A8 correction is not loadable by ComfyUI")
        cb = getattr(p, "codebook", None)
        out = {"weight": qt._qdata.cpu(),
               "weight_s_rel": p.scale.cpu(),
               "weight_s_channel": p.s_channel.cpu(),
               "codebook": cb.cpu() if cb is not None else None}
        del qt, x
        return out

    rows = max(1, _CHUNK_ELEMS_GROUPED // k)
    if n <= rows or (bits == 4 and codebook and not caps.get("codebook_tensor")):
        if n > rows:
            logging.warning(
                f"Quantize: this comfy-kitchen cannot share a codebook across chunks; "
                f"quantizing a {n}x{k} tensor in one call (needs more VRAM).")
        part = call(w2)
        weight, s_rel, s_ch, cb = part["weight"], part["weight_s_rel"], part["weight_s_channel"], part["codebook"]
    else:
        cb, use_cb = None, None
        if bits == 4 and codebook:
            idx = torch.linspace(0, n - 1, steps=min(n, rows)).round().long()
            probe = call(w2.index_select(0, idx))
            cb = probe["codebook"]
            use_cb = cb is not None
            del probe
        ws, srs, scs = [], [], []
        for r0 in range(0, n, rows):
            part = call(w2[r0:r0 + rows],
                        cb_tensor=cb.to(device) if cb is not None else None, use_cb=use_cb)
            if cb is not None and (part["codebook"] is None
                                   or not torch.allclose(part["codebook"], cb)):
                raise RuntimeError("comfy-kitchen did not reuse the shared codebook across chunks")
            ws.append(part["weight"])
            srs.append(part["weight_s_rel"])
            scs.append(part["weight_s_channel"])
        weight, s_rel, s_ch = torch.cat(ws, 0), torch.cat(srs, 0), torch.cat(scs, 0)

    if weight.shape[1] != k * bits // 8:
        raise RuntimeError(
            f"packed width {weight.shape[1]} != K*{bits}/8 = {k * bits // 8} — "
            "this comfy-kitchen ignored the bit width; update it.")
    out = {"weight": weight.reshape(*shp[:-1], k * bits // 8),
           "weight_s_rel": s_rel.reshape(*shp[:-1], k // int(group)),
           "weight_s_channel": s_ch.reshape(*shp[:-1])}
    if cb is not None:
        out["weight_codebook"] = cb
    return out


def quant_conf(spec):
    """Per-layer conf dict in the key order of Comfy-Org / Kijai checkpoints."""
    if spec["format"] == FORMAT_INT8:
        conf = {"format": FORMAT_INT8, "convrot": spec["cg"] is not None}
        if spec["cg"] is not None:
            conf["convrot_groupsize"] = int(spec["cg"])
    else:
        conf = {"format": spec["format_name"], "group_size": int(spec["group"]),
                "convrot": True, "convrot_groupsize": GROUPED_CONVROT}
    if spec.get("num_experts"):
        conf["num_experts"] = int(spec["num_experts"])
    return conf


def conf_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


# =============================================================================
# Exact runtime decode (bit-compatible with kitchen's eager decode)
# =============================================================================

def flatten_parts(parts):
    out = {}
    for name, t in parts.items():
        if t is None:
            continue
        if name == "weight_codebook":
            out[name] = t
        elif name == "weight_s_channel":
            out[name] = t.reshape(-1)
        else:
            out[name] = t.reshape(-1, t.shape[-1])
    return out


def unpack_codes(packed, k, bits):
    """Packed int8 [n, k*bits/8] -> unsigned codes int32 [n, k].
    Nibble plane: even col = low nibble. 6-bit adds a k/4-byte plane holding
    each code's top 2 bits (col c -> byte c//4, bits 2*(c%4))."""
    p = packed.view(torch.uint8) if packed.dtype == torch.int8 else packed
    n = p.shape[0]
    codes = torch.empty((n, k), dtype=torch.int32, device=p.device)
    lo = p[:, : k // 2]
    codes[:, 0::2] = (lo & 0x0F).to(torch.int32)
    codes[:, 1::2] = (lo >> 4).to(torch.int32)
    if bits == 6:
        hi = p[:, k // 2: k // 2 + k // 4].to(torch.int32)
        top = torch.stack([(hi >> (2 * j)) & 0x3 for j in range(4)], dim=-1).reshape(n, k)
        codes |= top << 4
    return codes


@torch.no_grad()
def decode_grouped(parts, k, bits, group, device, rotated=False, rows=None):
    """Flattened grouped parts -> float32 weight [n, k] (original basis unless
    rotated=True). int8 grid: round(clamp(level * s_rel)) * s_channel."""
    w = parts["weight"]
    sr = parts["weight_s_rel"]
    sc = parts["weight_s_channel"]
    if rows is not None:
        w, sr, sc = w.index_select(0, rows), sr.index_select(0, rows), sc.index_select(0, rows)
    w = w.to(device)
    n = w.shape[0]
    codes = unpack_codes(w, k, bits)
    cb = parts.get("weight_codebook")
    if bits == 4 and cb is not None:
        lvl = cb.to(device=device, dtype=torch.float32)[codes.long()]
    else:
        lvl = codes.to(torch.float32) - float(1 << (bits - 1))
    del codes
    s = sr.to(device=device).to(torch.float32)
    q8 = (lvl.view(n, k // group, group) * s.unsqueeze(-1)).round_().clamp_(-127, 127)
    w_rot = q8.view(n, k) * sc.to(device=device, dtype=torch.float32).view(n, 1)
    return w_rot if rotated else rotate_rows(w_rot, GROUPED_CONVROT)


@torch.no_grad()
def decode_int8(parts, cg, device, rotated=False, rows=None):
    q = parts["weight"]
    s = parts["weight_scale"]
    if rows is not None:
        q, s = q.index_select(0, rows), s.index_select(0, rows)
    w_rot = q.to(device=device, dtype=torch.float32) * s.to(device=device, dtype=torch.float32)
    return w_rot if rotated else rotate_rows(w_rot, cg)


def decode_spec(parts, spec, device, rotated=False, rows=None):
    flat = flatten_parts(parts)
    if spec["format"] == FORMAT_INT8:
        return decode_int8(flat, spec["cg"], device, rotated=rotated, rows=rows)
    return decode_grouped(flat, spec["k"], spec["bits"], spec["group"], device,
                          rotated=rotated, rows=rows)


@torch.no_grad()
def verify_grouped_decoder(src, spec, kitchen, scale_dtype, codebook, scale_search, device):
    """Quantize a 64-row slice with kitchen, compare kitchen's own dequantize
    with ours. Returns (ok, rel_diff)."""
    QuantizedTensor, caps = kitchen
    k = spec["k"]
    w = src.reshape(-1, k)[:64].to(device)
    if w.dtype not in (torch.bfloat16, torch.float16):
        w = w.to(torch.bfloat16)
    kw = _kitchen_kwargs(spec["bits"], spec["group"], scale_dtype, codebook, scale_search, caps)
    qt = QuantizedTensor.from_float(w.contiguous(), "AsymW4A8Int8Layout", **kw)
    ref = qt.dequantize().float().reshape(-1, k)
    p = qt._params
    parts = {"weight": qt._qdata, "weight_s_rel": p.scale, "weight_s_channel": p.s_channel,
             "weight_codebook": getattr(p, "codebook", None)}
    mine = decode_grouped(parts, k, spec["bits"], spec["group"], device)
    rel = float((mine - ref).norm() / ref.norm().clamp(min=1e-12))
    return rel < 2e-2, rel


# =============================================================================
# Error metrics
# =============================================================================

@torch.no_grad()
def error_metrics(test, ref, channel_dim):
    """Primary: Rel-RMSE, SNR, Outlier %, per-channel MSE (max/std) + worst-
    channel relative RMSE. Local check (JSON only): MSE, MAE, max error.
    channel_dim: 0 for weights [N_out, K], 1 for activations [T, N_out]."""
    test = test.float()
    ref = ref.float()
    err = test - ref
    ref_pow = ref.pow(2).mean().clamp(min=1e-30)
    mse_raw = err.pow(2).mean()
    mse = mse_raw.clamp(min=1e-30)
    ch_mse = err.pow(2).mean(dim=1 - channel_dim)
    return {
        "rel_rmse": float((mse_raw / ref_pow).sqrt()),
        "snr_db": float(10.0 * torch.log10(ref_pow / mse)),
        "outlier_pct": float((err.abs() > 0.1 * ref.std().clamp(min=1e-30)).float().mean() * 100.0),
        "ch_mse_max": float(ch_mse.max()),
        "ch_mse_std": float(ch_mse.std()) if ch_mse.numel() > 1 else 0.0,
        "ch_rel_max": float((ch_mse.max() / ref_pow).sqrt()),
        "local_check": {"mse": float(mse_raw), "mae": float(err.abs().mean()),
                        "max_err": float(err.abs().max())},
    }


@torch.no_grad()
def hessian_weighted(dw_rot, w_rot, hdiag):
    """Diagonal-Hessian weighted distance: E||dW x||^2 ~ sum_k h_k ||dW[:,k]||^2."""
    h = hdiag.to(device=dw_rot.device, dtype=torch.float32)
    num = (dw_rot.pow(2).sum(0) * h).sum().clamp(min=1e-30)
    den = (w_rot.pow(2).sum(0) * h).sum().clamp(min=1e-30)
    return {"h_rel_rmse": float((num / den).sqrt()), "h_snr_db": float(10.0 * torch.log10(den / num))}


def _sample_rows(n, k):
    if n * k <= _METRIC_MAX_ELEMS:
        return None
    m = max(1, _METRIC_MAX_ELEMS // k)
    return torch.linspace(0, n - 1, steps=m).round().long()


@torch.no_grad()
def weight_error(src, parts, spec, device):
    k = spec["k"]
    ref2 = src.reshape(-1, k)
    rows = _sample_rows(ref2.shape[0], k)
    ref = (ref2.index_select(0, rows) if rows is not None else ref2).to(device=device, dtype=torch.float32)
    deq = decode_spec(parts, spec, device, rows=rows)
    m = error_metrics(deq, ref, channel_dim=0)
    if rows is not None:
        m["sampled_rows"] = int(rows.numel())
    return m


def fmt_short(spec):
    return {FORMAT_INT8: "int8", FORMAT_W4A8: "w4a8", FORMAT_W6A8: "w6a8"}.get(
        spec["format"], spec["format"])


def log_weight_line(spec, base, m):
    logging.info(
        f"  [{fmt_short(spec):4s}] {base}  rel {m['rel_rmse']:.4f}  snr {m['snr_db']:5.1f}dB  "
        f"outl {m['outlier_pct']:.3f}%  chMSE max {m['ch_mse_max']:.3e} std {m['ch_mse_std']:.3e}"
        + (f"  (sampled {m['sampled_rows']} rows)" if m.get("sampled_rows") else ""))


def generic_name(base):
    return re.sub(r"\.\d+(?=\.|$)", ".N", base)


# =============================================================================
# Streaming safetensors I/O
# =============================================================================

_ST2TORCH = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
             "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
             "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
if hasattr(torch, "float8_e4m3fn"):
    _ST2TORCH["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _ST2TORCH["F8_E5M2"] = torch.float8_e5m2
_TORCH2ST = {v: k for k, v in _ST2TORCH.items()}


class SourceReader:
    """Tensor-at-a-time reader over a (memory-mapped) .safetensors file."""

    def __init__(self, path):
        if not str(path).lower().endswith(".safetensors"):
            raise RuntimeError(f"Only .safetensors files are supported: {path}")
        from safetensors import safe_open
        self.path = path
        self._f = safe_open(path, framework="pt", device="cpu")
        self.metadata = dict(self._f.metadata() or {})
        self.keys = list(self._f.keys())
        self._info = {}

    def info(self, key):
        if key not in self._info:
            sl = self._f.get_slice(key)
            st = sl.get_dtype()
            self._info[key] = (tuple(sl.get_shape()), _ST2TORCH.get(st), st)
        return self._info[key]

    def shape(self, key):
        return self.info(key)[0]

    def dtype(self, key):
        return self.info(key)[1]

    def itemsize(self, key):
        dt = self.dtype(key)
        return torch.empty((), dtype=dt).element_size() if dt is not None else 1

    def get(self, key):
        return self._f.get_tensor(key)

    def meta_state_dict(self):
        return {k: torch.empty(self.shape(k), dtype=self.dtype(k) or torch.uint8, device="meta")
                for k in self.keys}


class StreamingSafetensorsWriter:
    """Writes tensors to a spill file as they are produced (RAM ~ one tensor),
    then assembles the final file with tensors ordered by (alignment desc,
    name) like safetensors.save_file, and atomically renames it into place.
    Needs ~2x the output size of free disk space while running."""

    _COPY_CHUNK = 64 << 20

    def __init__(self, final_path):
        self.final_path = final_path
        self.spill_path = final_path + ".spill.tmp"
        self.tmp_path = final_path + ".tmp"
        self._f = open(self.spill_path, "wb")
        self.entries = []
        self.names = set()
        self.pos = 0

    def add(self, name, tensor):
        if name in self.names:
            raise RuntimeError(f"duplicate tensor name in output: {name}")
        t = tensor.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        st = _TORCH2ST.get(t.dtype)
        if st is None:
            raise RuntimeError(f"unsupported dtype {t.dtype} for {name}")
        nbytes = t.numel() * t.element_size()
        if nbytes:
            buf = t.reshape(-1).view(torch.uint8).numpy()
            mv = memoryview(buf)
            for o in range(0, nbytes, self._COPY_CHUNK):
                self._f.write(mv[o:o + self._COPY_CHUNK])
        self.entries.append((name, st, list(t.shape), self.pos, nbytes, t.element_size()))
        self.names.add(name)
        self.pos += nbytes

    def finalize(self, metadata=None):
        self._f.close()
        order = sorted(self.entries, key=lambda e: (-e[5], e[0]))
        header, off = {}, 0
        for name, st, shape, _sp, nbytes, _isz in order:
            header[name] = {"dtype": st, "shape": shape, "data_offsets": [off, off + nbytes]}
            off += nbytes
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hb += b" " * ((-len(hb)) % 8)
        with open(self.tmp_path, "wb") as out, open(self.spill_path, "rb") as src:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            for _name, _st, _shape, spill_off, nbytes, _isz in order:
                src.seek(spill_off)
                remaining = nbytes
                while remaining:
                    chunk = src.read(min(remaining, self._COPY_CHUNK))
                    if not chunk:
                        raise RuntimeError("spill file truncated")
                    out.write(chunk)
                    remaining -= len(chunk)
        os.replace(self.tmp_path, self.final_path)
        os.remove(self.spill_path)

    def abort(self):
        try:
            self._f.close()
        except Exception:
            pass
        for p in (self.spill_path, self.tmp_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# =============================================================================
# Plan helpers
# =============================================================================

def grouped_spec(fmt, format_name, k, n, group):
    return {"format": fmt, "format_name": format_name, "bits": GROUPED_BITS[fmt],
            "group": int(group), "cg": GROUPED_CONVROT, "k": int(k), "n": int(n)}


def int8_spec(k, n):
    return {"format": FORMAT_INT8, "format_name": FORMAT_INT8, "bits": 8, "group": None,
            "cg": pick_convrot_group(k), "k": int(k), "n": int(n)}


def entry_bytes(e, reader_itemsize, scale_dtype_bytes):
    """Projected output bytes of one plan entry (dry run)."""
    numel = 1
    for d in e["shape"]:
        numel *= int(d)
    if e["action"] in ("copy", "bf16"):
        return numel * reader_itemsize
    spec = e["spec"]
    rows = numel // spec["k"]
    if spec["format"] == FORMAT_INT8:
        return numel + rows * 4 + 96
    return (numel * spec["bits"] // 8 + rows * (spec["k"] // spec["group"]) * scale_dtype_bytes
            + rows * 4 + 64 + 128)


def print_plan(plan, label, scale_dtype_bytes, src_bytes):
    groups = {}
    total = 0
    for e in plan:
        if e["action"] in ("copy",) and not e.get("is_weight"):
            key = ("(passthrough: norms / biases / buffers / non-2D)", "copy", "")
        else:
            spec = e.get("spec")
            fmt = fmt_short(spec) if spec else e["action"]
            cg = f"cg{spec['cg']}" if spec and spec.get("cg") else ""
            if spec and spec.get("group") and spec["format"] != FORMAT_INT8:
                cg = f"g{spec['group']} {cg}"
            if spec and spec.get("num_experts"):
                cg += f" E={spec['num_experts']}"
            key = (generic_name(e.get("out_base") or e["out_key"]), fmt, cg)
        g = groups.setdefault(key, [0, 0])
        g[0] += 1
        b = entry_bytes(e, e["itemsize"], scale_dtype_bytes)
        g[1] += b
        total += b
    logging.info(f"{label}: plan ({len(plan)} source tensors)")
    for (name, fmt, cg), (cnt, b) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        logging.info(f"  {fmt:5s} {cg:16s} {cnt:5d}x  {b / 2**30:8.3f} GiB  {name}")
    logging.info(f"{label}: projected output {total / 2**30:.2f} GiB "
                 f"(source {src_bytes / 2**30:.2f} GiB, {total / max(1, src_bytes):.1%})")
    return total


def execute_plan(plan, writer, device, label, conf_style, kitchen=None, grouped_opts=None,
                 measure_weights=False):
    """Run a plan: stream every source tensor through its action into the
    writer. Returns (qmeta_layers, weight_records, specs_by_base, decoder_ok)."""
    import comfy.utils
    opts = grouped_opts or {}
    pbar = comfy.utils.ProgressBar(len(plan))
    qmeta, records, specs = {}, {}, {}
    decoder_ok = None
    n_int8 = n_grp = 0
    for e in plan:
        src = e["reader"].get(e["src_key"])
        act = e["action"]
        if act in ("copy", "bf16"):
            writer.add(e["out_key"], src)
            del src
            pbar.update(1)
            continue
        spec = e["spec"]
        base = e["out_base"]
        if act == "int8":
            parts = quantize_int8_convrot(src, spec["cg"], device)
            n_int8 += 1
        else:
            if decoder_ok is None and measure_weights:
                try:
                    decoder_ok, rel = verify_grouped_decoder(
                        src, spec, kitchen, opts["scale_dtype"], opts.get("codebook", True),
                        opts.get("scale_search", True), device)
                    (logging.info if decoder_ok else logging.warning)(
                        f"{label}: grouped decoder self-check vs kitchen dequantize: "
                        f"rel diff {rel:.2e} -> {'OK' if decoder_ok else 'MISMATCH — grouped error metrics disabled'}")
                except Exception as ex:
                    decoder_ok = False
                    logging.warning(f"{label}: decoder self-check failed ({ex}); grouped error metrics disabled")
            parts = quantize_grouped(src, spec["bits"], spec["group"], opts["scale_dtype"],
                                     opts.get("codebook", True), opts.get("scale_search", True),
                                     kitchen, device)
            n_grp += 1
        for suffix, t in parts.items():
            writer.add(f"{base}.{suffix}", t)
        conf = quant_conf(spec)
        if conf_style == "tensor":
            writer.add(f"{base}.comfy_quant", conf_tensor(conf))
        else:
            qmeta[base] = conf
        specs[base] = dict(spec, src_key=e["src_key"], reader=e["reader"], shape=tuple(e["shape"]))
        if measure_weights and (act == "int8" or decoder_ok):
            try:
                m = weight_error(src, parts, spec, device)
                records.setdefault(base, {})["weight"] = m
                log_weight_line(spec, base, m)
            except Exception as ex:
                logging.warning(f"{label}: weight error for {base} failed: {ex}")
        del src, parts
        pbar.update(1)
        if device.type == "cuda" and (n_int8 + n_grp) % 16 == 0:
            torch.cuda.empty_cache()
    logging.info(f"{label}: quantized {n_grp} grouped + {n_int8} int8 layers.")
    return qmeta, records, specs, (decoder_ok is not False)


# =============================================================================
# Fake-quant calibration session (runtime-faithful A8 + decoded weights)
# =============================================================================

def _suffix_score(a, b):
    s = 0
    while s < len(a) and s < len(b) and a[-1 - s] == b[-1 - s]:
        s += 1
    score = float(s)
    if s < len(a) and s < len(b):
        x, y = a[-1 - s], b[-1 - s]
        if x.endswith(y) or y.endswith(x):
            score += 0.5
    return s, score


def map_modules(root, specs, want="linear"):
    """Map module names under 'root' to spec bases by longest dotted-suffix
    match + shape check. Ambiguous or duplicate matches are dropped."""
    index = {}
    for base in specs:
        p = base.split(".")
        index.setdefault(p[-1], []).append((base, p))
    found = {}
    for name, mod in root.named_modules():
        if want == "linear":
            if not isinstance(mod, torch.nn.Linear):
                continue
            shape = (getattr(mod, "out_features", None), getattr(mod, "in_features", None))
        elif want == "embedding":
            if not isinstance(mod, torch.nn.Embedding):
                continue
            shape = (getattr(mod, "num_embeddings", None), getattr(mod, "embedding_dim", None))
        else:  # fused expert bank
            w = getattr(mod, "weight", None)
            if not (isinstance(w, torch.Tensor) and w.ndim == 3):
                continue
            shape = tuple(w.shape)
        parts = name.split(".")
        best, best_score, tie = None, -1.0, False
        for base, bp in index.get(parts[-1], []):
            s, score = _suffix_score(parts, bp)
            if s < 3 and s != len(bp):
                continue
            sp = specs[base]
            if want == "experts":
                if tuple(sp["shape"]) != shape:
                    continue
            elif shape[0] is not None and (sp["n"], sp["k"]) != tuple(shape):
                continue
            if score > best_score:
                best, best_score, tie = base, score, False
            elif score == best_score:
                tie = True
        if best is not None and not tie:
            found.setdefault(best, []).append((name, mod))
    return {base: v[0] for base, v in found.items() if len(v) == 1}


class FakeQuantSession:
    """Hooks that make a high-precision model behave like the quantized file:
    Linear outputs are replaced by A8(ConvRot(x)) @ decoded_W_rot^T (+bias),
    embeddings by decoded int8 rows, fused expert banks by decoded weights.
    Reference passes (quant=False) accumulate the diagonal Hessian E[x_rot^2]."""

    def __init__(self, out_path, specs, decoder_ok=True):
        self.out = SourceReader(out_path)
        self._out_keys = set(self.out.keys)
        self.specs = specs
        self.decoder_ok = decoder_ok
        self.quant = False
        self.step = None
        self.collect_local = True
        self.hsum, self.hcount = {}, {}
        self.local = {}
        self.handles = []
        self._emb_cache = {}
        self._saved = {}
        self.active = set()

    # ---- weights from the written file ----
    def parts(self, base):
        sp = self.specs[base]
        names = (["weight", "weight_scale"] if sp["format"] == FORMAT_INT8 else
                 ["weight", "weight_s_rel", "weight_s_channel", "weight_codebook"])
        out = {}
        for n in names:
            key = f"{base}.{n}"
            if key in self._out_keys:
                out[n] = self.out.get(key)
        return out

    def usable(self, base):
        sp = self.specs.get(base)
        return sp is not None and (sp["format"] == FORMAT_INT8 or self.decoder_ok)

    def weight(self, base, device, rotated):
        return decode_spec(self.parts(base), self.specs[base], device, rotated=rotated)

    # ---- hooks ----
    def install_linear(self, mapping):
        for base, (name, mod) in mapping.items():
            if not self.usable(base):
                continue
            self.handles.append(mod.register_forward_hook(self._linear_hook(base)))
            self.active.add(base)
        return len(self.active)

    def _linear_hook(self, base):
        sp = self.specs[base]
        k, cg = sp["k"], sp["cg"]

        def hook(mod, inputs, output):
            x = inputs[0] if inputs else None
            if not (torch.is_tensor(x) and torch.is_tensor(output)) or x.shape[-1] != k:
                return None
            with torch.no_grad():
                xr = rotate_rows(x.reshape(-1, k).float(), cg)
                if not self.quant:
                    self.hsum[base] = self.hsum.get(base, 0) + xr.pow(2).sum(0)
                    self.hcount[base] = self.hcount.get(base, 0) + xr.shape[0]
                    return None
                a_s = (xr.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_(min=1e-30)
                xq = (xr / a_s).round_().clamp_(-128, 127).mul_(a_s)
                y = xq @ self.weight(base, x.device, rotated=True).T
                if getattr(mod, "bias", None) is not None:
                    y = y + mod.bias.to(device=y.device, dtype=torch.float32)
                if self.collect_local:
                    rec = self.local.setdefault(base, [])
                    if len(rec) < _LOCAL_RECORDS_PER_LAYER:
                        m = error_metrics(y, output.reshape(-1, output.shape[-1]), channel_dim=1)
                        m["step"] = self.step
                        rec.append(m)
                return y.to(output.dtype).reshape(output.shape)
        return hook

    def install_embedding(self, base, mod):
        if not self.usable(base) or self.specs[base]["format"] != FORMAT_INT8:
            return False

        def hook(m, inputs, output):
            if not self.quant:
                return None
            ids = inputs[0] if inputs else None
            if not torch.is_tensor(ids):
                return None
            with torch.no_grad():
                if base not in self._emb_cache:
                    p = self.parts(base)
                    self._emb_cache[base] = (p["weight"], p["weight_scale"])
                q, s = self._emb_cache[base]
                flat = ids.reshape(-1).long().cpu()
                rows = rotate_rows(q.index_select(0, flat).to(output.device, torch.float32)
                                   * s.index_select(0, flat).to(output.device, torch.float32),
                                   self.specs[base]["cg"])
                return rows.to(output.dtype).reshape(output.shape)
        self.handles.append(mod.register_forward_hook(hook))
        self.active.add(base)
        return True

    def install_expert_swaps(self, mapping):
        count = 0
        for base, (name, mod) in mapping.items():
            if not self.usable(base):
                continue

            def pre(m, inputs, _b=base):
                w = getattr(m, "weight", None)
                if (self.quant and _b not in self._saved and isinstance(w, torch.Tensor)
                        and w.device.type != "meta"):
                    with torch.no_grad():
                        sp = self.specs[_b]
                        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                        deq = self.weight(_b, dev, rotated=False).view(*sp["shape"])
                        self._saved[_b] = w.data.clone()
                        w.data.copy_(deq.to(device=w.device, dtype=w.dtype))
                        del deq
                return None

            def post(m, inputs, output, _b=base):
                saved = self._saved.pop(_b, None)
                if saved is not None:
                    with torch.no_grad():
                        m.weight.data.copy_(saved)
                return None
            self.handles.append(mod.register_forward_pre_hook(pre))
            self.handles.append(mod.register_forward_hook(post))
            self.active.add(base)
            count += 1
        return count

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        self._emb_cache = {}

    # ---- Hessian-weighted weight distances ----
    @torch.no_grad()
    def hessian_metrics(self, device):
        out = {}
        for base, hs in self.hsum.items():
            sp = self.specs[base]
            try:
                hdiag = hs / max(1, self.hcount.get(base, 1))
                src = sp["reader"].get(sp["src_key"]).reshape(-1, sp["k"])
                rows = _sample_rows(src.shape[0], sp["k"])
                ref = (src.index_select(0, rows) if rows is not None else src).to(device, torch.float32)
                w_rot = rotate_rows(ref, sp["cg"])
                q_rot = decode_spec(self.parts(base), sp, device, rotated=True, rows=rows)
                out[base] = hessian_weighted(q_rot - w_rot, w_rot, hdiag)
                del src, ref, w_rot, q_rot
            except Exception as ex:
                logging.warning(f"Hessian metric for {base} failed: {ex}")
        return out


def summarize_local(local):
    """Collapse per-call activation records into per-layer means / worst."""
    out = {}
    for base, recs in local.items():
        if not recs:
            continue
        n = len(recs)

        def mean(key):
            return sum(r[key] for r in recs) / n
        worst = max(recs, key=lambda r: r["rel_rmse"])
        out[base] = {"calls": n, "rel_rmse": mean("rel_rmse"), "snr_db": mean("snr_db"),
                     "outlier_pct": mean("outlier_pct"),
                     "ch_mse_max": max(r["ch_mse_max"] for r in recs),
                     "ch_mse_std": mean("ch_mse_std"),
                     "ch_rel_max": max(r["ch_rel_max"] for r in recs),
                     "worst_rel_rmse": worst["rel_rmse"], "worst_step": worst.get("step"),
                     "local_check": {"mse": sum(r["local_check"]["mse"] for r in recs) / n,
                                     "mae": sum(r["local_check"]["mae"] for r in recs) / n,
                                     "max_err": max(r["local_check"]["max_err"] for r in recs)}}
    return out


def log_summary(label, records, key="weight", top=10):
    """Per-category means + worst layers (Hessian-weighted when available)."""
    cats = {}
    for base, rec in records.items():
        m = rec.get(key)
        if not m:
            continue
        c = cats.setdefault(generic_name(base), [])
        c.append(m)
    if not cats:
        return
    logging.info(f"{label}: {key} error by layer category (mean)")
    for name, ms in sorted(cats.items()):
        n = len(ms)
        logging.info(f"  {n:4d}x  rel {sum(m['rel_rmse'] for m in ms) / n:.4f}  "
                     f"snr {sum(m['snr_db'] for m in ms) / n:5.1f}dB  "
                     f"outl {sum(m['outlier_pct'] for m in ms) / n:.3f}%  {name}")

    def score(item):
        rec = item[1]
        h = rec.get("hessian")
        return h["h_rel_rmse"] if h else rec[key]["rel_rmse"]
    ranked = sorted(((b, r) for b, r in records.items() if r.get(key)), key=score, reverse=True)[:top]
    logging.info(f"{label}: worst {len(ranked)} layers"
                 + (" (Hessian-weighted)" if any(r.get("hessian") for _, r in ranked) else ""))
    for base, rec in ranked:
        h = rec.get("hessian")
        hs = f"  H-rel {h['h_rel_rmse']:.4f} H-snr {h['h_snr_db']:5.1f}dB" if h else ""
        logging.info(f"  rel {rec[key]['rel_rmse']:.4f}  snr {rec[key]['snr_db']:5.1f}dB{hs}  {base}")


def write_report(out_path, report):
    path = re.sub(r"\.safetensors$", "", out_path) + "_quant_report.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=lambda o: float(o) if torch.is_tensor(o) else str(o))
        logging.info(f"Quant report written: {path}")
    except Exception as ex:
        logging.warning(f"Could not write quant report: {ex}")

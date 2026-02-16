import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Callable

_DEBUG_MODE = False
_DEBUG_FORWARD = False

if __import__('os').environ.get("INT8_DEBUG_MODE", "").lower() in ("1", "true", "yes"):
    _DEBUG_MODE = True


def _debug(msg: str):
    """Minimal debug logger to avoid repeated _DEBUG_MODE checks."""
    if _DEBUG_MODE:
        print(msg)


def _as_device(t: Tensor | float, device: torch.device) -> Tensor | float:
    """Ensure tensor is on target device."""
    if isinstance(t, Tensor) and t.device != device:
        return t.to(device)
    return t


def _reshape_scale_for_matmul(scale: Tensor | float, rows: int, cols: int, ndim: int) -> Tensor | float:
    """Reshape scale tensor for proper broadcasting in matmul operations."""
    if not isinstance(scale, Tensor):
        return scale
    if scale.numel() == 1:
        return scale.view(())
    if ndim == 1:
        return scale.view(-1)
    if ndim == 2:
        if scale.numel() == rows:
            return scale.view(-1, 1)
        if scale.numel() == cols:
            return scale.view(1, -1)
    return scale


def _get_chunk_rows(n_rows: int, n_cols: int, target_elements: int) -> int:
    """Calculate optimal chunk rows for memory-efficient processing."""
    return max(1, target_elements // max(1, n_cols))


def _chunked_apply_rows(
    x2d: Tensor,
    out: Tensor,
    chunk_rows: int,
    fn: Callable[[Tensor, int, int], Tensor],
) -> Tensor:
    """Apply fn to row chunks, writing results to out[start:end]."""
    for start in range(0, x2d.shape[0], chunk_rows):
        end = min(start + chunk_rows, x2d.shape[0])
        out[start:end] = fn(x2d[start:end], start, end)
    return out


def _scalar_or_vec(s: Tensor | float, rows: int) -> Tensor:
    """Convert scalar or single-element tensor to vector of given length."""
    if not isinstance(s, Tensor):
        return torch.full((rows,), float(s))
    if s.numel() == 1:
        return torch.full((rows,), float(s.item()), device=s.device, dtype=s.dtype)
    return s

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

_triton_kernels = None
_hadamard_quip_kernels = None

def _get_triton_kernels():
    """Lazy import of triton kernels to avoid import errors when Triton not available."""
    global _triton_kernels
    if _triton_kernels is None and _TRITON_AVAILABLE:
        try:
            from .triton_kernels import triton_int8_linear
            _triton_kernels = triton_int8_linear
        except ImportError as e:
            if _DEBUG_MODE:
                print(f"[DEBUG] Failed to import triton_int8_linear: {e}")
        except Exception as e:
            if _DEBUG_MODE:
                print(f"[DEBUG] Unexpected error importing triton kernels: {type(e).__name__}: {e}")
    return _triton_kernels


def _get_hadamard_quip_kernels():
    """Lazy import of Hadamard-QuIP kernels."""
    global _hadamard_quip_kernels
    if _hadamard_quip_kernels is None and _TRITON_AVAILABLE:
        try:
            from .triton_kernels import triton_hadamard_quip_linear, pytorch_hadamard_quip_linear
            _hadamard_quip_kernels = (triton_hadamard_quip_linear, pytorch_hadamard_quip_linear)
        except ImportError as e:
            if _DEBUG_MODE:
                print(f"[DEBUG] Failed to import Hadamard-QuIP kernels: {e}")
        except Exception as e:
            if _DEBUG_MODE:
                print(f"[DEBUG] Unexpected error importing Hadamard-QuIP kernels: {type(e).__name__}: {e}")
    return _hadamard_quip_kernels


_quantize_int8_jit = None
_dequantize_jit = None

def _get_quantize_int8_jit():
    """Lazy initialization of JIT-compiled quantize_int8."""
    global _quantize_int8_jit
    if _quantize_int8_jit is None:
        try:
            @torch.jit.script
            def _quantize_int8_script(x: torch.Tensor, scale: float) -> torch.Tensor:
                return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
            _quantize_int8_jit = _quantize_int8_script
        except RuntimeError as e:
            print(f"[INT8 WARNING] JIT compilation failed for quantize_int8: {e}")
            print(f"[INT8 WARNING] Falling back to non-JIT implementation (may be slower)")
        except Exception as e:
            print(f"[INT8 WARNING] Unexpected error in JIT quantize_int8: {type(e).__name__}: {e}")
            print(f"[INT8 WARNING] Falling back to non-JIT implementation (may be slower)")
    return _quantize_int8_jit


def _get_dequantize_jit():
    """Lazy initialization of JIT-compiled dequantize."""
    global _dequantize_jit
    if _dequantize_jit is None:
        try:
            @torch.jit.script
            def _dequantize_script(q: torch.Tensor, scale: float) -> torch.Tensor:
                return q.float().mul(scale)
            _dequantize_jit = _dequantize_script
        except RuntimeError as e:
            print(f"[INT8 WARNING] JIT compilation failed for dequantize: {e}")
            print(f"[INT8 WARNING] Falling back to non-JIT implementation (may be slower)")
        except Exception as e:
            print(f"[INT8 WARNING] Unexpected error in JIT dequantize: {type(e).__name__}: {e}")
            print(f"[INT8 WARNING] Falling back to non-JIT implementation (may be slower)")
    return _dequantize_jit


def quantize_int8(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to INT8 using the provided scale."""
    if isinstance(scale, (int, float)):
        jit_fn = _get_quantize_int8_jit()
        if jit_fn is not None:
            try:
                return jit_fn(x, float(scale))
            except (RuntimeError, ValueError, TypeError):
                _debug("[DEBUG] JIT quantize failed; fallback")
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_chunked(x: torch.Tensor, scale: torch.Tensor | float, chunk_size: int) -> torch.Tensor:
    """Quantize a tensor to INT8 with chunking for memory efficiency."""
    if x.numel() <= chunk_size:
        return quantize_int8(x, scale)

    x2d = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2d, dtype=torch.int8)

    chunk_rows = _get_chunk_rows(x2d.shape[0], x2d.shape[1], chunk_size)

    def _quant_chunk(chunk: Tensor, start: int, end: int) -> Tensor:
        if isinstance(scale, Tensor) and scale.numel() > 1:
            if scale.ndim == 2:
                chunk_scale = scale[start:end]
            else:
                chunk_scale = scale[start:end].view(-1, 1)
        else:
            chunk_scale = scale
        return quantize_int8(chunk, chunk_scale)

    return _chunked_apply_rows(x2d, out, chunk_rows, _quant_chunk).reshape_as(x)


def _check_nan_inf(tensor: torch.Tensor, name: str, location: str = ""):
    """Check tensor for NaN or Inf values."""
    if not _DEBUG_MODE:
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        loc_str = f" [{location}]" if location else ""
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"[DEBUG]{loc_str} NaN detected in {name}: {nan_count} values")
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            print(f"[DEBUG]{loc_str} Inf detected in {name}: {inf_count} values")


def _apply_hadamard_transform(w: Tensor, hadamard_size: int, sign_vec: Tensor | None = None) -> Tensor:
    """Apply Hadamard transform with fallback to PyTorch implementation."""
    if hadamard_size <= 0:
        return w
    try:
        from .triton_kernels import triton_hadamard_transform
        w = triton_hadamard_transform(w, normalize=True)
    except (ImportError, RuntimeError, ValueError) as e:
        _debug(f"[DEBUG] Hadamard transform fallback: {type(e).__name__}")
        w = _pytorch_fwht_simple(w)
    if sign_vec is not None and sign_vec.shape[0] >= hadamard_size:
        w = w * sign_vec[:hadamard_size].unsqueeze(0)
    return w


def dequantize(q: Tensor, scale: float | Tensor, quip_s_u: Tensor | None = None, quip_s_v: Tensor | None = None,
               hadamard_quip: bool = False, hadamard_size_in: int = 0, hadamard_size_out: int = 0,
               sign_row: Tensor | None = None, sign_col: Tensor | None = None) -> Tensor:
    """Dequantize INT8 tensor to float."""
    total_elements = q.numel()

    rows = q.shape[0] if q.ndim >= 1 else 1
    cols = q.shape[1] if q.ndim >= 2 else 1

    # Handle 1D safely (bypass Hadamard/QuIP paths)
    if q.ndim == 1:
        scale = _reshape_scale_for_matmul(scale, rows, cols, q.ndim)
        scale = _as_device(scale, q.device)
        return q.float().mul(scale)

    # --- Hadamard-QuIP path ---
    if hadamard_quip and (hadamard_size_in > 0 or hadamard_size_out > 0):
        scale = _reshape_scale_for_matmul(scale, rows, cols, q.ndim)
        scale = _as_device(scale, q.device)
        if sign_row is not None:
            sign_row = _as_device(sign_row, q.device)
        if sign_col is not None:
            sign_col = _as_device(sign_col, q.device)

        w = q.float() * scale
        dtype = w.dtype
        N, K = w.shape

        # Pad to Hadamard dimensions
        if hadamard_size_out > 0 and N < hadamard_size_out:
            w = torch.nn.functional.pad(w, (0, 0, 0, hadamard_size_out - N))
        if hadamard_size_in > 0 and K < hadamard_size_in:
            w = torch.nn.functional.pad(w, (0, hadamard_size_in - K, 0, 0))

        # Apply Hadamard transforms
        w = _apply_hadamard_transform(w.T, hadamard_size_out, sign_row).T
        w = _apply_hadamard_transform(w, hadamard_size_in, sign_col)

        # Slice back to original dimensions
        if hadamard_size_out > 0 and N < hadamard_size_out:
            w = w[:N, :]
        if hadamard_size_in > 0 and K < hadamard_size_in:
            w = w[:, :K]

        return w.to(dtype)

    # --- QuIP path ---
    if quip_s_u is not None and quip_s_v is not None:
        scale = _reshape_scale_for_matmul(scale, rows, cols, q.ndim)
        scale = _as_device(scale, q.device)

        w = q.float() * scale
        N, K = w.shape
        u = quip_s_u.to(device=w.device, dtype=w.dtype)
        v = quip_s_v.to(device=w.device, dtype=w.dtype)

        if u.numel() == N and v.numel() == K:
            res = u.unsqueeze(1) * w * v.unsqueeze(0)
            return res.squeeze() if q.ndim == 1 else res
        if u.numel() == N * N and v.numel() == K * K:
            return (u.view(N, N) @ w @ v.view(K, K).T).squeeze()
        _debug(f"[QuIP# WARNING] Unexpected sign vector sizes: u={u.shape}, v={v.shape}, w={w.shape}")
        return w

    # --- JIT fast path ---
    if isinstance(scale, (int, float)) and total_elements <= CHUNK_THRESHOLD_ELEMENTS:
        jit_fn = _get_dequantize_jit()
        if jit_fn is not None:
            try:
                return jit_fn(q, float(scale))
            except (RuntimeError, ValueError, TypeError):
                _debug("[DEBUG] JIT dequantize failed; fallback")

    # --- Chunked path ---
    if total_elements > CHUNK_THRESHOLD_ELEMENTS:
        q2d = q.reshape(-1, q.shape[-1])
        out = torch.empty_like(q2d, dtype=torch.float32)
        chunk_rows = _get_chunk_rows(q2d.shape[0], q2d.shape[1], CHUNK_TARGET_ELEMENTS)

        def _dq_chunk(chunk: Tensor, start: int, end: int) -> Tensor:
            if isinstance(scale, Tensor) and scale.numel() > 1:
                chunk_scale = scale[start:end] if scale.ndim == 2 else scale[start:end].view(-1, 1)
            else:
                chunk_scale = scale
            chunk_scale = _as_device(chunk_scale, chunk.device)
            return chunk.float().mul(chunk_scale)

        return _chunked_apply_rows(q2d, out, chunk_rows, _dq_chunk).reshape_as(q)

    # --- Normal path ---
    scale = _reshape_scale_for_matmul(scale, *q.shape[:2], q.ndim)
    scale = _as_device(scale, q.device)
    return q.float().mul(scale)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _pytorch_fwht_simple(x: torch.Tensor) -> torch.Tensor:
    """Simple PyTorch FWHT for fallback dequantization."""
    import math
    original_shape = x.shape
    n = original_shape[-1]
    
    if not _is_power_of_two(n):
        raise ValueError(f"FWHT requires power-of-2 dimension, got {n}")
    
    x = x.reshape(-1, n)
    batch = x.shape[0]
    
    output = torch.empty_like(x)
    
    h = 1
    while h < n:
        half = n // (2 * h)
        
        x_view = x.view(batch, half, 2, h)
        out_view = output.view(batch, half, 2, h)
        
        a = x_view[:, :, 0, :]
        b = x_view[:, :, 1, :]
        
        out_view[:, :, 0, :] = a + b
        out_view[:, :, 1, :] = a - b
        
        x, output = output, x
        h *= 2
    
    result = x.reshape(original_shape)
    result = result / math.sqrt(n)
    
    return result


def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8_chunked(x, scale, CHUNK_TARGET_ELEMENTS), scale


def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8_chunked(x, scale, CHUNK_TARGET_ELEMENTS), scale


def convert_zimage_diffusers_state_dict(sd):
    """Convert z-image state dict from Diffusers INT8 format to ComfyUI format."""
    import re
    new_sd = {}
    attn_groups = {}
    
    print(f"[z-image] Starting state dict conversion for {len(sd)} keys...")
    
    for key, value in sd.items():
        new_key = re.sub(r"all_x_embedder\.\d+-\d+\.", "x_embedder.", key)
        new_key = re.sub(r"all_final_layer\.\d+-\d+\.", "final_layer.", new_key)
        
        if new_key.endswith(".scale_weight"):
            new_key = new_key.replace(".scale_weight", ".weight_scale")
        
        attn_match = re.match(r"(.+\.attention)\.(to_q|to_k|to_v)\.(weight|weight_scale|bias|quip_s_u|quip_s_v)$", new_key)
        if attn_match:
            prefix = attn_match.group(1)
            qkv_type = attn_match.group(2)
            param_type = attn_match.group(3)
            if prefix not in attn_groups:
                attn_groups[prefix] = {}
            attn_groups[prefix][f"{qkv_type}.{param_type}"] = value
            continue
        
        if ".attention.to_out.0." in new_key:
            new_key = new_key.replace(".attention.to_out.0.", ".attention.out.")
        
        if ".attention.norm_q." in new_key:
            new_key = new_key.replace(".attention.norm_q.", ".attention.q_norm.")
        if ".attention.norm_k." in new_key:
            new_key = new_key.replace(".attention.norm_k.", ".attention.k_norm.")
            
        new_sd[new_key] = value
    
    for prefix, params in attn_groups.items():
        if "to_q.weight" in params and "to_k.weight" in params and "to_v.weight" in params:
            q_w, k_w, v_w = params["to_q.weight"], params["to_k.weight"], params["to_v.weight"]
            
            for qkv_type in ["to_q", "to_k", "to_v"]:
                params.pop(f"{qkv_type}.comfy_quant", None)
            
            has_quip = any(f"{t}.quip_s_u" in params for t in ["to_q", "to_k", "to_v"])
            
            if has_quip:
                def get_dequant(t_name, w):
                    s = params.get(f"{t_name}.weight_scale", 1.0)
                    u = params.get(f"{t_name}.quip_s_u")
                    v = params.get(f"{t_name}.quip_s_v")
                    
                    if u is None or v is None:
                        return dequantize(w, s).to(torch.bfloat16)
                    
                    return dequantize(w, s, u, v).to(torch.bfloat16)
                
                q_dq = get_dequant("to_q", q_w)
                k_dq = get_dequant("to_k", k_w)
                v_dq = get_dequant("to_v", v_w)
                
                if q_dq.ndim == 1: q_dq = q_dq.unsqueeze(0)
                if k_dq.ndim == 1: k_dq = k_dq.unsqueeze(0)
                if v_dq.ndim == 1: v_dq = v_dq.unsqueeze(0)
                
                new_sd[f"{prefix}.qkv.weight"] = torch.cat([q_dq, k_dq, v_dq], dim=0)
                print(f"[z-image] Dequantized and fused QuIP# QKV for {prefix}. Shapes: Q={q_dq.shape}, K={k_dq.shape}, V={v_dq.shape}, Fused={new_sd[f'{prefix}.qkv.weight'].shape}")
            else:
                new_sd[f"{prefix}.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                
                if "to_q.weight_scale" in params and "to_k.weight_scale" in params and "to_v.weight_scale" in params:
                    q_s, k_s, v_s = params["to_q.weight_scale"], params["to_k.weight_scale"], params["to_v.weight_scale"]
                    new_sd[f"{prefix}.qkv.weight_scale"] = torch.cat([
                        _scalar_or_vec(q_s, q_w.shape[0]),
                        _scalar_or_vec(k_s, k_w.shape[0]),
                        _scalar_or_vec(v_s, v_w.shape[0])
                    ], dim=0)
            
            if "to_q.bias" in params and "to_k.bias" in params and "to_v.bias" in params:
                new_sd[f"{prefix}.qkv.bias"] = torch.cat([params["to_q.bias"], params["to_k.bias"], params["to_v.bias"]], dim=0)
        else:
            for pk, pv in params.items():
                nk = pk.replace("to_q.", "q.").replace("to_k.", "k.").replace("to_v.", "v.")
                new_sd[f"{prefix}.{nk}"] = pv
    
    return new_sd


def is_diffusers_zimage_format(sd):
    """Check if state dict uses Diffusers-style z-image naming."""
    for key in sd.keys():
        if "all_x_embedder." in key or "all_final_layer." in key or ".attention.to_q." in key:
            return True
    return False


def strip_comfy_quant_keys(sd):
    """Strip .comfy_quant metadata keys from state dict to prevent unexpected key warnings."""
    keys_to_remove = [k for k in sd.keys() if ".comfy_quant" in k]
    for k in keys_to_remove:
        sd.pop(k, None)
    return sd


try:
    from torch.compiler import disable as compiler_disable
except ImportError:
    def compiler_disable(fn=None, recursive=True):
        if fn is None:
            return lambda f: f
        return fn

CHUNK_THRESHOLD_ELEMENTS = 67_108_864
CHUNK_TARGET_ELEMENTS = 33_554_432

_ENABLE_CUDA_SYNC = os.environ.get("INT8_ENABLE_CUDA_SYNC", "0") == "1"

_CLEAR_CACHE_STRATEGY = os.environ.get("INT8_CLEAR_CACHE", "auto")

_HADAMARD_QUIP_ENABLED = os.environ.get("INT8_HADAMARD_QUIP", "0").lower() in ("1", "true", "yes", "on") and \
                          os.environ.get("INT8_DISABLE_HADAMARD_QUIP", "").lower() not in ("1", "true", "yes")


def set_hadamard_quip_enabled(enabled: bool):
    """Enable or disable the Hadamard-QuIP kernel at runtime."""
    global _HADAMARD_QUIP_ENABLED
    _HADAMARD_QUIP_ENABLED = enabled
    status = "enabled" if enabled else "disabled"
    print(f"[INT8] Hadamard-QuIP kernel {status}")


def is_hadamard_quip_enabled() -> bool:
    """Check if Hadamard-QuIP kernel is currently enabled."""
    return _HADAMARD_QUIP_ENABLED


_DISABLE_HADAMARD_QUIP = not _HADAMARD_QUIP_ENABLED


def warmup_int8_kernels(model, device="cuda", dtype=torch.bfloat16):
    print("[INT8] Warming up kernels...")
    
    _get_quantize_int8_jit()
    _get_dequantize_jit()
    
    if _TRITON_AVAILABLE:
        _get_triton_kernels()
        _get_hadamard_quip_kernels()
    
    with torch.no_grad():
        dummy = torch.randn(1, 4096, device=device, dtype=dtype)
        
        for module in model.modules():
            if hasattr(module, '_is_quantized') and module._is_quantized:
                try:
                    _ = module(dummy[:, :module.in_features] if dummy.shape[-1] >= module.in_features
                              else torch.randn(1, module.in_features, device=device, dtype=dtype))
                    break
                except:
                    pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[INT8] Kernel warmup complete")


def _should_clear_cache():
    """Check if we should clear CUDA cache based on memory pressure."""
    if _CLEAR_CACHE_STRATEGY == "always":
        return True
    if _CLEAR_CACHE_STRATEGY == "never":
        return False
    if not torch.cuda.is_available():
        return False
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if reserved > 0:
        utilization = allocated / reserved
        return utilization < 0.5
    return False


def _maybe_synchronize():
    """Synchronize CUDA only if explicitly enabled."""
    if _ENABLE_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


if _DEBUG_MODE:
    def _log_memory(msg: str, is_forward: bool = False):
        if is_forward and not _DEBUG_FORWARD:
            return
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] {msg}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_mem:.2f}GB")

    def _log_tensor_size(name: str, t: Tensor, is_forward: bool = False):
        if is_forward and not _DEBUG_FORWARD:
            return
        if torch.cuda.is_available():
            size_mb = t.numel() * t.element_size() / 1024**2
            print(f"[TENSOR] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, size={size_mb:.2f}MB")

else:
    def _log_memory(msg: str, is_forward: bool = False):
        pass

    def _log_tensor_size(name: str, t: Tensor, is_forward: bool = False):
        pass


@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None,
                         compute_dtype: torch.dtype, chunk_size: int = 0,
                         has_lora: bool = False, offload_to_cpu: bool = False) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    output_dtype = compute_dtype if (has_lora and offload_to_cpu) else (torch.float32 if has_lora else compute_dtype)

    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])

    if chunk_size > 0 and x_2d.shape[0] > chunk_size:
        out = torch.empty((x_2d.shape[0], weight.shape[0]), device=x.device, dtype=output_dtype)
        _chunked_apply_rows(
            x_2d, out, chunk_size,
            lambda c, s, e: int8_forward_dynamic(c, weight, weight_scale, bias, compute_dtype)
        )
        return out.reshape(*x_shape[:-1], out.shape[-1])

    x = x_2d
    _check_nan_inf(x, "input x", "int8_forward_dynamic")

    triton_kernels = _get_triton_kernels()
    if triton_kernels is not None and x.ndim >= 2:
        try:
            _log_kernel_usage("w8a8_dynamic_triton", "int8_forward_dynamic")
            return triton_kernels(x, weight, weight_scale, bias, compute_dtype, use_fp32_output=has_lora)
        except (RuntimeError, ValueError) as e:
            _debug(f"[DEBUG] Triton fallback: {type(e).__name__}: {e}")

    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8.contiguous(), weight.T.contiguous())
    scale = x_scale * weight_scale

    if res.numel() > CHUNK_THRESHOLD_ELEMENTS:
        res_f = torch.empty_like(res, dtype=output_dtype)
        chunk_rows = _get_chunk_rows(res.shape[0], res.shape[1], CHUNK_TARGET_ELEMENTS)

        def _scale_chunk(chunk: Tensor, start: int, end: int) -> Tensor:
            if scale.numel() == 1 or scale.shape[0] == 1:
                return chunk.float().mul(scale).to(output_dtype)
            return chunk.float().mul(scale[start:end]).to(output_dtype)

        res_f = _chunked_apply_rows(res, res_f, chunk_rows, _scale_chunk)
    else:
        res_f = res.float().mul(scale).to(output_dtype)

    if bias is not None:
        res_f.add_(bias.to(output_dtype))

    _check_nan_inf(res_f, "output", "int8_forward_dynamic")
    return res_f


@torch.no_grad()
def int8_forward_static(x: Tensor, weight: Tensor, weight_scale: float | Tensor, input_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype, chunk_size: int = 0, has_lora: bool = False, offload_to_cpu: bool = False) -> Tensor:
    """Forward with static (learned) activation quantization."""
    output_dtype = compute_dtype if (has_lora and offload_to_cpu) else (torch.float32 if has_lora else compute_dtype)

    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])

    if chunk_size > 0 and x_2d.shape[0] > chunk_size:
        out = torch.empty((x_2d.shape[0], weight.shape[0]), device=x.device, dtype=output_dtype)
        for i in range(0, x_2d.shape[0], chunk_size):
            chunk = x_2d[i:i+chunk_size]
            out[i:i+chunk_size] = int8_forward_static(chunk, weight, weight_scale, input_scale, bias, compute_dtype, chunk_size=0, has_lora=has_lora, offload_to_cpu=offload_to_cpu)
            del chunk
        return out.reshape(*x_shape[:-1], out.shape[-1])

    x = x_2d
    x_8 = quantize_int8(x, input_scale)
    x_8_contig = x_8.contiguous() if not x_8.is_contiguous() else x_8
    weight_T = weight.T.contiguous() if not weight.T.is_contiguous() else weight.T
    res = torch._int_mm(x_8_contig, weight_T)
    del x_8, x_8_contig, weight_T
    
    scale = weight_scale * input_scale
    
    total_elements = res.numel()
    
    if total_elements > CHUNK_THRESHOLD_ELEMENTS:
        chunk_rows = max(1, (CHUNK_TARGET_ELEMENTS // res.shape[1]))
        
        res_scaled = torch.empty_like(res, dtype=output_dtype)
        
        for start_row in range(0, res.shape[0], chunk_rows):
            end_row = min(start_row + chunk_rows, res.shape[0])
            chunk_int32 = res[start_row:end_row]
            res_scaled[start_row:end_row] = chunk_int32.float().mul_(scale).to(output_dtype)
            del chunk_int32
    else:
        res_scaled = res.float().mul_(scale).to(output_dtype)
    
    del res
    
    if bias is not None:
        res_scaled.add_(bias.to(output_dtype))
    
    _check_nan_inf(res_scaled, "output", "int8_forward_static")
    
    return res_scaled


@torch.no_grad()
def chunked_lora_forward(x: Tensor, down: Tensor, up: Tensor, alpha: float, output: Tensor, offset: int = 0, size: int = 0):
    """Memory-efficient float LoRA forward pass using chunking."""
    ref = output
    if x.device != ref.device:
        x = x.to(device=ref.device)
    if x.dtype != ref.dtype:
        x = x.to(dtype=ref.dtype)
    
    if down.device != ref.device or down.dtype != ref.dtype:
        down = down.to(device=ref.device, dtype=ref.dtype)
    if up.device != ref.device or up.dtype != ref.dtype:
        up = up.to(device=ref.device, dtype=ref.dtype)
    
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    
    chunk_rows = max(1, CHUNK_TARGET_ELEMENTS // max(down.shape[0], up.shape[0]))
    
    is_fp32 = (x.dtype == torch.float32)
    old_allow_tf32 = None
    if is_fp32 and torch.cuda.is_available():
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    
    try:
        if x_2d.shape[0] <= chunk_rows:
            lora_out = F.linear(F.linear(x_2d, down), up)
            
            if size > 0:
                output_view = output.reshape(-1, output.shape[-1])
                if lora_out.shape[1] != size:
                    if lora_out.shape[1] > size:
                        lora_out = lora_out[:, :size]
                    else:
                        lora_out = torch.nn.functional.pad(lora_out, (0, size - lora_out.shape[1]))
                output_view[:, offset:offset+size].add_(lora_out, alpha=alpha)
            else:
                lora_out = lora_out.reshape(*x_shape[:-1], lora_out.shape[-1])
                output.add_(lora_out, alpha=alpha)
        else:
            output_view = output.reshape(-1, output.shape[-1])
            for i in range(0, x_2d.shape[0], chunk_rows):
                end = min(i + chunk_rows, x_2d.shape[0])
                x_chunk = x_2d[i:end]
                lora_out = F.linear(F.linear(x_chunk, down), up)
                
                if size > 0:
                    if lora_out.shape[1] != size:
                        if lora_out.shape[1] > size:
                            lora_out = lora_out[:, :size]
                        else:
                            lora_out = torch.nn.functional.pad(lora_out, (0, size - lora_out.shape[1]))
                    output_view[i:end, offset:offset+size].add_(lora_out, alpha=alpha)
                else:
                    output_view[i:end].add_(lora_out, alpha=alpha)
                del lora_out
                
                if _ENABLE_CUDA_SYNC and i % (chunk_rows * 4) == 0:
                    torch.cuda.synchronize()
    finally:
        if old_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


@torch.no_grad()
def chunked_int8_lora_forward(x: Tensor, down: Tensor, up: Tensor, down_scale: float | Tensor, up_scale: float | Tensor, alpha: float, output: Tensor, offset: int = 0, size: int = 0):
    """Memory-efficient INT8 LoRA forward pass using chunking."""
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    
    inter_dtype = output.dtype if output.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    
    chunk_rows = max(1, CHUNK_TARGET_ELEMENTS // max(down.shape[0], up.shape[0]))
    
    if x_2d.shape[0] <= chunk_rows:
        x_int8, x_scale = quantize_int8_axiswise(x_2d, dim=-1)
        x_int8_contig = x_int8.contiguous() if not x_int8.is_contiguous() else x_int8
        down_T = down.T.contiguous() if not down.T.is_contiguous() else down.T
        lora_inter = torch._int_mm(x_int8_contig, down_T)
        del x_int8, x_int8_contig, down_T
        
        lora_inter = lora_inter.to(dtype=torch.float32)
        lora_inter.mul_(x_scale * down_scale)
        lora_inter = lora_inter.to(inter_dtype)
        del x_scale
        
        up_float = (up.float() * up_scale).to(inter_dtype)
        lora_out = F.linear(lora_inter, up_float)
        del lora_inter, up_float
        
        if size > 0:
            output_view = output.reshape(-1, output.shape[-1])
            if lora_out.shape[1] != size:
                if lora_out.shape[1] > size:
                    lora_out = lora_out[:, :size]
                else:
                    lora_out = torch.nn.functional.pad(lora_out, (0, size - lora_out.shape[1]))
            output_view[:, offset:offset+size].add_(lora_out.to(output.dtype), alpha=alpha)
        else:
            lora_out = lora_out.reshape(*x_shape[:-1], lora_out.shape[-1])
            output.add_(lora_out.to(output.dtype), alpha=alpha)
        del lora_out
    else:
        output_view = output.reshape(-1, output.shape[-1])
        for i in range(0, x_2d.shape[0], chunk_rows):
            end = min(i + chunk_rows, x_2d.shape[0])
            x_chunk = x_2d[i:end]
            
            x_int8, x_scale = quantize_int8_axiswise(x_chunk, dim=-1)
            x_int8_contig = x_int8.contiguous() if not x_int8.is_contiguous() else x_int8
            down_T = down.T.contiguous() if not down.T.is_contiguous() else down.T
            lora_inter = torch._int_mm(x_int8_contig, down_T)
            del x_int8, x_int8_contig, down_T
            
            lora_inter = lora_inter.to(dtype=torch.float32)
            lora_inter.mul_(x_scale * down_scale)
            lora_inter = lora_inter.to(inter_dtype)
            del x_scale
            
            up_float = (up.float() * up_scale).to(inter_dtype)
            lora_out = F.linear(lora_inter, up_float)
            del lora_inter, up_float
            
            if size > 0:
                if lora_out.shape[1] != size:
                    if lora_out.shape[1] > size:
                        lora_out = lora_out[:, :size]
                    else:
                        lora_out = torch.nn.functional.pad(lora_out, (0, size - lora_out.shape[1]))
                output_view[i:end, offset:offset+size].add_(lora_out.to(output.dtype), alpha=alpha)
            else:
                output_view[i:end].add_(lora_out.to(output.dtype), alpha=alpha)
                
            del lora_out
            
            if _ENABLE_CUDA_SYNC and i % (chunk_rows * 4) == 0:
                torch.cuda.synchronize()


try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False

_loading_stats = {"int8_direct": 0, "quantized_on_fly": 0, "excluded": 0, "quantize_time": 0.0}

_inference_kernel_stats = {
    "hadamard_quip_triton": 0,
    "hadamard_quip_pytorch": 0,
    "hadamard_quip_dequant": 0,
    "w8a8_static": 0,
    "w8a8_dynamic": 0,
    "w8a8_dynamic_triton": 0,
    "dequant_fallback": 0,
    "non_quantized": 0,
}

_LOG_KERNEL_PER_LAYER = os.environ.get("INT8_LOG_KERNEL", "0") == "1"

_shown_initial_kernel_info = False


def _log_kernel_usage(kernel_type: str, layer_name: str = ""):
    """Log which kernel is being used for inference."""
    global _shown_initial_kernel_info
    _inference_kernel_stats[kernel_type] = _inference_kernel_stats.get(kernel_type, 0) + 1
    
    if _LOG_KERNEL_PER_LAYER:
        print(f"[INT8 KERNEL] {kernel_type}: {layer_name}")
        _shown_initial_kernel_info = True


def print_kernel_summary():
    """Print a summary of kernel usage during inference."""
    total = sum(_inference_kernel_stats.values())
    if total == 0:
        return
    
    print("\n[INT8 KERNEL SUMMARY] Inference kernel usage:")
    print("-" * 50)
    for kernel, count in sorted(_inference_kernel_stats.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {kernel:30s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':30s}: {total:5d}")
    print("-" * 50)


def reset_kernel_stats():
    """Reset kernel usage statistics."""
    global _shown_initial_kernel_info
    for key in _inference_kernel_stats:
        _inference_kernel_stats[key] = 0
    _shown_initial_kernel_info = False
    print("[INT8] Kernel statistics reset")


def cast_int8_weights(layer, device):
    """Cast INT8 weights to target device, supporting offloading."""
    weight = layer.weight.to(device)

    weight_scale = getattr(layer, "weight_scale", None)
    if isinstance(weight_scale, torch.Tensor):
        weight_scale = weight_scale.to(weight.device)

    input_scale = getattr(layer, "input_scale", None)
    if isinstance(input_scale, torch.Tensor):
        input_scale = input_scale.to(weight.device)

    bias = getattr(layer, "bias", None)
    if bias is not None:
        bias = bias.to(weight.device)

    return weight, weight_scale, input_scale, bias


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """Custom ComfyUI operations for INT8 tensorwise quantization."""
        excluded_names = []
        offload_to_cpu = False
        chunk_size = 0
        auto_convert_to_int8 = True
        debug_mode = False
        
        class Linear(manual_cast.Linear):
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self.input_scale = None
                self._is_quantized = False
                self._weight_scale_reshaped = False
                self.compute_dtype = torch.bfloat16
                self.comfy_cast_weights = True
                self.lora_patches = []
                
                self._hadamard_quip = False
                self._hadamard_size_in = 0
                self._hadamard_size_out = 0
                self._sign_row = None
                self._sign_col = None
            
            def reset_parameters(self):
                return None
            
            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """Load INT8 weights and scales directly from state dict."""
                weight_key = prefix + "weight"
                weight_scale_keys = [
                    prefix + "weight_scale",
                    prefix + "scale",
                    prefix + "weight.scale",
                    prefix + "weight.weight_scale",
                    prefix + "scale_weight",
                ]
                weight_scale = None
                found_weight_scale_key = None
                for key in weight_scale_keys:
                    if key in state_dict:
                        if weight_scale is not None:
                            print(f"[INT8 WARNING] Multiple weight_scale keys found for {prefix}: "
                                  f"'{found_weight_scale_key}' and '{key}'. Using '{key}'.")
                        weight_scale = state_dict.pop(key)
                        found_weight_scale_key = key
                
                input_scale_keys = [
                    prefix + "input_scale",
                    prefix + "act_scale",
                    prefix + "input.scale",
                    prefix + "input.input_scale",
                ]
                input_scale = None
                found_input_scale_key = None
                for key in input_scale_keys:
                    if key in state_dict:
                        if input_scale is not None:
                            print(f"[INT8 WARNING] Multiple input_scale keys found for {prefix}: "
                                  f"'{found_input_scale_key}' and '{key}'. Using '{key}'.")
                        input_scale = state_dict.pop(key)
                        found_input_scale_key = key

                quip_s_u = state_dict.pop(prefix + "quip_s_u", None)
                quip_s_v = state_dict.pop(prefix + "quip_s_v", None)
                
                hadamard_quip = state_dict.pop(prefix + "hadamard_quip", False)
                hadamard_size_in = state_dict.pop(prefix + "hadamard_size_in", 0)
                hadamard_size_out = state_dict.pop(prefix + "hadamard_size_out", 0)
                sign_row = state_dict.pop(prefix + "sign_row", None)
                sign_col = state_dict.pop(prefix + "sign_col", None)
                
                bias_key = prefix + "bias"
                
                state_dict.pop(prefix + "comfy_quant", None)
                
                weight_tensor = state_dict.pop(weight_key, None)
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is None:
                        print(f"INT8 Loader: WARNING - Found INT8 weight but NO SCALE for {prefix.rstrip('.')}")
                    
                    if Int8TensorwiseOps.debug_mode and weight_scale is not None:
                        if isinstance(weight_scale, torch.Tensor):
                            scale_info = f"tensor{tuple(weight_scale.shape)}"
                            if torch.isnan(weight_scale).any():
                                scale_info += " [NaN DETECTED!]"
                            if torch.isinf(weight_scale).any():
                                scale_info += " [INF DETECTED!]"
                            if (weight_scale == 0).any():
                                scale_info += " [ZERO DETECTED!]"
                        else:
                            scale_info = f"scalar={weight_scale}"
                        print(f"[DEBUG] Loading {prefix.rstrip('.')}: dtype={weight_tensor.dtype}, scale={scale_info}")
                    
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_hadamard_quip = hadamard_quip if isinstance(hadamard_quip, bool) else (hadamard_quip.item() if isinstance(hadamard_quip, torch.Tensor) else False)
                        
                        if is_excluded:
                            self._is_quantized = False
                            if is_hadamard_quip:
                                dequant_weight = dequantize(
                                    weight_tensor, weight_scale,
                                    hadamard_quip=True,
                                    hadamard_size_in=int(hadamard_size_in) if isinstance(hadamard_size_in, torch.Tensor) else hadamard_size_in,
                                    hadamard_size_out=int(hadamard_size_out) if isinstance(hadamard_size_out, torch.Tensor) else hadamard_size_out,
                                    sign_row=sign_row,
                                    sign_col=sign_col
                                ).to(torch.bfloat16)
                            elif quip_s_u is not None:
                                dequant_weight = dequantize(weight_tensor, weight_scale, quip_s_u, quip_s_v).to(torch.bfloat16)
                            else:
                                dequant_weight = dequantize(weight_tensor, weight_scale).to(torch.bfloat16)
                            self.weight = nn.Parameter(dequant_weight, requires_grad=False)
                            _loading_stats["excluded"] += 1
                            if is_hadamard_quip:
                                print(f"INT8 Loader: Dequantized Hadamard-QuIP layer {prefix.rstrip('.')}")
                            elif quip_s_u is not None:
                                print(f"INT8 Loader: Dequantized QuIP# layer {prefix.rstrip('.')}")
                        elif is_hadamard_quip:
                            if not _HADAMARD_QUIP_ENABLED:
                                self._is_quantized = True
                                self._hadamard_quip = False
                                dequant_weight = dequantize(
                                    weight_tensor, weight_scale,
                                    hadamard_quip=True,
                                    hadamard_size_in=int(hadamard_size_in) if isinstance(hadamard_size_in, torch.Tensor) else hadamard_size_in,
                                    hadamard_size_out=int(hadamard_size_out) if isinstance(hadamard_size_out, torch.Tensor) else hadamard_size_out,
                                    sign_row=sign_row,
                                    sign_col=sign_col
                                )
                                q_weight, q_scale = quantize_int8_tensorwise(dequant_weight)
                                self.weight = nn.Parameter(q_weight, requires_grad=False)
                                weight_scale = q_scale
                                _loading_stats["int8_direct"] += 1
                                print(f"INT8 Loader: Converted Hadamard-QuIP layer {prefix.rstrip('.')} to standard W8A8")
                            else:
                                self._is_quantized = True
                                self._hadamard_quip = True
                                self._hadamard_size_in = int(hadamard_size_in) if isinstance(hadamard_size_in, torch.Tensor) else hadamard_size_in
                                self._hadamard_size_out = int(hadamard_size_out) if isinstance(hadamard_size_out, torch.Tensor) else hadamard_size_out
                                if sign_row is not None:
                                    if "_sign_row" in self.__dict__:
                                        del self.__dict__["_sign_row"]
                                    self.register_buffer("_sign_row", sign_row)
                                if sign_col is not None:
                                    if "_sign_col" in self.__dict__:
                                        del self.__dict__["_sign_col"]
                                    self.register_buffer("_sign_col", sign_col)
                                self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                                _loading_stats["int8_direct"] += 1
                                print(f"INT8 Loader: Loaded Hadamard-QuIP layer {prefix.rstrip('.')} (kept INT8)")
                        elif quip_s_u is not None:
                            self._is_quantized = False
                            dequant_weight = dequantize(weight_tensor, weight_scale, quip_s_u, quip_s_v).to(torch.bfloat16)
                            self.weight = nn.Parameter(dequant_weight, requires_grad=False)
                            _loading_stats["excluded"] += 1
                            print(f"INT8 Loader: Dequantized QuIP# layer {prefix.rstrip('.')}")
                        else:
                            self._is_quantized = True
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                            _loading_stats["int8_direct"] += 1
                        
                        if weight_scale is not None:
                            if hasattr(self, "weight_scale"):
                                if "weight_scale" in self._buffers:
                                    del self._buffers["weight_scale"]
                                else:
                                    del self.weight_scale
                            
                            if isinstance(weight_scale, torch.Tensor):
                                if weight_scale.numel() == self.out_features:
                                    weight_scale = weight_scale.float().reshape(-1)
                                elif weight_scale.numel() == 1:
                                    weight_scale = weight_scale.float().view(())
                                else:
                                    if weight_scale.numel() < self.out_features:
                                        pad_size = self.out_features - weight_scale.numel()
                                        last_val = weight_scale[-1].item()
                                        weight_scale = torch.cat([
                                            weight_scale.float(),
                                            torch.full((pad_size,), last_val, dtype=torch.float32, device=weight_scale.device)
                                        ])
                                    else:
                                        weight_scale = weight_scale[:self.out_features].float()
                            else:
                                weight_scale = torch.tensor(weight_scale, dtype=torch.float32)
                            
                            self.register_buffer("weight_scale", weight_scale)
                            self._weight_scale_reshaped = True
                        
                        if input_scale is not None:
                            if hasattr(self, "input_scale"):
                                if "input_scale" in self._buffers:
                                    del self._buffers["input_scale"]
                                else:
                                    del self.input_scale
                            
                            if isinstance(input_scale, torch.Tensor):
                                input_scale = input_scale.float().mean().view(())
                            else:
                                input_scale = torch.tensor(input_scale, dtype=torch.float32)
                                
                            self.register_buffer("input_scale", input_scale)
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        if not Int8TensorwiseOps.auto_convert_to_int8:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                            is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                            
                            if is_excluded or is_dim1:
                                if Int8TensorwiseOps.debug_mode:
                                    print(f"Skipping dynamic quantization for {prefix.rstrip('.')} ({'excluded' if is_excluded else 'dim1/1D'})")
                                self._is_quantized = False
                                self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                                _loading_stats["excluded"] += 1
                            else:
                                if Int8TensorwiseOps.debug_mode:
                                    print(f"Auto-converting to INT8: {prefix.rstrip('.')} ({weight_tensor.dtype} -> INT8)")
                                
                                import time as _time_mod
                                _q_start = _time_mod.perf_counter()
                                q_weight, q_scale = quantize_int8_tensorwise(weight_tensor)
                                self.weight = nn.Parameter(q_weight, requires_grad=False)
                                self.weight_scale = q_scale
                                self._is_quantized = True
                                _loading_stats["quantized_on_fly"] += 1
                                _loading_stats["quantize_time"] += _time_mod.perf_counter() - _q_start
                    else:
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)
                
                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None
                
                for k in list(missing_keys):
                    if k.startswith(prefix) and (k.endswith(".weight_scale") or k.endswith(".input_scale")):
                        missing_keys.remove(k)
                
                for k in list(unexpected_keys):
                    if k.startswith(prefix) and (".quip_s_u" in k or ".quip_s_v" in k or ".comfy_quant" in k):
                        unexpected_keys.remove(k)
            
            def forward(self, x: Tensor) -> Tensor:
                """Forward pass using torch._int_mm for quantized weights."""
                if Int8TensorwiseOps.debug_mode and self._is_quantized:
                    if self.weight_scale is not None:
                        if isinstance(self.weight_scale, torch.Tensor):
                            if torch.isnan(self.weight_scale).any():
                                print(f"[DEBUG FORWARD] NaN weight_scale in {self}! Input shape: {x.shape}")
                            if torch.isinf(self.weight_scale).any():
                                print(f"[DEBUG FORWARD] Inf weight_scale in {self}! Input shape: {x.shape}")
                            if (self.weight_scale == 0).any():
                                print(f"[DEBUG FORWARD] Zero weight_scale in {self}! Input shape: {x.shape}")
                
                if not self._is_quantized:
                    _log_kernel_usage("non_quantized", self.__class__.__name__)
                    
                    lora_patches = getattr(self, "lora_patches", [])
                    has_lora = len(lora_patches) > 0
                    
                    compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                    acc_dtype = compute_dtype if (has_lora and Int8TensorwiseOps.offload_to_cpu) else (torch.float32 if has_lora else compute_dtype)
                    
                    if _DEBUG_MODE and has_lora:
                        print(f"[DEBUG LoRA NON-QUANTIZED] {self}: x={x.dtype}, compute={compute_dtype}, acc={acc_dtype}")
                    
                    if _COMFY_OPS_AVAILABLE:
                        weight, bias, offload_stream = cast_bias_weight(
                            self, x, offloadable=True
                        )
                        if has_lora:
                            weight = weight.to(acc_dtype)
                            if bias is not None:
                                bias = bias.to(acc_dtype)
                            x_compute = x.to(acc_dtype)
                        else:
                            x_compute = x
                        out = F.linear(x_compute, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                    else:
                        weight = self.weight.to(x.device, dtype=acc_dtype if has_lora else x.dtype)
                        bias = getattr(self, "bias", None)
                        if bias is not None:
                            bias = bias.to(x.device, dtype=acc_dtype if has_lora else x.dtype)
                        x_compute = x.to(acc_dtype) if has_lora else x
                        out = F.linear(x_compute, weight, bias)
                    
                    if lora_patches:
                        x_shape = x.shape
                        x_2d = x.reshape(-1, x_shape[-1])
                        
                        if _DEBUG_MODE:
                            print(f"[DEBUG LoRA APPLY NON-QUANTIZED] x_2d={x_2d.dtype}, out={out.dtype}")
                        
                        if x_2d.dtype != out.dtype:
                            _debug(f"[DEBUG LoRA APPLY NON-QUANTIZED] Casting x_2d from {x_2d.dtype} to {out.dtype}")
                            x_2d = x_2d.to(out.dtype)
                        
                        for idx, patch_data in enumerate(lora_patches):
                            if not isinstance(patch_data, (list, tuple)):
                                if _DEBUG_MODE:
                                    print(f"[DEBUG LoRA] Skipping invalid patch {idx}: not a list/tuple")
                                continue
                            
                            if len(patch_data) < 3:
                                if _DEBUG_MODE:
                                    print(f"[DEBUG LoRA] Skipping invalid patch {idx}: needs at least 3 elements, got {len(patch_data)}")
                                continue
                            
                            down, up, alpha = patch_data[0], patch_data[1], patch_data[2]
                            down_scale, up_scale = None, None
                            offset, size = 0, 0
                            
                            if len(patch_data) >= 5:
                                down_scale, up_scale = patch_data[3], patch_data[4]
                            if len(patch_data) >= 7:
                                offset, size = patch_data[5], patch_data[6]
                            elif len(patch_data) == 6:
                                offset = patch_data[5]
                                size = 0
                            
                            if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                                if _DEBUG_MODE:
                                    print(f"[DEBUG LoRA] Skipping patch {idx}: down/up are not tensors")
                                continue
                            
                            is_int8 = down.dtype == torch.int8 and up.dtype == torch.int8
                            
                            if _DEBUG_MODE:
                                print(f"[DEBUG LoRA PATCH {idx} NON-QUANTIZED] is_int8={is_int8}, down={down.dtype}, up={up.dtype}, target={out.dtype}")
                            
                            if is_int8 and down_scale is not None and up_scale is not None:
                                x_shape = x.shape
                                x_2d = x.reshape(-1, x_shape[-1])
                                
                                if x_2d.shape[0] > 16:
                                    down_int8 = down.to(device=out.device, non_blocking=True)
                                    up_int8 = up.to(device=out.device, non_blocking=True)
                                    down_scale_t = down_scale.to(device=out.device, non_blocking=True) if isinstance(down_scale, Tensor) else down_scale
                                    up_scale_t = up_scale.to(device=out.device, non_blocking=True) if isinstance(up_scale, Tensor) else up_scale
                                    
                                    chunked_int8_lora_forward(
                                        x_2d, down_int8, up_int8,
                                        down_scale_t, up_scale_t,
                                        alpha, out,
                                        offset=offset, size=size
                                    )
                                    
                                    del down_int8, up_int8, down_scale_t, up_scale_t
                                    
                                    if out.numel() > CHUNK_THRESHOLD_ELEMENTS and _should_clear_cache():
                                        torch.cuda.empty_cache()
                                else:
                                    d = dequantize(down, down_scale).to(device=out.device, dtype=out.dtype)
                                    u = dequantize(up, up_scale).to(device=out.device, dtype=out.dtype)
                                    
                                    is_fp32 = (out.dtype == torch.float32)
                                    if is_fp32 and torch.cuda.is_available():
                                        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
                                        torch.backends.cuda.matmul.allow_tf32 = False
                                        torch.set_float32_matmul_precision("highest")
                                    
                                    try:
                                        lora_out = F.linear(F.linear(x_2d, d), u)
                                        if size > 0:
                                            out_view = out.reshape(-1, out.shape[-1])
                                            if lora_out.shape[1] != size:
                                                if lora_out.shape[1] > size:
                                                    lora_out = lora_out[:, :size]
                                                else:
                                                    lora_out = torch.nn.functional.pad(lora_out, (0, size - lora_out.shape[1]))
                                            out_view[:, offset:offset+size].add_(lora_out, alpha=alpha)
                                        else:
                                            lora_out = lora_out.reshape(*out.shape)
                                            out.add_(lora_out, alpha=alpha)
                                        del lora_out
                                    finally:
                                        if is_fp32 and torch.cuda.is_available():
                                            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
                                    
                                    del d, u
                            else:
                                d = down.to(device=out.device, dtype=out.dtype)
                                u = up.to(device=out.device, dtype=out.dtype)
                                
                                chunked_lora_forward(x_2d, d, u, alpha, out, offset=offset, size=size)
                                
                                del d, u
                    
                    if has_lora and out.dtype != compute_dtype:
                        _debug(f"[DEBUG LoRA CAST NON-QUANTIZED] Casting output from {out.dtype} to {compute_dtype}")
                        out = out.to(compute_dtype)
                    elif _DEBUG_MODE and has_lora:
                        print(f"[DEBUG LoRA CAST NON-QUANTIZED] Output already in {compute_dtype}")
                    
                    return out
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                weight, weight_scale, input_scale, bias = cast_int8_weights(self, x_2d.device)
                
                lora_patches = getattr(self, "lora_patches", [])
                has_lora = len(lora_patches) > 0
                acc_dtype = compute_dtype if (has_lora and Int8TensorwiseOps.offload_to_cpu) else (torch.float32 if has_lora else compute_dtype)
                
                if bias is not None and has_lora and bias.dtype != acc_dtype:
                    bias = bias.to(acc_dtype)
                
                if _DEBUG_MODE and has_lora:
                    print(f"[DEBUG LoRA] Layer: {self}, x={x.dtype}, compute={compute_dtype}, acc={acc_dtype}, weight={weight.dtype}, bias={bias.dtype if bias is not None else None}")
                
                if self._hadamard_quip and x_2d.shape[0] > 16:
                    hadamard_kernels = _get_hadamard_quip_kernels()
                    if hadamard_kernels is not None:
                        triton_hadamard_quip_linear, pytorch_hadamard_quip_linear = hadamard_kernels
                        
                        sign_row = getattr(self, "_sign_row", None)
                        sign_col = getattr(self, "_sign_col", None)
                        
                        N, K = weight.shape
                        needs_padding = False
                        if self._hadamard_size_out > 0 and N < self._hadamard_size_out:
                            needs_padding = True
                        if self._hadamard_size_in > 0 and K < self._hadamard_size_in:
                            needs_padding = True
                        
                        if needs_padding:
                            _debug(f"[DIAG] Padding weight from {weight.shape} to Hadamard dimensions")
                            if self._hadamard_size_out > 0 and N < self._hadamard_size_out:
                                weight = torch.nn.functional.pad(weight, (0, 0, 0, self._hadamard_size_out - N))
                            if self._hadamard_size_in > 0 and K < self._hadamard_size_in:
                                weight = torch.nn.functional.pad(weight, (0, self._hadamard_size_in - K, 0, 0))
                            _debug(f"[DIAG] Padded weight shape: {weight.shape}")
                        
                        if sign_row is not None and sign_row.device != x_2d.device:
                            sign_row = sign_row.to(x_2d.device)
                        if sign_col is not None and sign_col.device != x_2d.device:
                            sign_col = sign_col.to(x_2d.device)
                        
                        if _DEBUG_MODE:
                            print(f"[DIAG] Hadamard-QuIP forward: x_2d={x_2d.device}, weight={weight.device}")
                            if sign_row is not None:
                                print(f"[DIAG]   sign_row={sign_row.device}, shape={tuple(sign_row.shape)}")
                            if sign_col is not None:
                                print(f"[DIAG]   sign_col={sign_col.device}, shape={tuple(sign_col.shape)}")
                        
                        try:
                            _log_kernel_usage("hadamard_quip_triton", self.__class__.__name__)
                            y = triton_hadamard_quip_linear(
                                x_2d, weight, weight_scale, bias, acc_dtype,
                                hadamard_size_in=self._hadamard_size_in,
                                hadamard_size_out=self._hadamard_size_out,
                                sign_row=sign_row,
                                sign_col=sign_col,
                                out_features=self.out_features,
                                use_fp32_output=has_lora
                            )
                            if y.shape[-1] != self.out_features:
                                _debug(f"[DIAG] Slicing output from {y.shape[-1]} to {self.out_features}")
                                y = y[:, :self.out_features]
                        except Exception as e:
                            print(f"[DIAG FALLBACK] Triton failed: {type(e).__name__}: {e}")
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                reserved = torch.cuda.memory_reserved() / 1024**3
                                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                                print(f"[DIAG FALLBACK] Memory: Alloc={allocated:.2f}GB, Res={reserved:.2f}GB, Total={total:.2f}GB")
                                print(f"[DIAG FALLBACK] Shapes: input={tuple(x_2d.shape)}, weight={tuple(weight.shape)}, Hadamard={self._hadamard_size_in}/{self._hadamard_size_out}")
                                torch.cuda.empty_cache()
                                print(f"[DIAG FALLBACK] Cache cleared, retrying with PyTorch...")
                            _debug(f"[DEBUG] Hadamard-QuIP Triton failed, falling back to PyTorch: {e}")
                            try:
                                _log_kernel_usage("hadamard_quip_pytorch", self.__class__.__name__)
                                y = pytorch_hadamard_quip_linear(
                                    x_2d, weight, weight_scale, bias, acc_dtype,
                                    hadamard_size_in=self._hadamard_size_in,
                                    hadamard_size_out=self._hadamard_size_out,
                                    sign_row=sign_row,
                                    sign_col=sign_col,
                                    out_features=self.out_features
                                )
                                if y.shape[-1] != self.out_features:
                                    _debug(f"[DIAG] Slicing output from {y.shape[-1]} to {self.out_features}")
                                    y = y[:, :self.out_features]
                                print(f"[DIAG FALLBACK] PyTorch fallback succeeded!")
                            except torch.OutOfMemoryError as oom_e:
                                print(f"[DIAG FALLBACK] PyTorch fallback also failed with OOM: {oom_e}")
                                raise
                    else:
                        _log_kernel_usage("hadamard_quip_dequant", self.__class__.__name__)
                        sign_row = getattr(self, "_sign_row", None)
                        sign_col = getattr(self, "_sign_col", None)
                        if sign_row is not None and sign_row.device != x_2d.device:
                            sign_row = sign_row.to(x_2d.device)
                        if sign_col is not None and sign_col.device != x_2d.device:
                            sign_col = sign_col.to(x_2d.device)
                        
                        w_float = dequantize(
                            weight, weight_scale,
                            hadamard_quip=True,
                            hadamard_size_in=self._hadamard_size_in,
                            hadamard_size_out=self._hadamard_size_out,
                            sign_row=sign_row,
                            sign_col=sign_col
                        ).to(x.dtype)
                        y = F.linear(x_2d, w_float, bias)
                        del w_float
                elif x_2d.shape[0] > 16:
                    if input_scale is not None:
                        _log_kernel_usage("w8a8_static", self.__class__.__name__)
                        y = int8_forward_static(
                            x_2d, weight, weight_scale,
                            input_scale, bias, acc_dtype,
                            chunk_size=Int8TensorwiseOps.chunk_size,
                            has_lora=bool(lora_patches),
                            offload_to_cpu=Int8TensorwiseOps.offload_to_cpu
                        )
                    else:
                        _log_kernel_usage("w8a8_dynamic", self.__class__.__name__)
                        y = int8_forward_dynamic(
                            x_2d, weight, weight_scale,
                            bias, acc_dtype,
                            chunk_size=Int8TensorwiseOps.chunk_size,
                            has_lora=bool(lora_patches),
                            offload_to_cpu=Int8TensorwiseOps.offload_to_cpu
                        )
                else:
                    _log_kernel_usage("dequant_fallback", self.__class__.__name__)
                    if self._hadamard_quip:
                        sign_row = getattr(self, "_sign_row", None)
                        sign_col = getattr(self, "_sign_col", None)
                        if sign_row is not None and sign_row.device != x_2d.device:
                            sign_row = sign_row.to(x_2d.device)
                        if sign_col is not None and sign_col.device != x_2d.device:
                            sign_col = sign_col.to(x_2d.device)
                        
                        w_float = dequantize(
                            weight, weight_scale,
                            hadamard_quip=True,
                            hadamard_size_in=self._hadamard_size_in,
                            hadamard_size_out=self._hadamard_size_out,
                            sign_row=sign_row,
                            sign_col=sign_col
                        ).to(x.dtype)
                    else:
                        w_float = dequantize(weight, weight_scale).to(x.dtype)
                    y = F.linear(x_2d, w_float, bias)
                    del w_float
                
                lora_patches = getattr(self, "lora_patches", [])
                if lora_patches:
                    _log_memory(f"before LoRA application ({len(lora_patches)} patches)", is_forward=True)
                    
                    if _DEBUG_MODE:
                        print(f"[DEBUG LoRA APPLY] quantized={self._is_quantized}, x={x.dtype}, x_2d={x_2d.dtype}, y={y.dtype}, compute={compute_dtype}, acc={acc_dtype}")
                    
                    if x_2d.dtype != y.dtype:
                        _debug(f"[DEBUG LoRA APPLY] Casting x_2d from {x_2d.dtype} to {y.dtype}")
                        x_2d = x_2d.to(y.dtype)
                    
                    for idx, patch_data in enumerate(lora_patches):
                        if not isinstance(patch_data, (list, tuple)):
                            if _DEBUG_MODE:
                                print(f"[DEBUG LoRA] Skipping invalid patch {idx}: not a list/tuple")
                            continue
                        
                        if len(patch_data) < 3:
                            if _DEBUG_MODE:
                                print(f"[DEBUG LoRA] Skipping invalid patch {idx}: needs at least 3 elements, got {len(patch_data)}")
                            continue
                        
                        down, up, alpha = patch_data[0], patch_data[1], patch_data[2]
                        down_scale, up_scale = None, None
                        offset, size = 0, 0
                        
                        if len(patch_data) >= 5:
                            down_scale, up_scale = patch_data[3], patch_data[4]
                        if len(patch_data) >= 7:
                            offset, size = patch_data[5], patch_data[6]
                        elif len(patch_data) == 6:
                            offset = patch_data[5]
                            size = 0
                        
                        if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                            if _DEBUG_MODE:
                                print(f"[DEBUG LoRA] Skipping patch {idx}: down/up are not tensors")
                            continue
                        
                        is_int8 = down.dtype == torch.int8 and up.dtype == torch.int8
                        
                        if _DEBUG_MODE:
                            print(f"[DEBUG LoRA PATCH {idx}] is_int8={is_int8}, down={down.dtype}{tuple(down.shape)}, up={up.dtype}{tuple(up.shape)}")
                            if is_int8:
                                print(f"[DEBUG LoRA PATCH {idx}]   scales: down={down_scale}, up={up_scale}")
                            print(f"[DEBUG LoRA PATCH {idx}]   target={y.dtype}, alpha={alpha}")
                        
                        if is_int8 and down_scale is not None and up_scale is not None:
                            _log_memory(f"INT8 LoRA {idx} start", is_forward=True)
                            
                            if x_2d.shape[0] > 16:
                                down_int8 = down.to(device=y.device, non_blocking=True)
                                up_int8 = up.to(device=y.device, non_blocking=True)
                                down_scale_t = down_scale.to(device=y.device, non_blocking=True) if isinstance(down_scale, Tensor) else down_scale
                                up_scale_t = up_scale.to(device=y.device, non_blocking=True) if isinstance(up_scale, Tensor) else up_scale
                                
                                _log_tensor_size(f"LoRA {idx} down (INT8)", down_int8, is_forward=True)
                                _log_tensor_size(f"LoRA {idx} up (INT8)", up_int8, is_forward=True)
                                
                                chunked_int8_lora_forward(
                                    x_2d, down_int8, up_int8, 
                                    down_scale_t, up_scale_t, 
                                    alpha, y,
                                    offset=offset, size=size
                                )
                                
                                del down_int8, up_int8, down_scale_t, up_scale_t
                                
                                if y.numel() > CHUNK_THRESHOLD_ELEMENTS and _should_clear_cache():
                                    torch.cuda.empty_cache()
                            else:
                                d = dequantize(down, down_scale).to(device=y.device, dtype=y.dtype)
                                u = dequantize(up, up_scale).to(device=y.device, dtype=y.dtype)
                                
                                is_fp32 = (y.dtype == torch.float32)
                                if is_fp32 and torch.cuda.is_available():
                                    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
                                    torch.backends.cuda.matmul.allow_tf32 = False
                                    torch.set_float32_matmul_precision("highest")
                                
                                try:
                                    lora_out = F.linear(F.linear(x_2d, d), u)
                                    if size > 0:
                                        y_view = y.reshape(-1, y.shape[-1])
                                        y_view[:, offset:offset+size].add_(lora_out.reshape(-1, size), alpha=alpha)
                                    else:
                                        y.add_(lora_out, alpha=alpha)
                                    del lora_out
                                finally:
                                    if is_fp32 and torch.cuda.is_available():
                                        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
                                
                                del d, u
                            
                            _log_memory(f"INT8 LoRA {idx} end", is_forward=True)
                        else:
                            target_dtype = y.dtype
                            d = down.to(device=y.device, dtype=target_dtype).contiguous()
                            u = up.to(device=y.device, dtype=target_dtype).contiguous()
                            
                            chunked_lora_forward(x_2d, d, u, alpha, y, offset=offset, size=size)
                            
                            del d, u
                    _log_memory("after LoRA application", is_forward=True)

                if has_lora and y.dtype != compute_dtype:
                    _debug(f"[DEBUG LoRA CAST] Casting output from {y.dtype} to {compute_dtype}")
                    y = y.to(compute_dtype)
                elif _DEBUG_MODE and has_lora:
                    print(f"[DEBUG LoRA CAST] Output already in {compute_dtype}")

                del weight, weight_scale, input_scale, bias
                
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        class GroupNorm(manual_cast.GroupNorm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class LayerNorm(manual_cast.LayerNorm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class Conv2d(manual_cast.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
            
            def forward(self, x):
                out = super().forward(x)
                lora_patches = getattr(self, "lora_patches", [])
                if lora_patches:
                    for patch_data in lora_patches:
                        if len(patch_data) == 3:
                            down, up, alpha = patch_data
                        else:
                            down, up, alpha, _, _ = patch_data
                        
                        d = down.to(device=out.device, dtype=out.dtype)
                        u = up.to(device=out.device, dtype=out.dtype)
                        lora_out = F.conv2d(F.conv2d(x, d), u)
                        out.add_(lora_out, alpha=alpha)
                        del lora_out, d, u
                return out
        
        class Conv3d(manual_cast.Conv3d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class Embedding(manual_cast.Embedding):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")

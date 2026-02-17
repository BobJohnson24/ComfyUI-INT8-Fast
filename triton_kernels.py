import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math
import gc
from typing import Optional
import os

def gpu_cc(device=None):
    """Get GPU compute capability as integer (e.g., 86, 89, 75)."""
    if device is None:
        device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def gpu_family(device=None):
    """
    Get GPU architecture family.
    Returns: 'ada' (RTX 40xx), 'ampere' (A100/RTX 30xx), 'turing' (RTX 20xx), or 'older'
    """
    cc = gpu_cc(device)
    if cc >= 89:
        return "ada"      # RTX 40xx (SM89)
    if cc >= 80:
        return "ampere"   # SM80/86 (A100 / RTX 30xx)
    if cc >= 75:
        return "turing"   # RTX 20xx (SM75)
    return "older"


# Cache for the selected kernel to avoid repeated lookups
_gpu_arch_cache = None

def get_gpu_arch():
    """Get cached GPU architecture."""
    global _gpu_arch_cache
    if _gpu_arch_cache is None and torch.cuda.is_available():
        _gpu_arch_cache = gpu_family()
    return _gpu_arch_cache


# Environment variable controls for kernel selection
_FORCE_KERNEL = os.environ.get("INT8_FORCE_KERNEL", "").lower()
_LOG_KERNEL_SELECTION = os.environ.get("INT8_LOG_KERNEL", "0") == "1"

_shown_kernel_info = False

def pick_gemm_kernel_with_logging(device=None):
    """Select the best GEMM kernel with optional logging."""
    global _shown_kernel_info

    if not torch.cuda.is_available():
        return _int8_matmul_dequant_kernel

    # Re-read environment variable to allow runtime override
    force_kernel = os.environ.get("INT8_FORCE_KERNEL", "").lower()
    if force_kernel:
        if force_kernel == "ampere":
            if not _shown_kernel_info and _LOG_KERNEL_SELECTION:
                print(f"[INT8 KERNEL] Forced Ampere autotuned kernel")
                _shown_kernel_info = True
            return _int8_gemm_dequant_ampere
        elif force_kernel in ("fixed", "fallback"):
            if not _shown_kernel_info and _LOG_KERNEL_SELECTION:
                print(f"[INT8 KERNEL] Forced fixed fallback kernel")
                _shown_kernel_info = True
            return _int8_matmul_dequant_kernel
    
    arch = gpu_family(device)
    kernel = _KERNEL_TABLE.get(arch, _int8_matmul_dequant_kernel)
    
    if not _shown_kernel_info and _LOG_KERNEL_SELECTION:
        cc = gpu_cc(device)
        kernel_name = "Ampere autotuned" if kernel == _int8_gemm_dequant_ampere else "Fixed fallback"
        print(f"[INT8 KERNEL] Detected GPU: {arch} (CC {cc})")
        print(f"[INT8 KERNEL] Selected: {kernel_name}")
        _shown_kernel_info = True
    
    return kernel


# Replace the original pick_gemm_kernel with the logging version
pick_gemm_kernel = pick_gemm_kernel_with_logging


# Hadamard QuIP Kernel Implementation
# Based on QuIP paper - uses Hadamard transforms for INT8 quantization.
# Formula: W = D1 @ H @ W_q @ H.T @ D2 (H is self-inverse, no storage needed)

_HADAMARD_DIAGNOSTICS = os.environ.get("HADAMARD_DIAGNOSTICS", "0") == "1"

# Helper Functions

_HADAMARD_CHUNK_SIZE = int(os.environ.get("HADAMARD_CHUNK_SIZE", "2048"))

def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return (n > 0) and (n & (n - 1) == 0)


def next_power_of_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    if n <= 0:
        return 1
    return 2 ** (n - 1).bit_length()


def _maybe_defragment_memory(required_bytes: int, verbose: bool = False) -> bool:
    """Check for memory fragmentation and defragment if needed."""
    if not torch.cuda.is_available():
        return False
    
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    
    free_memory = total - allocated
    free_in_reserved = reserved - allocated
    
    if free_memory >= required_bytes:
        if reserved > 0:
            fragmentation_ratio = free_in_reserved / reserved
            if fragmentation_ratio > 0.3 and free_in_reserved < required_bytes:
                if verbose:
                    print(f"[DEFRAG] Fragmentation detected: {fragmentation_ratio:.1%}")
                    print(f"[DEFRAG] Required: {required_bytes/1024**3:.2f}GB, Free: {free_in_reserved/1024**3:.2f}GB")
                
                gc.collect()
                torch.cuda.empty_cache()
                
                if verbose:
                    new_reserved = torch.cuda.memory_reserved()
                    print(f"[DEFRAG] Cache cleared: {reserved/1024**3:.2f}GB -> {new_reserved/1024**3:.2f}GB")
                
                return True
    
    return False


def _safe_contiguous_clone(x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """Safely create a contiguous clone with fragmentation handling."""
    if not x.is_contiguous():
        required_bytes = x.numel() * x.element_size()
        _maybe_defragment_memory(required_bytes, verbose=verbose)
        
        try:
            return x.contiguous().clone()
        except torch.OutOfMemoryError:
            if verbose:
                print("[SAFE CLONE] First attempt failed, trying aggressive defrag...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            return x.contiguous().clone()
    else:
        required_bytes = x.numel() * x.element_size()
        _maybe_defragment_memory(required_bytes, verbose=verbose)
        
        try:
            return x.clone()
        except torch.OutOfMemoryError:
            if verbose:
                print("[SAFE CLONE] First attempt failed, trying aggressive defrag...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            return x.clone()


# Kernel 3: Fast Walsh-Hadamard Transform (FWHT)

@triton.jit
def _hadamard_stage_kernel(
    in_ptr,
    out_ptr,
    n: tl.constexpr,
    stage: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single stage of butterfly Hadamard transform."""
    pid = tl.program_id(0)
    batch_offset = pid * n
    
    h = 1 << stage
    
    for tile_start in range(0, n, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        
        x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
        
        pair_offs = offs ^ h
        pair_mask = pair_offs < n
        x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
        
        is_upper = (offs & h) == 0
        new_val = tl.where(is_upper, x + x_pair, x_pair - x)
        
        tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)


@triton.jit
def _hadamard_normalize_kernel(
    x_ptr,
    out_ptr,
    n: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply normalization after all butterfly stages."""
    pid = tl.program_id(0)
    batch_offset = pid * n
    
    for tile_start in range(0, n, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        
        x = tl.load(x_ptr + batch_offset + offs, mask=mask, other=0.0)
        x = x * scale
        tl.store(out_ptr + batch_offset + offs, x, mask=mask)


@triton.jit
def _hadamard_stage_with_signs_kernel(
    in_ptr,
    out_ptr,
    sign_row_ptr,
    sign_col_ptr,
    n: tl.constexpr,
    stage: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_ROW_SIGNS: tl.constexpr,
    HAS_COL_SIGNS: tl.constexpr,
    APPLY_SIGNS: tl.constexpr,
):
    """Single butterfly stage with optional sign multiplication on first stage."""
    pid = tl.program_id(0)
    batch_offset = pid * n
    
    if APPLY_SIGNS and HAS_ROW_SIGNS:
        sign_r = tl.load(sign_row_ptr + pid)
    
    h = 1 << stage
    
    for tile_start in range(0, n, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        
        x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
        
        if APPLY_SIGNS:
            if HAS_ROW_SIGNS:
                x = x * sign_r
            if HAS_COL_SIGNS:
                sign_c = tl.load(sign_col_ptr + offs, mask=mask, other=1.0)
                x = x * sign_c
        
        pair_offs = offs ^ h
        pair_mask = pair_offs < n
        x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
        
        if APPLY_SIGNS:
            if HAS_ROW_SIGNS:
                x_pair = x_pair * sign_r
            if HAS_COL_SIGNS:
                sign_c_pair = tl.load(sign_col_ptr + pair_offs, mask=pair_mask, other=1.0)
                x_pair = x_pair * sign_c_pair
        
        is_upper = (offs & h) == 0
        new_val = tl.where(is_upper, x + x_pair, x_pair - x)
        
        tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)


@triton.jit
def _copy_kernel(
    in_ptr,
    out_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple copy kernel for buffer management."""
    pid = tl.program_id(0)
    batch_offset = pid * n
    
    for tile_start in range(0, n, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        
        x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
        tl.store(out_ptr + batch_offset + offs, x, mask=mask)


_hadamard_configs = [
    triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}),
    triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
    triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
    triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 16}),
]

@triton.autotune(
    configs=_hadamard_configs,
    key=['n'],
)
@triton.jit
def _hadamard_autotuned_stage_kernel(
    in_ptr,
    out_ptr,
    n: tl.constexpr,
    stage: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned single stage butterfly operation."""
    pid = tl.program_id(0)
    batch_offset = pid * n
    
    h = 1 << stage
    
    for tile_start in range(0, n, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        
        x = tl.load(in_ptr + batch_offset + offs, mask=mask, other=0.0)
        pair_offs = offs ^ h
        pair_mask = pair_offs < n
        x_pair = tl.load(in_ptr + batch_offset + pair_offs, mask=pair_mask, other=0.0)
        
        is_upper = (offs & h) == 0
        new_val = tl.where(is_upper, x + x_pair, x_pair - x)
        
        tl.store(out_ptr + batch_offset + offs, new_val, mask=mask)


def triton_hadamard_transform(
    x: torch.Tensor,
    normalize: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Fast Hadamard Transform using Triton kernel."""
    n = x.shape[-1]
    batch = x.numel() // n
    
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be power of 2, got {n}")
    
    if _HADAMARD_DIAGNOSTICS and torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3
        tensor_size_gb = x.numel() * x.element_size() / 1024**3
        print(f"[DIAG HADAMARD] Input shape: {x.shape}, dtype: {x.dtype}")
        print(f"[DIAG HADAMARD] Tensor size: {tensor_size_gb:.2f} GB")
        print(f"[DIAG HADAMARD] Memory before: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
    
    inplace_requested = output is not None and output.data_ptr() == x.data_ptr()
    
    # Guard against unsafe in-place on non-contiguous tensors
    if inplace_requested and not x.is_contiguous():
        raise ValueError("In-place Hadamard requires contiguous tensor.")
    
    if inplace_requested:
        if not x.is_contiguous():
            output.copy_(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        x = output
    else:
        try:
            x = _safe_contiguous_clone(x, verbose=_HADAMARD_DIAGNOSTICS)
            if _HADAMARD_DIAGNOSTICS and torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                print(f"[DIAG HADAMARD] After clone: Allocated={allocated_after:.2f}GB")
        except torch.OutOfMemoryError as e:
            if _HADAMARD_DIAGNOSTICS and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[DIAG HADAMARD] OOM during clone!")
                print(f"[DIAG HADAMARD] Allocated: {allocated:.2f}GB / {total:.2f}GB")
            raise
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if output is None:
        out = torch.empty_like(x)
    else:
        out = output
    
    if inplace_requested:
        temp = torch.empty_like(x)
        buf_a = x
        buf_b = temp
    else:
        buf_a = x
        buf_b = out
    
    log_n = int(math.log2(n))
    BLOCK_SIZE = min(n, 2048)
    grid = (batch,)
    
    for stage in range(log_n):
        if stage % 2 == 0:
            in_buf, out_buf = buf_a, buf_b
        else:
            in_buf, out_buf = buf_b, buf_a
        
        _hadamard_autotuned_stage_kernel[grid](
            in_buf, out_buf,
            n=n,
            stage=stage,
        )
    
    if normalize:
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        scale = 1.0 / math.sqrt(n)
        _hadamard_normalize_kernel[grid](
            final_buf, out,
            n=n,
            scale=scale,
            BLOCK_SIZE=BLOCK_SIZE,
            num_stages=1,
        )
    else:
        if (log_n - 1) % 2 == 0:
            final_buf = buf_b
        else:
            final_buf = buf_a
        
        if final_buf.data_ptr() != out.data_ptr():
            _copy_kernel[grid](
                final_buf, out,
                n=n,
                BLOCK_SIZE=BLOCK_SIZE,
                num_stages=1,
            )
    
    return out


def fast_hadamard_transform_2d(
    x: torch.Tensor,
    normalize: bool = True,
    inplace: bool = False
) -> torch.Tensor:
    """Apply Fast Hadamard Transform to both dimensions: result = H @ X @ H.T"""
    x_orig = x
    if not inplace:
        x = x.clone()

    # First pass: transform columns WITHOUT normalization
    x = triton_hadamard_transform(x, normalize=False, output=x if inplace else None)

    # Second pass: transform rows; must be contiguous
    x_t = x.transpose(-2, -1).contiguous()
    x_t = triton_hadamard_transform(x_t, normalize=normalize)
    x_out = x_t.transpose(-2, -1).contiguous()

    if inplace:
        x_orig.copy_(x_out)
        return x_orig
    return x_out

# Kernel: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)


@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row_ptr = x_ptr + row_idx * n_elements
    y_row_ptr = y_ptr + row_idx * n_elements

    max_val = 0.0
    for i in tl.static_range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        local_max = tl.max(abs_x, axis=0)
        max_val = tl.maximum(max_val, local_max)

    scale = tl.maximum(max_val / 127.0, 1e-30)

    for i in tl.static_range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        q_f = tl.clamp(x / scale, -128.0, 127.0)
        q_i = libdevice.rint(q_f).to(tl.int32)
        tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)

    tl.store(s_ptr + row_idx, scale.to(tl.float32))


def triton_quantize_rowwise(x: torch.Tensor):
    """
    Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch] (float32) - scale is flat vector for kernel compatibility
    """
    rows, cols = x.shape
    y = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 4096 if cols > 4096 else triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128: BLOCK_SIZE = 128
    
    grid = (rows,)
    _quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
    # Ensure scale is flat, contiguous, and float32 for kernel compatibility
    return y, s.reshape(-1).contiguous().to(torch.float32)


# Kernel: INT8 GEMM + Fused Dequantization Epilogue
# Uses fixed config - autotune was removed due to correctness issues with large matrices.

_FIXED_BLOCK_M = 128
_FIXED_BLOCK_N = 128
_FIXED_BLOCK_K = 32
_FIXED_GROUP_SIZE_M = 8
_FIXED_NUM_WARPS = 4
_FIXED_NUM_STAGES = 4

# Ampere-optimized blockwise configuration
_AMPERE_BLOCKWISE_BLOCK_M = 128
_AMPERE_BLOCKWISE_BLOCK_N = 128
_AMPERE_BLOCKWISE_BLOCK_K = 64
_AMPERE_BLOCKWISE_NUM_WARPS = 8
_AMPERE_BLOCKWISE_NUM_STAGES = 4
_AMPERE_BLOCKWISE_GROUP_M = 8

@triton.jit
def _int8_matmul_dequant_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_PER_CHANNEL_SCALE: tl.constexpr,
):
    """
    Computes: C = ((A * B) * (scale_a * scale_b)) + bias
    A: [M, K] int8
    B: [N, K] int8 (transposed access via strides)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=1.0)
    if HAS_PER_CHANNEL_SCALE:
        scale_b = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    else:
        scale_b = tl.load(b_scale_ptr)

    c = acc.to(tl.float32) * (scale_a[:, None] * scale_b[None, :])

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Ampere-Tuned INT8 GEMM Kernel
# Optimized for Ampere (SM80/86) - larger BK=64, more warps for int8 dot performance.
# B matrix is packed as [K, N] for direct tl.dot without transpose.

_AMPERE_BLOCK_M = 128
_AMPERE_BLOCK_N = 128
_AMPERE_BLOCK_K = 64
_AMPERE_GROUP_M = 8
_AMPERE_NUM_WARPS = 8
_AMPERE_NUM_STAGES = 4

def _kernel_grid_ampere(M, N):
    """Grid calculation for fixed Ampere block sizes."""
    return (triton.cdiv(M, _AMPERE_BLOCK_M) * triton.cdiv(N, _AMPERE_BLOCK_N),)

@triton.jit
def _int8_gemm_dequant_ampere(
    a_ptr, b_ptr, c_ptr,                    # A: [M,K] int8, B: [K,N] int8 packed
    a_scale_ptr, b_scale_ptr, bias_ptr,     # a_scale: [M], b_scale: [1] or [N], bias: [N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,                   # B strides for [K, N]
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    HAS_PER_CHANNEL_SCALE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
        BM: tl.constexpr = _AMPERE_BLOCK_M,
    BN: tl.constexpr = _AMPERE_BLOCK_N,
    BK: tl.constexpr = _AMPERE_BLOCK_K,
    GROUP_M: tl.constexpr = _AMPERE_GROUP_M,
):
    """
    Ampere-tuned INT8 GEMM with fused dequantization.
    Uses tl.make_block_ptr for pipelining and better performance on SM80/86.
    Weight matrix B is packed as [K, N] for direct tl.dot without transpose.
    
    A: [M, K] int8
    B: [K, N] int8 (packed, contiguous in K dimension)
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)

    # Swizzled grouping for better L2 cache utilization
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    # Use block pointers for pipelined loads (Ampere-optimized)
    # A is [M, K] with row-major layout
    A = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0),
    )
    # B is [K, N] with column-major access pattern (K changes fastest)
    B = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(0, 1),
    )

    acc = tl.zeros((BM, BN), dtype=tl.int32)

    # Main loop with pipelined loads
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(A, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
        b = tl.load(B, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
        # Direct dot product: a [BM, BK] @ b [BK, BN] -> [BM, BN]
        acc += tl.dot(a, b)
        A = tl.advance(A, (0, BK))
        B = tl.advance(B, (BK, 0))

    # Fused epilogue: dequantize, apply bias
    a_s = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
    if HAS_PER_CHANNEL_SCALE:
        b_s = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    else:
        b_s = tl.load(b_scale_ptr).to(tl.float32)

    out = acc.to(tl.float32) * (a_s[:, None] * b_s[None, :])

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out += bias[None, :]

    out = out.to(OUT_DTYPE)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel dispatch table
_KERNEL_TABLE = {
    "ada": _int8_gemm_dequant_ampere,
    "ampere": _int8_gemm_dequant_ampere,
    "turing": _int8_gemm_dequant_ampere,  # Uses fallback configs in autotune
    "older": _int8_matmul_dequant_kernel,  # Use fixed kernel for older GPUs
}


def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=None, use_fp32_output: bool = False):
    """
    Fused pipeline for W8A8 Linear Layer with architecture-specific kernel selection.
    
    Args:
        x: Input tensor
        weight: INT8 weight tensor
        weight_scale: Scale for weight dequantization
        bias: Optional bias tensor
        compute_dtype: Target compute dtype (defaults to x.dtype to avoid FP16/BF16 mismatch)
        use_fp32_output: If True, output FP32 for LoRA accumulation precision
    """
    # Default compute_dtype to input dtype (avoids hardcoded FP16/BF16 mismatch)
    if compute_dtype is None:
        compute_dtype = x.dtype
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    
    M, K = x_2d.shape
    N = weight.shape[0]

    x_int8, x_scale = triton_quantize_rowwise(x_2d)

    # Use FP32 output dtype when requested for LoRA accumulation
    output_dtype = torch.float32 if use_fp32_output else compute_dtype
    output = torch.empty((M, N), device=x.device, dtype=output_dtype)
    
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    elif weight_scale.numel() == 1:
        if weight_scale.device != x.device:
            weight_scale = weight_scale.to(x.device).reshape(1)
        else:
            weight_scale = weight_scale.reshape(1)
    else:
        weight_scale = weight_scale.reshape(-1).contiguous()

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x
    has_per_channel_scale = weight_scale.numel() > 1
    
    # Select kernel based on GPU architecture
    kernel = pick_gemm_kernel(x.device)
    
    # Determine output dtype for Triton
    # When use_fp32_output is True (e.g., for LoRA accumulation), force FP32 output
    if use_fp32_output:
        out_tl = tl.float32
    elif compute_dtype == torch.float16:
        out_tl = tl.float16
    elif compute_dtype == torch.bfloat16:
        out_tl = tl.bfloat16
    else:
        out_tl = tl.float32
    
    if kernel == _int8_gemm_dequant_ampere:
        # Pack weight from [N, K] to [K, N] for optimal access pattern
        if weight.stride(0) != weight.shape[1] or weight.stride(1) != 1:
            weight_packed = weight.t().contiguous()
        else:
            weight_packed = weight.t()
        
        # Use autotuned kernel with dynamic grid
        grid = _kernel_grid_ampere(M, N)
        kernel[grid](
            a_ptr=x_int8,
            b_ptr=weight_packed,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=weight_scale,
            bias_ptr=bias_ptr,
            M=M, N=N, K=K,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=weight_packed.stride(0), stride_bn=weight_packed.stride(1),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            HAS_BIAS=has_bias,
            HAS_PER_CHANNEL_SCALE=has_per_channel_scale,
            OUT_DTYPE=out_tl,
        )
    else:
        # Use fixed fallback kernel for older GPUs
        grid = (triton.cdiv(M, _FIXED_BLOCK_M) * triton.cdiv(N, _FIXED_BLOCK_N),)
        kernel[grid](
            a_ptr=x_int8,
            b_ptr=weight,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=weight_scale,
            bias_ptr=bias_ptr,
            M=M, N=N, K=K,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=weight.stride(1), stride_bn=weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            BLOCK_M=_FIXED_BLOCK_M,
            BLOCK_N=_FIXED_BLOCK_N,
            BLOCK_K=_FIXED_BLOCK_K,
            GROUP_SIZE_M=_FIXED_GROUP_SIZE_M,
            HAS_BIAS=has_bias,
            HAS_PER_CHANNEL_SCALE=has_per_channel_scale,
            num_warps=_FIXED_NUM_WARPS,
            num_stages=_FIXED_NUM_STAGES,
        )
    
    return output.reshape(x_shape_orig[:-1] + (N,))


def triton_hadamard_quip_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = None,
    hadamard_size_in: int = 0,
    hadamard_size_out: int = 0,
    sign_row: Optional[torch.Tensor] = None,
    sign_col: Optional[torch.Tensor] = None,
    out_features: Optional[int] = None,
    use_fp32_output: bool = False,
) -> torch.Tensor:
    """
    Fused pipeline for Hadamard-QuIP W8A8 Linear Layer.
    
    Mathematical Foundation:
        W' = H_row @ W @ H_col  (stored weight)
        W = H_row @ W' @ H_col  (original weight, since H is self-inverse)
        
    Computation Order:
        1. Apply column signs: x_s = x * D_col
        2. Apply FWHT to input: x_h = FWHT(x_s)
        3. Quantize activation: x_q = quantize(x_h)
        4. INT8 matmul: y_h = matmul(x_q, W.T)
        5. Dequantize
        6. Apply FWHT to output: y_h2 = FWHT(y_h)
        7. Apply row signs: y = y_h2 * D_row
    
    Args:
        x: Input tensor
        weight: INT8 weight tensor
        weight_scale: Scale for weight dequantization
        bias: Optional bias tensor
        compute_dtype: Target compute dtype (defaults to x.dtype to avoid FP16/BF16 mismatch)
        hadamard_size_in: Size of Hadamard transform for input dimension
        hadamard_size_out: Size of Hadamard transform for output dimension
        sign_row: Row signs for Hadamard-QuIP (diagonal D1)
        sign_col: Column signs for Hadamard-QuIP (diagonal D2)
        out_features: Original output features (for padding removal)
        use_fp32_output: If True, output FP32 for LoRA accumulation precision
    """
    # Default compute_dtype to input dtype (avoids hardcoded FP16/BF16 mismatch)
    if compute_dtype is None:
        compute_dtype = x.dtype
    
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    
    M, K = x_2d.shape
    N = weight.shape[0]
    
    if out_features is not None:
        original_N = out_features
    elif hadamard_size_out > 0 and bias is not None:
        original_N = bias.shape[0]
    else:
        original_N = None
    
    if hadamard_size_in > 0 or hadamard_size_out > 0:
        if hadamard_size_in > 0:
            assert is_power_of_two(hadamard_size_in), \
                f"hadamard_size_in ({hadamard_size_in}) must be a power of 2"
        if hadamard_size_out > 0:
            assert is_power_of_two(hadamard_size_out), \
                f"hadamard_size_out ({hadamard_size_out}) must be a power of 2"
        
        if hadamard_size_out > 0 and hadamard_size_in > 0:
            N_orig, K_orig = weight.shape
            if N_orig != hadamard_size_out or K_orig != hadamard_size_in:
                # Pad weight dimensions
                if N_orig < hadamard_size_out:
                    weight = torch.nn.functional.pad(weight, (0, 0, 0, hadamard_size_out - N_orig))
                    # CRITICAL: Also pad weight_scale to match padded weight dimensions
                    if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() > 1:
                        if weight_scale.numel() == N_orig:
                            # Pad scale vector with small epsilon to avoid division issues
                            pad_size = hadamard_size_out - N_orig
                            # Use the last scale value for padding (better than zeros)
                            last_scale = weight_scale[-1].item() if weight_scale.numel() > 0 else 1e-30
                            weight_scale = torch.cat([
                                weight_scale.reshape(-1),
                                torch.full((pad_size,), last_scale, dtype=weight_scale.dtype, device=weight_scale.device)
                            ])
                
                if K_orig < hadamard_size_in:
                    weight = torch.nn.functional.pad(weight, (0, hadamard_size_in - K_orig, 0, 0))
                
                # Update N to reflect padded dimensions
                N = weight.shape[0]
        
        if sign_row is not None and hadamard_size_out > 0:
            assert sign_row.shape[0] >= hadamard_size_out, \
                f"sign_row length ({sign_row.shape[0]}) must be >= hadamard_size_out ({hadamard_size_out})"
        
        if sign_col is not None and hadamard_size_in > 0:
            assert sign_col.shape[0] >= hadamard_size_in, \
                f"sign_col length ({sign_col.shape[0]}) must be >= hadamard_size_in ({hadamard_size_in})"
        
        if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() > 1:
            assert weight_scale.numel() == N, \
                f"Per-channel weight_scale length ({weight_scale.numel()}) must match weight output dim ({N}) after padding"
    
    # Chunked processing for large batches to prevent OOM
    global _HADAMARD_CHUNK_SIZE
    if M > _HADAMARD_CHUNK_SIZE and _HADAMARD_CHUNK_SIZE > 0:
        if _HADAMARD_DIAGNOSTICS:
            print(f"[DIAG CHUNK] Processing batch: {M} rows in chunks of {_HADAMARD_CHUNK_SIZE}")

        output_chunks = []

        for start_idx in range(0, M, _HADAMARD_CHUNK_SIZE):
            end_idx = min(start_idx + _HADAMARD_CHUNK_SIZE, M)
            chunk = x_2d[start_idx:end_idx]

            output_chunk = triton_hadamard_quip_linear(
                chunk, weight, weight_scale, None, compute_dtype,
                hadamard_size_in=hadamard_size_in,
                hadamard_size_out=hadamard_size_out,
                sign_row=sign_row,
                sign_col=sign_col,
                out_features=original_N,   # <-- critical: propagate out_features
                use_fp32_output=use_fp32_output
            )
            
            output_chunks.append(output_chunk)
            del chunk
            
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
        
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        
        output = torch.cat(output_chunks, dim=0)
        del output_chunks
        
        # Add bias in the correct dtype (output might be FP32 for LoRA accumulation)
        if bias is not None:
            output = output + bias.to(output.dtype)
        
        return output.reshape(x_shape_orig[:-1] + (output.shape[1],))
    
    # Apply Hadamard transform to input if specified
    if hadamard_size_in > 0 and is_power_of_two(hadamard_size_in):
        if K < hadamard_size_in:
            pad_size = hadamard_size_in - K
            x_padded = torch.nn.functional.pad(x_2d, (0, pad_size))
        else:
            x_padded = x_2d
        
        x_signed = x_padded
        if sign_col is not None:
            sign_col_expanded = sign_col[:hadamard_size_in]
            x_signed = x_signed * sign_col_expanded.unsqueeze(0)
        
        x_hadamard = triton_hadamard_transform(x_signed, normalize=True)
        x_for_quant = x_hadamard
    else:
        x_for_quant = x_2d
    
    x_int8, x_scale = triton_quantize_rowwise(x_for_quant)
    # Ensure activation scale is flat, contiguous, and float32 for kernel compatibility
    # The kernel assumes a_scale_ptr is a flat vector of length M (offs_m indexing)
    x_scale = x_scale.reshape(-1).contiguous().to(torch.float32)
    
    # Use FP32 output dtype when requested for LoRA accumulation
    output_dtype = torch.float32 if use_fp32_output else compute_dtype
    output = torch.empty((M, N), device=x.device, dtype=output_dtype)
    
    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    elif weight_scale.numel() == 1:
        if weight_scale.device != x.device:
            weight_scale = weight_scale.to(x.device).reshape(1)
        else:
            weight_scale = weight_scale.reshape(1)
    else:
        weight_scale = weight_scale.reshape(-1).contiguous()
    
    # For Hadamard-QuIP: disable bias in GEMM when output transform is needed
    # Bias will be added ONCE at the end, in the correct space
    has_output_hadamard = hadamard_size_out > 0 and is_power_of_two(hadamard_size_out)
    fused_bias = bias if not has_output_hadamard else None
    has_bias = fused_bias is not None
    bias_ptr = fused_bias if has_bias else x
    has_per_channel_scale = weight_scale.numel() > 1
    
    # Select kernel based on GPU architecture
    kernel = pick_gemm_kernel(x.device)
    
    # Determine output dtype for Triton
    # When use_fp32_output is True (e.g., for LoRA accumulation), force FP32 output
    if use_fp32_output:
        out_tl = tl.float32
    elif compute_dtype == torch.float16:
        out_tl = tl.float16
    elif compute_dtype == torch.bfloat16:
        out_tl = tl.bfloat16
    else:
        out_tl = tl.float32
    
    actual_K = x_int8.shape[1]
    weight_K = weight.shape[1]
    if actual_K != weight_K:
        if _HADAMARD_DIAGNOSTICS:
            print(f"[DIAG SHAPE MISMATCH] x_int8 K={actual_K}, weight K={weight_K}")
        if actual_K < weight_K:
            x_int8 = torch.nn.functional.pad(x_int8, (0, weight_K - actual_K))
            actual_K = weight_K
        elif actual_K > weight_K:
            x_int8 = x_int8[:, :weight_K]
            actual_K = weight_K
    
    if kernel == _int8_gemm_dequant_ampere:
        # Pack weight from [N, K] to [K, N] for optimal access pattern
        if weight.stride(0) != weight.shape[1] or weight.stride(1) != 1:
            weight_packed = weight.t().contiguous()
        else:
            weight_packed = weight.t()
        
        # Use autotuned kernel with dynamic grid
        grid = _kernel_grid_ampere(M, N)
        kernel[grid](
            a_ptr=x_int8,
            b_ptr=weight_packed,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=weight_scale,
            bias_ptr=bias_ptr,
            M=M, N=N, K=actual_K,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=weight_packed.stride(0), stride_bn=weight_packed.stride(1),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            HAS_BIAS=has_bias,
            HAS_PER_CHANNEL_SCALE=has_per_channel_scale,
            OUT_DTYPE=out_tl,
        )
    else:
        # Use fixed fallback kernel for older GPUs
        grid = (triton.cdiv(M, _FIXED_BLOCK_M) * triton.cdiv(N, _FIXED_BLOCK_N),)
        kernel[grid](
            a_ptr=x_int8,
            b_ptr=weight,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=weight_scale,
            bias_ptr=bias_ptr,
            M=M, N=N, K=actual_K,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=weight.stride(1), stride_bn=weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            BLOCK_M=_FIXED_BLOCK_M,
            BLOCK_N=_FIXED_BLOCK_N,
            BLOCK_K=_FIXED_BLOCK_K,
            GROUP_SIZE_M=_FIXED_GROUP_SIZE_M,
            HAS_BIAS=has_bias,
            HAS_PER_CHANNEL_SCALE=has_per_channel_scale,
            num_warps=_FIXED_NUM_WARPS,
            num_stages=_FIXED_NUM_STAGES,
        )
    
    # Apply inverse Hadamard transform to output if specified
    if hadamard_size_out > 0 and is_power_of_two(hadamard_size_out):
        if N < hadamard_size_out:
            pad_size = hadamard_size_out - N
            output_padded = torch.nn.functional.pad(output, (0, pad_size))
        else:
            output_padded = output
        
        output_hadamard = triton_hadamard_transform(output_padded, normalize=True)
        
        if sign_row is not None:
            sign_row_expanded = sign_row[:hadamard_size_out]
            output_hadamard = output_hadamard * sign_row_expanded.unsqueeze(0)
        
        if original_N is not None and original_N < hadamard_size_out:
            output = output_hadamard[:, :original_N]
        elif N < hadamard_size_out:
            output = output_hadamard[:, :N]
        else:
            output = output_hadamard
    
    # Add bias ONCE at the very end, in the original output space
    # (after inverse Hadamard transform and sign application)
    if bias is not None and has_output_hadamard:
        output = output + bias.to(output.dtype)
    
    return output.reshape(x_shape_orig[:-1] + (output.shape[1],))


# PyTorch Fallback for Hadamard-QuIP (when Triton not available)

def pytorch_hadamard_quip_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float16,
    hadamard_size_in: int = 0,
    hadamard_size_out: int = 0,
    sign_row: Optional[torch.Tensor] = None,
    sign_col: Optional[torch.Tensor] = None,
    out_features: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch fallback for Hadamard-QuIP linear layer.
    
    IMPORTANT: This function assumes `weight` is in Hadamard domain (W_q),
    NOT spatial domain. This matches the Triton implementation which computes:
        y = FWHT( matmul( FWHT(x * sign_col), W_q^T ) ) * sign_row
    
    If you have a weight in spatial domain W = H @ W_q @ H, you must convert
    it to Hadamard domain before calling this function.
    """
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    
    M, K = x_2d.shape
    N = weight.shape[0]
    
    if out_features is not None:
        original_N = out_features
    elif hadamard_size_out > 0 and bias is not None:
        original_N = bias.shape[0]
    else:
        original_N = None
    
    # Apply Hadamard transform to input if specified
    # This transforms x from spatial domain to Hadamard domain
    if hadamard_size_in > 0 and is_power_of_two(hadamard_size_in):
        if K < hadamard_size_in:
            pad_size = hadamard_size_in - K
            x_padded = torch.nn.functional.pad(x_2d, (0, pad_size))
        else:
            x_padded = x_2d
        
        if sign_col is not None:
            sign_col_expanded = sign_col[:hadamard_size_in]
            x_padded = x_padded * sign_col_expanded.unsqueeze(0)
        
        x_hadamard = _pytorch_fwht(x_padded)
        x_for_matmul = x_hadamard
    else:
        x_for_matmul = x_2d
    
    # Weight is in Hadamard domain (W_q) - dequantize and matmul in Hadamard space
    w_float = weight.float() * weight_scale
    output = torch.matmul(x_for_matmul.to(compute_dtype), w_float.T.to(compute_dtype))
    
    # Apply inverse Hadamard to output (transform back to spatial domain)
    if hadamard_size_out > 0 and is_power_of_two(hadamard_size_out):
        if N < hadamard_size_out:
            pad_size = hadamard_size_out - N
            required_bytes = output.shape[0] * hadamard_size_out * output.element_size()
            _maybe_defragment_memory(required_bytes, verbose=_HADAMARD_DIAGNOSTICS)
            output_padded = torch.nn.functional.pad(output, (0, pad_size))
        else:
            output_padded = output
        
        output_hadamard = _pytorch_fwht(output_padded)
        
        if sign_row is not None:
            sign_row_expanded = sign_row[:hadamard_size_out]
            output_hadamard = output_hadamard * sign_row_expanded.unsqueeze(0)
        
        if original_N is not None and original_N < hadamard_size_out:
            output = output_hadamard[:, :original_N]
        elif N < hadamard_size_out:
            output = output_hadamard[:, :N]
        else:
            output = output_hadamard
    
    if bias is not None:
        output = output + bias.to(output.dtype)
    
    return output.reshape(x_shape_orig[:-1] + (output.shape[1],))


def _pytorch_fwht(x: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of Fast Walsh-Hadamard Transform."""
    original_shape = x.shape
    n = original_shape[-1]
    
    if not is_power_of_two(n):
        raise ValueError(f"Last dimension must be power of 2, got {n}")
    
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


# Kernel Numerical Accuracy Tests

@torch.no_grad()
def ref_int8_linear(x_int8, x_scale, w_int8, w_scale, out_dtype=torch.float32):
    """
    Reference implementation for INT8 linear in float.
    
    Formula: Y = (A_q * s_A) @ (B_q * s_B).T
    where s_A is per-row and s_B is per-output-channel if per-channel.
    
    Args:
        x_int8: [M, K] int8 - quantized activations
        x_scale: [M] fp32 - per-row activation scales
        w_int8: [N, K] int8 - quantized weights
        w_scale: [N] fp32 or [1] - per-channel or per-tensor weight scales
        out_dtype: output dtype
    
    Returns:
        Y: [M, N] - dequantized output
    """
    # x_int8: [M,K] int8, x_scale: [M] fp32
    # w_int8: [N,K] int8, w_scale: [N] fp32 or [1]
    x_f = x_int8.float() * x_scale.view(-1, 1)
    if w_scale.numel() == 1:
        w_f = w_int8.float() * w_scale.view(1, 1)
    else:
        w_f = w_int8.float() * w_scale.view(-1, 1)
    y = x_f @ w_f.t()
    return y.to(out_dtype)


def compare_kernels(x, w_int8, w_scale, compute_dtype=torch.bfloat16, force_shapes=None):
    """
    Compare Ampere kernel vs fallback kernel vs float reference.
    
    Args:
        x: Input tensor [..., K] (will be flattened to 2D)
        w_int8: INT8 weight tensor [N, K]
        w_scale: Weight scale [N] or scalar
        compute_dtype: Compute dtype for kernels
        force_shapes: Optional tuple (M, N, K) to override auto-detected shapes
    
    Returns:
        dict with comparison metrics
    """
    import os
    
    x2 = x.reshape(-1, x.shape[-1])
    x_int8, x_scale = triton_quantize_rowwise(x2)
    x_scale = x_scale.reshape(-1).contiguous().to(torch.float32)
    w_scale = w_scale.reshape(-1).contiguous().to(torch.float32)

    # Force Ampere kernel
    os.environ["INT8_FORCE_KERNEL"] = "ampere"
    global _shown_kernel_info
    _shown_kernel_info = False  # Reset to show kernel selection
    y_amp = triton_int8_linear(x2, w_int8, w_scale, compute_dtype=compute_dtype)
    
    # Force fallback kernel
    os.environ["INT8_FORCE_KERNEL"] = "fallback"
    _shown_kernel_info = False
    y_fbk = triton_int8_linear(x2, w_int8, w_scale, compute_dtype=compute_dtype)
    
    # Clear force to restore normal behavior
    os.environ["INT8_FORCE_KERNEL"] = ""
    
    # Reference implementation
    y_ref = ref_int8_linear(x_int8, x_scale, w_int8, w_scale, out_dtype=torch.float32)

    ya = y_amp.float()
    yf = y_fbk.float()
    yr = y_ref.float()
    
    results = {
        "amp_vs_ref_max_abs": (ya - yr).abs().max().item(),
        "fbk_vs_ref_max_abs": (yf - yr).abs().max().item(),
        "amp_vs_fbk_max_abs": (ya - yf).abs().max().item(),
        "amp_vs_ref_mean_abs": (ya - yr).abs().mean().item(),
        "fbk_vs_ref_mean_abs": (yf - yr).abs().mean().item(),
        "amp_vs_fbk_mean_abs": (ya - yf).abs().mean().item(),
        "amp_vs_ref_rel_error": ((ya - yr).abs() / (yr.abs() + 1e-8)).max().item(),
        "fbk_vs_ref_rel_error": ((yf - yr).abs() / (yr.abs() + 1e-8)).max().item(),
        "y_amp": y_amp,
        "y_fbk": y_fbk,
        "y_ref": y_ref,
    }
    
    return results


def print_comparison_results(results):
    """Print formatted comparison results."""
    print("=" * 60)
    print("W8A8 Kernel Numerical Comparison Results")
    print("=" * 60)
    print(f"Ampere vs Reference max abs error: {results['amp_vs_ref_max_abs']:.6f}")
    print(f"Fallback vs Reference max abs error: {results['fbk_vs_ref_max_abs']:.6f}")
    print(f"Ampere vs Fallback max abs error: {results['amp_vs_fbk_max_abs']:.6f}")
    print("-" * 60)
    print(f"Ampere vs Reference mean abs error: {results['amp_vs_ref_mean_abs']:.6f}")
    print(f"Fallback vs Reference mean abs error: {results['fbk_vs_ref_mean_abs']:.6f}")
    print(f"Ampere vs Fallback mean abs error: {results['amp_vs_fbk_mean_abs']:.6f}")
    print("-" * 60)
    print(f"Ampere vs Reference max relative error: {results['amp_vs_ref_rel_error']:.6f}")
    print(f"Fallback vs Reference max relative error: {results['fbk_vs_ref_rel_error']:.6f}")
    print("=" * 60)
    
    # Numerical closeness check
    if results['amp_vs_fbk_max_abs'] < 0.1:
        print("✓ Ampere and Fallback kernels are NUMERICALLY CLOSE")
    else:
        print("✗ WARNING: Ampere and Fallback kernels differ significantly!")
    
    if results['amp_vs_ref_max_abs'] < 1.0:
        print("✓ Ampere kernel matches reference within tolerance")
    else:
        print("✗ WARNING: Ampere kernel deviates from reference!")
        
    if results['fbk_vs_ref_max_abs'] < 1.0:
        print("✓ Fallback kernel matches reference within tolerance")
    else:
        print("✗ WARNING: Fallback kernel deviates from reference!")
    print("=" * 60)


def test_kernel_accuracy(shapes=None, device="cuda", compute_dtype=torch.bfloat16):
    """
    Run kernel accuracy test with specified matrix shapes.
    
    Args:
        shapes: List of (M, N, K) tuples to test, or None for default shapes
        device: Device to run on
        compute_dtype: Compute dtype for kernels
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel test")
        return
    
    default_shapes = [
        (128, 256, 512),
        (256, 512, 768),
        (512, 1024, 1024),
        (1024, 2048, 2048),
        (311, 3072, 27648),  # Large matrix from the codebase
    ]
    
    shapes = shapes or default_shapes
    
    print(f"\nRunning kernel accuracy tests on {device}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"Testing {len(shapes)} shape configurations\n")
    
    for M, N, K in shapes:
        print(f"\nShape: M={M}, N={N}, K={K}")
        print("-" * 40)
        
        # Generate random input and weights
        torch.manual_seed(42)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        
        # Quantize weights
        w_abs_max = w.abs().amax(dim=1, keepdim=True)
        w_scale = (w_abs_max / 127.0).clamp(min=1e-30).reshape(-1)
        w_int8 = (w / w_scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
        
        results = compare_kernels(x, w_int8, w_scale, compute_dtype=compute_dtype)
        print_comparison_results(results)
        
        # Cleanup
        del x, w, w_int8, w_scale
        del results['y_amp'], results['y_fbk'], results['y_ref']
        torch.cuda.empty_cache()


# =============================================================================
# Block-wise INT8 Quantization Kernels
# =============================================================================
# These kernels perform INT8 matrix multiplication with block-wise scales.
# Weights stay in INT8 during inference for ~4x memory savings vs BF16.
# Block-wise scales have shape [N, num_blocks, 1] where num_blocks = K // block_size

# Cache for packed weights to avoid re-packing on every forward pass
_weight_packing_cache = {}
_MAX_CACHE_ENTRIES = 8  # Limit cache size for low VRAM


def _get_cache_key(weight, weight_scale):
    """Generate cache key from weight tensor identity and shape."""
    return (id(weight), weight.shape, weight_scale.shape if isinstance(weight_scale, torch.Tensor) else weight_scale)


def clear_weight_packing_cache():
    """Clear the weight packing cache. Call when running low on memory."""
    global _weight_packing_cache
    _weight_packing_cache.clear()
    gc.collect()


def pack_weight_for_blockwise(weight_nk: torch.Tensor) -> torch.Tensor:
    """
    Pack weight from [N, K] to [K, N] layout for coalesced access.
    
    Args:
        weight_nk: INT8 weight tensor [N, K]
    
    Returns:
        Packed weight [K, N] contiguous along N dimension
    """
    # Transpose and ensure contiguous
    # This is a one-time cost per layer during model loading or first forward
    weight_kn = weight_nk.t().contiguous()
    return weight_kn


def pack_scales_for_blockwise(scale_n_blocks: torch.Tensor) -> torch.Tensor:
    """
    Pack scales from [N, num_blocks] to [num_blocks, N] layout.
    
    Args:
        scale_n_blocks: Block-wise scales [N, num_blocks] or [N, num_blocks, 1]
    
    Returns:
        Packed scales [num_blocks, N] contiguous along N
    """
    # Remove trailing dim if present [N, num_blocks, 1] -> [N, num_blocks]
    if scale_n_blocks.ndim == 3:
        scale_n_blocks = scale_n_blocks.squeeze(-1)
    
    # Transpose to [num_blocks, N] and ensure contiguous
    scale_blocks_n = scale_n_blocks.t().contiguous()
    return scale_blocks_n


def get_packed_weight_and_scales(weight: torch.Tensor, weight_scale: torch.Tensor,
                                  use_cache: bool = True) -> tuple:
    """
    Get packed weight and scales for optimized blockwise matmul.
    
    For low VRAM:
    - Uses caching to avoid re-packing
    - Cache is LRU with limited entries
    - Can be disabled if memory is critical
    
    Args:
        weight: INT8 weight [N, K]
        weight_scale: Block-wise scales [N, num_blocks, 1] or [N, num_blocks]
        use_cache: Whether to cache packed tensors
    
    Returns:
        Tuple of (packed_weight [K, N], packed_scales [num_blocks, N],
                  num_blocks, block_size)
    """
    global _weight_packing_cache
    
    if use_cache:
        cache_key = _get_cache_key(weight, weight_scale)
        if cache_key in _weight_packing_cache:
            return _weight_packing_cache[cache_key]
    
    # Get dimensions
    N, K = weight.shape
    
    # Handle 3D scales [N, num_blocks, 1]
    if weight_scale.ndim == 3:
        num_blocks = weight_scale.shape[1]
    else:
        num_blocks = weight_scale.shape[1] if weight_scale.ndim >= 2 else 1
    
    block_size = K // num_blocks
    
    # Pack weight and scales
    weight_packed = pack_weight_for_blockwise(weight)
    scales_packed = pack_scales_for_blockwise(weight_scale)
    
    result = (weight_packed, scales_packed, num_blocks, block_size)
    
    if use_cache:
        # Manage cache size for low VRAM
        if len(_weight_packing_cache) >= _MAX_CACHE_ENTRIES:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_weight_packing_cache))
            del _weight_packing_cache[oldest_key]
        
        _weight_packing_cache[cache_key] = result
    
    return result


def prepack_weights_for_inference(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple:
    """
    Pre-pack weights and scales during model loading to avoid first-call overhead.
    
    For low VRAM scenarios, call this during model loading so the transpose/contiguous
    operations don't happen during the first forward pass.
    
    Args:
        weight: INT8 weight [N, K] in original layout
        weight_scale: Block-wise scales [N, num_blocks, 1] or [N, num_blocks]
    
    Returns:
        Tuple of (packed_weight [K, N], packed_scales [num_blocks, N])
        Store these and pass them to the kernel directly for best performance.
    
    Example:
        # During model loading
        weight_packed, scales_packed = prepack_weights_for_inference(weight, scale)
        layer.weight = nn.Parameter(weight_packed, requires_grad=False)
        layer.register_buffer('weight_scale_packed', scales_packed)
    """
    weight_packed = pack_weight_for_blockwise(weight)
    scales_packed = pack_scales_for_blockwise(weight_scale)
    return weight_packed, scales_packed


@triton.jit
def _blockwise_int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K, num_blocks, block_size,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scale_n, stride_scale_b,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Computes: C = sum over K blocks of ((A * B) * (scale_a * scale_b_block))
    A: [M, K] int8 (activations)
    B: [N, K] int8 (weights, transposed access via strides)
    b_scale: [N, num_blocks] float32 (block-wise scales)
    a_scale: [M, 1] float32 (per-token activation scales)
    
    Fixed version: Properly accumulates INT32 results within each scale block
    before dequantization to avoid precision loss from partial dequantization.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize FP32 accumulator for final output
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load activation scales for this tile [BLOCK_M]
    scale_a = tl.load(a_scale_ptr + offs_am)
    
    # Calculate how many BLOCK_K tiles fit in one scale block
    # This determines how many K tiles share the same scale
    tiles_per_scale_block = block_size // BLOCK_K
    
    # Calculate tiles per scale block (compile-time constant when possible)
    tiles_per_scale_block = block_size // BLOCK_K
    
    # Iterate through scale blocks first (outer loop)
    for block_idx in range(0, num_blocks):
        # Initialize INT32 accumulator for this scale block
        # We accumulate INT32 results within each scale block before dequantizing
        block_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        
        # Compute starting K offset for this scale block
        block_k_start = block_idx * block_size
        
        # Inner loop: accumulate INT32 matmul within this scale block
        for tile_idx in range(0, tiles_per_scale_block):
            k_offset = block_k_start + tile_idx * BLOCK_K
            
            # Create masks for valid K elements
            # When k_offset >= K, all masks will be False and loads will return 0
            k_mask_a = offs_k[None, :] < K - k_offset
            k_mask_b = offs_k[:, None] < K - k_offset
            
            # Update pointers for this K tile
            a_tile_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                                   (offs_k[None, :] + k_offset) * stride_ak)
            b_tile_ptrs = b_ptr + ((offs_k[:, None] + k_offset) * stride_bk +
                                   offs_bn[None, :] * stride_bn)
            
            a = tl.load(a_tile_ptrs, mask=k_mask_a, other=0)
            b = tl.load(b_tile_ptrs, mask=k_mask_b, other=0)
            
            # Accumulate INT32 matmul result
            block_accumulator += tl.dot(a, b)
        
        # Load block-wise scale for this block [BLOCK_N]
        scale_b = tl.load(b_scale_ptr + offs_bn * stride_scale_n + block_idx * stride_scale_b)
        
        # Dequantize once per scale block: int32 -> fp32, apply scales
        block_result = block_accumulator.to(tl.float32)
        total_scale = scale_a[:, None] * scale_b[None, :]
        block_result = block_result * total_scale
        
        # Accumulate to output
        accumulator += block_result

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        accumulator = accumulator + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _blockwise_int8_matmul_optimized_kernel(
    a_ptr,                    # [M, K] int8 activations
    b_packed_ptr,             # [K, N] int8 weights (pre-packed, contiguous along N)
    c_ptr,                    # [M, N] output
    a_scale_ptr,              # [M] float32 per-token activation scales
    b_scale_packed_ptr,       # [num_blocks, N] float32 block-wise scales (pre-transposed)
    bias_ptr,                 # [N] optional bias
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,     # stride_bn should be 1 for packed layout
    stride_cm, stride_cn,
    stride_scale_blk, stride_scale_n,  # stride_scale_n should be 1 for packed layout
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,      # Compile-time constant: num_blocks
    BLOCK_SIZE: tl.constexpr,      # Compile-time constant: K // num_blocks
    TILES_PER_BLOCK: tl.constexpr, # Compile-time constant: block_size // BLOCK_K
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,      # True if K % BLOCK_K == 0
    EVEN_M: tl.constexpr,      # True if M % BLOCK_M == 0
    EVEN_N: tl.constexpr,      # True if N % BLOCK_N == 0
):
    """
    Optimized block-wise INT8 matmul with pre-packed layouts.
    
    Memory Layouts (pre-packed for coalescing):
    - B: [K, N] contiguous along N (was [N, K] originally)
    - Scales: [num_blocks, N] contiguous along N (was [N, num_blocks] originally)
    
    Loop Structure:
    - Outer loop over scale blocks (num_blocks)
    - Inner loop over tiles within each block (tiles_per_block)
    - Accumulate int32 within block, dequantize once per block
    
    Grid: 2D (pid_m, pid_n) for simpler indexing
    
    Fixed version: Properly handles partial tiles at K boundary to avoid
    including garbage values in accumulation.
    """
    # 2D launch grid - simpler than grouped 1D
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets with proper bounds checking (no modulo wraparound)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for valid elements
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Alignment hints for better vectorization
    # These tell Triton that indices are multiples of certain values
    tl.multiple_of(offs_n, 8)
    tl.max_contiguous(offs_n, BLOCK_N)

    # Load activation scales once [BLOCK_M] with mask
    scale_a = tl.load(a_scale_ptr + offs_m, mask=mask_m, other=0.0)
    
    # Initialize output accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Outer loop over scale blocks
    # Using static_range when NUM_BLOCKS is constexpr allows unrolling
    for blk in tl.static_range(0, NUM_BLOCKS):
        # Accumulate int32 within this scale block
        acc_int32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        
        # Compute K offset for this block
        block_k_start = blk * BLOCK_SIZE
        
        # Inner loop over tiles within this block
        # Using static_range when TILES_PER_BLOCK is constexpr
        for t in tl.static_range(0, TILES_PER_BLOCK):
            k_offset = block_k_start + t * BLOCK_K
            
            offs_kt = tl.arange(0, BLOCK_K)
            k_ids = k_offset + offs_kt
            
            # Create K mask - critical for correctness when K boundary is within this tile
            # When k_offset >= K, all elements are masked out (loads return 0)
            if EVEN_K:
                mask_k = k_ids < K
            else:
                mask_k = k_ids < K
            
            # Load activation tile [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                              k_ids[None, :] * stride_ak)
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
            
            # Load weight tile [BLOCK_K, BLOCK_N] from packed layout
            # B is [K, N] with stride_bn=1, so loads are contiguous along N
            b_ptrs = b_packed_ptr + (k_ids[:, None] * stride_bk +
                                      offs_n[None, :] * stride_bn)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
            
            # INT8 matmul accumulation
            acc_int32 += tl.dot(a, b)
        
        # Load block-wise scale for this block [BLOCK_N]
        # Scales are [num_blocks, N] with stride_scale_n=1, so loads are contiguous
        scale_b = tl.load(b_scale_packed_ptr + blk * stride_scale_blk + offs_n,
                          mask=mask_n, other=0.0)
        
        # Dequantize once per block: int32 -> fp32, apply scales
        block_result = acc_int32.to(tl.float32)
        total_scale = scale_a[:, None] * scale_b[None, :]
        block_result = block_result * total_scale
        
        # Accumulate to output
        accumulator += block_result
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator = accumulator + bias[None, :]

    # Store output with proper masking
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, accumulator)
    else:
        tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _blockwise_int8_matmul_ampere(
    a_ptr,                    # [M, K] int8 activations
    b_packed_ptr,             # [K, N] int8 weights (pre-packed, contiguous along N)
    c_ptr,                    # [M, N] output
    a_scale_ptr,              # [M] float32 per-token activation scales
    b_scale_packed_ptr,       # [num_blocks, N] float32 block-wise scales (pre-transposed)
    bias_ptr,                 # [N] optional bias
    M, N, K,
    num_blocks,               # Number of scale blocks (runtime value)
    block_size,               # Size of each scale block (K // num_blocks)
    stride_am, stride_ak,
    stride_bk, stride_bn,     # stride_bn should be 1 for packed layout
    stride_cm, stride_cn,
    stride_scale_blk, stride_scale_n,  # stride_scale_n should be 1 for packed layout
    HAS_BIAS: tl.constexpr = False,
    BM: tl.constexpr = _AMPERE_BLOCKWISE_BLOCK_M,
    BN: tl.constexpr = _AMPERE_BLOCKWISE_BLOCK_N,
    BK: tl.constexpr = _AMPERE_BLOCKWISE_BLOCK_K,
    GROUP_M: tl.constexpr = _AMPERE_BLOCKWISE_GROUP_M,
):
    """
    Ampere-optimized block-wise INT8 matmul with pipelined loads.
    
    Memory Layouts (pre-packed for coalescing):
    - A: [M, K] row-major
    - B: [K, N] contiguous along N (packed from [N, K])
    - Scales: [num_blocks, N] contiguous along N (packed from [N, num_blocks])
    
    Uses tl.make_block_ptr for pipelined memory access on Ampere (SM80/86).
    Grid: 1D with grouped swizzling for L2 cache optimization.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    
    # Swizzled grouping for better L2 cache utilization (same as tensorwise Ampere kernel)
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load activation scales once [BM]
    scale_a = tl.load(a_scale_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)
    
    # Initialize FP32 accumulator
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    
    # Calculate tiles per scale block
    tiles_per_scale_block = block_size // BK
    
    # Outer loop over scale blocks
    for block_idx in range(0, num_blocks):
        # INT32 accumulator for this scale block
        block_acc = tl.zeros((BM, BN), dtype=tl.int32)
        
        # Starting K offset for this scale block
        block_k_start = block_idx * block_size
        
        # Use block pointers for pipelined loads within this scale block
        # A block pointer for [BM, BK] tiles
        A_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BM, block_k_start), block_shape=(BM, BK), order=(1, 0),
        )
        # B block pointer for [BK, BN] tiles (B is [K, N] packed)
        B_block_ptr = tl.make_block_ptr(
            base=b_packed_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(block_k_start, pid_n * BN), block_shape=(BK, BN), order=(0, 1),
        )
        
        # Inner loop over tiles within this scale block
        for _ in range(0, tiles_per_scale_block):
            # Load A and B tiles with pipelining
            a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
            b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.int8)
            
            # INT8 matmul accumulation
            block_acc += tl.dot(a, b)
            
            # Advance block pointers
            A_block_ptr = tl.advance(A_block_ptr, (0, BK))
            B_block_ptr = tl.advance(B_block_ptr, (BK, 0))
        
        # Load block-wise scale for this block [BN]
        scale_b = tl.load(
            b_scale_packed_ptr + block_idx * stride_scale_blk + offs_n,
            mask=mask_n, other=1.0
        ).to(tl.float32)
        
        # Dequantize once per block
        block_result = block_acc.to(tl.float32)
        total_scale = scale_a[:, None] * scale_b[None, :]
        block_result = block_result * total_scale
        
        # Accumulate to output
        accumulator += block_result
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]
    
    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def triton_blockwise_int8_linear(
    x: torch.Tensor,           # [*, K] activations
    weight: torch.Tensor,      # [N, K] INT8 weights
    weight_scale: torch.Tensor, # [N, num_blocks, 1] block-wise scales
    bias: torch.Tensor = None,
    compute_dtype: torch.dtype = torch.float16,
    use_optimized: bool = False,  # Use optimized kernel with pre-packed layouts (default False for stability)
) -> torch.Tensor:
    """
    Block-wise INT8 linear layer with on-the-fly dequantization.
    
    Keeps weights in INT8 during inference for ~4x memory savings vs BF16.
    Block-wise scales are applied per K-block during the matmul.
    
    Args:
        x: Input activations [*, K] (float16/bfloat16/float32)
        weight: INT8 weights [N, K]
        weight_scale: Block-wise scales [N, num_blocks, 1]
        bias: Optional bias [N]
        compute_dtype: Output dtype
        use_optimized: Whether to use the optimized kernel with pre-packed layouts
    
    Returns:
        Output tensor [*, N] in compute_dtype
    """
    import os
    import sys
    _BLOCKWISE_DEBUG = os.environ.get("INT8_DEBUG_MODE", "0") == "1"
    _DIAG = os.environ.get("INT8_DEBUG_DIAG", "0") == "1"
    
    if _DIAG or _BLOCKWISE_DEBUG:
        print(f"[BLOCKWISE] ENTRY: x.shape={x.shape}, weight.shape={weight.shape}, weight_scale.shape={weight_scale.shape}", flush=True)
        sys.stdout.flush()
    
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])
    
    M, K = x_2d.shape
    N = weight.shape[0]
    
    if _BLOCKWISE_DEBUG:
        print(f"[BLOCKWISE] Input: M={M}, K={K}, N={N}, weight_scale_shape={weight_scale.shape}")
    
    # Quantize activations to INT8 with per-token scales
    x_int8, x_scale = triton_quantize_rowwise(x_2d)

    # Prepare output buffer
    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)
    
    if _BLOCKWISE_DEBUG:
        print(f"[BLOCKWISE] x_int8_shape={x_int8.shape}, x_scale_shape={x_scale.shape}")
    
    if use_optimized:
        # Use optimized kernel with pre-packed layouts
        try:
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Using optimized kernel path")
            
            # Get packed weight and scales (cached for efficiency)
            weight_packed, scales_packed, num_blocks, block_size = \
                get_packed_weight_and_scales(weight, weight_scale, use_cache=True)
            
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Packed: num_blocks={num_blocks}, block_size={block_size}")
            
            # Calculate tiles per block for constexpr
            # For correctness, block_size must be evenly divisible by BLOCK_K
            if block_size < _FIXED_BLOCK_K or block_size % _FIXED_BLOCK_K != 0:
                reason = "< BLOCK_K" if block_size < _FIXED_BLOCK_K else "not divisible by BLOCK_K"
                raise ValueError(
                    f"block_size ({block_size}) {reason} ({_FIXED_BLOCK_K}). "
                    f"Optimized kernel requires block_size to be >= BLOCK_K and evenly divisible. "
                    f"Falling back to original kernel."
                )
            
            tiles_per_block = block_size // _FIXED_BLOCK_K
            
            # Check if dimensions are evenly divisible (for mask optimization)
            even_m = (M % _FIXED_BLOCK_M) == 0
            even_n = (N % _FIXED_BLOCK_N) == 0
            even_k = (K % _FIXED_BLOCK_K) == 0
            
            # Use 2D grid for simpler indexing
            grid = (triton.cdiv(M, _FIXED_BLOCK_M), triton.cdiv(N, _FIXED_BLOCK_N))
            
            has_bias = bias is not None
            bias_ptr = bias if has_bias else x_int8
            
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Launching optimized kernel: grid={grid}, tiles_per_block={tiles_per_block}")
                torch.cuda.synchronize()
            
            # Launch optimized kernel with 2D grid
            # Note: NUM_BLOCKS, BLOCK_SIZE, TILES_PER_BLOCK are constexpr for unrolling
            _blockwise_int8_matmul_optimized_kernel[grid](
                a_ptr=x_int8,
                b_packed_ptr=weight_packed,
                c_ptr=output,
                a_scale_ptr=x_scale,
                b_scale_packed_ptr=scales_packed,
                bias_ptr=bias_ptr,
                M=M, N=N, K=K,
                stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
                stride_bk=weight_packed.stride(0), stride_bn=weight_packed.stride(1),
                stride_cm=output.stride(0), stride_cn=output.stride(1),
                stride_scale_blk=scales_packed.stride(0), stride_scale_n=scales_packed.stride(1),
                BLOCK_M=_FIXED_BLOCK_M,
                BLOCK_N=_FIXED_BLOCK_N,
                BLOCK_K=_FIXED_BLOCK_K,
                NUM_BLOCKS=num_blocks,
                BLOCK_SIZE=block_size,
                TILES_PER_BLOCK=tiles_per_block,
                HAS_BIAS=has_bias,
                EVEN_K=even_k,
                EVEN_M=even_m,
                EVEN_N=even_n,
                num_warps=_FIXED_NUM_WARPS,
                num_stages=_FIXED_NUM_STAGES,
            )
            
            if _BLOCKWISE_DEBUG:
                torch.cuda.synchronize()
                print(f"[BLOCKWISE] Optimized kernel completed successfully")
            
            return output.reshape(x_shape_orig[:-1] + (N,))
            
        except Exception as e:
            # Fallback to original kernel if optimized fails
            # This can happen if shapes don't align or for edge cases
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Optimized kernel failed: {e}, falling back to original kernel")
            use_optimized = False
    
    if not use_optimized:
        # Get block-wise parameters from scale tensor
        num_blocks = weight_scale.shape[1]
        block_size = K // num_blocks
        
        # Check if we can use the Ampere-optimized kernel (includes Ada)
        gpu_arch = gpu_family()
        is_ampere_or_newer = gpu_arch in ("ampere", "ada")
        can_use_ampere = (is_ampere_or_newer and
                         block_size >= _AMPERE_BLOCKWISE_BLOCK_K and
                         block_size % _AMPERE_BLOCKWISE_BLOCK_K == 0)
        
        if can_use_ampere:
            # Use Ampere-optimized kernel with pre-packed layouts
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Using Ampere-optimized kernel path")
            
            try:
                # Pack weight from [N, K] to [K, N] for optimal access
                if weight.stride(0) != weight.shape[1] or weight.stride(1) != 1:
                    weight_packed = weight.t().contiguous()
                else:
                    weight_packed = weight.t()
                
                # Pack scales from [N, num_blocks] to [num_blocks, N]
                w_scale_2d = weight_scale.squeeze(-1)
                if w_scale_2d.stride(0) != 1:
                    scales_packed = w_scale_2d.t().contiguous()
                else:
                    scales_packed = w_scale_2d.t()
                
                # Grid with grouped swizzling (same as tensorwise Ampere kernel)
                grid = (triton.cdiv(M, _AMPERE_BLOCKWISE_BLOCK_M) * triton.cdiv(N, _AMPERE_BLOCKWISE_BLOCK_N),)
                
                has_bias = bias is not None
                bias_ptr = bias if has_bias else x_int8
                
                if _DIAG or _BLOCKWISE_DEBUG:
                    print(f"[BLOCKWISE] Launching Ampere kernel: grid={grid}, num_blocks={num_blocks}, block_size={block_size}", flush=True)
                    sys.stdout.flush()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                import time as _time
                _kernel_start = _time.perf_counter()
                
                _blockwise_int8_matmul_ampere[grid](
                    a_ptr=x_int8,
                    b_packed_ptr=weight_packed,
                    c_ptr=output,
                    a_scale_ptr=x_scale,
                    b_scale_packed_ptr=scales_packed,
                    bias_ptr=bias_ptr,
                    M=M, N=N, K=K,
                    num_blocks=num_blocks,
                    block_size=block_size,
                    stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
                    stride_bk=weight_packed.stride(0), stride_bn=weight_packed.stride(1),
                    stride_cm=output.stride(0), stride_cn=output.stride(1),
                    stride_scale_blk=scales_packed.stride(0), stride_scale_n=scales_packed.stride(1),
                    HAS_BIAS=has_bias,
                    num_warps=_AMPERE_BLOCKWISE_NUM_WARPS,
                    num_stages=_AMPERE_BLOCKWISE_NUM_STAGES,
                )
                
                if _DIAG or _BLOCKWISE_DEBUG:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _kernel_elapsed = _time.perf_counter() - _kernel_start
                    print(f"[BLOCKWISE] Ampere kernel execution completed in {_kernel_elapsed*1000:.2f}ms", flush=True)
                    sys.stdout.flush()
                
            except Exception as e:
                if _BLOCKWISE_DEBUG:
                    print(f"[BLOCKWISE] Ampere kernel failed: {e}, falling back to original kernel")
                can_use_ampere = False
        
        if not can_use_ampere:
            # Fallback to original kernel
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Using original (non-optimized) kernel path")
            
            if _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Original kernel: num_blocks={num_blocks}, block_size={block_size}")
            
            # Validate: block_size must be >= BLOCK_K and divisible by BLOCK_K
            if block_size < _FIXED_BLOCK_K:
                raise ValueError(
                    f"block_size ({block_size}) < BLOCK_K ({_FIXED_BLOCK_K}). "
                    f"Blockwise kernels require block_size >= BLOCK_K. "
                    f"Consider using fewer blocks (num_blocks <= {K // _FIXED_BLOCK_K})."
                )
            
            if block_size % _FIXED_BLOCK_K != 0:
                raise ValueError(
                    f"block_size ({block_size}) must be divisible by BLOCK_K ({_FIXED_BLOCK_K}). "
                    f"Consider using a block count where {K} // num_blocks is divisible by {_FIXED_BLOCK_K}. "
                    f"Valid num_blocks for K={K}: {[b for b in range(1, K+1) if K % b == 0 and (K // b) % _FIXED_BLOCK_K == 0][:10]}..."
                )
            
            # Prepare scale tensor: [N, num_blocks, 1] -> [N, num_blocks]
            w_scale_2d = weight_scale.squeeze(-1).contiguous()

            # Launch kernel
            grid = (triton.cdiv(M, _FIXED_BLOCK_M) * triton.cdiv(N, _FIXED_BLOCK_N), )
            
            has_bias = bias is not None
            bias_ptr = bias if has_bias else x_int8
            
            if _DIAG or _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] Launching original kernel: grid={grid}, num_blocks={num_blocks}, block_size={block_size}", flush=True)
                sys.stdout.flush()
            
            if _DIAG or _BLOCKWISE_DEBUG:
                print(f"[BLOCKWISE] About to compile/launch Triton kernel...", flush=True)
                sys.stdout.flush()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            import time as _time
            _kernel_start = _time.perf_counter()
            
            _blockwise_int8_matmul_kernel[grid](
                a_ptr=x_int8,
                b_ptr=weight,
                c_ptr=output,
                a_scale_ptr=x_scale,
                b_scale_ptr=w_scale_2d,
                bias_ptr=bias_ptr,
                M=M, N=N, K=K,
                num_blocks=num_blocks,
                block_size=block_size,
                stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
                stride_bk=weight.stride(1), stride_bn=weight.stride(0),
                stride_cm=output.stride(0), stride_cn=output.stride(1),
                stride_scale_n=w_scale_2d.stride(0), stride_scale_b=w_scale_2d.stride(1),
                BLOCK_M=_FIXED_BLOCK_M,
                BLOCK_N=_FIXED_BLOCK_N,
                BLOCK_K=_FIXED_BLOCK_K,
                GROUP_SIZE_M=_FIXED_GROUP_SIZE_M,
                HAS_BIAS=has_bias,
                num_warps=_FIXED_NUM_WARPS,
                num_stages=_FIXED_NUM_STAGES,
            )
            
            if _DIAG or _BLOCKWISE_DEBUG:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _kernel_elapsed = _time.perf_counter() - _kernel_start
                print(f"[BLOCKWISE] Kernel execution completed in {_kernel_elapsed*1000:.2f}ms", flush=True)
                sys.stdout.flush()
    
    if _DIAG or _BLOCKWISE_DEBUG:
        print(f"[BLOCKWISE] RETURN: output.shape={output.shape}", flush=True)
        sys.stdout.flush()
    
    return output.reshape(x_shape_orig[:-1] + (N,))


if __name__ == "__main__":
    # Run kernel accuracy tests when executed directly
    test_kernel_accuracy()

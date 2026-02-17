"""
ComfyUI-Wan-INT8 - Fast INT8 Quantization for ComfyUI

Provides:
- Int8TensorwiseOps: Custom operations for direct int8 weight loading
- WanVideoINT8Loader: Load int8 quantized diffusion models
- Block-wise INT8 quantization support for improved accuracy

Uses torch._int_mm and Triton kernels for blazing fast inference.
Supports both tensor-wise and block-wise quantization modes.
"""

import torch

def _apply_cudagraph_compatibility_fix():
    """
    Patch PyTorch's CUDA graph memory pool checking to work with cudaMallocAsync.
    This allows torch.compile to work without requiring PYTORCH_ALLOC_CONF=backend:native.
    """
    
    if hasattr(torch._C, '_cuda_checkPoolLiveAllocations'):
        _original_check = torch._C._cuda_checkPoolLiveAllocations
        def _safe_check(*args, **kwargs):
            try:
                return _original_check(*args, **kwargs)
            except RuntimeError as e:
                if "cudaMallocAsync" in str(e):
                    return True
                raise
        torch._C._cuda_checkPoolLiveAllocations = _safe_check
    
    if hasattr(torch._C, '_cuda_memorySnapshot'):
        _original_snapshot = torch._C._cuda_memorySnapshot
        def _safe_snapshot(*args, **kwargs):
            try:
                return _original_snapshot(*args, **kwargs)
            except RuntimeError as e:
                if "cudaMallocAsync" in str(e) or "snapshot" in str(e).lower():
                    return {"segments": [], "device_traces": []}
                raise
        torch._C._cuda_memorySnapshot = _safe_snapshot
    
    try:
        from torch._inductor import cudagraph_trees
        if hasattr(cudagraph_trees, 'check_memory_pool'):
            _original_check_pool = cudagraph_trees.check_memory_pool
            def _safe_check_pool(*args, **kwargs):
                try:
                    return _original_check_pool(*args, **kwargs)
                except RuntimeError as e:
                    if "cudaMallocAsync" in str(e):
                        return None
                    raise
            cudagraph_trees.check_memory_pool = _safe_check_pool
    except ImportError:
        pass

_apply_cudagraph_compatibility_fix()

try:
    from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout

    class Int8TensorwiseLayout(QuantizedLayout):
        """Minimal layout class to satisfy ComfyUI's registry requirements."""
        class Params:
            def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                self.scale = scale
                self.orig_dtype = orig_dtype
                self.orig_shape = orig_shape
            
            def clone(self):
                import torch
                return Int8TensorwiseLayout.Params(
                    scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                    orig_dtype=self.orig_dtype,
                    orig_shape=self.orig_shape
                )

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {"": qdata, "weight_scale": params.scale}
        
        @classmethod  
        def dequantize(cls, qdata, params):
            return qdata.float() * params.scale

    register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

    cur_config = QUANT_ALGOS.get("int8_tensorwise")
    if cur_config is None:
        QUANT_ALGOS["int8_tensorwise"] = {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
        }
    else:
        cur_config["comfy_tensor_layout"] = "Int8TensorwiseLayout"
        cur_config["parameters"] = {"weight_scale", "input_scale"}

except ImportError:
    pass

try:
    from .int8_quant import (
        Int8TensorwiseOps,
        set_hadamard_quip_enabled,
        is_hadamard_quip_enabled,
        print_kernel_summary,
        reset_kernel_stats,
    )
    
    # Export block-wise kernel utilities for advanced users
    try:
        from .triton_kernels import (
            triton_blockwise_int8_linear,
            clear_weight_packing_cache,
        )
    except ImportError:
        triton_blockwise_int8_linear = None
        clear_weight_packing_cache = None
    
    _hadamard_status = "enabled" if is_hadamard_quip_enabled() else "disabled"
    print(f"[ComfyUI-Wan-INT8] Hadamard-QuIP kernel: {_hadamard_status}")
    print(f"[ComfyUI-Wan-INT8] Use set_hadamard_quip_enabled(True/False) to toggle")
    print(f"[ComfyUI-Wan-INT8] Use print_kernel_summary() to see inference kernel usage")
except ImportError:
    Int8TensorwiseOps = None
    set_hadamard_quip_enabled = None
    is_hadamard_quip_enabled = None
    print_kernel_summary = None
    reset_kernel_stats = None

from .int8_unet_loader import WanVideoINT8Loader
from .wan_lora_loader import WanLoRALoader
from .wan_lora_loader_with_clip import WanLoRALoaderWithCLIP
 
NODE_CLASS_MAPPINGS = {
    "WanVideoINT8Loader": WanVideoINT8Loader,
    "WanLoRALoaderINT8": WanLoRALoader,
    "WanLoRALoaderWithCLIPINT8": WanLoRALoaderWithCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoINT8Loader": "Wan Video Loader (INT8)",
    "WanLoRALoaderINT8": "Wan LoRA Loader (INT8)",
    "WanLoRALoaderWithCLIPINT8": "Wan LoRA Loader with CLIP (INT8)",
}

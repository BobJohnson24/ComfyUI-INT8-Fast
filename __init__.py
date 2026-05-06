"""
INT8 Fast - INT8 Tensorwise Quantization for ComfyUI

Provides:
- Int8TensorwiseOps: Custom operations for direct int8 weight loading
- OTUNetLoaderW8A8: Load int8 quantized diffusion models

Uses torch._int_mm for fast inference.
"""

import logging
import torch

# =============================================================================
# Layout Registration
# =============================================================================

def _register_layouts():
    """
    Register the Int8Tensorwise layout with ComfyUI's model management.
    """
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

        # Register the class
        register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

        # Register the Algo Config
        QUANT_ALGOS.setdefault(
            "int8_tensorwise",
            {
                "storage_t": torch.int8,
                # We include input_scale here so ComfyUI extracts it from checkpoints if present,
                # even though our LinearW8A8 implementation explicitly ignores it.
                "parameters": {"weight_scale", "input_scale"},
                "comfy_tensor_layout": "Int8TensorwiseLayout",
            }
        )
        
    except ImportError:
        logging.warning("INT8 Fast: ComfyUI Quantization system not found (Update ComfyUI?)")
    except Exception as e:
        logging.error(f"INT8 Fast: Failed to register layouts: {e}")


# =============================================================================
# Module Initialization
# =============================================================================

# 1. Register Layouts
_register_layouts()

# 2. Export Custom Ops (for external use)
try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

# 3. Node Mappings
# Wrap imports in try/except to prevent total failure if dependencies are missing
try:
    from .int8_unet_loader import UNetLoaderINTW8A8
    from .int8_lora import INT8LoraLoader, INT8LoraLoaderStack
    from .int8_save import INT8ModelSave
    
    NODE_CLASS_MAPPINGS = {
        "OTUNetLoaderW8A8": UNetLoaderINTW8A8,
        "INT8LoraLoader": INT8LoraLoader,
        "INT8LoraLoaderStack": INT8LoraLoaderStack,
        "INT8ModelSave": INT8ModelSave,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "OTUNetLoaderW8A8": "Load Diffusion Model INT8 (W8A8)",
        "INT8LoraLoader": "Load LoRA INT8 (Stochastic)",
        "INT8LoraLoaderStack": "INT8 LoRA Stack (Stochastic)",
        "INT8ModelSave": "Save Int8 Model",
    }
except ImportError as e:
    logging.error(f"Int88: Failed to import nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "Int8TensorwiseOps",
]
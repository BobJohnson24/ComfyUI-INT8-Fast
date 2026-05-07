import torch
from torch import Tensor, nn
import torch.nn.functional as F
import comfy.model_patcher
import comfy.lora
import comfy.utils

# Add this at the top of your file
try:
    from .int8_fused_kernel import triton_int8_linear
    from .int8_fused_kernel import triton_int8_linear_per_row
    from .int8_fused_kernel import triton_quantize_rowwise
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    print("Triton not found, falling back to torch._int_mm")

# Runtime toggle — set by Int8TensorwiseOps.use_triton via the loader node
_use_triton = True

# QuaRot Configuration
QUAROT_GROUP_SIZE = 256  # Must be a power of 4 for Regular Hadamard (e.g. 16, 64, 256)

# --- Quantization Utils ---

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale




# --- LinearW8A8 Core ---

@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    
    # --- FAST PATH: Triton Fused Kernel ---
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    # Quantize activations per row (dynamic)
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    
    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)
    
    # Dequantize: (res * weight_scale * x_scale)
    # Note: Creating intermediate Float tensors here is VRAM heavy
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled


@torch.no_grad()
def int8_forward_dynamic_per_row(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    """Forward with dynamic per-token activation quantization and per-row weight quantization.
    
    Args:
        x: Input activations [batch, in_features]
        weight: INT8 weight matrix [out_features, in_features]
        weight_scale: Per-row weight scales [out_features, 1]
        bias: Optional bias
        compute_dtype: Output dtype
    """
    # --- FAST PATH: Triton Fused Kernel (per-row) ---
    if _TRITON_AVAILABLE and _use_triton and x.is_cuda:
        return triton_int8_linear_per_row(x, weight, weight_scale, bias, compute_dtype)

    # --- SLOW PATH: Standard PyTorch ---
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)

    # INT8 Matmul (Outputs Int32)
    res = torch._int_mm(x_8, weight.T)  # [batch, out_features]
    
    # Dequantize with per-row weight scales
    # res[i,j] = sum_k(x_8[i,k] * weight[j,k]) * x_scale[i] * weight_scale[j]
    # Broadcasting: res * x_scale * weight_scale.T
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.T).to(compute_dtype)
    
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)
    return res_scaled




# =============================================================================
# INT8 LoRA Adapter - High Precision, Low RAM Patching
# =============================================================================




# =============================================================================
# Int8TensorwiseOps - ComfyUI Custom Operations
# =============================================================================

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    class Int8TensorwiseOps(manual_cast):
        """
        Custom ComfyUI operations for INT8 tensorwise quantization.
        """
        excluded_names = []
        dynamic_quantize = False # Manual toggle for on-the-fly quantization
        enable_quarot = False # Toggle for QuaRot Hadamard rotation
        use_triton = True  # Toggle for Triton fused kernel (mirrors _use_triton)
        _is_prequantized = False # Keep this as a status flag, but don't use for detection
        dynamic_lora = False # If True, apply LoRA dynamically at inference; if False, bake into INT8 weights at load time
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer('weight_scale', None)
                self._is_quantized = False
                self._is_per_row = False  # Track quantization granularity
                self._use_quarot = False  # Track if QuaRot was applied
                self._weight_scale_scalar = None  # For scalar (non-tensor) scales
                self.compute_dtype = torch.bfloat16
                self.lora_patches = []  # List of (down_scaled, up, start, size) set by INT8ModelPatcher
            
            def reset_parameters(self):
                return None
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"
                
                def pop_metadata(sd, p, k):
                    v = sd.pop(p + k, None)
                    if v is not None: return v
                    v = sd.pop("model." + p + k, None)
                    if v is not None: return v
                    if p.startswith("model."):
                        v = sd.pop(p[6:] + k, None)
                        if v is not None: return v
                    if p.startswith("diffusion_model."):
                        v = sd.pop("diffusion_model." + p + k, None)
                        if v is not None: return v
                    return None

                weight_scale = pop_metadata(state_dict, prefix, "weight_scale")
                comfy_quant_tensor = pop_metadata(state_dict, prefix, "comfy_quant")

                weight_tensor = state_dict.pop(weight_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
                if comfy_quant_tensor is not None:
                    try:
                        import json
                        quant_conf = json.loads(bytes(comfy_quant_tensor.tolist()).decode('utf-8'))
                        if quant_conf.get("quarot", False):
                            self._use_quarot = True
                            Int8TensorwiseOps.enable_quarot = True  # Propagate globally for LoRAs
                            if "quarot_groupsize" in quant_conf:
                                self._quarot_groupsize = quant_conf["quarot_groupsize"]
                                Int8TensorwiseOps._global_quarot_groupsize = self._quarot_groupsize
                    except Exception:
                        pass
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Load Quantized
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        Int8TensorwiseOps._is_prequantized = True # Found a quantized layer
                        
                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                # Scalar scale — store as float for speed
                                self._weight_scale_scalar = weight_scale.float().item()
                                self.weight_scale = None
                                self._is_per_row = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = True
                            else:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = False
                        else:
                            self._weight_scale_scalar = float(weight_scale)
                            self.weight_scale = None
                            self._is_per_row = False
                            
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # Load High-Precision
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        
                        if is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # Quantize on the fly
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            
                            # Log the first time we quantize in this loader pass
                            if not hasattr(Int8TensorwiseOps, '_logged_otf'):
                                print(f"INT8 Fast: Quantizing on-the-fly (QuaRot: {getattr(Int8TensorwiseOps, 'enable_quarot', False)})")
                                Int8TensorwiseOps._logged_otf = True

                            # Cast to float32 before rotation and scale computation, may be snake oil but can it hurt?
                            w_gpu = weight_tensor.to(device, non_blocking=True).float()
                            
                            self._use_quarot = False
                            if getattr(Int8TensorwiseOps, "enable_quarot", False) and self.in_features % QUAROT_GROUP_SIZE == 0:
                                try:
                                    import logging
                                    from .quarot import build_hadamard, rotate_weight
                                    H = build_hadamard(QUAROT_GROUP_SIZE, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=QUAROT_GROUP_SIZE)
                                    self._use_quarot = True
                                except ImportError as e:
                                    import logging
                                    logging.warning(f"INT8 Fast: QuaRot enabled but quarot module error: {e}")
                                    
                            q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)

                            self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                            self.register_buffer('weight_scale', q_scale.cpu())
                            self._weight_scale_scalar = None
                            self._is_quantized = True
                            self._is_per_row = True
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

                # Update archived model dtypes so VBAR geometry uses the correct
                # sizes. archive_model_dtypes runs before state_dict loading, so
                # weight_comfy_model_dtype is stale (e.g. bfloat16 instead of int8).
                # Without this, VBAR allocates 2x the needed memory and the cast
                # buffer path misinterprets int8 data as bfloat16.
                if self.weight is not None:
                    self.weight_comfy_model_dtype = self.weight.dtype
                if self.weight_scale is not None:
                    self.weight_scale_comfy_model_dtype = self.weight_scale.dtype
                if self.bias is not None:
                    self.bias_comfy_model_dtype = self.bias.dtype

            def _get_weight_scale(self):
                """Get weight scale, preferring scalar if available."""
                if self._weight_scale_scalar is not None:
                    return self._weight_scale_scalar
                return self.weight_scale

            def convert_weight(self, _weight, inplace=False):
                if not self._is_quantized:
                    return _weight
                return self.weight

            def set_weight(self, out_weight, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if not self._is_quantized:
                    new_weight = out_weight.to(self.weight.dtype)
                    if return_weight:
                        return new_weight

                    if inplace_update:
                        self.weight.data.copy_(new_weight)
                    else:
                        self.weight = nn.Parameter(new_weight, requires_grad=False)
                    return

                if out_weight.dtype == torch.int8:
                    if return_weight:
                        return out_weight

                    if inplace_update:
                        self.weight.data.copy_(out_weight)
                    else:
                        self.weight = nn.Parameter(out_weight, requires_grad=False)
                    return

                # Re-quantize if fallback occurred
                new_weight = quantize_int8(out_weight, self._get_weight_scale())
                
                if return_weight:
                    return new_weight

                if inplace_update:
                    self.weight.data.copy_(new_weight)
                else:
                    self.weight = nn.Parameter(new_weight, requires_grad=False)

            def set_bias(self, out_bias, inplace_update=False, seed=0, return_weight=False, **kwargs):
                if out_bias is None: return None
                
                new_bias = out_bias
                if return_weight:
                    return new_bias

                if inplace_update:
                    if self.bias is not None:
                        self.bias.data.copy_(new_bias)
                else:
                    self.bias = nn.Parameter(new_bias, requires_grad=False)

            def forward(self, x: Tensor) -> Tensor:
                """Fast forward using torch._int_mm for quantized weights."""
                
                # Check if ComfyUI needs to manage weight transfer (VBAR, offloading, LoRA patches, etc.)
                # This mirrors the base class check in disable_weight_init.Linear.forward()
                need_cast = self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0
                
                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    else:
                        return F.linear(x, self.weight, self.bias)
                
                # INT8 quantized path
                if need_cast:
                    # VBAR / offload / lowvram path
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True
                    )
                else:
                    # Fast path: weights already on GPU, no functions to apply
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None
                
                w_scale = self._get_weight_scale()
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)
                
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                if getattr(self, "_use_quarot", False):
                    from .quarot import build_hadamard, rotate_activation
                    group_size = getattr(self, "_quarot_groupsize", QUAROT_GROUP_SIZE)
                    H = build_hadamard(group_size, device=x.device, dtype=x.dtype)
                    x_2d = rotate_activation(x_2d, H, group_size=group_size)
                
                # Sync the loader toggle to the module-level flag read by the forward fns
                import sys as _sys
                _mod = _sys.modules[__name__]
                _mod._use_triton = Int8TensorwiseOps.use_triton

                if x_2d.shape[0] > 16:
                    if self._is_per_row:
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)
                
                # Dynamic LoRA Path — handles split QKV via per-patch offsets
                for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                    lD = lora_down.to(x.device, non_blocking=True)
                    lU = lora_up.to(x.device, non_blocking=True)
                    lora_x = F.linear(x_2d.to(lD.dtype), lD)
                    lora_y = F.linear(lora_x, lU)  # [batch, slice_size or full_out]
                    if lora_start is not None:
                        y[:, lora_start:lora_start + lora_size] = (
                            y[:, lora_start:lora_start + lora_size] + lora_y.to(y.dtype)
                        )
                    else:
                        y = y + lora_y.to(y.dtype)
                
                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        # Pass-through for other layers
        class GroupNorm(manual_cast.GroupNorm): pass
        class LayerNorm(manual_cast.LayerNorm): pass
        class Conv2d(manual_cast.Conv2d): pass
        class Conv3d(manual_cast.Conv3d): pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
        class Embedding(manual_cast.Embedding): pass
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2: return cls.Conv2d(*args, **kwargs)
            elif dims == 3: return cls.Conv3d(*args, **kwargs)
            else: raise ValueError(f"unsupported dimensions: {dims}")

# =============================================================================
# INT8 Model Patcher - Unified LoRA Handling
# =============================================================================

class INT8ModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher that intercepts patching for INT8 layers.
    Routes patching through either a bake-in path (dequant-patch-requant)
    or a dynamic path (runtime injection), depending on the dynamic_lora toggle.
    """
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches:
            return

        # Check if this is one of our INT8 modules
        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int8_module = hasattr(module, "_is_quantized") and module._is_quantized
        patches = self.patches[key]



        if is_int8_module:
            if not Int8TensorwiseOps.dynamic_lora:
                # --- BAKE-IN LORA PATH (Dequant → Patch → Quant) ---
                # Works with the native ComfyUI LoRA Loader (and also INT8LoraLoader).
                # All patches are applied in float space via ComfyUI's standard mechanism,
                # then the result is re-quantized back to INT8.

                weight_int8 = comfy.utils.get_attr(self.model, key)
                scale = module._get_weight_scale()

                if device_to is None:
                    device_to = weight_int8.device

                # Save original weight so unpatch_model can restore it.
                # Must use the same namedtuple format as ComfyUI's base patcher
                # (collections.namedtuple('Dimension', ['weight', 'inplace_update']))
                # otherwise unpatch_model crashes with AttributeError on bk.inplace_update.
                if key not in self.backup:
                    import collections
                    BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                    self.backup[key] = BackupEntry(
                        weight=weight_int8.to(device=self.offload_device, copy=inplace_update),
                        inplace_update=inplace_update,
                    )

                # 1. Dequantize to float (move scale to device_to since it lives on CPU)
                if isinstance(scale, torch.Tensor):
                    scale = scale.to(device_to)
                weight_float = dequantize(weight_int8.to(device_to), scale)

                # 2. Handle QuaRot: de-rotate into weight space before patching
                use_quarot = getattr(module, "_use_quarot", False)
                if use_quarot:
                    group_size = getattr(module, "_quarot_groupsize", QUAROT_GROUP_SIZE)
                    try:
                        from .quarot import build_hadamard, rotate_weight
                        H = build_hadamard(group_size, device=device_to, dtype=weight_float.dtype)
                        weight_float = rotate_weight(weight_float, H, group_size=group_size)
                    except ImportError:
                        pass

                # 3. Patch in float space using ComfyUI's standard mechanism.
                # calculate_weight handles LoRA, LoHA, LoKR, DoRA, etc.
                patches_list = self.patches.get(key, [])
                patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

                # 4. Handle QuaRot: re-rotate
                if use_quarot:
                    patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)

                # 5. Re-quantize back to INT8 using the original scale
                patched_weight_int8 = quantize_int8(patched_weight_float, scale)

                # 6. Move back to original device and store
                patched_weight_int8 = patched_weight_int8.to(weight_int8.device)

                if return_weight:
                    return patched_weight_int8

                if inplace_update:
                    weight_int8.data.copy_(patched_weight_int8)
                else:
                    comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight_int8, requires_grad=False))
                return

            else:
                # --- DYNAMIC LORA PATH ---
                # Build a list of (down_scaled, up, start, size) per patch.
                # Keeping patches separate preserves the offset info needed for
                # fused QKV layers where each of Q/K/V targets a different output slice.
                weight = comfy.utils.get_attr(self.model, key)
                device = weight.device if weight is not None else self.offload_device
                lora_patches = []
                for p in patches:
                    strength_patch = p[0]  # float
                    adapter = p[1]         # the LoRA adapter object
                    strength_model = p[2]  # float
                    offset = p[3] if len(p) > 3 else None  # (dim, start, size) or None

                    if not hasattr(adapter, "weights"):
                        continue

                    strength = strength_patch * strength_model
                    weights = adapter.weights
                    # Standard LoRA: (up, down, alpha, mid, dora_scale, reshape)
                    if len(weights) == 6:
                        up, down, alpha, mid, dora, reshape = weights
                        rank = down.shape[0] if down.ndim >= 2 else 1
                        scale = (alpha / rank) * strength if alpha is not None else strength

                        down_scaled = down.flatten(1) * scale
                        if mid is not None:
                            down_scaled = torch.mm(mid.flatten(1), down.flatten(1)) * scale

                        # If this layer has QuaRot applied, rotate the 'down' matrix
                        # so the LoRA delta is coherent with the rotated weight basis:
                        #   W_rot = W @ H^T  =>  ΔW_rot = ΔW @ H^T  =>  rotate down only
                        if getattr(module, "_use_quarot", False) and down_scaled.shape[1] % QUAROT_GROUP_SIZE == 0:
                            try:
                                from .quarot import build_hadamard, rotate_weight
                                group_size = getattr(module, "_quarot_groupsize", QUAROT_GROUP_SIZE)
                                H = build_hadamard(group_size, device=down_scaled.device, dtype=down_scaled.dtype)
                                down_scaled = rotate_weight(down_scaled, H, group_size=group_size)
                            except ImportError:
                                pass

                        # Extract offset: which output rows this patch targets
                        start, size = None, None
                        if offset is not None:
                            _dim, start, size = offset  # dim is always 0 for linear weights

                        lora_patches.append((down_scaled.to(device), up.flatten(1).to(device), start, size))

                module.lora_patches = lora_patches
                return  # Skip standard weight-merging path

        # --- NON-INT8 MODULE PATH ---
        return super().patch_weight_to_device(key, device_to, inplace_update)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = INT8ModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = INT8ModelPatcher
        self.__class__ = src_cls
        return n

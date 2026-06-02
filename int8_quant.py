import torch
from torch import Tensor, nn
import torch.nn.functional as F
import logging
import comfy.model_patcher
import comfy.memory_management
import comfy.model_management
import comfy.lora
import comfy.utils

try:
    import comfy_aimdo.host_buffer
    import comfy_aimdo.torch
    _AIMDO_FILE_SLICE_LOAD = True
except Exception:
    _AIMDO_FILE_SLICE_LOAD = False

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

# ConvRot Configuration
CONVROT_GROUP_SIZE = 256  # Must be a power of 4 for Regular Hadamard (e.g. 16, 64, 256)

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

def tensor_to_device_file_slice(tensor: Tensor, device: torch.device) -> Tensor:
    if (
        not _AIMDO_FILE_SLICE_LOAD
        or tensor.device.type != "cpu"
        or device is None
        or device.type != "cuda"
    ):
        return tensor.to(device, non_blocking=True)

    size = tensor.numel() * tensor.element_size()
    if size == 0:
        return tensor.to(device, non_blocking=True)

    hostbuf = comfy_aimdo.host_buffer.HostBuffer(size)
    host_tensor = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)
    host_view = host_tensor[:size].view(dtype=tensor.dtype).view(tensor.shape)
    if comfy.memory_management.read_tensor_file_slice_into(tensor, host_view):
        out = torch.empty_like(tensor, device=device)
        out.copy_(host_view, non_blocking=False)
        return out

    return tensor.to(device, non_blocking=True)

def stochastic_round_int8_delta(x: Tensor, scale: float | Tensor, seed: int = 0) -> Tensor:
    """
    Quantize a delta tensor to INT8 using stochastic rounding.
    Used for LoRA deltas to minimize quantization error.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    
    # Scale to INT8 range — move scale to x's device to handle CPU-stored scales
    if isinstance(scale, torch.Tensor):
        scale = scale.to(x.device)
    x_scaled = x / scale
    
    # Stochastic rounding
    x_floor = torch.floor(x_scaled)
    fraction = x_scaled - x_floor
    del x_scaled # High-precision input no longer needed
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_floor.shape, generator=generator, device=x.device, dtype=x_floor.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    
    del random_vals
    del fraction
    del x_floor
    
    return torch.clamp(x_rounded, -128, 127).to(torch.int8)



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
        enable_convrot = False # Toggle for ConvRot Hadamard rotation
        use_triton = True  # Toggle for Triton fused kernel (mirrors _use_triton)
        _is_prequantized = False # Keep this as a status flag, but don't use for detection
        lora_mode = "None" # None/Stochastic bake into INT8 weights; Dynamic applies LoRA at inference
        dynamic_lora = False # If True, apply LoRA dynamically at inference; if False, bake into INT8 weights at load time
        lora_patches = {} # Map of model_key -> patch list (from load_lora)
        lora_strength = 1.0
        dynamic_load_device = None # Set by the loader when Aimdo should avoid a full CPU staging copy
        skeleton_meta_init = False # Temporary mode for LoRA key-map discovery
        
        class Linear(manual_cast.Linear):
            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                if getattr(Int8TensorwiseOps, "skeleton_meta_init", False):
                    nn.Module.__init__(self)
                    self.in_features = in_features
                    self.out_features = out_features
                    tensor_kwargs = {"device": "meta"}
                    if dtype is not None:
                        tensor_kwargs["dtype"] = dtype
                    self.weight = nn.Parameter(torch.empty((out_features, in_features), **tensor_kwargs), requires_grad=False)
                    self.bias = nn.Parameter(torch.empty((out_features,), **tensor_kwargs), requires_grad=False) if bias else None
                    self.weight_comfy_model_dtype = dtype
                    self.bias_comfy_model_dtype = dtype
                # Preserve ComfyUI's Windows/Aimdo lazy-init path. The base
                # disable_weight_init.Linear only takes this path for classes
                # that do not override _load_from_state_dict; this INT8 class
                # does override it, so calling super() would allocate full
                # skeleton weights during Pre-LoRA key-map discovery.
                elif comfy.model_management.WINDOWS and comfy.memory_management.aimdo_enabled:
                    nn.Module.__init__(self)
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weight = None
                    self.bias = None
                    self.comfy_need_lazy_init_bias = bias
                    self.weight_comfy_model_dtype = dtype
                    self.bias_comfy_model_dtype = dtype
                else:
                    super().__init__(in_features, out_features, bias, device, dtype)
                self.register_buffer('weight_scale', None)
                self._is_quantized = False
                self._is_per_row = False  # Track quantization granularity
                self._use_convrot = False  # Track if ConvRot was applied
                self._weight_scale_scalar = None  # For scalar (non-tensor) scales
                self.compute_dtype = torch.bfloat16
                self.comfy_cast_weights = False
                self.lora_patches = []  # List of (down_scaled, up, start, size) set by INT8ModelPatcher
            
            def reset_parameters(self):
                return None

            @staticmethod
            def _normalize_lora_key(key):
                if not isinstance(key, str):
                    return key
                for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                    if key.startswith(p):
                        return key[len(p):]
                return key

            @staticmethod
            def _is_bias_key(key):
                return isinstance(key, str) and key.endswith(".bias")

            @staticmethod
            def _format_lora_patches(patches):
                formatted = []
                for patch in patches or []:
                    if len(patch) == 4:
                        v, offset, function, strength = patch
                    else:
                        v, offset, function = patch
                        strength = getattr(Int8TensorwiseOps, "lora_strength", 1.0)
                    formatted.append((strength, v, 1.0, offset, function))
                return formatted

            def _apply_int8_lora_patches(self, tensor, key, patches, device):
                if not patches or tensor.dtype == torch.int8:
                    return tensor

                temp_dtype = comfy.model_management.lora_compute_dtype(device)
                tensor_temp = tensor_to_device_file_slice(tensor, device).to(dtype=temp_dtype)
                return comfy.lora.calculate_weight(self._format_lora_patches(patches), tensor_temp, key)

            def finalize_pending_int8(self):
                pending = getattr(self, "_pending_int8_finalize", None)
                if pending is None:
                    return False

                weight_key = pending["weight_key"]
                device = pending.get("device")
                if device is None:
                    device = torch.device("cuda") if torch.cuda.is_available() else self.weight.device

                weight_tensor = self.weight.detach()
                weight_tensor = self._apply_int8_lora_patches(weight_tensor, weight_key, pending.get("lora_patches"), device)

                if pending["quantize"]:
                    if not hasattr(Int8TensorwiseOps, '_logged_otf'):
                        print(f"INT8 Fast: Quantizing on-the-fly (ConvRot: {pending.get('enable_convrot', False)})")
                        Int8TensorwiseOps._logged_otf = True

                    w_gpu = tensor_to_device_file_slice(weight_tensor, device).float()

                    self._use_convrot = False
                    if pending.get("enable_convrot", False) and self.in_features % CONVROT_GROUP_SIZE == 0:
                        try:
                            from .convrot import build_hadamard, rotate_weight
                            H = build_hadamard(CONVROT_GROUP_SIZE, device=w_gpu.device, dtype=w_gpu.dtype)
                            w_gpu = rotate_weight(w_gpu, H, group_size=CONVROT_GROUP_SIZE)
                            self._use_convrot = True
                            # Stamp the active groupsize so INT8ModelSave can
                            # round-trip it deterministically.
                            self._convrot_groupsize = CONVROT_GROUP_SIZE
                        except ImportError as e:
                            logging.warning(f"INT8 Fast: ConvRot enabled but convrot module error: {e}")

                    q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)
                    self.weight = nn.Parameter(q_weight.cpu(), requires_grad=False)
                    self.register_buffer('weight_scale', q_scale.cpu())
                    self._weight_scale_scalar = None
                    self._is_quantized = True
                    self._is_per_row = True
                    del w_gpu, q_weight, q_scale
                else:
                    self.weight = nn.Parameter(weight_tensor.cpu(), requires_grad=False)

                self.weight_comfy_model_dtype = self.weight.dtype
                if self.weight_scale is not None:
                    self.weight_scale_comfy_model_dtype = self.weight_scale.dtype
                if self.bias is not None:
                    self.bias_comfy_model_dtype = self.bias.dtype

                delattr(self, "_pending_int8_finalize")
                return True
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = prefix + "weight"
                
                # Utility to normalize keys by stripping common prefixes
                def normalize_key(key):
                    return self._normalize_lora_key(key)

                def apply_lora_patches(tensor, key):
                    if self._is_bias_key(key) or not Int8TensorwiseOps.lora_patches or tensor.dtype == torch.int8:
                        return tensor
                    nk = normalize_key(key)
                    patches = Int8TensorwiseOps.lora_patches.get(nk)
                    if patches:
                        # Track applied patches
                        if not hasattr(Int8TensorwiseOps, 'applied_lora_patches'):
                            Int8TensorwiseOps.applied_lora_patches = set()
                        Int8TensorwiseOps.applied_lora_patches.add(nk)

                        # Print only if multiple sub-patches map to the same layer
                        # if "weight" in key and len(patches) > 1:
                        #     print(f"INT8 Fast: Baking multiple LoRA parts into {nk} ({len(patches)} sub-patches)")
                            
                        # ComfyUI dynamically patches during inference using lora_compute_dtype()
                        # On most modern GPUs, this evaluates to torch.float16. 
                        # We simulate that exact intermediate cast here to achieve a 1:1 binary match.
                        device = getattr(Int8TensorwiseOps, "dynamic_load_device", None)
                        if device is None:
                            device = tensor.device
                        result_temp = self._apply_int8_lora_patches(tensor, key, patches, device)
                        return result_temp.to(tensor.dtype)
                    return tensor

                def source_tensor(tensor):
                    if tensor is not None and getattr(Int8TensorwiseOps, "dynamic_load_device", None) is not None:
                        return tensor.cpu()
                    return tensor

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
                bias_tensor = state_dict.pop(bias_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
                quant_conf_parsed = None
                if comfy_quant_tensor is not None:
                    try:
                        import json
                        quant_conf_parsed = json.loads(bytes(comfy_quant_tensor.tolist()).decode('utf-8'))
                        if quant_conf_parsed.get("convrot", False):
                            self._use_convrot = True
                            Int8TensorwiseOps.enable_convrot = True  # Propagate globally for LoRA
                            if "convrot_groupsize" in quant_conf_parsed:
                                self._convrot_groupsize = int(quant_conf_parsed["convrot_groupsize"])
                                Int8TensorwiseOps._global_convrot_groupsize = self._convrot_groupsize
                    except Exception:
                        pass
                
                pending_weight_lora_patches = None
                if weight_tensor is not None and weight_tensor.dtype != torch.int8:
                    pending_weight_lora_patches = Int8TensorwiseOps.lora_patches.get(normalize_key(weight_key))

                defer_weight_lora = (
                    getattr(Int8TensorwiseOps, "dynamic_load_device", None) is not None
                    and pending_weight_lora_patches
                )

                # Apply LoRA patches to weight and bias once. With Aimdo, large
                # weight patches are deferred until KSampler/model load time so
                # the loader node stays cheap and VBAR geometry is finalized once.
                if weight_tensor is not None and not defer_weight_lora:
                    weight_tensor = apply_lora_patches(weight_tensor, weight_key)
                if bias_tensor is not None:
                    bias_tensor = apply_lora_patches(bias_tensor, bias_key)
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Load Quantized
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        Int8TensorwiseOps._is_prequantized = True # Found a quantized layer

                        # Optional explicit hint from saved comfy_quant
                        per_row_hint = None
                        if isinstance(quant_conf_parsed, dict) and "per_row" in quant_conf_parsed:
                            per_row_hint = bool(quant_conf_parsed["per_row"])

                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                # Scalar scale — store as float for speed
                                self._weight_scale_scalar = weight_scale.float().item()
                                self.register_buffer('weight_scale', weight_scale.float().reshape(1))
                                self._weight_scale_scalar = None
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = True if per_row_hint is None else per_row_hint
                            else:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._weight_scale_scalar = None
                                self._is_per_row = False if per_row_hint is None else per_row_hint
                        else:
                            self.weight_scale = nn.Parameter(
                                torch.tensor(float(weight_scale), dtype=torch.float32), 
                                requires_grad=False
                            )
                            self.weight_scale = None
                            self._is_per_row = False if per_row_hint is None else per_row_hint
                            
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn):
                        # Load High-Precision
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                        should_quantize = not (is_excluded or is_dim1 or not Int8TensorwiseOps.dynamic_quantize)
                        defer_finalize = (
                            getattr(Int8TensorwiseOps, "dynamic_load_device", None) is not None
                            and (should_quantize or pending_weight_lora_patches)
                        )
                        
                        if defer_finalize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
                            self._pending_int8_finalize = {
                                "weight_key": weight_key,
                                "quantize": should_quantize,
                                "lora_patches": pending_weight_lora_patches,
                                "device": getattr(Int8TensorwiseOps, "dynamic_load_device", None),
                                "enable_convrot": getattr(Int8TensorwiseOps, "enable_convrot", False),
                            }
                            if pending_weight_lora_patches:
                                if not hasattr(Int8TensorwiseOps, 'applied_lora_patches'):
                                    Int8TensorwiseOps.applied_lora_patches = set()
                                Int8TensorwiseOps.applied_lora_patches.add(normalize_key(weight_key))
                        elif not should_quantize:
                            self._is_quantized = False
                            self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
                        else:
                            # Quantize on the fly
                            device = getattr(Int8TensorwiseOps, "dynamic_load_device", None)
                            if device is None:
                                device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                            
                            # Log the first time we quantize in this loader pass
                            if not hasattr(Int8TensorwiseOps, '_logged_otf'):
                                print(f"INT8 Fast: Quantizing on-the-fly (ConvRot: {getattr(Int8TensorwiseOps, 'enable_convrot', False)})")
                                Int8TensorwiseOps._logged_otf = True

                            # Cast to float32 before rotation and scale computation
                            w_gpu = weight_tensor.to(device, non_blocking=True).float()
                            
                            self._use_convrot = False
                            if getattr(Int8TensorwiseOps, "enable_convrot", False) and self.in_features % CONVROT_GROUP_SIZE == 0:
                                try:
                                    import logging
                                    from .convrot import build_hadamard, rotate_weight
                                    H = build_hadamard(CONVROT_GROUP_SIZE, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=CONVROT_GROUP_SIZE)
                                    self._use_convrot = True
                                    # Stamp the active groupsize on the module so
                                    # INT8ModelSave can serialize it deterministically
                                    # (instead of relying on the loader's default).
                                    self._convrot_groupsize = CONVROT_GROUP_SIZE
                                except ImportError as e:
                                    import logging
                                    logging.warning(f"INT8 Fast: ConvRot enabled but convrot module error: {e}")
                                    
                            q_weight, q_scale = quantize_int8_axiswise(w_gpu, dim=1)

                            q_weight = q_weight.cpu()
                            q_scale = q_scale.cpu()

                            self.weight = nn.Parameter(q_weight, requires_grad=False)
                            self.register_buffer('weight_scale', q_scale)
                            self._weight_scale_scalar = None
                            self._is_quantized = True
                            self._is_per_row = True
                            del w_gpu, q_weight, q_scale
                    else:
                        self._is_quantized = False
                        self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
                else:
                    missing_keys.append(weight_key)
                
                # Assign bias if it exists (already patched if needed)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(source_tensor(bias_tensor), requires_grad=False)
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
                
                if getattr(self, "_use_convrot", False):
                    from .convrot import build_hadamard, rotate_activation
                    group_size = getattr(self, "_convrot_groupsize", CONVROT_GROUP_SIZE)
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

import inspect
try:
    _prefetch_sig = inspect.signature(comfy.lora.prefetch_prepared_value)
    _use_new_prefetch = len(_prefetch_sig.parameters) == 5
except Exception:
    _use_new_prefetch = False


class INT8LowVramPatch:
    is_lowvram_patch = True

    def __init__(self, key, patches, module, lora_mode):
        self.key = key
        self.patches = patches
        self.module = module
        self.lora_mode = lora_mode
        self.prepared_patches = None

    def memory_required(self):
        if not _use_new_prefetch:
            return 0
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, *args, **kwargs):
        if _use_new_prefetch:
            # 0.22.0+ signature: prepare(self, destination, stream, copy=True, commit=True)
            destination = args[0] if len(args) > 0 else kwargs.get("destination")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")
            copy = args[2] if len(args) > 2 else kwargs.get("copy", True)
            commit = args[3] if len(args) > 3 else kwargs.get("commit", True)

            counter = [0]
            prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            if commit:
                self.prepared_patches = prepared_patches
            return prepared_patches
        else:
            # 0.21.1- signature: prepare(self, allocate_buffer, stream)
            allocate_buffer = args[0] if len(args) > 0 else kwargs.get("allocate_buffer")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")

            self.prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], allocate_buffer, stream), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            return self.prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        scale = self.module._get_weight_scale()
        if isinstance(scale, torch.Tensor):
            scale = scale.to(weight.device)

        weight_float = dequantize(weight, scale)

        use_convrot = getattr(self.module, "_use_convrot", False)
        if use_convrot:
            group_size = getattr(self.module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
            try:
                from .convrot import build_hadamard, rotate_weight
                H = build_hadamard(group_size, device=weight.device, dtype=weight_float.dtype)
                weight_float = rotate_weight(weight_float, H, group_size=group_size)
            except ImportError:
                use_convrot = False

        patched_weight_float = comfy.lora.calculate_weight(
            patches,
            weight_float,
            self.key,
            intermediate_dtype=weight_float.dtype,
        )

        if use_convrot:
            patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)

        if self.lora_mode == "Stochastic":
            return stochastic_round_int8_delta(
                patched_weight_float,
                scale,
                seed=comfy.utils.string_to_seed(self.key),
            )
        return quantize_int8(patched_weight_float, scale)


class INT8ModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher that intercepts patching for INT8 layers.
    Routes patching through either a bake-in path (dequant-patch-requant)
    or a dynamic path (runtime injection), depending on the dynamic_lora toggle.
    """
    def finalize_pending_int8(self):
        finalized = 0
        for module in self.model.modules():
            finalize = getattr(module, "finalize_pending_int8", None)
            if finalize is not None and finalize():
                finalized += 1
        if finalized > 0:
            self.size = 0
            #logging.info(f"INT8 Fast: Finalized {finalized} deferred INT8 layer(s) at model load time.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches and not force_cast:
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

        # Check if this is one of our INT8 modules
        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int8_module = hasattr(module, "_is_quantized") and module._is_quantized
        patches = self.patches.get(key, [])

        if is_int8_module and Int8TensorwiseOps.Linear._is_bias_key(key):
            return comfy.utils.get_attr(self.model, key) if return_weight else None

        if is_int8_module:
            if not Int8TensorwiseOps.dynamic_lora:
                # --- BAKE-IN LORA PATH (Dequant → Patch → Quant) ---
                # Works with the native ComfyUI LoRA Loader (and also INT8LoraLoader).
                # All patches are applied in float space via ComfyUI's standard mechanism,
                # then the result is re-quantized back to INT8.

                # Identify current weight in the model
                current_weight = comfy.utils.get_attr(self.model, key)
                scale = module._get_weight_scale()

                if device_to is None:
                    device_to = current_weight.device

                # ALWAYS use the weight from backup as the source if it exists to prevent additive stacking.
                # If it doesn't exist, this is the first patch, so create it from the current model weight.
                if key not in self.backup:
                    import collections
                    BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                    self.backup[key] = BackupEntry(
                        weight=current_weight.to(device=self.offload_device, copy=inplace_update),
                        inplace_update=inplace_update,
                    )
                    source_weight = current_weight
                else:
                    # Use existing backup as source
                    source_weight = self.backup[key].weight

                # 1. Dequantize to float (move scale to device_to since it lives on CPU)
                if isinstance(scale, torch.Tensor):
                    scale = scale.to(device_to)
                weight_float = dequantize(source_weight.to(device_to), scale)

                # 2. Handle ConvRot: de-rotate into weight space before patching
                use_convrot = getattr(module, "_use_convrot", False)
                if use_convrot:
                    group_size = getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                    try:
                        from .convrot import build_hadamard, rotate_weight
                        H = build_hadamard(group_size, device=device_to, dtype=weight_float.dtype)
                        weight_float = rotate_weight(weight_float, H, group_size=group_size)
                    except ImportError:
                        pass

                # 3. Patch in float space using ComfyUI's standard mechanism.
                # calculate_weight handles LoRA, LoHA, LoKR, DoRA, etc.
                patches_list = self.patches.get(key, [])
                patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

                # 4. Handle ConvRot: re-rotate
                if use_convrot:
                    patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)

                # 5. Re-quantize back to INT8 using the original scale
                if getattr(Int8TensorwiseOps, "lora_mode", "None") == "Stochastic":
                    patched_weight_int8 = stochastic_round_int8_delta(patched_weight_float, scale)
                else:
                    patched_weight_int8 = quantize_int8(patched_weight_float, scale)

                # 6. Move back to original device and store
                patched_weight_int8 = patched_weight_int8.to(current_weight.device)

                if return_weight:
                    return patched_weight_int8

                if inplace_update:
                    current_weight.data.copy_(patched_weight_int8)
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

                        # If this layer has ConvRot applied, rotate the 'down' matrix
                        # so the LoRA delta is coherent with the rotated weight basis:
                        #   W_rot = W @ H^T  =>  ΔW_rot = ΔW @ H^T  =>  rotate down only
                        if getattr(module, "_use_convrot", False) and down_scaled.shape[1] % CONVROT_GROUP_SIZE == 0:
                            try:
                                from .convrot import build_hadamard, rotate_weight
                                group_size = getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
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
                if return_weight:
                    return weight
                return  # Skip standard weight-merging path

        # --- NON-INT8 MODULE PATH ---
        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

    def load(self, *args, **kwargs):
        self.finalize_pending_int8()

        save_materialized = bool(getattr(self, "_int8_save_materialized_lora", False))

        if not Int8TensorwiseOps.dynamic_lora and not save_materialized:
            for k in list(self.backup):
                if k in self.patches:
                    try:
                        module = comfy.utils.get_attr(self.model, k.rsplit('.', 1)[0])
                    except AttributeError:
                        module = None
                    if hasattr(module, "_is_quantized") and module._is_quantized:
                        bk = self.backup.pop(k)
                        if bk.inplace_update:
                            dest = comfy.utils.get_attr(self.model, k)
                            dest.data.copy_(bk.weight)
                        else:
                            comfy.utils.set_attr(self.model, k, bk.weight)

        # Cleanup: Revert any keys that are in backup but no longer in patches (stale patches)
        # This ensures that when a LoRA is disabled, the model returns to its base state.
        stale_keys = [k for k in self.backup if k not in self.patches]
        for k in stale_keys:
            bk = self.backup.pop(k)
            if bk.inplace_update:
                dest = comfy.utils.get_attr(self.model, k)
                dest.data.copy_(bk.weight)
            else:
                comfy.utils.set_attr(self.model, k, bk.weight)
        
        # Cleanup: Clear stale dynamic LoRA patches.
        # This prevents LoRA from "sticking" when dynamic_lora is toggled or LoRAs are disabled.
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_patches") and module.lora_patches:
                # If dynamic LoRA is disabled globally, or if this module has no active patches, clear them.
                if not Int8TensorwiseOps.dynamic_lora or (name + ".weight") not in self.patches:
                    module.lora_patches = []

        res = super().load(*args, **kwargs) if hasattr(super(), "load") else None
        
        device_to = kwargs.get("device_to", args[0] if len(args) > 0 else self.model.device)
        
        for name, module in self.model.named_modules():
            if hasattr(module, "_is_quantized") and module._is_quantized:
                weight_key = name + ".weight"
                
                if weight_key in self.patches:
                    if save_materialized:
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = None
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                    elif Int8TensorwiseOps.dynamic_lora:
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = None
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                        self.patch_weight_to_device(weight_key, device_to=device_to)
                    else:
                        lowvram_patch = INT8LowVramPatch(
                            weight_key,
                            self.patches,
                            module,
                            getattr(Int8TensorwiseOps, "lora_mode", "None"),
                        )
                        pin_state = getattr(self.model, "dynamic_pins", {}).get(self.load_device, None)
                        if pin_state is not None:
                            lowvram_patch._pin_state = pin_state
                        module.weight_lowvram_function = lowvram_patch
                    
        return res

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        
        if src_cls is INT8ModelPatcher:
            return super().clone(*args, **kwargs)
            
        if not issubclass(src_cls, INT8ModelPatcher):
            name = f"INT8_{src_cls.__name__}"
            dynamic_cls = type(name, (INT8ModelPatcher, src_cls), {})
        else:
            dynamic_cls = src_cls
            
        self.__class__ = dynamic_cls
        
        # Provide a fallback for non-dynamic delegates (e.g. for KJNodes)
        if getattr(self, "cached_patcher_init", None) is None:
            self.cached_patcher_init = (lambda *a, **kw: self, ())
            
        n = super().clone(*args, **kwargs)
        
        # If disable_dynamic is True, the core strips dynamic wrappers. We must re-apply INT8!
        disable_dyn = kwargs.get("disable_dynamic", False)
        if len(args) > 0:
            disable_dyn = args[0]
            
        if disable_dyn and not issubclass(n.__class__, INT8ModelPatcher):
            new_cls = type(f"INT8_{n.__class__.__name__}", (INT8ModelPatcher, n.__class__), {})
            n.__class__ = new_cls

        self.__class__ = src_cls
        return n

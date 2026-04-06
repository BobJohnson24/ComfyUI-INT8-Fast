import torch
from torch import Tensor, nn
import torch.nn.functional as F

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
    
    # Speed optimization: Create random values directly on the target device
    random_vals = torch.rand(x_scaled.shape, generator=generator, device=x.device, dtype=x_scaled.dtype)
    x_rounded = torch.where(random_vals < fraction, x_floor + 1, x_floor)
    
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
# INT8 LoRA Adapter - High Precision, Low RAM Patching
# =============================================================================

def reconstruct_lora_diff(weights, target_shape, device, dtype, strength):
    """
    Reconstructs the low-rank delta in high precision for different adapter types.
    Supports LoRA, LoHA, and LoKR.
    """
    import comfy.model_management
    
    def cast(t):
        return comfy.model_management.cast_to_device(t, device, dtype)

    v = weights
    # -------------------------------------------------------------------------
    # Case 1: LoRA (used by LoRAAdapter)
    # weights: (up, down, alpha, mid, dora_scale, reshape)
    # -------------------------------------------------------------------------
    if len(v) == 6:
        up, down, alpha, mid, dora, reshape = v
        up_f = cast(up)
        down_f = cast(down)
        rank = down.shape[0]
        scale = (alpha / rank) * strength if alpha is not None else strength
        
        if mid is not None:
            mid_f = cast(mid)
            # LoCon/LoHA style Tucker: up @ mid @ down
            lora_diff = torch.mm(up_f.flatten(1), torch.mm(mid_f.flatten(1), down_f.flatten(1)))
        else:
            lora_diff = torch.mm(up_f.flatten(1), down_f.flatten(1))
            
        return lora_diff.reshape(target_shape), scale

    # -------------------------------------------------------------------------
    # Case 2: LoHA (used by LoHAAdapter)
    # weights: (w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale)
    # -------------------------------------------------------------------------
    elif len(v) == 8:
        w1a, w1b, alpha, w2a, w2b, t1, t2, dora = v
        rank = w1b.shape[0]
        scale = (alpha / rank) * strength if alpha is not None else strength
        
        if t1 is not None:
             m1 = torch.einsum("i j k l, j r, i p -> p r k l", cast(t1), cast(w1b), cast(w1a))
             m2 = torch.einsum("i j k l, j r, i p -> p r k l", cast(t2), cast(w2b), cast(w2a))
        else:
             m1 = torch.mm(cast(w1a), cast(w1b))
             m2 = torch.mm(cast(w2a), cast(w2b))
        
        lora_diff = (m1 * m2)
        return lora_diff.reshape(target_shape), scale

    # -------------------------------------------------------------------------
    # Case 3: LoKR (used by LoKrAdapter)
    # weights: (w1, w2, alpha, w1a, w1b, w2a, w2b, t2, dora_scale)
    # -------------------------------------------------------------------------
    elif len(v) == 9:
        w1, w2, alpha, w1a, w1b, w2a, w2b, t2, dora = v
        dim = None
        
        if w1 is None:
            dim = w1b.shape[0]
            w1_f = torch.mm(cast(w1a), cast(w1b))
        else:
            w1_f = cast(w1)
            
        if w2 is None:
            dim = w2b.shape[0]
            if t2 is None:
                w2_f = torch.mm(cast(w2a), cast(w2b))
            else:
                w2_f = torch.einsum("i j k l, j r, i p -> p r k l", cast(t2), cast(w2b), cast(w2a))
        else:
            w2_f = cast(w2)
            
        if len(w2_f.shape) == 4:
            w1_f = w1_f.unsqueeze(2).unsqueeze(2)
            
        scale = (alpha / dim) * strength if (alpha is not None and dim is not None) else strength
        lora_diff = torch.kron(w1_f, w2_f)
        return lora_diff.reshape(target_shape), scale

    return None, 1.0


try:
    from comfy.weight_adapter.lora import LoRAAdapter
    _LORA_ADAPTER_AVAILABLE = True
except ImportError:
    _LORA_ADAPTER_AVAILABLE = False

if _LORA_ADAPTER_AVAILABLE:
    class INT8LoRAPatchAdapter(LoRAAdapter):
        """
        Specialized LoRA adapter that patches INT8 weights IN-PLACE in INT8 space.
        """
        def __init__(self, loaded_keys, weights, weight_scale, seed=0):
            super().__init__(loaded_keys, weights)
            self.weight_scale = weight_scale
            self.seed = seed

        def _get_effective_scale(self, delta_f, offset):
            """Slice weight_scale to match delta_f when dealing with merged QKV LoRAs.
            
            ComfyUI narrows the full weight to a Q/K/V slice before calling us,
            but weight_scale is still the full merged tensor, so we need to match it.
            """
            ws = self.weight_scale
            if not isinstance(ws, torch.Tensor) or ws.numel() == 1:
                return ws  # scalar – always fine
            # offset = (dim, start, size) as set by ComfyUI for merged QKV layers
            if offset is not None and ws.shape[0] != delta_f.shape[0]:
                dim, start, size = offset
                if dim == 0:
                    return ws.narrow(0, start, size)
            # No offset, but still mismatched (e.g. separate-weight LoRA vs merged model):
            # try to detect which chunk by matching row count
            if ws.shape[0] != delta_f.shape[0] and ws.shape[0] % delta_f.shape[0] == 0:
                # We don't know which chunk, best we can do is use a scale subset.
                # Row-0 is a reasonable default; this path is a rare edge case.
                return ws[: delta_f.shape[0]]
            return ws

        def _get_effective_scale(self, delta_f, offset):
            """Slice weight_scale to match delta_f when dealing with merged QKV LoRAs.
            
            ComfyUI narrows the full weight to a Q/K/V slice before calling us,
            but weight_scale is still the full merged tensor, so we need to match it.
            """
            ws = self.weight_scale
            if not isinstance(ws, torch.Tensor) or ws.numel() == 1:
                return ws  # scalar – always fine
            
            # offset = (dim, start, size) as set by ComfyUI for merged QKV layers
            if offset is not None:
                dim, start, size = offset
                if dim < ws.dim() and ws.shape[dim] > start:
                    # Double check if size matches, or if we can safely narrow
                    narrow_size = min(size, ws.shape[dim] - start)
                    return ws.narrow(dim, start, narrow_size)
            
            # No offset, but still mismatched (e.g. separate-weight LoRA vs merged model):
            # try to detect which chunk by matching row count
            if ws.shape[0] != delta_f.shape[0] and ws.shape[0] % delta_f.shape[0] == 0:
                return ws[: delta_f.shape[0]]
                
            return ws

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            if intermediate_dtype == torch.int8:
                intermediate_dtype = torch.float32
                
            # Compute LoRA Delta in high-precision on GPU
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
            # Unified reconstruction for LoRA, LoHA, LoKR
            lora_diff, scale = reconstruct_lora_diff(
                self.weights, weight.shape, comp_device, intermediate_dtype, strength
            )
            
            if lora_diff is None:
                return weight
            
            # Apply Patch
            if weight.dtype == torch.int8:
                # --- INT8 SPACE PATCHING ---
                delta_f = lora_diff * scale

                # If QuaRot was applied to this layer's weights, rotate the delta into the same
                # basis (ΔW @ H^T) so the update is coherent: W_rot + ΔW_rot = (W + ΔW) @ H^T
                if getattr(Int8TensorwiseOps, 'enable_quarot', False) and weight.shape[1] % 128 == 0:
                    try:
                        from .quarot import build_hadamard, rotate_weight
                        H = build_hadamard(128, device=comp_device, dtype=delta_f.dtype)
                        delta_f = rotate_weight(delta_f, H, group_size=128)
                    except ImportError:
                        pass

                eff_scale = self._get_effective_scale(delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(delta_f, eff_scale, self.seed)
                
                # Perform integer addition (int32 for safety) then clamp
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                # Fallback: Standard Float Patching
                return weight + (lora_diff * scale).to(weight.device, weight.dtype)

    class INT8MergedLoRAPatchAdapter(LoRAAdapter):
        """
        Adapter that merges multiple LoRAs in float space BEFORE applying a single
        stochastic rounding step. This is much more precise for LoRA stacks.
        """
        def __init__(self, patches, weight_scale, seed=0):
            # We need to satisfy the base LoRAAdapter constructor.
            # We use the first patch's keys/weights as a reference.
            first_patch_adapter = patches[0][0]
            super().__init__(first_patch_adapter.loaded_keys, first_patch_adapter.weights)
            
            # patches is a list of (adapter, strength)
            self.patches = patches
            self.weight_scale = weight_scale
            self.seed = seed

        def _get_effective_scale(self, delta_f, offset):
            """Same slice-logic as INT8LoRAPatchAdapter._get_effective_scale."""
            ws = self.weight_scale
            if not isinstance(ws, torch.Tensor) or ws.numel() == 1:
                return ws
            if offset is not None and ws.shape[0] != delta_f.shape[0]:
                dim, start, size = offset
                if dim == 0:
                    return ws.narrow(0, start, size)
            if ws.shape[0] != delta_f.shape[0] and ws.shape[0] % delta_f.shape[0] == 0:
                return ws[: delta_f.shape[0]]
            return ws

        def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=torch.float32, original_weight=None):
            if intermediate_dtype == torch.int8:
                intermediate_dtype = torch.float32
                
            # Note: 'strength' from ComfyUI is ignored here as we use internal lora_strengths
            device = weight.device
            comp_device = torch.device("cuda") if torch.cuda.is_available() else device
            
            total_delta_f = None
            
            for adapter, lora_strength in self.patches:
                # Unified reconstruction for each adapter in the stack
                delta, scale = reconstruct_lora_diff(
                    adapter.weights, weight.shape, comp_device, intermediate_dtype, lora_strength
                )
                
                if delta is None: continue
                
                if total_delta_f is None:
                    total_delta_f = delta * scale
                else:
                    total_delta_f += delta * scale
            
            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                # One single stochastic rounding step for all combined LoRAs

                # If QuaRot was applied to this layer's weights, rotate the combined delta into
                # the same basis (ΔW @ H^T) so the update is coherent: W_rot + ΔW_rot = (W + ΔW) @ H^T
                if getattr(Int8TensorwiseOps, 'enable_quarot', False) and weight.shape[1] % 128 == 0:
                    try:
                        from .quarot import build_hadamard, rotate_weight
                        H = build_hadamard(128, device=comp_device, dtype=total_delta_f.dtype)
                        total_delta_f = rotate_weight(total_delta_f, H, group_size=128)
                    except ImportError:
                        pass

                eff_scale = self._get_effective_scale(total_delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(total_delta_f, eff_scale, self.seed)
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                return torch.clamp(res, -128, 127).to(torch.int8).to(device)
            else:
                return weight + total_delta_f.to(device, weight.dtype)


# =============================================================================
# Dynamic LoRA Synchronization Hook
# =============================================================================

class DynamicLoRAHook:
    """
    Hook registered on the diffusion_model to synchronize dynamic LoRA attributes
    with the current ModelPatcher context at the start of each forward pass.
    """
    def __init__(self):
        self.current_lora_id = None

    def pre_forward(self, module, input_args, input_kwargs):
        # 1. Try to find transformer_options
        transformer_options = input_kwargs.get("transformer_options", {})
        if not transformer_options:
            # Fallback for models that pass it in context
            context = input_args[2] if len(input_args) > 2 else None
            if isinstance(context, dict) and "transformer_options" in context:
                transformer_options = context["transformer_options"]
        
        dynamic_loras = transformer_options.get("dynamic_loras", [])
        
        # 2. Generate a unique ID for this set of LoRAs
        # We use handles/strengths to detect changes
        lora_id = hash(tuple((id(d["patches"]), d["strength"]) for d in dynamic_loras)) if dynamic_loras else None
        
        if lora_id == self.current_lora_id:
            return None # Already synchronized
            
        # 3. Synchronize all linear layers
        self.apply_composition(module, dynamic_loras)
        self.current_lora_id = lora_id
        return None

    def apply_composition(self, diffusion_model, dynamic_loras):
        # Pre-group patches by layer
        layer_patches = {}
        if dynamic_loras:
            for entry in dynamic_loras:
                strength = entry["strength"]
                for key, adapter in entry["patches"].items():
                    if key not in layer_patches: layer_patches[key] = []
                    layer_patches[key].append((adapter, strength))

        # Update all modules
        for name, module in diffusion_model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            
            # Find patches for this module
            # ComfyUI keys are often 'diffusion_model.path.to.weight' or 'path.to.weight'
            possible_keys = [f"diffusion_model.{name}.weight", f"{name}.weight"]
            patches = None
            for pk in possible_keys:
                if pk in layer_patches:
                    patches = layer_patches[pk]
                    break
            
            if not patches:
                module.lora_A = None
                module.lora_B = None
                module.lora_alpha = None
                continue

            # Compose
            all_A = []
            all_B = []
            for adapter, strength in patches:
                v = adapter.weights
                up, down, alpha, mid = v[0], v[1], v[2], v[3]
                rank = down.shape[0] if down.ndim >= 2 else 1
                scale = (alpha / rank) * strength if alpha is not None else strength
                
                curr_A = down
                if mid is not None:
                    curr_A = torch.mm(mid.flatten(1), down.flatten(1)).reshape(down.shape)
                
                all_A.append(curr_A * scale)
                all_B.append(up)
            
            if all_A:
                device = getattr(module, "weight", torch.tensor(0)).device
                module.lora_A = torch.cat(all_A, dim=0).to(device)
                module.lora_B = torch.cat(all_B, dim=1).to(device)
                module.lora_alpha = None
            else:
                module.lora_A = None
                module.lora_B = None

    @classmethod
    def register(cls, diffusion_model):
        if not hasattr(diffusion_model, "_dynamic_lora_hook"):
            hook = cls()
            diffusion_model._dynamic_lora_hook = hook
            diffusion_model.register_forward_pre_hook(hook.pre_forward, with_kwargs=True)
        return diffusion_model._dynamic_lora_hook


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
        
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer('weight_scale', None)
                self._is_quantized = False
                self._is_per_row = False  # Track quantization granularity
                self._use_quarot = False  # Track if QuaRot was applied
                self._weight_scale_scalar = None  # For scalar (non-tensor) scales
                self.compute_dtype = torch.bfloat16
                self.lora_A = None
                self.lora_B = None
                self.lora_alpha = None
            
            def reset_parameters(self):
                return None
            
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"
                
                weight_scale = state_dict.pop(scale_key, None)
                state_dict.pop(prefix + "comfy_quant", None)
                weight_tensor = state_dict.pop(weight_key, None)

                # Pop input_scale to clean state_dict, but ignore it
                _ = state_dict.pop(input_scale_key, None)
                
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
                            # Cast to float32 before rotation and scale computation, may be snake oil but can it hurt?
                            w_gpu = weight_tensor.to(device, non_blocking=True).float()
                            
                            self._use_quarot = False
                            if getattr(Int8TensorwiseOps, "enable_quarot", False) and self.in_features % 128 == 0:
                                try:
                                    import logging
                                    from .quarot import build_hadamard, rotate_weight
                                    H = build_hadamard(128, device=w_gpu.device, dtype=w_gpu.dtype)
                                    w_gpu = rotate_weight(w_gpu, H, group_size=128)
                                    self._use_quarot = True
                                except ImportError as e:
                                    import logging
                                    logging.warning(f"Int88: QuaRot enabled but quarot module error: {e}")
                                    
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
                from .int8_quant import stochastic_round_int8_delta
                new_weight = stochastic_round_int8_delta(out_weight, self._get_weight_scale(), seed)
                
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
                    H = build_hadamard(128, device=x.device, dtype=x.dtype)
                    x_2d = rotate_activation(x_2d, H, group_size=128)
                
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
                
                # Dynamic LoRA Path
                if self.lora_A is not None and self.lora_B is not None:
                    # Ensure LoRA tensors are on the same device as x
                    lA = self.lora_A.to(x.device, non_blocking=True)
                    lB = self.lora_B.to(x.device, non_blocking=True)
                    
                    lora_x = F.linear(x_2d.to(lA.dtype), lA)
                    lora_y = F.linear(lora_x, lB)
                    
                    if self.lora_alpha is not None:
                        lora_y = lora_y * self.lora_alpha
                    
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

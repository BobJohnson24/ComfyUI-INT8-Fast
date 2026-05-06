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
                
            # Compute LoRA Delta in high-precision. 
            # Respect ComfyUI's device selection for computation.
            device = weight.device
            import comfy.model_management
            comp_device = comfy.model_management.get_torch_device()
            
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
                del lora_diff # Free memory immediately

                # If QuaRot was applied to this layer's weights, rotate the delta into the same
                # basis (ΔW @ H^T) so the update is coherent: W_rot + ΔW_rot = (W + ΔW) @ H^T
                group_size = getattr(Int8TensorwiseOps, '_global_quarot_groupsize', QUAROT_GROUP_SIZE)
                if getattr(Int8TensorwiseOps, 'enable_quarot', False) and weight.shape[1] % group_size == 0:
                    try:
                        from .quarot import build_hadamard, rotate_weight
                        H = build_hadamard(group_size, device=comp_device, dtype=delta_f.dtype)
                        delta_f = rotate_weight(delta_f, H, group_size=group_size)
                    except ImportError:
                        pass

                eff_scale = self._get_effective_scale(delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(delta_f, eff_scale, self.seed)
                del delta_f # Free high-precision delta
                
                # Perform integer addition (int32 for safety) then clamp
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                del delta_int8 # Free temporary INT8 delta
                
                final_weight = torch.clamp(res, -128, 127).to(torch.int8).to(device)
                del res
                return final_weight
            else:
                # Fallback: Standard Float Patching
                final_weight = weight + (lora_diff * scale).to(weight.device, weight.dtype)
                del lora_diff
                return final_weight

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
                
                del delta # Free high-precision intermediate
            
            if total_delta_f is None:
                return weight

            if weight.dtype == torch.int8:
                # One single stochastic rounding step for all combined LoRAs

                # If QuaRot was applied to this layer's weights, rotate the combined delta into
                # the same basis (ΔW @ H^T) so the update is coherent: W_rot + ΔW_rot = (W + ΔW) @ H^T
                group_size = getattr(Int8TensorwiseOps, '_global_quarot_groupsize', QUAROT_GROUP_SIZE)
                if getattr(Int8TensorwiseOps, 'enable_quarot', False) and weight.shape[1] % group_size == 0:
                    try:
                        from .quarot import build_hadamard, rotate_weight
                        H = build_hadamard(group_size, device=comp_device, dtype=total_delta_f.dtype)
                        total_delta_f = rotate_weight(total_delta_f, H, group_size=group_size)
                    except ImportError:
                        pass

                eff_scale = self._get_effective_scale(total_delta_f, offset)
                delta_int8 = stochastic_round_int8_delta(total_delta_f, eff_scale, self.seed)
                del total_delta_f # Free combined high-precision delta
                
                res = weight.to(comp_device, torch.int32) + delta_int8.to(comp_device, torch.int32)
                del delta_int8 # Free temporary INT8 delta
                
                final_weight = torch.clamp(res, -128, 127).to(torch.int8).to(device)
                del res
                return final_weight
            else:
                final_weight = weight + total_delta_f.to(device, weight.dtype)
                del total_delta_f
                return final_weight


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
    If a standard LoRA is applied, it uses the dynamic adder path.
    If a specialized INT8 stochastic LoRA is applied, it uses the merged path.
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

        # ComfyUI patch format: (strength_patch, adapter, strength_model, offset, function)
        # Check if any patch is our specialized stochastic adapter (index 1 = adapter)
        has_stochastic = any(
            "INT8LoRAPatchAdapter" in str(type(p[1])) or "INT8MergedLoRAPatchAdapter" in str(type(p[1]))
            for p in patches
        )

        if is_int8_module and not has_stochastic:
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

        # --- STANDARD / STOCHASTIC PATH ---
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

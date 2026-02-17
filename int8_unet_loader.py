import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import warnings

warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm mode.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.utils")

try:
    from .int8_quant import Int8TensorwiseOps, _loading_stats
except ImportError:
    Int8TensorwiseOps = None
    _loading_stats = {"int8_direct": 0, "quantized_on_fly": 0, "excluded": 0, "quantize_time": 0.0}


class WanVideoINT8Loader:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    Inference uses fast torch._int_mm for blazing speed.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (["wan", "flux2", "z-image", "chroma", "ltx2", "qwen", "anima"],),
                "quantization_mode": (["auto", "tensorwise", "blockwise"], {"default": "auto"}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
                "auto_convert_to_int8": (["enable", "disable"], {"default": "enable"}),
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "WanVideo/INT8"
    DESCRIPTION = "Load INT8 tensorwise quantized models with fast torch._int_mm inference."

    def load_unet(self, unet_name, model_type, quantization_mode, offload_to_cpu, auto_convert_to_int8, debug_mode):
        import comfy.model_management
        import gc
        from comfy.sd import load_diffusion_model
        from . import int8_quant
        from .int8_quant import strip_comfy_quant_keys
        
        # Guard: Ensure Int8TensorwiseOps imported successfully
        if Int8TensorwiseOps is None:
            raise RuntimeError(
                "Int8TensorwiseOps failed to import. "
                "Check console for the underlying ImportError. "
                "Common causes: missing dependencies (e.g., torch, numpy), "
                "or errors in int8_quant.py."
            )
        
        is_debug = debug_mode
        Int8TensorwiseOps.debug_mode = is_debug
        int8_quant._DEBUG_MODE = is_debug
        
        comfy.model_management.unload_all_models()
        gc.collect()
        try:
            comfy.model_management.soft_empty_cache()
        except RuntimeError as e:
            if "checkPoolLiveAllocations" not in str(e):
                raise
        
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        model_options = {"custom_operations": Int8TensorwiseOps}
        
        offload_enabled = (offload_to_cpu == "enable")
        Int8TensorwiseOps.offload_to_cpu = offload_enabled
        
        Int8TensorwiseOps.auto_convert_to_int8 = (auto_convert_to_int8 == "enable")
        Int8TensorwiseOps.quantization_mode = quantization_mode
        
        Int8TensorwiseOps.excluded_names = []
        _loading_stats["int8_direct"] = 0
        _loading_stats["quantized_on_fly"] = 0
        _loading_stats["excluded"] = 0
        _loading_stats["quantize_time"] = 0.0
        
        if model_type == "flux2":
            Int8TensorwiseOps.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 'final_layer',
                'double_stream_modulation_img', 'double_stream_modulation_txt',
                'single_stream_modulation'
            ]
        elif model_type == "z-image":
            Int8TensorwiseOps.excluded_names = [
                'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner', 
                'final_layer', 'noise_refiner', 'adaLN',
                'x_pad_token',
            ]
        elif model_type == "wan":
            Int8TensorwiseOps.excluded_names = [
                'patch_embed', 'text_projection', 'time_projection', 'head',
                'modulation', 'guidance', 'img_emb', 'txt_emb', 'time_emb',
                'final_layer', 'output_projection'
            ]
        elif model_type == "chroma":
            Int8TensorwiseOps.excluded_names = [
                'distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in', 'nerf_image_embedder',
                'nerf_blocks', 'nerf_final_layer_conv', '__x0__',
            ]
        elif model_type == "qwen":
            Int8TensorwiseOps.excluded_names = [
                'time_text_embed', 'img_in', 'norm_out', 'proj_out', 'txt_in'
            ]
        elif model_type == "ltx2":
            Int8TensorwiseOps.excluded_names = [
                'adaln_single', 'audio_adaln_single', 'audio_caption_projection', 'audio_patchify_proj', 'audio_proj_out',
                'audio_scale_shift_table', 'av_ca_a2v_gate_adaln_single', 'av_ca_audio_scale_shift_adaln_single', 'av_ca_v2a_gate_adaln_single',
                'av_ca_video_scale_shift_adaln_single', 'caption_projection', 'patchify_proj', 'proj_out', 'scale_shift_table',
            ]
        elif model_type == "anima":
            Int8TensorwiseOps.excluded_names = [
                "t_embedder", "x_embedder", "final_layer", "llm_adapter", "adaln_modulation",
            ]

        import time
        import comfy.utils
        from .int8_quant import convert_zimage_diffusers_state_dict, is_diffusers_zimage_format
        
        start_time = time.time()
        
        original_load_torch_file = comfy.utils.load_torch_file
        
        def patched_load_torch_file(*args, **kwargs):
            if is_debug:
                print(f"[z-image] patched_load_torch_file called for {args[0]}")
            result = original_load_torch_file(*args, **kwargs)
            
            if isinstance(result, tuple):
                sd, metadata = result
                # Strip .comfy_quant metadata keys for all models to prevent warnings
                sd = strip_comfy_quant_keys(sd)
                if is_diffusers_zimage_format(sd):
                    if is_debug:
                        print("[z-image] Diffusers format detected in tuple, converting...")
                    return convert_zimage_diffusers_state_dict(sd), metadata
                return sd, metadata
            
            sd = result
            # Strip .comfy_quant metadata keys for all models to prevent warnings
            sd = strip_comfy_quant_keys(sd)
            if is_diffusers_zimage_format(sd):
                if is_debug:
                    print("[z-image] Diffusers format detected, converting...")
                return convert_zimage_diffusers_state_dict(sd)
            return sd
        
        comfy.utils.load_torch_file = patched_load_torch_file
        
        try:
            model = load_diffusion_model(unet_path, model_options=model_options)
        finally:
            comfy.utils.load_torch_file = original_load_torch_file
            
        end_time = time.time()
        
        if is_debug:
            print(f"[DEBUG] Model load took: {end_time - start_time:.2f}s")
            print(f"[DEBUG] Stats: {_loading_stats['int8_direct']} layers loaded directly, "
                  f"{_loading_stats['quantized_on_fly']} quantized on-fly "
                  f"({_loading_stats['quantize_time']:.2f}s), "
                  f"{_loading_stats['excluded']} excluded")
        
        if is_debug:
            print("\n" + "="*60)
            print("[DEBUG] Detailed Model Inspection")
            print("="*60)
            
            if hasattr(model, 'model'):
                inner_model = model.model
            else:
                inner_model = model
            
            total_modules = 0
            linear_modules = []
            nan_scales = []
            inf_scales = []
            zero_scales = []
            
            for name, module in inner_model.named_modules():
                total_modules += 1
                if hasattr(module, 'weight'):
                    if hasattr(module, '_is_quantized') and module._is_quantized:
                        linear_modules.append((name, module))
                        
                        if hasattr(module, 'weight_scale'):
                            scale = module.weight_scale
                            if isinstance(scale, torch.Tensor):
                                if torch.isnan(scale).any():
                                    nan_scales.append((name, scale))
                                if torch.isinf(scale).any():
                                    inf_scales.append((name, scale))
                                if (scale == 0).any():
                                    zero_scales.append((name, scale))
            
            print(f"[DEBUG] Total modules: {total_modules}")
            print(f"[DEBUG] INT8 quantized Linear layers: {len(linear_modules)}")
            
            if nan_scales:
                print(f"\n[DEBUG] WARNING: {len(nan_scales)} layers have NaN scales:")
                for name, scale in nan_scales[:10]:
                    print(f"  - {name}: scale={scale}")
                if len(nan_scales) > 10:
                    print(f"  ... and {len(nan_scales) - 10} more")
            
            if inf_scales:
                print(f"\n[DEBUG] WARNING: {len(inf_scales)} layers have Inf scales:")
                for name, scale in inf_scales[:10]:
                    print(f"  - {name}: scale={scale}")
                if len(inf_scales) > 10:
                    print(f"  ... and {len(inf_scales) - 10} more")
            
            if zero_scales:
                print(f"\n[DEBUG] WARNING: {len(zero_scales)} layers have zero scales:")
                for name, scale in zero_scales[:10]:
                    print(f"  - {name}: scale={scale}")
                if len(zero_scales) > 10:
                    print(f"  ... and {len(zero_scales) - 10} more")
            
            print(f"\n[DEBUG] Sample INT8 layers:")
            for name, module in linear_modules[:20]:
                w_shape = tuple(module.weight.shape)
                scale = getattr(module, 'weight_scale', 'N/A')
                if isinstance(scale, torch.Tensor):
                    scale_str = f"{scale.item():.6f}" if scale.numel() == 1 else f"tensor{tuple(scale.shape)}"
                else:
                    scale_str = str(scale)
                print(f"  {name}: weight={w_shape}, scale={scale_str}")
            
            print(f"\n[DEBUG] Weight value inspection:")
            issues_found = []
            for name, module in linear_modules:
                weight = module.weight
                if torch.isnan(weight).any():
                    nan_count = torch.isnan(weight).sum().item()
                    issues_found.append(f"{name}: {nan_count} NaN values in weight")
                if torch.isinf(weight).any():
                    inf_count = torch.isinf(weight).sum().item()
                    issues_found.append(f"{name}: {inf_count} Inf values in weight")
            
            if issues_found:
                print("[DEBUG] Weight issues found:")
                for issue in issues_found[:10]:
                    print(f"  - {issue}")
                if len(issues_found) > 10:
                    print(f"  ... and {len(issues_found) - 10} more")
            else:
                print("[DEBUG] No NaN/Inf values found in INT8 weights")
            
            print("="*60 + "\n")
        
        return (model,)

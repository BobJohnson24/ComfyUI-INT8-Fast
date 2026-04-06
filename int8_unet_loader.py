import os
import torch
import folder_paths
import comfy.sd
import comfy.utils

from .int8_quant import Int8TensorwiseOps


class UNetLoaderINTW8A8:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    Inference uses fast torch._int_mm for blazing speed. (insert rocket emoji, fire emoji to taste)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp16", "bf16"],),
                "model_type": (["flux2", "z-image", "chroma", "wan", "ltx2", "qwen"],),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False}),
                "enable_quarot": ("BOOLEAN", {"default": False, "tooltip": "Enable QuaRot (Hadamard rotation) for better quantization."}),
                #"enable_triton": ("BOOLEAN", {"default": True, "tooltip": "Use the Triton fused INT8 kernel. Disable to fall back to torch._int_mm (useful for debugging or unsupported GPUs)."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load INT8 tensorwise quantized models with fast torch._int_mm inference."

    def load_unet(self, unet_name, weight_dtype, model_type, on_the_fly_quantization, enable_quarot=False):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Use Int8TensorwiseOps for proper direct int8 loading
        model_options = {"custom_operations": Int8TensorwiseOps}
        
        # We need to peek at the model type to set exclusions for Flux
        # ComfyUI loads metadata before the full model
        from comfy.sd import load_diffusion_model
        
        # Set quantization flags
        Int8TensorwiseOps.excluded_names = []
        Int8TensorwiseOps.dynamic_quantize = on_the_fly_quantization
        Int8TensorwiseOps.enable_quarot = enable_quarot
        Int8TensorwiseOps.use_triton = True
        Int8TensorwiseOps._is_prequantized = False
        
        # Check explicit model_type for exclusions
        if model_type == "flux2":
            Int8TensorwiseOps.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 'final_layer', 
                'double_stream_modulation_img', 'double_stream_modulation_txt', 
                'single_stream_modulation',
            ]
        elif model_type == "z-image":
            Int8TensorwiseOps.excluded_names = [
                'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner', 
                'final_layer', 'noise_refiner', 'adaLN',
                'x_pad_token', 'layers.0.',
            ]
        elif model_type == "chroma":
            Int8TensorwiseOps.excluded_names = [
                'distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in', 'nerf_image_embedder',
                 'nerf_blocks', 'nerf_final_layer_conv', '__x0__', 'nerf_final_layer_conv',
            ]
        elif model_type == "qwen":
            Int8TensorwiseOps.excluded_names = [
                'time_text_embed', 'img_in', 'norm_out', 'proj_out', 'txt_in'
            ]
        elif model_type == "wan":
            Int8TensorwiseOps.excluded_names = [
                'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection' 'head',
                'img_emb',
            ]
        elif model_type == "ltx2":
            Int8TensorwiseOps.excluded_names = [
                'adaln_single', 'audio_adaln_single', 'audio_caption_projection', 'audio_patchify_proj', 'audio_proj_out',
                'audio_scale_shift_table', 'av_ca_a2v_gate_adaln_single', 'av_ca_audio_scale_shift_adaln_single', 'av_ca_v2a_gate_adaln_single',
                'av_ca_video_scale_shift_adaln_single', 'caption_projection', 'patchify_proj', 'proj_out', 'scale_shift_table',
            ]
            #print(f"Applying model-specific exclusions to Int8TensorwiseOps: {Int8TensorwiseOps.excluded_names}")

        # Load model directly - Int8TensorwiseOps handles int8 weights natively
        model = load_diffusion_model(unet_path, model_options=model_options)
        
        return (model,)


import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_detection
import comfy.lora
import comfy.lora_convert
import comfy.memory_management
import comfy.model_management
import logging

from .int8_quant import Int8TensorwiseOps


class UNetLoaderINTW8A8:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp16", "bf16"],),
                "model_type": (["flux2", "z-image", "z-image-l2p", "chroma", "wan", "ltx2", "qwen", "ernie", "anima"], {"tooltip": "Only used for on the fly quantization, to filter sensitive layers."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize a higher precision model to INT8. If the selected model is already INT8 keep unchecked."}),
                "enable_convrot": ("BOOLEAN", {"default": True, "tooltip": "Enable ConvRot for better quantization. ~1.1x slower, but near-GGUF_Q8 quality."}),
                "lora_mode": (["None", "Stochastic", "Dynamic"], {"default": "None", "tooltip": "None bakes LoRA patches with normal rounding which is the default behavior. Stochastic bakes with stochastic INT8 rounding, which can occasionally be closer to the BF16+lora baseline. Dynamic applies LoRA at inference time, which is slow and only works for conventional lora."}),
            },
            "optional": {
                "pre_lora": ("PRE_LORA",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load and Quantize INT8 models with fast triton inference."

    def load_unet(self, unet_name, weight_dtype, model_type, on_the_fly_quantization, enable_convrot=False, lora_mode="None", pre_lora=None):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        # Backward compatibility for workflows saved with the old dynamic_lora boolean widget.
        if isinstance(lora_mode, bool):
            lora_mode = "Dynamic" if lora_mode else "None"
        lora_mode = str(lora_mode)
        if lora_mode not in {"None", "Stochastic", "Dynamic"}:
            lora_mode = "None"
        
        if pre_lora is not None:
            loras_to_load = pre_lora if isinstance(pre_lora, list) else [pre_lora]
        else:
            loras_to_load = []
        
        # Use Int8TensorwiseOps for proper direct int8 loading
        model_options = {"custom_operations": Int8TensorwiseOps}
        
        # We need to peek at the model type to set exclusions for Flux
        # ComfyUI loads metadata before the full model
        
        # Set quantization flags
        Int8TensorwiseOps.excluded_names = []
        Int8TensorwiseOps.dynamic_quantize = on_the_fly_quantization
        Int8TensorwiseOps.enable_convrot = enable_convrot
        Int8TensorwiseOps.use_triton = True
        Int8TensorwiseOps._is_prequantized = False
        Int8TensorwiseOps.lora_mode = lora_mode
        Int8TensorwiseOps.dynamic_lora = lora_mode == "Dynamic"
        Int8TensorwiseOps.dynamic_load_device = None
        if comfy.memory_management.aimdo_enabled and (on_the_fly_quantization or len(loras_to_load) > 0):
            Int8TensorwiseOps.dynamic_load_device = comfy.model_management.get_torch_device()
            logging.info(f"INT8 Fast: Aimdo dynamic loading active, using {Int8TensorwiseOps.dynamic_load_device} as a per-layer bake/quant work device.")
        if hasattr(Int8TensorwiseOps, "_logged_otf"):
            delattr(Int8TensorwiseOps, "_logged_otf")
        
        # Check explicit model_type for exclusions
        if model_type == "flux2":
            Int8TensorwiseOps.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 
                'double_stream_modulation_img', 'double_stream_modulation_txt', 
                'single_stream_modulation',
            ]
        elif model_type == "z-image":
            Int8TensorwiseOps.excluded_names = [
                'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner', 
                'final_layer', 'noise_refiner', 'adaLN',
                'x_pad_token', 'layers.0.',
            ]
        elif model_type == "z-image-l2p":
            Int8TensorwiseOps.excluded_names = [
        'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner',
        'noise_refiner', 'adaLN', 'x_pad_token',
        'local_decoder',
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
        elif model_type == "ernie":
            Int8TensorwiseOps.excluded_names = [
                'time', 'x_embedder', 'text_proj', 'adaLN',
            ]
        elif model_type == "anima":
            Int8TensorwiseOps.excluded_names = [
                'embed', 'llm', 'adaln',
            ]
        elif model_type == "hidream o1":
            Int8TensorwiseOps.excluded_names = [
                'embed', 'language_model.layers.35.mlp',
            ]
        elif model_type == "wan":
            Int8TensorwiseOps.excluded_names = [
                'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head',
                'img_emb',
            ]
        elif model_type == "ltx2":
            Int8TensorwiseOps.excluded_names = [
                'adaln', 'embedding', 'patchify', 'to_gate_logits', 'proj_out', 'model.audio', 'model.video', 'model.av', 'model.patch', 'model.proj', 'shift', #'transformer_blocks.0.', 'transformer_blocks.46.', # 'transformer_blocks.1.', 'transformer_blocks.47.',
            ]

        # Load state dict once to detect model and prepare LoRA
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        
        # Pre-load LoRA if selected to bake it during quantization
        Int8TensorwiseOps.lora_patches = {}
        if len(loras_to_load) > 0:
            grouped_patches = {}
            for lora in loras_to_load:
                lora_name = lora.get("lora_name", "None")
                lora_strength = lora.get("lora_strength", 1.0)
                
                if lora_name == "None":
                    continue
                    
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                lora_data = comfy.lora_convert.convert_lora(lora_data) # Handle various LoRA formats
                
                # Use a skeleton model to build the proper key_map.
                # This is fast because Int8TensorwiseOps.Linear skips weight initialization.
                unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
                m_config = comfy.model_detection.model_config_from_unet(sd, unet_prefix, metadata=metadata)
                
                # Fallback: some models like Flux might fail with the extracted prefix
                if m_config is None and unet_prefix != "":
                    m_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
                    if m_config is not None:
                        unet_prefix = ""

                if m_config is not None:
                    m_config.custom_operations = Int8TensorwiseOps
                    Int8TensorwiseOps.skeleton_meta_init = True
                    try:
                        skeleton_model = m_config.get_model(sd, unet_prefix)
                    finally:
                        Int8TensorwiseOps.skeleton_meta_init = False
                    key_map = comfy.lora.model_lora_keys_unet(skeleton_model, {})

                    
                    patch_dict = comfy.lora.load_lora(lora_data, key_map)
                    
                    # Normalize keys and group patches by target layer to support offsets/functions
                    # We want the keys to match what the model's Linear._load_from_state_dict will see.
                    def normalize_key(key):
                        if not isinstance(key, str):
                            return key
                        for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                            if key.startswith(p):
                                return key[len(p):]
                        return key
                    
                    for k, v in patch_dict.items():
                        target_key = k
                        offset = None
                        function = None
                        if isinstance(k, tuple):
                            target_key = k[0]
                            if len(k) > 1: offset = k[1]
                            if len(k) > 2: function = k[2]
                        
                        nk = normalize_key(target_key)
                        if nk not in grouped_patches:
                            grouped_patches[nk] = []
                        grouped_patches[nk].append((v, offset, function, lora_strength))
                else:
                    logging.warning(f"INT8 Fast: Could not detect model type for LoRA mapping.")
                
                del lora_data
            
            if grouped_patches:
                Int8TensorwiseOps.lora_patches = grouped_patches
                logging.info(f"INT8 Fast: Prepared {len(grouped_patches)} layer patches for baking.")
            
        # Load model using the already-loaded state dict
        try:
            Int8TensorwiseOps.applied_lora_patches = set()
            model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
            
            # Print unmatched keys to help with debugging
            if Int8TensorwiseOps.lora_patches:
                unmatched = set(Int8TensorwiseOps.lora_patches.keys()) - Int8TensorwiseOps.applied_lora_patches
                if unmatched:
                    print(f"INT8 Fast: {len(unmatched)} LoRA keys were NOT matched:")
                    for k in sorted(unmatched):
                        print(f"  unmatched: {k}")
                else:
                    action = "scheduled for deferred baking" if Int8TensorwiseOps.dynamic_load_device is not None else "successfully baked"
                    #print(f"INT8 Fast: All {len(Int8TensorwiseOps.lora_patches)} LoRA keys {action}!")
        finally:
            # Always clear patches after load to avoid sticking
            dynamic_load_device = Int8TensorwiseOps.dynamic_load_device
            Int8TensorwiseOps.lora_patches = {}
            Int8TensorwiseOps.dynamic_load_device = None
            if dynamic_load_device is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(Int8TensorwiseOps, 'applied_lora_patches'):
                delattr(Int8TensorwiseOps, 'applied_lora_patches')
        
        # Wrap in custom patcher for unified LoRA support
        from .int8_quant import INT8ModelPatcher
        model = INT8ModelPatcher.clone(model)
        model._safetensors_metadata = metadata  # stash for save node
        
        return (model,)


class PreLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "lora_name_1": (["None"] + folder_paths.get_filename_list("loras"), ),
                "lora_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("PRE_LORA",)
    FUNCTION = "load_pre_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Pre-load a LoRA to bake it during quantization."

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    def load_pre_lora(self, **kwargs):
        loras = []
        
        # ComfyUI strips dynamic inputs that aren't in INPUT_TYPES.
        # We can recover them from the raw prompt dictionary.
        prompt = kwargs.get("prompt", {})
        node_id = kwargs.get("id", None)
        
        if prompt and node_id and node_id in prompt:
            node_inputs = prompt[node_id].get("inputs", {})
        else:
            node_inputs = kwargs

        if "lora_name" in node_inputs:
            name = node_inputs["lora_name"]
            strength = round(node_inputs.get("lora_strength", 1.0), 2)
            if name != "None" and strength != 0.0:
                loras.append({"lora_name": name, "lora_strength": strength})
                
        i = 1
        while True:
            name_key = f"lora_name_{i}"
            strength_key = f"lora_strength_{i}"
            if name_key in node_inputs:
                name = node_inputs[name_key]
                strength = round(node_inputs.get(strength_key, 1.0), 2)
                if name != "None" and strength != 0.0:
                    loras.append({"lora_name": name, "lora_strength": strength})
                i += 1
            else:
                break
                
        return (loras,)



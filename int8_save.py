import folder_paths
import comfy.sd
import json
import os
from comfy.cli_args import args

class INT8ModelSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "int8_models/INT8_Model"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "loaders"

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)

        metadata = {}
        if not args.disable_metadata:
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        extra_keys = {}
        import torch
        
        patched_modules = []
        
        # We need to peek at the model's actual modules to save comfy_quant and weight_scale
        if hasattr(model, "model") and hasattr(model.model, "named_modules"):
            for name, module in model.model.named_modules():
                if getattr(module, "_is_quantized", False):
                    # 1. Comfy Quant Hint
                    quant_conf = {"quarot": getattr(module, "_use_quarot", False)}
                    if hasattr(module, "_quarot_groupsize"):
                        quant_conf["quarot_groupsize"] = module._quarot_groupsize
                        
                    # Prepend 'model.' as comfy.sd.save_checkpoint typically adds this to all weights
                    # but may not add it to extra_keys. This ensures they stay alongside weights.
                    prefix = "model." + name + "." if name else "model."
                    
                    extra_keys[prefix + "comfy_quant"] = torch.tensor(
                        list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8
                    )
                    
                    # 2. Handle scalar weight_scale which is not registered as a buffer
                    if getattr(module, "_weight_scale_scalar", None) is not None:
                        extra_keys[prefix + "weight_scale"] = torch.tensor(module._weight_scale_scalar)

                    # 3. Temporarily bypass ComfyUI's LazyCastingParam to prevent crash on int8 tensors
                    had_flag = hasattr(module, "comfy_patched_weights")
                    old_flag = getattr(module, "comfy_patched_weights", False)
                    patched_modules.append((module, had_flag, old_flag))
                    module.comfy_patched_weights = True

        try:
            comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata, extra_keys=extra_keys)
        finally:
            # Restore module states so we don't break dynamic VRAM management
            for module, had_flag, old_flag in patched_modules:
                if had_flag:
                    module.comfy_patched_weights = old_flag
                else:
                    delattr(module, "comfy_patched_weights")
                    
        return {}

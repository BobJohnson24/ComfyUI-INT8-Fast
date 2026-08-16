import folder_paths
import comfy.sd
import comfy.model_patcher
import comfy.model_management
import json
import os
import logging
import torch
from comfy.cli_args import args


def _is_dynamic_lora_enabled():
    try:
        from .int8_quant import Int8TensorwiseOps
        return bool(getattr(Int8TensorwiseOps, "dynamic_lora", False))
    except Exception:
        return False


def _resolve_source_metadata(model):
    """Walk the patcher clone chain and the inner model object to recover the
    original safetensors metadata that was stashed by UNetLoaderINTW8A8.

    ComfyUI's ``ModelPatcher.clone()`` builds a fresh patcher and does not copy
    over arbitrary attributes set on the source patcher. INT8GroupedLora and
    other downstream nodes call ``model.clone()``, which would otherwise drop
    the ``_safetensors_metadata`` stash and produce a checkpoint missing the
    ``int8_quantized`` / ``int8_model_type`` / ``config`` (LTX2) flags that the
    loader relies on for round-trips.
    """
    seen = set()

    def _walk(m):
        if m is None or id(m) in seen:
            return None
        seen.add(id(m))
        meta = getattr(m, "_safetensors_metadata", None)
        if isinstance(meta, dict) and meta:
            return meta
        inner = getattr(m, "model", None)
        if inner is not None:
            inner_meta = getattr(inner, "_int8_source_metadata", None)
            if isinstance(inner_meta, dict) and inner_meta:
                return inner_meta
        parent = getattr(m, "parent", None)
        return _walk(parent)

    return _walk(model)


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
        # Preserve source safetensors metadata (int8_quantized, int8_model_type,
        # ltx2 config, etc.). Walk the patcher chain because INT8GroupedLora's
        # clone strips this attribute from the local patcher.
        src_meta = _resolve_source_metadata(model)
        if isinstance(src_meta, dict):
            metadata.update(src_meta)
        if not src_meta:
            logging.warning(
                "INT8 Save: source safetensors metadata could not be located on the patcher chain. "
                "The output checkpoint will be saved without int8_quantized/int8_model_type/config metadata, "
                "which may break re-loading for some models (notably LTX2)."
            )
        # if not args.disable_metadata:
        #     metadata["prompt"] = prompt_info
        #     if extra_pnginfo is not None:
        #         for x in extra_pnginfo:
        #             metadata[x] = json.dumps(extra_pnginfo[x])

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        extra_keys = {}

        patched_modules = []
        patched_module_ids = set()

        def mark_module_for_direct_save(module):
            module_id = id(module)
            if module_id in patched_module_ids:
                return
            had_flag = hasattr(module, "comfy_patched_weights")
            old_flag = getattr(module, "comfy_patched_weights", False)
            patched_modules.append((module, had_flag, old_flag))
            patched_module_ids.add(module_id)
            module.comfy_patched_weights = True

        def module_has_int8_param(module):
            for attr in ("weight", "bias"):
                tensor = getattr(module, attr, None)
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int8:
                    return True
            return False

        def iter_model_modules(model_patcher):
            if hasattr(model_patcher, "model") and hasattr(model_patcher.model, "named_modules"):
                yield from model_patcher.model.named_modules()

        def materialize_int8_lora_patches(model_patcher):
            """Bake active non-dynamic INT8 LoRA low-VRAM functions into weights.

            The newer ComfyUI low-VRAM integration stores non-dynamic INT8 LoRA
            patches as module.weight_lowvram_function instead of immediately
            mutating the int8 parameter. That is correct for sampling memory,
            but this save node marks int8 modules as directly savable, which
            intentionally bypasses LazyCastingParam. Without this pre-pass the
            checkpoint would serialize the base int8 weight and drop the LoRA.
            """
            if _is_dynamic_lora_enabled() or not hasattr(model_patcher, "patch_weight_to_device"):
                return

            patches = getattr(model_patcher, "patches", None)
            if not patches:
                return

            load_device = getattr(model_patcher, "load_device", None)
            materialized = 0
            for name, module in iter_model_modules(model_patcher):
                if not getattr(module, "_is_quantized", False):
                    continue

                weight_key = name + ".weight" if name else "weight"
                if weight_key not in patches:
                    continue

                try:
                    current_weight = getattr(module, "weight", None)
                    device_to = load_device if load_device is not None else getattr(current_weight, "device", None)
                    model_patcher.patch_weight_to_device(weight_key, device_to=device_to)
                    if hasattr(module, "weight_lowvram_function"):
                        module.weight_lowvram_function = None
                    materialized += 1
                except Exception as e:
                    logging.warning(
                        f"INT8 Save: failed to materialize LoRA patch for {weight_key}: {e}. "
                        "The saved checkpoint may miss this LoRA patch."
                    )

            if materialized > 0:
                logging.info(f"INT8 Save: materialized {materialized} INT8 LoRA patched weight(s) before saving.")

        # Finalize any deferred INT8 layers (Aimdo/Windows deferred-load path sets
        # _pending_int8_finalize instead of quantizing immediately). Without this,
        # those modules still have _is_quantized=False at save time and no
        # comfy_quant keys are emitted.
        finalize_fn = getattr(model, "finalize_pending_int8", None)
        if finalize_fn is not None:
            finalize_fn()

        # CRITICAL: Apply any pending LoRA / model patches BEFORE collecting
        # extra_keys and BEFORE save_checkpoint runs its own load_models_gpu().
        #
        # Why: when LoRAs were stacked via INT8GroupedLora (or the standard
        # LoRA loader), the patches live on the model patcher and are only
        # baked into the int8 weights when ``patch_model`` runs. ``save_checkpoint``
        # internally calls ``load_models_gpu`` which does trigger the bake, but
        # we also need to observe the post-bake module state to emit accurate
        # ``comfy_quant`` / scalar ``weight_scale`` extra_keys (and to be sure
        # ``module.comfy_patched_weights`` is set so ``model_state_dict_for_saving``
        # emits the int8 weight directly instead of wrapping it in a
        # LazyCastingParam, which assumes float dtypes).
        #
        # ``force_full_load=True`` keeps every patched layer on-device so we
        # see consistent int8 weights for every module, even on lowvram setups.
        try:
            comfy.model_management.load_models_gpu([model], force_full_load=True)
        except Exception as e:
            logging.warning(
                f"INT8 Save: full-load pre-pass failed ({e}); falling back to "
                "default load_models_gpu without force_full_load."
            )
            try:
                comfy.model_management.load_models_gpu([model])
            except Exception as e2:
                logging.warning(
                    f"INT8 Save: load_models_gpu fallback also failed ({e2}); "
                    "continuing best-effort. The saved checkpoint may be "
                    "incomplete if LoRA patches were not applied."
                )

        # Re-finalize after load_models_gpu in case any aimdo deferred layers
        # were materialized only during the load pass.
        if finalize_fn is not None:
            finalize_fn()

        materialize_int8_lora_patches(model)

        # Collect comfy_quant and (scalar) weight_scale extra_keys based on
        # the post-patch module state.
        if hasattr(model, "model"):
            for name, module in iter_model_modules(model):
                if module_has_int8_param(module):
                    # ComfyUI's LazyCastingParam subclasses torch.nn.Parameter
                    # with requires_grad=True by default, which is invalid for
                    # int8 tensors. Mark all int8 modules for direct save.
                    mark_module_for_direct_save(module)

                if getattr(module, "_is_quantized", False):
                    use_convrot = bool(getattr(module, "_use_convrot", False))
                    # comfy.ops dispatches on this key; without it core's
                    # UNETLoader raises "Unknown quantization format" and the
                    # checkpoint only loads through our own loader.
                    quant_conf = {"format": "int8_tensorwise", "convrot": use_convrot}
                    # Always emit a groupsize when convrot is on, even if the
                    # module is using the default. Older save paths only wrote
                    # this field when ``_convrot_groupsize`` had been set
                    # explicitly, which left on-the-fly-quantized layers with
                    # an unspecified groupsize and forced the loader to fall
                    # back to ``CONVROT_GROUP_SIZE`` (which happens to match
                    # today, but is fragile if the default ever changes).
                    if use_convrot:
                        try:
                            from .int8_quant import CONVROT_GROUP_SIZE
                        except Exception:
                            CONVROT_GROUP_SIZE = 256
                        quant_conf["convrot_groupsize"] = int(
                            getattr(module, "_convrot_groupsize", CONVROT_GROUP_SIZE)
                        )

                    # Track granularity so the loader can pick the matching
                    # forward kernel without re-inspecting tensor shapes.
                    quant_conf["per_row"] = bool(getattr(module, "_is_per_row", False))

                    # Prepend 'model.' as comfy.sd.save_checkpoint adds this
                    # prefix to all weights; extra_keys are NOT auto-prefixed
                    # so we must do it ourselves to keep them aligned with the
                    # owning weight tensor.
                    prefix = "model." + name + "." if name else "model."

                    extra_keys[prefix + "comfy_quant"] = torch.tensor(
                        list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8
                    )

                    # Handle scalar weight_scale which is not registered as a
                    # persistent buffer (so it would be missing from
                    # state_dict() entirely).
                    if getattr(module, "_weight_scale_scalar", None) is not None:
                        extra_keys[prefix + "weight_scale"] = torch.tensor(module._weight_scale_scalar)

                    mark_module_for_direct_save(module)

        original_lazy_new = comfy.model_patcher.LazyCastingParam.__new__
        original_lazy_piece_new = comfy.model_patcher.LazyCastingParamPiece.__new__

        def lazy_casting_param_new(cls, model, key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        def lazy_casting_param_piece_new(cls, caster, state_dict_key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        had_save_flag = hasattr(model, "_int8_save_materialized_lora")
        old_save_flag = getattr(model, "_int8_save_materialized_lora", False)

        try:
            model._int8_save_materialized_lora = True
            comfy.model_patcher.LazyCastingParam.__new__ = staticmethod(lazy_casting_param_new)
            comfy.model_patcher.LazyCastingParamPiece.__new__ = staticmethod(lazy_casting_param_piece_new)
            comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata, extra_keys=extra_keys)
        finally:
            comfy.model_patcher.LazyCastingParam.__new__ = original_lazy_new
            comfy.model_patcher.LazyCastingParamPiece.__new__ = original_lazy_piece_new
            if had_save_flag:
                model._int8_save_materialized_lora = old_save_flag
            else:
                try:
                    delattr(model, "_int8_save_materialized_lora")
                except AttributeError:
                    pass
            # Restore module states so we don't break dynamic VRAM management
            for module, had_flag, old_flag in patched_modules:
                if had_flag:
                    module.comfy_patched_weights = old_flag
                else:
                    try:
                        delattr(module, "comfy_patched_weights")
                    except AttributeError:
                        pass

        return {}

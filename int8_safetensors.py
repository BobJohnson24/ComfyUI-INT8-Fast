import json
import logging

import torch
import comfy.utils


def _parse_comfy_quant_tensor(tensor):
    try:
        return json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except Exception:
        return None


def _get_optional_tensor(state_dict, key):
    value = state_dict.get(key)
    if value is not None:
        return value
    return None


def _infer_int8_layer(weight, scale):
    if weight is None or scale is None or weight.dtype != torch.int8:
        return None
    return {"format": "int8_tensorwise"}


def prepare_int8_state_dict(state_dict, metadata=None, model_prefix=""):
    """Normalize quantized checkpoints before loading with custom INT8 ops.

    ComfyUI skips convert_old_quants() when custom_operations is set. That is
    normally correct for custom ops, but it means older INT8 files or files with
    only safetensors metadata never get their .comfy_quant sidecar tensors.
    This keeps the metadata conversion local and dependency-free.
    """
    metadata = {} if metadata is None else dict(metadata)

    try:
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, model_prefix=model_prefix, metadata=metadata
        )
    except Exception as e:
        logging.warning("INT8 Fast: Comfy quant metadata conversion failed: %s", e)

    layers = {}

    quant_metadata = metadata.get("_quantization_metadata")
    if quant_metadata:
        try:
            parsed = json.loads(quant_metadata)
            layers.update(parsed.get("layers", {}))
        except Exception as e:
            logging.warning("INT8 Fast: Could not parse _quantization_metadata: %s", e)

    for key, value in list(state_dict.items()):
        if key.endswith(".comfy_quant"):
            layer_name = key[: -len(".comfy_quant")]
            parsed = _parse_comfy_quant_tensor(value)
            if parsed is not None:
                layers[layer_name] = parsed

    for key, weight in list(state_dict.items()):
        if not key.endswith(".weight"):
            continue
        layer_name = key[: -len(".weight")]

        scale = _get_optional_tensor(state_dict, layer_name + ".weight_scale")
        if scale is None:
            scale = _get_optional_tensor(state_dict, layer_name + ".scale_weight")
            if scale is not None:
                state_dict[layer_name + ".weight_scale"] = scale

        if layer_name in layers:
            continue

        layer_conf = _infer_int8_layer(weight, scale)
        if layer_conf is not None:
            layers[layer_name] = layer_conf

    if layers:
        for layer_name, layer_conf in layers.items():
            key = layer_name + ".comfy_quant"
            if key not in state_dict:
                state_dict[key] = torch.tensor(
                    list(json.dumps(layer_conf).encode("utf-8")),
                    dtype=torch.uint8,
                )
        metadata["_quantization_metadata"] = json.dumps({"layers": layers})

    return state_dict, metadata


def load_int8_torch_file(path):
    """Load a checkpoint through Comfy's native loader and normalize INT8 metadata."""
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    return prepare_int8_state_dict(state_dict, metadata)

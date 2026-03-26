# Comfy INT8 Acceleration

This node speeds up Flux2, Chroma, Z-Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It's unlikely to be faster than proper FP8 on 40-Series and above. 
Works with lora*, torch compile (needed to get full speedup).

*LoRAs need to be applied using one of the following methods:

### Option 1: Included INT8 LoRA Node (Recommended for Speed)
- **Performance:** Faster inference
- **Quality:** Possibly slightly lower quality
- Use the included INT8 LoRA node

### Option 2A: Included Int8 Dynamic LoRa Node
- **Performance:** ~1.15x slower due to dynamic calculations
- **Quality:** Possibly slightly higher quality

### Option 2B: Comfy's native Lora Bypass Node
- **Performance:** ~1.15x slower due to dynamic calculations
- **Quality:** Possibly slightly higher quality

Pre-quantized checkpoints are recommended for most architectures.

**Shoutout to [vistralis](https://huggingface.co/vistralis) for these:** 
Make sure to update the node to use them as int8 row-wise was added.

| Model | Link |
|-------|------|
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

**My own:**

| Model | Link |
|-------|------|
| Chroma1-HD | [Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise) |
| Z-Image-Base | [Download](https://huggingface.co/bertbobson/Z-Image-Base-INT8-QUIP) |
| Z-Image-Turbo | [Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise) |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |


# Metrics:

Measured on a 3090 at 1024x1024, 26 steps with Flux2 Klein Base 9B.

| Format | Speed (s/it) | Relative Speedup |
|-------|--------------|------------------|
| bf16 | 2.07 | 1.00× |
| bf16 compile | 2.24 | 0.92× |
| fp8 | 2.06 | 1.00× |
| int8 | 1.64 | 1.26× |
| int8 compile | 1.04 | 1.99× |
| gguf8_0 compile | 2.03 | 1.02× |

Measured on an 8gb 5060, same settings:

| Format | Speed (s/it) | Relative Speedup |
|-------|--------------|------------------|
| fp8 | 3.04 | 1.00× |
| fp8 fast | 3.00 | 1.00× |
| fp8 compile | couldn't get to work | ??× |
| int8 | 2.53 | 1.20× |
| int8 compile | 2.25 | 1.35× |


# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!


## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of a day

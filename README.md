# Comfy INT8 Acceleration

This node speeds up Flux2, Chroma, Z-Image, Ernie Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It's unlikely to be faster than proper FP8 on 40-Series and above. 
Works with lora, torch compile (needed to get full speedup).

Pre-quantized checkpoints were recommended for most architectures, but on-the-fly quantization with QuaRot is better in all cases.
However, QuaRot is also a little slower, so these prequantized models are still useful.

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
| Z-Image-Base* | ~~[Download](https://huggingface.co/bertbobson/Z-Image-Base-INT8-QUIP)~~ 
| Z-Image-Turbo | [Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise) |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

*Z-Image Base weights have been Deprecated in favor of Quarot OTF, which is higher quality.


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

## newgrit1004 for the QuaRot code I basically copied
https://github.com/newgrit1004/ComfyUI-ZImage-Triton

## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of a day

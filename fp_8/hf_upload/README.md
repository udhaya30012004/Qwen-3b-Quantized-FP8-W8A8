---
base_model: Qwen/Qwen2.5-3B
base_model_relation: quantized
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
language:
- en
tags:
- fp8
- w8a8
- quantized
- post-training-quantization
- compressed-tensors
- llm-compressor
- vllm
- qwen2.5
---

# Qwen2.5-3B-FP8-W8A8-Dynamic

This repository contains a **Post-Training Quantized (PTQ) FP8 W8A8** checkpoint of [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B), produced with `llm-compressor` (`FP8_DYNAMIC` / `compressed-tensors` format) and verified for native FP8 Tensor Core serving in **vLLM**.

- **Quantization Method:** Pure one-shot PTQ — zero fine-tuning, zero LoRA, zero calibration dataset required.
- **Weights:** FP8 E4M3 (`float8_e4m3fn`), symmetric **per-channel** across all **252 Linear layers** (36 transformer blocks).
- **Activations:** FP8 E4M3, symmetric **dynamic per-token** scaling at inference time.
- **Unquantized (`bfloat16`):** Token embeddings, RMSNorms, and `lm_head`.
- **Disk Footprint:** `3.40 GB` (`3.17 GiB`) — **44.9% smaller** than the `6.17 GB` (`5.75 GiB`) BF16 baseline.
- **Full Benchmark Code & Study:** [`github.com/udhaya30012004/Qwen-3b-Quantized-FP8-W8A8`](https://github.com/udhaya30012004/Qwen-3b-Quantized-FP8-W8A8)

---

## Benchmark Summary (NVIDIA L4, 24 GB Ada Lovelace)

Evaluated with `vLLM 0.34.0` and `lm-evaluation-harness 0.4.13` (0-shot, exact paired McNemar tests vs BF16 baseline):

| Metric | BF16 Baseline | FP8 W8A8 Per-Channel (This Model) | Delta / Speedup |
|---|---:|---:|---:|
| **Safetensors Size** | `6,171.9 MB` (`5.75 GiB`) | **`3,399.5 MB` (`3.17 GiB`)** | **-44.9%** |
| **5-Task Zero-Shot Avg** | `68.89%` | **`69.01%`** | **`+0.11%` (no loss)** |
| **WikiText-2 Perplexity** | `7.782` | **`7.834`** | `+0.052` |
| **UltraChat Perplexity** | `4.144` | **`4.180`** | `+0.036` |
| **Mean Full-Vocab KL (Wiki / Chat)** | `0.0000` (ref) | **`0.0105` / `0.0093`** | Low logit shift |
| **Top-1 Token Agreement (Chat / Wiki)** | `100.0%` (ref) | **`95.63%` / `94.71%`** | High fidelity |
| **Single-User Decode (`B=1`)** | `25.55 ms/tok` (`39.0 tok/s`) | **`16.42 ms/tok` (`60.7 tok/s`)** | **1.56× faster** |
| **Peak Throughput (`B=128, in=128`)** | `2,503.6 tok/s` | **`3,373.5 tok/s`** | **1.35× (+34.7%)** |
| **Long-Prompt Throughput (`B=128, in=512`)** | `1,223.3 tok/s` | **`1,612.6 tok/s`** | **1.32× (+31.8%)** |
| **Prefill Time (`B=128, in=512`)** | `7.72 s` | **`5.78 s`** | **1.34× faster** |

---

## Visual Evaluation Results

### 1. Model Size & Compression (-44.9%)
![Model Size and Compression](assets/01_model_size_compression.png)

### 2. Zero-Shot Downstream Accuracy (`16,695` Questions)
![Overall Zero-Shot Accuracy](assets/02_zero_shot_accuracy.png)

| Benchmark | Metric | BF16 Baseline | FP8 W8A8 PC | Delta (McNemar) |
|---|---|---:|---:|---:|
| **HellaSwag** (`10,042` Qs) | `acc_norm` | `73.56%` | `73.52%` | `-0.04%` (ns) |
| **ARC-Easy** (`2,376` Qs) | `acc` | `77.31%` | `77.15%` | `-0.17%` (ns) |
| **ARC-Challenge** (`1,172` Qs) | `acc_norm` | `46.76%` | `46.93%` | `+0.17%` (ns) |
| **WinoGrande** (`1,267` Qs) | `acc` | `68.43%` | `68.98%` | `+0.55%` (ns) |
| **PIQA** (`1,838` Qs) | `acc` | `78.40%` | `78.45%` | `+0.05%` (ns) |
| **5-Task Average** | — | **`68.89%`** | **`69.01%`** | **`+0.11%` (Zero Drop)** |

*`ns` = not statistically significant (`p ≥ 0.05`, exact paired McNemar test).*

### 3. Next-Token Distribution Fidelity (Top-1 Token Agreement)
![Top-1 Token Agreement vs BF16](assets/03_token_agreement.png)

### 4. End-to-End Request Latency (Prefill vs Decode)
![End-to-End Latency Breakdown](assets/04_end_to_end_latency.png)

### 5. Serving Throughput Scaling (Batch 1 → 128 on NVIDIA L4)
![Throughput Scaling](assets/05_throughput_scaling.png)

---

## Quickstart Usage

### 1. Serving with vLLM (Recommended)

vLLM automatically detects the `compressed-tensors` FP8 configuration from `config.json`:

```bash
vllm serve UchihaUdhaya/Qwen2.5-3B-FP8-W8A8-Dynamic \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

Python offline inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="UchihaUdhaya/Qwen2.5-3B-FP8-W8A8-Dynamic")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

outputs = llm.generate(
    ["Explain FP8 dynamic quantization in two concise sentences."],
    params,
)
print(outputs[0].outputs[0].text)
```

### 2. Inference with Hugging Face `transformers`

```bash
pip install transformers accelerate compressed-tensors
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "UchihaUdhaya/Qwen2.5-3B-FP8-W8A8-Dynamic"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

inputs = tokenizer("Post-training quantization enables", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Hardware Requirements

- **Native FP8 Tensor Cores (Best Performance):** NVIDIA Ada Lovelace (`L4`, `L40`, `RTX 4090`, cc `8.9`), Hopper (`H100`, `H200`, cc `9.0`), or Blackwell (`B200`, cc `10.0`).
- **Older GPUs (Ampere / Turing):** Supported via software dequantization fallback (Marlin / weight-only path), without native FP8 GEMM acceleration.

---

## Quantization Configuration Details

- **Tool:** `llm-compressor==0.15.0`
- **Scheme:** `FP8_DYNAMIC` (`format: "float-quantized"`, recipe included in `recipe.yaml`)
- **Quantized Parameters:** `2,774,532,096` (`torch.float8_e4m3fn`) across `252` Linear layers (`252` per-channel `weight_scale` tensors, `0` static `input_scale` tensors).
- **Preserved Parameters:** `312,438,784` (`torch.bfloat16`) in embeddings, norms, and `lm_head`.

---

## License & Attribution

- Derived from [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B) under the **Apache 2.0** license.
- Quantized using [`vllm-project/llm-compressor`](https://github.com/vllm-project/llm-compressor).

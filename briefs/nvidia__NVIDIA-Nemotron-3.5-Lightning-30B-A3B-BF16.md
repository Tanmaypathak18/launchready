# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16: LaunchReady brief

Projected unless marked measured · generated 2026-11-02 by launchready 0.0.1 · sources at the bottom

## TL;DR

- **What it is:** hybrid Mamba MoE model from nvidia, 32.91B total / ~3.70B active per token (estimated), 262,144 tokens context, license openmdw-1.1.
- **What it needs:** weights 61.31 GiB at BF16, 20.08 GiB at NVFP4 and 23.53 GiB at GGUF Q4_K_M, plus 0.23 GiB per 32K-token sequence (projected).
- **Can you run it today:** supported in released vLLM, SGLang, llama.cpp and Transformers; fits one RTX 5090 at NVFP4 and GGUF Q4_K_M; BF16 for 8 users at 32K needs an A100 80GB.

## Identity

| | |
|---|---|
| Builder | nvidia |
| Hugging Face repo | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) @ `a9904d2` [1] |
| Repo created | 2026-08-01 (Hub creation date; public release can be later) [1] |
| Last modified | 2026-08-24 [1] |
| License | openmdw-1.1 [1] |
| Gated | no [1] |

## Architecture

| | |
|---|---|
| Family | hybrid Mamba MoE (nemotron_h) [2] |
| Layers | 52 (6 attention, 23 Mamba, 23 MLP/MoE-only) [2] |
| Parameters | 32.91B total / ~3.70B active per token (estimated) · 128 experts, 6 per token [3] |
| Attention | GQA, 32 query / 2 KV heads, head_dim 128 [2] |
| Context length | 262,144 tokens [2] |
| Extras | 1 multi-token-prediction (MTP) layer(s) in the checkpoint; drafter models for speculative decoding: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark), [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash) |

## What this means for developers

It stores 32.91B parameters (counted from safetensors headers; the name says 30B) but runs only ~3.70B per token: plan memory for the full 32.91B and expect the speed of a much smaller model.

Only 6 of its 52 layers keep a KV cache that grows with the conversation (6,144 bytes per token); its 23 Mamba layers keep a fixed 0.05 GiB state, so one 32K-token conversation needs just 0.23 GiB.

BF16 weights are 61.31 GiB: the smallest single GPU that fits them is the A100 80GB, not an RTX 5090.

On a single RTX 5090 (32 GB) it fits at NVFP4 (20.08 GiB Hub files, ~28 × 32K sequences) and GGUF Q4_K_M (23.53 GiB Hub files, ~13 × 32K sequences): projected, to be measured in Phase 5.

The architecture is registered in the latest release of vLLM, SGLang, llama.cpp and Transformers.

## Hardware requirements

All projected [4]. Weights use total parameters from safetensors headers [3]; "Hub files" is the real size of the matching repo on the Hub (file size, not measured GPU memory).

| Precision | Bits/weight | Weights GiB (projected) | Hub files GiB | Repo |
|---|---:|---:|---:|---|
| BF16 | 16 | 61.31 | 61.31 (+0%) | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) |
| FP8 | 8 | 30.65 | 30.08 (-2%) | [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8) |
| NVFP4 | 4.5 | 17.24 | 20.08 (+16%) | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) |
| GGUF Q8_0 | 8.5 | 32.57 | 33.31 (+2%) | [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) |
| GGUF Q4_K_M | 4.85 | 18.58 | 23.53 (+27%) | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) |

Memory per sequence (KV cache FP16, Mamba state float32), the same for every weight precision:

| Context | KV cache GiB | Mamba state GiB | Total GiB |
|---:|---:|---:|---:|
| 8,192 | 0.05 | 0.05 | 0.09 |
| 32,768 | 0.19 | 0.05 | 0.23 |
| 131,072 | 0.75 | 0.05 | 0.80 |

Smallest single NVIDIA GPU with native support that fits the weights plus N sequences:

| Precision | 8K × 1 | 8K × 8 | 32K × 1 | 32K × 8 | 128K × 1 | 128K × 8 |
|---|---|---|---|---|---|---|
| BF16 | A100 80GB | A100 80GB | A100 80GB | A100 80GB | A100 80GB | A100 80GB |
| FP8 | L40S | L40S | L40S | L40S | L40S | L40S |
| NVFP4 | RTX 5090 | RTX 5090 | RTX 5090 | RTX 5090 | RTX 5090 | RTX 5090 |
| GGUF Q8_0 | A100 40GB | L40S | A100 40GB | L40S | L40S | L40S |
| GGUF Q4_K_M | RTX 5090 | RTX 5090 | RTX 5090 | RTX 5090 | RTX 5090 | A100 40GB |

Fit model: usable = VRAM × 0.90 − weights − 2 GiB reserve; users = floor(usable ÷ per-sequence memory). Uses Hub file sizes where they exist, else projected weights. GPU capacity is the marketed size read as GiB. "—" = no single GPU in the catalog.
- GGUF rows assume the whole model on the GPU; llama.cpp/Ollama can offload layers to CPU RAM at lower speed

## Engine readiness

Checked against shallow sparse clones of each engine at its latest stable release and its default branch on 2026-11-02.

| Engine | Status | Version | Where | Notes |
|---|---|---|---|---|
| vLLM | release | v0.32.0 | [vllm/model_executor/models/registry.py:180](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/registry.py#L180) | also checked main @ 77d2790 |
| SGLang | release | v0.5.23 | [python/sglang/srt/models/nemotron_h.py:1361](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/nemotron_h.py#L1361) | also checked main @ 63c6de2 |
| llama.cpp | release | v0.8.0 | [conversion/nemotron.py:423](https://github.com/ggml-org/llama.cpp/blob/v0.8.0/conversion/nemotron.py#L423) | → Ollama via GGUF; also checked master @ 37b0f9c |
| Transformers | release | v5.22.1 | [src/transformers/models/auto/auto_mappings.py:482](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/auto/auto_mappings.py#L482) | also checked main @ 865e28f |

`release` = registered in the latest stable release · `main-only` = only on the default branch · `not-found` = registry checked, architecture absent · `unknown` = couldn't check. A registry entry means the engine knows the architecture; it doesn't prove every precision or feature works.

## Available variants

| Format | Repo | Size | Downloads |
|---|---|---|---:|
| NVFP4 | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | 20.08 GiB | 1,010,739 |
| FP8 | [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8) | 30.08 GiB | 969,304 |
| GGUF | [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q8_0 33.31 GiB (+2 more types) | 204,958 |
| GGUF | [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q4_K_M 23.53 GiB · Q8_0 32.60 GiB (+16 more types) | 51,567 |
| GGUF | [bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q4_K_M 23.73 GiB · Q8_0 34.63 GiB (+25 more types) | 39,091 |
| MLX | [TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit](https://huggingface.co/TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit) | 17.25 GiB | 35,056 |
| other | [useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16) | 16.60 GiB | 24,106 |
| GGUF | [AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q4_K_M 23.31 GiB · Q8_0 32.60 GiB (+5 more types) | 20,783 |
| GGUF | [mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-Q4_K_M-GGUF](https://huggingface.co/mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-Q4_K_M-GGUF) | Q4_K_M 22.83 GiB | 4,281 |
| GGUF | [vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | Q8_0 31.28 GiB (+5 more types) | 2,783 |

From the Hub filter `base_model:quantized:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` and a search of the nvidia org, most downloaded first (builder's own repos on top), on 2026-11-02. Sizes are file sizes on the Hub.

## Ecosystem impact

**Who it is for:** developers with one 32 GB-class GPU (RTX 5090) or Mac/local-model users. It stores 32.91B parameters but runs ~3.70B per token (estimated), and at NVFP4 it fits one RTX 5090 with room for ~28 × 32K-token sequences (projected).

**Where the ecosystem is today (2026-11-02):** the architecture is registered in the latest release of vLLM, SGLang, llama.cpp (so Ollama via GGUF) and Transformers. Quantized variants already exist for every major path: NVFP4 from nvidia (~1.0M downloads), FP8 from RedHatAI, GGUF from ggml-org, unsloth and bartowski, MLX from TensorFold. Builders outside NVIDIA shipped these, which suggests the community is treating it as a first-class model.

**What it means for NVIDIA:** at NVFP4 it is projected to fit on a 32 GiB RTX 5090, but it has not been measured on one yet. Drafter models (DSpark, DFlash) for speculative decoding are published alongside. The open license (openmdw-1.1) and no gating remove the usual access friction.

**Not covered here:** how it compares in quality or speed with similar-size open models. That needs measured benchmarks and will be covered in phase 5 of this repo.

## Open questions and risks

- **NVFP4 memory is higher than projected.** The real NVFP4 repo is 20.08 GiB vs 17.24 GiB projected (+16%), so some tensors likely stay in higher precision. The ~28-user figure for an RTX 5090 uses the real size, but it is still projected, not measured.
- **Mamba state is unmeasured.** The config says float32; vLLM may store or pad it differently. Phase 3 measures it.
- **Variant tags on the Hub can't be trusted.** Two repos tagged as NVFP4 variants (DSpark, DFlash) are speculative-decoding drafters, not quantized copies of this model. Check what a repo contains before relying on its tag.
- **Engine support means "registered", not "fully working".** A registry entry doesn't prove every precision, the MTP layer or drafter decoding works in each engine. Phase 5 tests this on vLLM and Ollama.
- **Community variants vary in quality.** Several GGUF and "uncensored" repos are from unknown builders; the brief lists sizes and downloads, not quality.
- **Hub dates are not release dates.** The repo was created 2026-08-01; public release may be later.

## Sources

1. Hugging Face repo and model info at this revision: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9
2. config.json: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/config.json
3. safetensors headers (parameter counts by dtype, via huggingface_hub.get_safetensors_metadata): https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9
4. launchready memory formulas (docs/roadmap.md sections 6.1-6.3): https://github.com/Tanmaypathak18/launchready/blob/main/docs/roadmap.md
5. vLLM registry at v0.32.0: https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/registry.py#L180
6. SGLang registry at v0.5.23: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/nemotron_h.py#L1361
7. llama.cpp registry at v0.8.0: https://github.com/ggml-org/llama.cpp/blob/v0.8.0/conversion/nemotron.py#L423
8. Transformers registry at v5.22.1: https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/auto/auto_mappings.py#L482
9. NVFP4 variant: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
10. drafter variant: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
11. drafter variant: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash
12. FP8 variant: https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8
13. GGUF variant: https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
14. GGUF variant: https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
15. GGUF variant: https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
16. MLX variant: https://huggingface.co/TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
17. other variant: https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16
18. GGUF variant: https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
19. GGUF variant: https://huggingface.co/mlasli/Nemotron-3.5-Lightning-30B-A3B-Heretic-Uncensored-Q4_K_M-GGUF
20. GGUF variant: https://huggingface.co/vcruz305/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF

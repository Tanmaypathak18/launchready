# Friction log

Write an entry here *before* fixing anything that breaks. Newest entries first.

## Template

```text
### <short title>
- When / where: date · GPU · driver · CUDA · engine + version · model + revision
- Exact command:
- What happened (error excerpt):
- Root cause:
- Fix or workaround:
- Severity: blocks / slows / cosmetic
- Filed: <issue or PR link>, or "not filed" + why
- Requirement this implies:
```

## Entries

### Lightning NVFP4 projection is 16% below the real NVFP4 repo
- When / where: 2026-11-02 · no GPU (projection on MacBook Air) · driver n/a · CUDA n/a · launchready 0.0.1 `fit` · nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ a9904d2 vs nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @ bee7596
- Exact command: `uv run launchready fit nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --context 32768 --users 8`
- What happened (error excerpt): projected NVFP4 weights 17.24 GiB; the NVFP4 repo's safetensors files total 21,561,882,284 bytes = 20.08 GiB (+16.5%). On an RTX 5090 at 32K that is ~29 users, not the projected 40.
- Root cause: weights are projected as total params × 4.5 bits, assuming every tensor is quantized. Real NVFP4 checkpoints keep some tensors (embeddings, lm_head, norms, router, likely others) in BF16.
- Fix or workaround: Phase 2 variants lookup: when a quantized variant exists, report its real file size next to the projection. Until then the output notes that real files will be larger.
- Severity: slows (overstates capacity on tight GPUs like the 5090)
- Filed: not filed: a gap in launchready's own projection, not an upstream bug
- Requirement this implies: fit must show real quantized-variant sizes when they exist and label pure projections as lower bounds.

### Lightning has 32.9B parameters, not the nominal 30B
- When / where: 2026-11-02 · no GPU · driver n/a · CUDA n/a · huggingface_hub `get_safetensors_metadata` · nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ a9904d2
- Exact command: `uv run launchready fit nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --json`
- What happened (error excerpt): safetensors headers count 32,913,266,240 parameters (BF16 32,913,263,168 + F32 3,072). BUILD_PLAN 6.5 used the nominal ~30B (BF16 ~60 GB, NVFP4 ~17 GB); real figures are BF16 ~65.8 GB and NVFP4 ~18.5 GB before the BF16 share.
- Root cause: the "30B" in the model name is a rounded label; the count includes untied embeddings and lm_head (~0.7B) and the multi-token-prediction layer.
- Fix or workaround: always size from safetensors headers, never the model name. Updated 6.5 and its test to 32.9B.
- Severity: cosmetic (BF16 still doesn't fit a 5090), but it moves every weight number by ~10%
- Filed: not filed: naming convention, not a bug
- Requirement this implies: briefs and recipes quote the header-derived parameter count, with the nominal name alongside.

### Mamba state dtype for Lightning is an assumption until measured
- When / where: 2026-11-02 · no GPU · driver n/a · CUDA n/a · launchready 0.0.1 `fit` · nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ a9904d2
- Exact command: `uv run launchready fit nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --context 32768 --users 8`
- What happened (error excerpt): config sets `mamba_ssm_cache_dtype: float32`, so fit sizes the whole Mamba state (conv + SSM) at FP32: 49,930,240 bytes (~0.05 GiB) per sequence across 23 Mamba layers, unlabeled as assumed. vLLM may keep the conv state at the model dtype and pad the state to its block size.
- Root cause: the config names one cache dtype; engines can store conv and SSM state differently and allocate per block, which the 6.2 formula doesn't model.
- Fix or workaround: measure in Phase 3 (vLLM KV-capacity and max-concurrency startup lines) and adjust the state formula; until then treat the state number as approximate.
- Severity: cosmetic at 32K (state is ~20% of per-sequence memory); matters more at short contexts where state dominates
- Filed: not filed: needs a measurement first
- Requirement this implies: fit separates conv and SSM state dtypes once engine behavior is measured.

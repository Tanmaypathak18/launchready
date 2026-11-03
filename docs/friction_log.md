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

### Lightning's drafter models are tagged as quantized variants of the base
- When / where: 2026-11-02 · no GPU · huggingface_hub 2.4.0 · nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash and -NVFP4-DSpark
- Exact command: `HfApi().list_models(filter="base_model:quantized:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")`
- What happened (error excerpt): both repos carry `base_model:quantized:<Lightning BF16>` and "NVFP4" in the name, but they're 1.2–1.3 GB, 6-layer models (`DFlashDraftModel`, `Qwen3DSparkModel`): speculative-decoding drafters, not the 32.9B weights. The variants table listed them as NVFP4 builds of Lightning.
- Root cause: the Hub has no "draft model of" relation, so drafters reuse `base_model:quantized`.
- Fix or workaround: classify repos named DFlash/DSpark/EAGLE/Medusa/draft as `drafter`; the brief also flags any variant far smaller than the projected size for its format.
- Severity: slows (wrong sizes in a brief; a developer could download a drafter thinking it's the model)
- Filed: not filed yet: candidate model-card suggestion to nvidia (say "drafter for …" in the card's first line)
- Requirement this implies: variants lookup must sanity-check size against the projection before calling something a variant.

### llama.cpp's stable release is buried under per-build prereleases
- When / where: 2026-11-02 · no GPU · GitHub releases API · ggml-org/llama.cpp · n/a
- Exact command: `gh api "repos/ggml-org/llama.cpp/releases?per_page=30"`
- What happened (error excerpt): all 30 entries are `b120xx pre=true`; launchready found no stable tag and checked `master` only. `releases/latest` returns `v0.8.0`.
- Root cause: llama.cpp publishes every CI build as a prerelease; semver releases (v0.8.0) are rare.
- Fix or workaround: ask `releases/latest` first (GitHub excludes prereleases there); only if that tag looks like an rc (vLLM's case) scan the last 100 releases for a stable one.
- Severity: slows (llama.cpp support shows as main-only when it's released)
- Filed: not filed: upstream release practice
- Requirement this implies: "released in llama.cpp" should name the release a developer actually installs; Ollama bundles its own llama.cpp, so Phase 5 checks the Ollama version separately.

### llama.cpp's default branch is `master`, not `main`
- When / where: 2026-11-02 · no GPU · git 2.50.1 · ggml-org/llama.cpp · n/a
- Exact command: `git clone --depth 1 --filter=blob:none --sparse --branch main https://github.com/ggml-org/llama.cpp.git`
- What happened (error excerpt): `fatal: Remote branch main not found in upstream origin`; llama.cpp showed as `unknown` in the readiness table.
- Root cause: BUILD_PLAN says "check main", but llama.cpp's development branch is `master`.
- Fix or workaround: clone the remote's default branch (no `--branch`) and label it with its real name, e.g. `master @ abc1234`.
- Severity: blocks (no llama.cpp/Ollama readiness)
- Filed: not filed: our assumption, not an upstream bug
- Requirement this implies: never hard-code branch names; read the default branch from the remote.

### Two engine registry paths in BUILD_PLAN 6.6 have moved
- When / where: 2026-11-02 · no GPU · driver n/a · CUDA n/a · llama.cpp main and v0.8.0; Transformers main and v5.22.1 · n/a
- Exact command: `gh api repos/ggml-org/llama.cpp/contents/convert_hf_to_gguf.py` and `gh api repos/huggingface/transformers/contents/src/transformers/models/auto/configuration_auto.py`
- What happened (error excerpt): `convert_hf_to_gguf.py` is now a 14 KB entry point that does `from conversion import ModelBase, ...`; the `@ModelBase.register("...")` decorators live in `conversion/*.py` (one file per family). `configuration_auto.py` now does `from .auto_mappings import CONFIG_MAPPING_NAMES`; the mapping is in the auto-generated `models/auto/auto_mappings.py`. A checker that greps only the 6.6 paths would report every architecture as not supported.
- Root cause: upstream refactors (llama.cpp split the converter into a package; Transformers generates the auto mappings from config classes).
- Fix or workaround: engines.py sparse-checks out the whole directory around each registry, looks for the registry's anchor in the expected file first, and if it isn't there, searches the tree and reports `moved: old → new`. Updated the 6.6 table.
- Severity: blocks (false "not-found" for every model)
- Filed: not filed: intentional upstream refactors
- Requirement this implies: engine checks must locate registries by symbol, not by a fixed path, and say when a path moved.

### vLLM's GitHub "latest release" is a release candidate
- When / where: 2026-11-02 · no GPU · vllm-project/vllm releases API · n/a
- Exact command: `gh api repos/vllm-project/vllm/releases/latest`
- What happened (error excerpt): returns `v0.33.0rc3` with `prerelease: false`; the latest stable is v0.32.0.
- Root cause: the rc isn't flagged as a prerelease on GitHub.
- Fix or workaround: engines.py skips tags matching rc/dev/alpha/beta as well as flagged prereleases.
- Severity: slows (would report rc-only support as "release")
- Filed: not filed: minor and possibly intentional
- Requirement this implies: "release" support means a stable tag a developer would `pip install`.

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

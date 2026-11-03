# Progress

One "Done so far" entry per phase, newest first: what shipped, decisions made, what's next.

## Phase 4: Developer feedback → requirements (2026-11-03)

- **Shipped:** `docs/requirements.md`: 36 GitHub issues and Hugging Face discussions (Aug 1 – Nov 3, 2026; vLLM, llama.cpp, Ollama, NVIDIA's Lightning repos, ggml-org and Unsloth GGUF repos), read by hand and grouped into 5 requirements with acceptance criteria and owners: GGUF contents vs labels (8), NVFP4 crashes on Blackwell/Jetson (5), hybrid Mamba speed features (9), wrong-output kernels (4), install/docs gaps (6). All 38 links checked against the GitHub and Hugging Face APIs on 2026-11-03 (dates and states match).
- **Decisions:** ranked by severity first (wrong output > blocks > slows), then thread count; counts are threads, not distinct bugs; sibling Nemotron models included only when the root cause is the shared `nemotron_h` / `nemotron_h_moe` architecture; chat template / tool calling (2 threads) left below the cut; no Lightning-specific OOM reports found.
- **Next:** Phase 5 launch kit, carrying in the five plan changes from requirements.md: real GGUF tensor types for Lightning (experts fall back to Q5_0), try the card's 5090 vLLM command then without `--kv-cache-dtype fp8`, Ollama at concurrency 1 plus `llama-server -np 8`, prefix caching off and vLLM pinned for headline numbers, and record `tokenizer.ggml.pre` for the benchmarked GGUF.

## Phase 3: Validate on free GPUs + first-user audits (2026-11-03)

- **Shipped:** `notebooks/01_memory_validation.ipynb` ran on Kaggle 2× T4 (driver 580.178.04, CUDA 13.0, vLLM 0.32.0); raw results in `results/` (CSV, environment, logs; chart rebuilt from the CSV with the fixed plot cell). Weights (vLLM "Model loading") within 1% for all three models; KV budget and concurrency within 4–6% for Qwen3-4B and OLMoE; granite-4.0-h-tiny concurrency +133% (budget ×1.72, per-sequence ×1.36). `launchready fit` now warns about the hybrid gap; 80 tests. Quickstart audits (`results/quickstart_audit/`, `notebooks/03_quickstart_audit.ipynb`): Lightning's llama.cpp recipe fails on a nonexistent `:Q4_K_M` tag (Mac); Qwen3-4B's Transformers quickstart needs undocumented `accelerate` and then offloads to disk on a 16 GB Mac, but worked as written on Kaggle 2× T4 in 115 s; Qwen3-4B's `vllm serve` line fails on `--enable-reasoning` in vLLM 0.32.0 despite the card's `vllm>=0.8.5`.
- **Decisions:** vLLM's "Model loading" line is the weights measurement (recorded in BUILD_PLAN); the Transformers stage crashed on our own bug (friction log), was fixed but not rerun on a GPU, and is optional and off by default. No retune from one hybrid model. Hybrid KV-token counts marked not comparable. vLLM timeout 30 min (granite took 951 s, mostly Mamba2 kernel warmup). Snippets run verbatim; workarounds noted but not run; nothing filed upstream (owner's call). Optional extras skipped: Qwen3-4B vLLM rerun without `--enable-reasoning`, Transformers GPU cross-check, Ollama audit 3a. Granite's +133% concurrency miss accepted as explained (2026-11-03): the KV-budget half (×1.72) is confirmed fro- **Next:** Phase 4, developer feedback → requirements (by hand).
- **Next:** Phase 4, developer feedback → requirements (by hand).
- Later: file an issue on the NVFP4 card about :Q4_K_M after re-checking it is still broken.

## Phase 2: Engine readiness + Nemotron brief (2026-11-02)

- **Shipped:** engines.py (shallow sparse clones of vLLM, SGLang, llama.cpp, Transformers at latest stable tag + default branch; release / main-only / not-found / unknown with file:line), variants.py (quantized variants with real Hub file sizes), `launchready brief` (Jinja2 template 7.1) with the Lightning brief and its hand-written sections, `launchready radar --since 30d` with the 2026-11-02 digest; 78 offline tests.
- **Decisions:** registries located by symbol, so moved paths are reported (llama.cpp → `conversion/*.py`, Transformers → `auto_mappings.py`); rc and per-build prerelease tags skipped; drafters (DFlash, DSpark) split from variants; brief fits use real Hub sizes when they exist (NVFP4 20.08 GiB vs 17.24 projected); radar keeps same-org fine-tunes, skips other orgs' fine-tunes, and filters on repo-creation dates (caveat in the digest footer).
- **Next:** Phase 3, validate projections on Kaggle T4 x2 (weights, KV capacity, Mamba state) and run 2–3 quickstart audits. Open gaps: active-parameter estimate runs high (~3.70B vs "A3B"); "registered" isn't "works" until Phase 5.

## Phase 1: Memory calculator (2026-11-02)

- **Shipped:** `launchready fit` (weights per precision, KV + Mamba state per sequence, max users per GPU) from config.json and safetensors headers only; hub.py with a JSON cache, arch.py for dense GQA, MoE, MLA, sliding-window and hybrid Mamba configs, memory.py, gpus.yaml; 34 offline tests incl. exact 6.5 KV values.
- **Decisions:** GPU capacity read as GiB; Mamba state FP32 unless the config names a dtype, labeled "assumed"; quantized weights = total params × bits (no BF16 share yet); MoE active params estimated as total minus idle routed experts, labeled "~".
- **Next:** Phase 2. Known gaps: projected NVFP4 Lightning 17.24 GiB vs 20.08 GiB real NVFP4 files (BF16 share); active params run high (Lightning ~3.7B, granite-h-tiny ~1.5B, DeepSeek-V3 ~40B); no tensor parallel, so DeepSeek-V3 fits no single GPU; calibrate reserve and GiB reading in Phase 3.

## Phase 0: Setup (2026-11-02)

- **Shipped:** uv-managed Python 3.11 scaffold with a Typer CLI (`fit`, `brief`, `radar` stubs that print "not implemented"), docstring-only modules, ruff + pytest smoke tests, GitHub Actions CI, Apache-2.0 LICENSE, README Status checklist, friction log template.
- **Decisions:** `typer` is the only runtime dependency; huggingface_hub, Jinja2 and PyYAML get added in the phase that needs them. Stub commands exit with code 1 so nobody mistakes them for working.
- **Next:** Phase 1 (memory calculator: `launchready fit`). Repo is public at Tanmaypathak18/launchready with CI green.

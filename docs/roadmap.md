# LaunchReady: Roadmap

**A small, open-source launch-readiness kit for open models on NVIDIA GPUs.** It sizes a new model's hardware needs, runs it on NVIDIA GPUs with the community's tools, measures it honestly, and explains the results to developers.

**Compute budget:** $0 for everything except one rented RTX 5090 session (~$5–10, hard cap $20).

| Phase | Status |
|---|---|
| 0. Setup | Done |
| 1. Memory calculator | Done |
| 2. Engine readiness + Nemotron brief | Done |
| 3. Validate on free GPUs + first-user audits | Done |
| 4. Developer feedback → requirements | Done |
| 5. Nemotron 3.5 Lightning launch kit | kit complete and validated by dry run on 2x T4 with Qwen3-4B; RTX 5090 Lightning run pending |

Details of what shipped in each phase are in [progress.md](progress.md).

---

## 1. What the project delivers

One repo with five deliverables:

1. **`launchready fit`**: a calculator that reads any model from Hugging Face (no weight download) and works out memory at BF16, FP8, NVFP4 and GGUF, plus which NVIDIA GPUs fit it.
2. **A model brief for NVIDIA Nemotron 3.5 Lightning** (30B total / 3B active hybrid Mamba-Transformer MoE, released Aug 11, 2026): capabilities, hardware needs, engine readiness, ecosystem impact.
3. **Proof the calculator is right**: predicted vs measured memory on free Kaggle GPUs.
4. **A launch kit**: verified vLLM and Ollama recipes for Lightning, benchmarked on a rented RTX 5090, with a quality check.
5. **A friction log and a requirements table** built from real developer issues.

**Customer:** developers and researchers: Hugging Face users, people running models locally, ML engineers serving models. Every output is written for them.

**Cut on purpose:** scheduled automation, an automated feedback pipeline, H100 benchmarks, TensorRT-LLM, fine-tuning, demo videos and slides.

---

## 2. Goals and deliverables

| Project goal | Deliverable |
|---|---|
| Enablement and launch readiness with model builders | Readiness checklist + recipes; issues or PRs filed upstream |
| Track releases; summarize capabilities, hardware needs, ecosystem impact | Nemotron brief + a one-off radar of recent releases |
| Requirements, launch plans, checklists | Readiness checklist; requirements table |
| Inference, evaluation, deployment | vLLM/Ollama recipes, benchmarks, quality check, pinned containers |
| Feedback → recurring issues → requirements | Hand-reviewed issues → requirements table |
| Test developer workflows, notebooks, Python | Quickstart audits, friction log, notebooks anyone can run |
| Docs | Brief, recipe README |

---

## 3. Ground rules

- **Labeled numbers.** Every number is *measured* (GPU, driver, CUDA, engine version, model revision, date) or *projected* (with the formula). This rule is not optional.
- **Reproducible.** Lockfile; pinned engine versions and container tags; model revision hashes; one command per result.
- **Raw data committed.** CSV/JSON next to every chart.
- **Runs for new users.** Notebooks pass "Run all" from a fresh session.
- **No weight downloads in the calculator.** config.json and safetensors headers only.
- **No secrets in git.** Public data only.

---

## 4. Compute

| Where | Use it for | Cost |
|---|---|---|
| Development laptop (Mac) | All code; Ollama hands-on; writing | $0 |
| Kaggle "GPU T4 x2" (2 × 16 GB) | Memory validation, quickstart audits, dry runs of the benchmark script | $0 |
| Rented RTX 5090, 32 GB, Blackwell (~6–8 h) | Lightning benchmarks: NVFP4 in vLLM, GGUF in Ollama/llama.cpp | ~$5–10 |

**Renting checklist:** Runpod or Vast.ai (per-minute billing) · on-demand, not spot · ~100 GB disk · CUDA 12.8+ image · set the spend cap first · download weights with `hf download` before starting the clock on benchmarks · copy results off before stopping. Rent only after the benchmark script has run end to end on Kaggle.

Mac numbers never go in published benchmarks.

---

## 5. Phases

### Phase 0: Setup ($0)

- [x] Public repo `launchready` (Apache-2.0), Python 3.11+ with uv, ruff, pytest.
- [x] Tokens: Hugging Face (accept Llama/Gemma licenses), Kaggle (GPU + internet enabled), GitHub.
- [x] Mac: install Ollama, run `ollama run llama3.1:8b --verbose`, note the eval rate.
- [x] Add the roadmap and `CLAUDE.md`; create `docs/friction_log.md` from template 7.3.

**Done when:** CI is green on a smoke test and Ollama answers on a local Mac.

### Phase 1: Memory calculator ($0)

**Goal:** `launchready fit <repo_id>` shows memory at each precision and which NVIDIA GPUs fit, without downloading weights.

- [x] `hub.py`: config.json (handle nested `text_config`), model info (date, license, gated), safetensors parameter counts (`huggingface_hub.get_safetensors_metadata`), file sizes. Cache on disk.
- [x] `arch.py`: normalize into an `Architecture` dataclass: layers and layer types (full attention, sliding window, Mamba), heads, KV heads, head_dim, context, MoE fields (total vs active), MLA fields, Mamba fields. Unknown fields → warning naming the field.
- [x] `memory.py`: weights per precision (6.1), KV/state per sequence (6.2), fit model (6.3), `--kv-dtype fp16|fp8`.
- [x] `data/gpus.yaml` from 6.4.
- [x] CLI: `launchready fit <id> [--context 32768] [--users 8] [--gpu rtx-5090] [--json]`, every output labeled "projected".
- [x] Tests first, using 6.5 and saved config fixtures.

**Done when:** reference tests pass and `launchready fit` gives sensible tables for dense, MoE, hybrid Mamba and MLA models.

### Phase 2: Engine readiness + the Nemotron brief ($0)

**Goal:** `launchready brief <repo_id>` writes a one-page brief a leader or developer can act on.

- [x] `engines.py`: shallow sparse clones of only the paths in 6.6, at `main` and the latest release tag, cached in `~/.cache/launchready`. Status per engine: `release`, `main-only`, `not-found`, `unknown`, with the matching file and line. Flag `auto_map` → "needs trust_remote_code".
- [x] Variants: FP8, NVFP4 and GGUF derivatives (Hub filter `base_model:quantized:<repo_id>` plus the builder's org), with sizes.
- [x] `brief.py`: render template 7.1 to `briefs/<org>__<model>.md`.
- [x] Write the **Ecosystem impact** and **Open questions** sections by hand.
- [x] `launchready radar --since 30d`: one-off list of recent root-model releases from the watchlist in 6.7, run by hand. No scheduling.

**Done when:** the Lightning brief exists with hand-written sections, every claim is sourced, and one radar digest exists.

### Phase 3: Validate on free GPUs + first-user audit ($0)

**Goal:** prove the calculator on real NVIDIA GPUs and act as the first user of official quickstarts.

- [x] `notebooks/01_memory_validation.ipynb` on Kaggle T4 x2 with 3 models: `Qwen/Qwen3-4B` (dense), `allenai/OLMoE-1B-7B-0924-Instruct` (MoE), `ibm-granite/granite-4.0-h-tiny` (hybrid Mamba). Over ~13 GB in FP16 → run across both GPUs with `--tensor-parallel-size 2`.
- [x] Per model: projected (`launchready fit`) vs measured — Transformers + `torch.cuda.max_memory_allocated()`, and vLLM's KV-capacity and max-concurrency startup lines (6.8).
  - Changed 2026-11-03: measured weights come from vLLM's "Model loading took X GiB" line (summed over TP ranks), not Transformers `max_memory_allocated()`; the Transformers stage is an optional cross-check, off by default (see docs/friction_log.md).
- [x] If something fails on T4, log it and move on.
- [x] Run 2–3 official quickstarts exactly as written (model-card snippets for vLLM or Transformers). Log every break; file an issue or PR where it genuinely helps.

**Done when:** predicted vs measured is in a CSV and chart with misses explained (aim: weights within ~5%, KV capacity within ~10–15%), and at least 2 audits are logged.

### Phase 4: Developer feedback → requirements ($0, by hand)

- [x] Read ~30 recent GitHub issues and Hugging Face discussions about Nemotron 3.5 Lightning (or its architecture) in vllm-project/vllm, ggml-org/llama.cpp, ollama/ollama and the model's Hugging Face repo.
- [x] Group them into themes (install/dependency, driver/CUDA, out of memory, chat template/tool calling, quantization, performance, correctness, docs gap).
- [x] Write `docs/requirements.md` from template 7.4: top 5 themes → requirements with acceptance criteria.

**Done when:** the table has 5 requirements, each with linked evidence.

This phase is done by hand on purpose: reading the threads is the work.

### Phase 5: Nemotron 3.5 Lightning launch kit (~$5–10)

1. [x] **Readiness checklist** (template 7.2) for Lightning: what must be true before it can be called "ready on NVIDIA GPUs."
2. [x] **Gap check:** read NVIDIA's model card and recipes. The kit adds independent measurements on a consumer GPU, first-user friction, cost per 1M tokens and plain-English guidance. Put that list at the top of the recipe README.
3. [x] **Recipes** (`recipes/nemotron-3.5-lightning/`): `vllm.sh` (NVFP4) and `ollama.md` (GGUF via `ollama run hf.co/...`), each stating GPU, engine version and model revision. Start from the vLLM version NVIDIA's card uses.
4. [x] **Benchmark + quality scripts** (`bench/`), per 6.9.
   - Changed 2026-11-03: GGUF is bartowski Q4_K_M (ggml-org has no Q4_K_M); its header says 6.19 bits/weight, so GGUF memory comes from the header, not 6.1. The card has no GSM8K/IFEval numbers, so quality compares NVFP4 (vLLM) with GGUF (llama.cpp) directly; thinking off for both.
5. [x] **Dry run** everything on Kaggle against `Qwen/Qwen3-4B`. All servers and runs finished; raw output in `results/dryrun_kaggle/` (not headline).
   - Quality fixed (2026-11-03): vLLM's 0.0 in run 2 was a harness problem (`deepseek_r1` left `content` null), not a score. Run 3 at `8b4ca32` used `qwen3` and gave real scores with 0 empty answers. See results/dryrun_kaggle/README.md.
6. [ ] **Rented RTX 5090 session:** NVFP4 in vLLM, GGUF in Ollama/llama.cpp, concurrency 1/8/32, one 32K-context point, drafter on/off at concurrency 1, quality subset, recipes run as written (`vllm.sh` and `ollama.md`, startup check). Not tested in this session: chat template, tool calling, soak, tokenizer comparison. NVIDIA's card points local devices to the GGUF, so if NVFP4 doesn't run on the 5090, that's a finding. Log it.
7. [ ] **Results notebook** (`notebooks/02_launch_kit_nemotron.ipynb`) that rebuilds every chart from committed results.

**Done when:** both recipes are verified on the 5090, every number is labeled measured, and the checklist is green or each exception is explained.

---

## 6. Technical reference

### 6.1 Bits per weight (effective)

| Format | Bits/weight | Notes |
|---|---|---|
| BF16 / FP16 | 16 | T4 has FP16 but no BF16, so use `--dtype half` |
| FP8 (E4M3) | ~8 | Native on Ada, Hopper, Blackwell |
| NVFP4 | 4.5 | 4-bit values + one FP8 scale per 16 values; native on Blackwell |
| GGUF Q8_0 | 8.5 | llama.cpp / Ollama |
| GGUF Q4_K_M | ~4.8–4.9 | Common 4-bit default for Ollama |

- Some tensors stay BF16 (embeddings, lm_head, norms, router), so check predictions against real quantized file sizes.
- Packed 4-bit weights are stored as uint8 (two per byte); unpack before reading dtype counts.
- MoE: memory scales with **total** parameters; compute per token with **active** parameters.

### 6.2 KV cache and state

- **Full attention**, per token per layer: `2 × kv_heads × head_dim × bytes`.
- **Sliding window:** same, capped at the window length.
- **MLA (DeepSeek-style):** `(kv_lora_rank + qk_rope_head_dim) × bytes` per token per layer.
- **Mamba-2:** no per-token cache; fixed state per sequence ≈ `(conv_kernel − 1) × conv_dim + mamba_heads × mamba_head_dim × state_size` elements, where `conv_dim = mamba_heads × mamba_head_dim + 2 × n_groups × state_size`. Engines may keep it in FP32.
- Total = per-sequence memory × users. An FP8 KV cache halves the attention part.

### 6.3 Fit model (vLLM-style)

- `KV budget = VRAM × 0.90 (vLLM default gpu_memory_utilization) − weights − runtime reserve` (start at 2 GiB; calibrate in Phase 3).
- `Max users at context C = floor(KV budget ÷ per-sequence memory at C)`.
- Compute in bytes, display GiB, and take VRAM from `nvidia-smi` for measured GPUs.

### 6.4 GPU catalog (starter `gpus.yaml`; verify against NVIDIA spec pages)

| GPU | Memory | Architecture | Native low precision |
|---|---|---|---|
| Tesla T4 (Kaggle) | 16 GB | Turing | FP16 (no BF16, FP8 or FP4) |
| RTX 4090 | 24 GB | Ada | FP8 |
| RTX 5090 | 32 GB | Blackwell | FP8, FP4 |
| RTX PRO 6000 Blackwell | 96 GB | Blackwell | FP8, FP4 |
| L40S | 48 GB | Ada | FP8 |
| A100 | 40 / 80 GB | Ampere | BF16 |
| H100 | 80 GB | Hopper | FP8 |
| H200 | 141 GB | Hopper | FP8 |
| B200 | 180 GB | Blackwell | FP8, FP4 |
| DGX Spark | 128 GB unified | Grace Blackwell | FP8, FP4 |

### 6.5 Reference values (tests must match)

| Case | Check | Expected |
|---|---|---|
| Llama 3.1 8B (32 layers, 8 KV heads, head_dim 128) | KV per token, FP16 | 131,072 bytes → 1 GiB per 8K-token sequence |
| Llama 3.1 70B (80 layers, 8 KV heads, 128) | KV per token, FP16 | 327,680 bytes → 40 GiB per 128K-token sequence |
| Llama 2 7B (32 KV heads, no GQA) | KV per token, FP16 | 524,288 bytes (4× Llama 3.1 8B) |
| DeepSeek-V3 (61 layers, MLA 512 + 64) | KV per token, BF16 | 70,272 bytes |
| Any 70B dense | Weights | BF16 140 GB · FP8 70 GB · NVFP4 ~39.4 GB |
| Nemotron 3.5 Lightning (32.9B per safetensors headers; "30B" is nominal) | Weights | BF16 ~65.8 GB (not on a 5090); NVFP4 ~18.5 GB + BF16 share (real NVFP4 repo: 21.6 GB) |

### 6.6 Where engines register architectures (confirm at the release tag)

| Engine | Repo | Look for |
|---|---|---|
| vLLM | vllm-project/vllm | `_*_MODELS` dicts in `registry.py` under `vllm/model_executor/models/` (not `_PREVIOUSLY_SUPPORTED_MODELS`) |
| SGLang | sgl-project/sglang | `EntryClass` under `python/sglang/srt/models/` |
| llama.cpp (→ Ollama via GGUF) | ggml-org/llama.cpp (default branch `master`) | `@ModelBase.register(...)` decorators in `conversion/*.py` (moved from `convert_hf_to_gguf.py`; verified at v0.8.0, 2026-11-02) |
| Transformers | huggingface/transformers | `CONFIG_MAPPING_NAMES` in `models/auto/auto_mappings.py` (moved from `configuration_auto.py`, which now imports it; verified at v5.22.1, 2026-11-02); `auto_map` = custom code |

"Latest release" = newest stable tag: skip rc/dev/alpha/beta tags even when GitHub doesn't flag them (vLLM), and skip per-build prereleases (llama.cpp's `bNNNN`).

### 6.7 Watchlist (verify handles)

nvidia · meta-llama · google · Qwen · deepseek-ai · mistralai · microsoft · openai · moonshotai · zai-org · MiniMaxAI · ibm-granite · allenai

### 6.8 Measured values to capture

- **vLLM:** startup lines for KV-cache capacity (tokens) and maximum concurrency.
- **llama.cpp:** KV buffer size at load. **Ollama:** `ollama ps`; `ollama run <model> --verbose`.
- **PyTorch:** `torch.cuda.max_memory_allocated()`.
- **GPU:** `nvidia-smi --query-gpu=memory.used,power.draw,utilization.gpu --format=csv -lms 500`.

### 6.9 Benchmark plan (RTX 5090)

| Dimension | Values |
|---|---|
| Engine / precision | vLLM + NVFP4; Ollama/llama.cpp + GGUF Q4_K_M |
| Concurrency | 1, 8, 32 |
| Workload | Chat (~512 in / 256 out); one long-context point (~32K in / 512 out) |
| Speculative decoding | NVIDIA's drafter off / on at concurrency 1 (vLLM) |
| Quality | GSM8K (250) + IFEval (200) per precision; compare to NVIDIA's published BF16-vs-NVFP4 gap |

- **Metrics:** TTFT and ITL (p50/p99), output tokens/s per user and total, GPU memory, and **$ per 1M output tokens = hourly price ÷ (tokens/s × 3,600) × 1,000,000**.
- **Rules:** warm up first; 3 repetitions for headline numbers; same max model length across engines; record versions.

```bash
aiperf profile \
  --model nemotron-lightning --url http://localhost:8000 \
  --endpoint-type chat --streaming \
  --tokenizer nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --concurrency 8 --request-count 200 --warmup-requests 10 \
  --isl 512 --osl 256
```

(Confirm flags for the pinned version with `aiperf profile --help`.)
Changed 2026-11-03: AIPerf 0.13.0 spells it `--warmup-request-count`; `bench/run_matrix.py` builds the command from flags checked against 0.13.0.

---

## 7. Templates

### 7.1 Model brief (`templates/brief.md.j2`)

```text
# <Model>: LaunchReady brief
Projected unless marked measured · generated <date> · sources at the bottom

## TL;DR
What it is · what it needs · can you run it today, and where

## Identity
Builder · release date · license · links

## Architecture
Dense / MoE / hybrid · total vs active parameters · attention type · context length · extras (MTP, drafter)

## What this means for developers
3–5 plain sentences generated from the numbers

## Hardware requirements
Precision × weights × KV per sequence at 8K / 32K / 128K × smallest NVIDIA GPU for 1 and 8 users

## Engine readiness
Engine × status × version × notes

## Available variants
Format × repo × size

## Ecosystem impact  (hand-written)
Who it competes with · which developer segment · what it means for NVIDIA

## Open questions and risks  (hand-written)

## Sources
```

### 7.2 Readiness checklist

- [ ] Loads in vLLM and Ollama/llama.cpp releases (or the exception is documented)
- [ ] Chat template correct (system prompt, multi-turn); tool calling works if supported
- [ ] Each precision verified on its target GPU; memory within tolerance of projection
- [ ] Quantized quality within an agreed gap to BF16
- [ ] Benchmarks reproducible from committed scripts with environment recorded
- [ ] Recipes copy-paste runnable on a fresh machine
- [ ] Recipe README includes top-5 troubleshooting

### 7.3 Friction-log entry (`docs/friction_log.md`)

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

### 7.4 Feedback → requirements (`docs/requirements.md`)

| Theme | # issues | Severity | Example links | Requirement (user story) | Acceptance criteria | Owner (engine / model builder / NVIDIA docs) |
|---|---|---|---|---|---|---|

---

## 8. Gotchas

- **T4:** no BF16 (`--dtype half`); some FP8, FP4, Mamba and MoE kernels need newer GPUs. Log and move on.
- **Version pins:** some vLLM releases have broken on T4; record the version that works.
- **Kaggle:** ~30 GPU-hours/week; check disk with `df -h`; keep tokens in Kaggle Secrets.
- **Gated repos** (Llama, Gemma): need token + license acceptance; report "gated" instead of crashing.
- **RTX 5090:** needs CUDA 12.8+ and recent engine builds; expect NVFP4 friction; log it.
- **Units:** don't mix GB and GiB in one table.
- **Licenses:** cite each model's license (NVIDIA OpenMDW for Lightning).

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| KV cache | Memory holding the conversation so far, per user; grows with context |
| GQA / MLA | Ways to shrink the KV cache: share K/V across heads, or compress them |
| MoE (total vs active) | Many experts stored, few run per token: memory for total, speed of active |
| Hybrid Mamba | Most layers keep a fixed-size state instead of a growing KV cache |
| NVFP4 | NVIDIA's 4-bit format with fine-grained scales; native on Blackwell |
| GGUF | llama.cpp/Ollama file format with built-in quantization |
| TTFT / ITL | Time to first token; time between later tokens |
| Tensor parallel | Splitting each layer across GPUs so a model fits |
| Speculative decoding | A small drafter guesses tokens; the big model verifies several at once |

---

## 10. Sources (checked November 2, 2026)

- Nemotron 3.5 Lightning: [NGC model card](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/models/nemotron-3.5-lightning) · [Hugging Face NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) · [Interfaze model page](https://interfaze.ai/models/nvidianvidia-nemotron-35-lightning-30b-a3b-bf16)
- AIPerf: [NVIDIA blog](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf) · [docs](https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking)
- [vLLM recipes](https://github.com/vllm-project/recipes)
- Free GPUs: [GPU Per Hour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) · [Kaggle T4 x2 guide](https://logicity.in/en/blog/how-to-run-70b-llms-free-using-kaggle-s-hidden-gpu-tier)
- GPU prices: [ComputePrices](https://computeprices.com/gpu) · [GPU Per Hour](https://gpuperhour.com/)
- [vLLM crash on T4 (microsoft/fara #57)](https://github.com/microsoft/fara/issues/57)

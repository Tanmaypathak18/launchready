# LaunchReady: Build Plan (Lean)

**A small, open-source launch-readiness kit for open models on NVIDIA GPUs.** It shows, rather than tells, that you can do NVIDIA's Product Manager – Open Models job: size a new model's hardware needs, run it on NVIDIA GPUs with the community's tools, measure it honestly, and explain it to developers.

- **Owner:** Tanmay Pathak
- **Build tool:** Claude Code
- **Plan date:** November 2, 2026
- **Budget:** $0 for everything except one rented RTX 5090 session (~$5–10). Hard cap $20.

---

## How to use this plan in Claude Code

1. Create a public GitHub repo named `launchready`. Save this file in it as `BUILD_PLAN.md`, and save section 8 as `CLAUDE.md` in the repo root. Claude Code loads `CLAUDE.md` at the start of every session.
2. Run one phase per session: paste that phase's prompt, ask Claude to show its plan before writing code, review the diff, run the tests, commit.
3. Tick a phase's boxes only when its "Done when" criteria pass.
4. When anything breaks, write the friction-log entry (template 7.3) *before* you fix it. That log becomes blog material and interview stories.
5. Phases 0–3 are already showable. If you want to send something early, send the brief and the validation results.

---

## 1. What you're building

One repo with five deliverables:

1. **`launchready fit`**: a calculator that reads any model from Hugging Face (no weight download) and works out memory at BF16, FP8, NVFP4 and GGUF, plus which NVIDIA GPUs fit it.
2. **A model brief for NVIDIA Nemotron 3.5 Lightning** (30B total / 3B active hybrid Mamba-Transformer MoE, released Aug 11, 2026): capabilities, hardware needs, engine readiness, ecosystem impact.
3. **Proof the calculator is right**: predicted vs measured memory on free Kaggle GPUs.
4. **A launch kit**: verified vLLM and Ollama recipes for Lightning, benchmarked on a rented RTX 5090, with a quality check.
5. **One blog post** that ties it together, plus a friction log and a requirements table built from real developer issues.

**Customer:** developers and researchers: Hugging Face users, people running models locally, ML engineers serving models. Every output is written for them.

**Cut on purpose:** scheduled automation, an automated feedback pipeline, H100 benchmarks, TensorRT-LLM, fine-tuning, demo videos and slides. Optional extras are in section 5.

---

## 2. JD coverage

| JD responsibility | What proves it |
|---|---|
| 1. Work with model builders on enablement and launch readiness | Readiness checklist + recipes; issues or PRs filed upstream |
| 2. Track releases; summarize capabilities, hardware needs, ecosystem impact | Nemotron brief + a one-off radar of recent releases |
| 3. Requirements, launch plans, checklists | Readiness checklist; requirements table |
| 4. Inference, evaluation, deployment | vLLM/Ollama recipes, benchmarks, quality check, pinned containers |
| 5. Feedback → recurring issues → requirements | Hand-reviewed issues → requirements table |
| 6. Test developer workflows, notebooks, Python | Quickstart audits, friction log, notebooks strangers can run |
| 7. Blogs, docs, social | Blog post + thread, brief, recipe README |

| What they need to see | Where it shows up |
|---|---|
| Hugging Face, PyTorch, Ollama, vLLM (SGLang optional) | Calculator (Hub API), validation (PyTorch memory APIs), recipes and benchmarks |
| Python in notebooks | Kaggle validation notebook, results notebook |
| Inference, evaluation, deployment, GPU acceleration | Memory math, precision formats, KV cache, tensor parallel, benchmarks |
| Technical writing | Brief, blog, recipe README |

---

## 3. Ground rules

- **Labeled numbers.** Every number is *measured* (GPU, driver, CUDA, engine version, model revision, date) or *projected* (with the formula). This is the one rule you can't cut.
- **Reproducible.** Lockfile; pinned engine versions and container tags; model revision hashes; one command per result.
- **Raw data committed.** CSV/JSON next to every chart.
- **Runs for strangers.** Notebooks pass "Run all" from a fresh session.
- **No weight downloads in the calculator.** config.json and safetensors headers only.
- **No secrets in git.** Public data only; nothing from Saayam.

---

## 4. Compute

| Where | Use it for | Cost |
|---|---|---|
| Your MacBook Air | All code; Ollama hands-on; writing | $0 |
| Kaggle "GPU T4 x2" (2 × 16 GB, phone verification) | Memory validation, quickstart audits, dry runs of the benchmark script | $0 |
| Rented RTX 5090, 32 GB, Blackwell (~6–8 h) | Lightning benchmarks: NVFP4 in vLLM, GGUF in Ollama/llama.cpp | ~$5–10 |

**Renting checklist:** Runpod or Vast.ai (per-minute billing) · on-demand, not spot · ~100 GB disk · CUDA 12.8+ image · set the spend cap first · download weights with `hf download` before starting the clock on benchmarks · copy results off before stopping. Rent only after the benchmark script has run end to end on Kaggle.

Mac numbers never go in published benchmarks.

---

## 5. Phases

### Phase 0: Setup ($0)

- [ ] Public repo `launchready` (Apache-2.0), Python 3.11+ with uv, ruff, pytest.
- [ ] Tokens: Hugging Face (accept Llama/Gemma licenses), Kaggle (verify phone; enable GPU + internet), GitHub.
- [ ] Mac: install Ollama, run `ollama run llama3.1:8b --verbose`, note the eval rate.
- [ ] Save `BUILD_PLAN.md` and `CLAUDE.md`; create `docs/friction_log.md` from template 7.3.

**Done when:** CI is green on a smoke test and Ollama answers on your Mac.

```text
Read BUILD_PLAN.md and CLAUDE.md. Scaffold a uv-managed Python 3.11 package `launchready` with a Typer CLI `launchready` (subcommands fit, brief, radar that print "not implemented"), ruff and pytest, one passing smoke test, a GitHub Actions ci.yml running ruff and pytest, an Apache-2.0 LICENSE, and a README with a one-paragraph description and a Status checklist mirroring the phases. Folders: launchready/ (hub.py, arch.py, memory.py, engines.py, brief.py, radar.py, cli.py, data/gpus.yaml), templates/, recipes/, bench/, notebooks/, briefs/, docs/, tests/. Docstrings only, no logic yet. Show me your plan first.
```

### Phase 1: Memory calculator ($0)

**Goal:** `launchready fit <repo_id>` shows memory at each precision and which NVIDIA GPUs fit, without downloading weights.

- [ ] `hub.py`: config.json (handle nested `text_config`), model info (date, license, gated), safetensors parameter counts (`huggingface_hub.get_safetensors_metadata`), file sizes. Cache on disk.
- [ ] `arch.py`: normalize into an `Architecture` dataclass: layers and layer types (full attention, sliding window, Mamba), heads, KV heads, head_dim, context, MoE fields (total vs active), MLA fields, Mamba fields. Unknown fields → warning naming the field.
- [ ] `memory.py`: weights per precision (6.1), KV/state per sequence (6.2), fit model (6.3), `--kv-dtype fp16|fp8`.
- [ ] `data/gpus.yaml` from 6.4.
- [ ] CLI: `launchready fit <id> [--context 32768] [--users 8] [--gpu rtx-5090] [--json]`, every output labeled "projected".
- [ ] Tests first, using 6.5 and saved config fixtures.

**Done when:** reference tests pass and `launchready fit` gives sensible tables for dense, MoE, hybrid Mamba and MLA models.

```text
Implement Phase 1 of BUILD_PLAN.md using the formulas in sections 6.1–6.5. Never download weights: use config.json and safetensors headers via huggingface_hub. Write the tests for section 6.5 first with saved config.json fixtures (no network in CI), then hub.py, arch.py, memory.py, data/gpus.yaml and `launchready fit`. Support dense GQA, MoE, MLA, sliding-window and hybrid Mamba layers; for any config field you can't interpret, warn and name it instead of guessing. Then run `launchready fit` on nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, Qwen/Qwen3-4B, deepseek-ai/DeepSeek-V3 and ibm-granite/granite-4.0-h-tiny and show me the tables. Flag anything that looks wrong.
```

### Phase 2: Engine readiness + the Nemotron brief ($0)

**Goal:** `launchready brief <repo_id>` writes a one-page brief a leader or developer can act on.

- [ ] `engines.py`: shallow sparse clones of only the paths in 6.6, at `main` and the latest release tag, cached in `~/.cache/launchready`. Status per engine: `release`, `main-only`, `not-found`, `unknown`, with the matching file and line. Flag `auto_map` → "needs trust_remote_code".
- [ ] Variants: FP8, NVFP4 and GGUF derivatives (Hub filter `base_model:quantized:<repo_id>` plus the builder's org), with sizes.
- [ ] `brief.py`: render template 7.1 to `briefs/<org>__<model>.md`.
- [ ] Write the **Ecosystem impact** and **Open questions** sections yourself.
- [ ] `launchready radar --since 30d`: one-off list of recent root-model releases from the watchlist in 6.7, run by hand. No scheduling.

**Done when:** the Lightning brief exists with your hand-written sections, every claim is sourced, and you have one radar digest.

```text
Implement Phase 2 of BUILD_PLAN.md. For engine support, use shallow sparse clones (git clone --depth 1 --filter=blob:none --sparse) of only the paths in section 6.6, check main and the latest release tag, cache under ~/.cache/launchready, and report each engine as release / main-only / not-found / unknown with the matching file and line. If a path has moved, find the new one and tell me. Add the quantized-variants lookup, build brief.py with Jinja2 from template 7.1, and wire `launchready brief`. Leave "Ecosystem impact" and "Open questions" as TODO sections. Then add `launchready radar --since 30d` that lists recent root models (skip quantized, GGUF, fine-tune and adapter repos) from the watchlist in 6.7 and writes radar/<date>.md. Generate the brief for nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 and show it to me.
```

### Phase 3: Validate on free GPUs + first-user audit ($0)

**Goal:** prove the calculator on real NVIDIA GPUs and act as the first user of official quickstarts.

- [x] `notebooks/01_memory_validation.ipynb` on Kaggle T4 x2 with 3 models: `Qwen/Qwen3-4B` (dense), `allenai/OLMoE-1B-7B-0924-Instruct` (MoE), `ibm-granite/granite-4.0-h-tiny` (hybrid Mamba). Over ~13 GB in FP16 → run across both GPUs with `--tensor-parallel-size 2`.
- [x] Per model: projected (`launchready fit`) vs measured — Transformers + `torch.cuda.max_memory_allocated()`, and vLLM's KV-capacity and max-concurrency startup lines (6.8).
  - Changed 2026-11-03: measured weights come from vLLM's "Model loading took X GiB" line (summed over TP ranks), not Transformers `max_memory_allocated()`; the Transformers stage is an optional cross-check, off by default (see docs/friction_log.md).
- [x] If something fails on T4, log it and move on.
- [x] Run 2–3 official quickstarts exactly as written (model-card snippets for vLLM or Transformers). Log every break; file an issue or PR where it genuinely helps.

**Done when:** predicted vs measured is in a CSV and chart with misses explained (aim: weights within ~5%, KV capacity within ~10–15%), and at least 2 audits are logged.

```text
Create notebooks/01_memory_validation.ipynb for a Kaggle "GPU T4 x2" session with internet on. Cell 1 installs pinned versions (launchready from my GitHub repo, torch, vllm) and prints the environment (nvidia-smi, driver, CUDA, versions). For each model in BUILD_PLAN.md Phase 3: record launchready's projection; measure weights with Transformers and torch.cuda.max_memory_allocated; start vLLM with --dtype half (tensor parallel 2 when FP16 weights exceed ~13 GB), parse its KV-cache capacity and max-concurrency log lines, then shut it down and free memory. Save results/memory_validation.csv with a measured/projected column and plot predicted vs measured. Skip failing models with a logged reason instead of crashing. It must pass "Run all" from a fresh session.
```

### Phase 4: Developer feedback → requirements ($0, by hand)

- [ ] Read ~30 recent GitHub issues and Hugging Face discussions about Nemotron 3.5 Lightning (or its architecture) in vllm-project/vllm, ggml-org/llama.cpp, ollama/ollama and the model's Hugging Face repo.
- [ ] Group them into themes (install/dependency, driver/CUDA, out of memory, chat template/tool calling, quantization, performance, correctness, docs gap).
- [ ] Write `docs/requirements.md` from template 7.4: top 5 themes → requirements with acceptance criteria.

**Done when:** the table has 5 requirements, each with linked evidence.

No prompt needed. This is your PM work, and doing it by hand is the point. Claude Code can help you collect the links.

### Phase 5: Nemotron 3.5 Lightning launch kit (~$5–10)

1. [ ] **Readiness checklist** (template 7.2) for Lightning: what must be true before you'd call it "ready on NVIDIA GPUs."
2. [ ] **Gap check:** read NVIDIA's model card and recipes. Your kit adds independent measurements on a consumer GPU, first-user friction, cost per 1M tokens and plain-English guidance. Put that list at the top of the recipe README.
3. [ ] **Recipes** (`recipes/nemotron-3.5-lightning/`): `vllm.sh` (NVFP4) and `ollama.md` (GGUF via `ollama run hf.co/...`), each stating GPU, engine version and model revision. Start from the vLLM version NVIDIA's card uses.
4. [ ] **Benchmark + quality scripts** (`bench/`), per 6.9.
5. [ ] **Dry run** everything on Kaggle against `Qwen/Qwen3-4B`.
6. [ ] **Rented RTX 5090 session:** NVFP4 in vLLM, GGUF in Ollama/llama.cpp, concurrency 1/8/32, one 32K-context point, drafter on/off at concurrency 1, quality subset. NVIDIA's card points local devices to the GGUF, so if NVFP4 doesn't run on the 5090, that's a finding. Log it.
7. [ ] **Results notebook** (`notebooks/02_launch_kit_nemotron.ipynb`) that rebuilds every chart from committed results.

**Done when:** both recipes are verified on the 5090, every number is labeled measured, and the checklist is green or each exception is explained.

```text
Implement Phase 5 steps 3–4 of BUILD_PLAN.md. Create recipes/nemotron-3.5-lightning/ (vllm.sh, ollama.md, README.md). Create bench/run_matrix.py that reads a small YAML matrix (section 6.9), starts and stops each server, waits for health, sends warm-up requests, runs AIPerf (confirm flags with `aiperf profile --help` for the pinned version), samples nvidia-smi every 500 ms, and writes one JSON per run plus results/summary.csv with the full environment. Compute TTFT and ITL p50/p99, output tokens/s per user and total, and $ per 1M output tokens from an --hourly-price argument. Create bench/quality.py running lm-eval GSM8K (250) and IFEval (200) against the OpenAI-compatible server. Then write run_session_5090.sh that runs everything unattended and prints where results are saved. Dry-run it against Qwen/Qwen3-4B on Kaggle first.
```

### Phase 6: Publish ($0)

- [ ] **Blog post:** "Running NVIDIA Nemotron 3.5 Lightning on an RTX 5090: what fits, how fast, and what broke." Include: the memory math in plain English, the predicted-vs-measured chart, recipes, benchmark charts, the quality check, cost per 1M tokens, and caveats.
- [ ] Publish on your site and as a Hugging Face community article; post a LinkedIn/X thread with one chart.
- [ ] README: quickstart, results, links to the brief, blog and requirements table.

**Done when:** the post is live and a stranger can reproduce one result from the README.

### Optional extras (only if they come cheap)

- **SGLang** recipe and benchmark on the same 5090 session.
- **72-hour drill:** if a notable model drops while you're building, run `launchready brief`, verify a recipe and publish a short post within 72 hours. Great interview story.
- **H100 session** (~$10) for a data-center comparison.

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
- Compute in bytes, display GiB, and take VRAM from `nvidia-smi` for GPUs you measure.

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

(Confirm flags for your pinned version with `aiperf profile --help`.)

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

## 8. `CLAUDE.md` starter (save to the repo root)

~~~markdown
# LaunchReady

Small open-source launch-readiness kit for open-weight LLMs on NVIDIA GPUs: sizes hardware needs, checks engine support, writes model briefs, and benchmarks NVIDIA Nemotron 3.5 Lightning. Full plan: BUILD_PLAN.md (read the current phase before starting).

## Audience
Developers and researchers: Hugging Face users, local-model users, ML engineers serving models. Write every output for them.

## Rules
- Never download model weights in the calculator; use config.json and safetensors headers only.
- Label every number "measured" (GPU, driver, CUDA, engine version, model revision, date) or "projected" (with the formula).
- When a config field can't be interpreted, warn and name it. Never guess silently.
- Pin versions and record the environment with every result.
- No secrets in git: read HF_TOKEN and GITHUB_TOKEN from the environment.
- Keep scope lean: don't add features beyond the current phase.
- When something breaks, add an entry to docs/friction_log.md before fixing it.

## Stack
Python 3.11+, uv, Typer, huggingface_hub, Jinja2, PyYAML, pytest, ruff. Benchmarks: AIPerf, lm-evaluation-harness. Tests use saved fixtures; CI has no network.

## Commands
- `uv run launchready fit <repo_id> --context 32768 --users 8`
- `uv run launchready brief <repo_id>`
- `uv run launchready radar --since 30d`
- `uv run pytest` and `uv run ruff check .`

## Conventions
- Small commits; update README Status when a phase's "Done when" passes.
- Compute in bytes, display GiB.
~~~

---

## 9. When it ships: resume bullets

Fill the brackets with real results only.

**LaunchReady: Open-Model Launch Readiness on NVIDIA GPUs** | Python, Hugging Face, PyTorch, vLLM, Ollama, AIPerf

- Projected GPU memory within [X]% of measured across [N] open models by building an open-source tool that sizes weights, KV cache and Mamba state at BF16, FP8 and NVFP4 from Hugging Face configs and checks engine support.
- Published a launch kit for NVIDIA Nemotron 3.5 Lightning with vLLM and Ollama recipes verified on an RTX 5090, benchmarks ([X] tokens/s at [Y] users, [Z] ms p99 TTFT) and an NVFP4 quality check within [Δ] of BF16.
- Turned [N] developer issues into 5 prioritized requirements and filed [N] upstream issues or PRs from first-user testing.

---

## 10. Gotchas

- **T4:** no BF16 (`--dtype half`); some FP8, FP4, Mamba and MoE kernels need newer GPUs. Log and move on.
- **Version pins:** some vLLM releases have broken on T4; record the version that works.
- **Kaggle:** phone verification; ~30 GPU-hours/week; check disk with `df -h`; keep tokens in Kaggle Secrets.
- **Gated repos** (Llama, Gemma): need token + license acceptance; report "gated" instead of crashing.
- **RTX 5090:** needs CUDA 12.8+ and recent engine builds; expect NVFP4 friction. Friction is content.
- **Units:** don't mix GB and GiB in one table.
- **Licenses:** cite each model's license (NVIDIA OpenMDW for Lightning).

---

## 11. Glossary

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

## 12. Sources (checked November 2, 2026)

- Nemotron 3.5 Lightning: [NGC model card](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/models/nemotron-3.5-lightning) · [Hugging Face NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) · [Interfaze model page](https://interfaze.ai/models/nvidianvidia-nemotron-35-lightning-30b-a3b-bf16)
- AIPerf: [NVIDIA blog](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf) · [docs](https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking)
- [vLLM recipes](https://github.com/vllm-project/recipes)
- Free GPUs: [GPU Per Hour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) · [Kaggle T4 x2 guide](https://logicity.in/en/blog/how-to-run-70b-llms-free-using-kaggle-s-hidden-gpu-tier)
- GPU prices: [ComputePrices](https://computeprices.com/gpu) · [GPU Per Hour](https://gpuperhour.com/)
- [vLLM crash on T4 (microsoft/fara #57)](https://github.com/microsoft/fara/issues/57)
- [Claude Code memory and CLAUDE.md](https://code.claude.com/docs/en/memory)

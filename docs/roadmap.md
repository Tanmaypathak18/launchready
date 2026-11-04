# LaunchReady: Product roadmap

**Status as of 2026-11-03:** the tools, validation, requirements and launch kit are built, and the [live tool](https://huggingface.co/spaces/tanmaypathak4/launchready) and [summary page](https://tanmaypathak18.github.io/launchready/) are public. [progress.md](progress.md) has the detail on what shipped.

---

## 1. Product overview

**Vision:** know whether a new open model will run on NVIDIA GPUs before launch, from evidence rather than from the model's name or card.

**Problem:** when a model drops, developers can't tell whether it will fit their GPU or whether the card's commands will work without first spending the download and the GPU time.

**Customers:** developers and researchers: Hugging Face users, people running models locally, ML engineers serving models. Every output is written for them.

**What we deliver:**

| Customer need | Deliverable | Status |
|---|---|---|
| Size a model without downloading it | `launchready fit` and the live tool: memory at BF16, FP8, NVFP4 and GGUF, plus which NVIDIA GPUs fit | Live |
| Summarize capabilities, hardware needs and ecosystem impact | A model brief for NVIDIA Nemotron 3.5 Lightning, plus a one-off radar of recent releases | Done |
| Trust the numbers | Projected vs measured memory on free Kaggle GPUs | Done, with one known miss |
| Launch readiness with model builders | Readiness checklist; vLLM and Ollama recipes, benchmarks, quality check and pinned containers | Built, dry-run only |
| Turn feedback into requirements | Friction log, quickstart audits and a requirements table from real developer issues | Done |

---

## 2. Outcomes and success measures

| Outcome | Measure | Target | Result |
|---|---|---|---|
| Weight projections are trustworthy | Projected vs measured weights | Within ~5% | Met: −0.9%, −0.1% and −0.3% measured on three models |
| Capacity projections are trustworthy | Projected vs measured KV capacity | Within ~10–15% | Met for dense and MoE. **Missed for hybrid Mamba:** +133% on users |
| Quickstart breaks are found before users hit them | Audits logged | At least 2 | Met: three breaks logged |
| Feedback becomes requirements | Requirements with linked evidence | 5 | Met: 5 from 36 threads |
| New users can reproduce a result | One result reproduced from the README | Works from a fresh clone | Met for the Qwen3-4B projection |
| Lightning is ready on an RTX 5090 | Readiness checklist | All checks ready, or each exception explained | **Not met:** 1 ready, 5 not ready, 21 unknown of 27 checks |

---

## 3. Milestones

| Milestone | Outcome | Status |
|---|---|---|
| Phase 0: Setup | Public repo with CI, tokens and a local Ollama | Done |
| Phase 1: Memory calculator | `launchready fit` sizes dense, MoE, hybrid Mamba and MLA models without downloading weights | Done |
| Phase 2: Engine readiness + Nemotron brief | `launchready brief` and `launchready radar`; the Lightning brief has hand-written sections and sources | Done |
| Phase 3: Validate on free GPUs + first-user audits | Projected vs measured in a CSV and chart with the misses explained; quickstart audits logged | Done |
| Phase 4: Developer feedback → requirements | Threads read by hand and grouped into 5 requirements with evidence | Done |
| Phase 5: Nemotron 3.5 Lightning launch kit | Checklist, gap check, recipes and benchmark scripts built; dry run on Kaggle against `Qwen/Qwen3-4B`, raw output in `results/dryrun_kaggle/` (not headline) | Done as a dry run |
| Live tool and summary page | Hugging Face Space and Pages site public | Done |

---

## 4. Risks and decisions

### 4.1 Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| Hybrid Mamba capacity runs high, and Lightning is hybrid | Teams plan for more users than the engine holds | The tool warns on every hybrid; the model isn't retuned from one result |
| Validation rests on three models on T4 | Accuracy elsewhere is unproven | Every number stays labeled; limits are stated in the README |
| Engines and cards change quickly | Recipes and audits go stale | Pinned versions and model revisions |
| Fixes confirmed on T4 only | They could still fail on Lightning | The checklist keeps these as unknown |

### 4.2 Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-03 | Measured weights come from vLLM's "Model loading took X GiB" line, summed over TP ranks. The Transformers stage is an optional cross-check, off by default | The Transformers stage crashed on our own bug ([friction log](friction_log.md)) |
| 2026-11-03 | GGUF is bartowski Q4_K_M, and its memory comes from the header | ggml-org has no Q4_K_M, and the header says 6.19 bits per weight |
| 2026-11-03 | Quality compares NVFP4 (vLLM) with GGUF (llama.cpp) directly, thinking off for both | The card has no GSM8K or IFEval numbers |
| 2026-11-03 | The fit model isn't retuned from one hybrid; the tool warns instead | One model isn't enough to calibrate on |
| 2026-11-03 | The live tool uses real Hub file sizes where they exist | Projected sizes flip the verdict for Lightning on an RTX 4090 |

### 4.3 Out of scope

Scheduled automation, an automated feedback pipeline, H100 benchmarks, TensorRT-LLM, fine-tuning, demo videos and slides.

---

## 5. How we run it

### 5.1 Product principles

- **Labeled numbers.** Every number is *measured* (GPU, driver, CUDA, engine version, model revision, date) or *projected* (with the formula). This rule is not optional.
- **Reproducible.** Lockfile; pinned engine versions and container tags; model revision hashes; one command per result.
- **Raw data committed.** CSV/JSON next to every chart.
- **Runs for new users.** Notebooks pass "Run all" from a fresh session.
- **No weight downloads in the calculator.** config.json and safetensors headers only.
- **No secrets in git.** Public data only.

### 5.2 Compute

| Where | Used for | Cost |
|---|---|---|
| Development laptop (Mac) | All code; Ollama hands-on; writing | $0 |
| Kaggle "GPU T4 x2" (2 × 16 GB) | Memory validation, quickstart audits, dry runs of the benchmark script | $0 |
| Hugging Face PRO | Hosting the live tool on CPU Basic | $9/month |

Mac numbers never go in published benchmarks.

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

### 6.9 Benchmark configuration (RTX 5090)

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

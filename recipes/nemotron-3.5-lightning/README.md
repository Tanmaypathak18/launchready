# NVIDIA Nemotron 3.5 Lightning on one RTX 5090: launch kit

Recipes, benchmark scripts and a quality check for [NVIDIA Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (32.9B total / ~3B active hybrid Mamba-Transformer MoE) on a single consumer Blackwell GPU.

**Status (2026-11-03):** scripts dry-run on Kaggle T4 with Qwen3-4B ([raw output, not headline](../../results/dryrun_kaggle/README.md)); fixes made after that dry run are tested offline but not re-run on a GPU. RTX 5090 session **not yet run**. See the [readiness checklist](../../docs/readiness_checklist.md) for what's ready, not ready and unknown. Every performance number below is *projected* until the 5090 session replaces it with *measured* values.

## What this kit adds to NVIDIA's model card

NVIDIA's [NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) gives vLLM recipes for DGX Spark, H100, GB200 and Ampere, an Ollama command, a llama.cpp command, and a BF16-vs-NVFP4 accuracy table. This kit adds:

1. **A consumer-GPU recipe, verified on the GPU.** The card has no RTX 5090 row. Two community reports say its FP8-KV recipe crashes on SM12x. The session runs the card's command as written, then without `--kv-cache-dtype fp8`, and records both.
2. **Independent measurements** of TTFT, inter-token latency, tokens/s per user and in total, and GPU memory and power, at 1, 8 and 32 users plus one 32K-token prompt, with the DSpark drafter on and off. Every number carries the GPU, driver, CUDA, engine version, model revision and date.
3. **Cost per 1M output tokens** at the rented GPU's real hourly price.
4. **What's really inside the GGUF.** The "Q4_K_M" files store 6.1–6.2 bits per weight (experts fall back to Q5_0/Q8_0), and all four repos checked carry the wrong pre-tokenizer tag. The card says neither.
5. **First-user friction** from following the card: wrong tags, single-slot Ollama, flags that break on consumer GPUs. It's logged in [`docs/friction_log.md`](../../docs/friction_log.md) and summarized in troubleshooting below.
6. **A like-for-like quality check** (GSM8K 250, IFEval 200) of NVFP4 in vLLM vs GGUF in llama.cpp, from the same harness and settings. The card publishes neither benchmark, and BF16 (61 GiB) doesn't fit a 5090, so NVIDIA's own BF16-vs-NVFP4 gap (IFBench 71.88 → 72.88) is context, not a baseline.

## Pinned versions

| | Version | Why |
|---|---|---|
| GPU | 1× RTX 5090, 32 GB, Blackwell SM120, CUDA 12.8+ driver | consumer Blackwell; NVFP4 is native |
| vLLM | `vllm/vllm-openai:v0.27.1` | the version NVIDIA's card uses; decode speed moved 16% between v0.28 and v0.29 on one Blackwell device ([vllm#59770](https://github.com/vllm-project/vllm/issues/59770)) |
| NVFP4 model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `bee7596` | |
| DSpark drafter | `…-NVFP4-DSpark` @ `6d127b5` | NVIDIA's speculative-decoding drafter |
| GGUF | `bartowski/…-GGUF` @ `f0eec22`, `Q4_K_M` | plain `llama-quantize` output; ggml-org has no Q4_K_M |
| llama.cpp | v0.8.0 | includes the pre-tokenizer fix (PR #30960) |
| Ollama | 0.40.14 | latest release on 2026-11-03 |
| AIPerf / lm-eval | 0.13.0 / 0.4.13 | run through `uvx`, pinned |
| Tokenizer for token counts | `nvidia/…-BF16` @ `a9904d2` | same for every engine |

## Recipes

- **vLLM, NVFP4:** [`vllm.sh`](vllm.sh). The default mode is NVIDIA's flags without FP8 KV, prefix caching off, max model length 40,960. `MODE=card` runs the card's recipe as written; `SPEC=dspark` adds the drafter.
- **Ollama and llama-server, GGUF:** [`ollama.md`](ollama.md). Ollama serves one request at a time for this model, so use `llama-server -np 8` for more users.

## Memory on a 32 GiB RTX 5090 (projected)

| Path | Weights | KV + Mamba state per 40,960-token sequence | Sequences that fit (projected) |
|---|---|---|---|
| vLLM, NVFP4, BF16 KV cache, `--gpu-memory-utilization 0.85` | 20.08 GiB (Hub files; 17.24 GiB if every weight were 4.5 bits) | 0.28 GiB (0.23 GiB attention KV at 2 bytes + 0.05 GiB Mamba state) | ~18 |
| llama-server, GGUF Q4_K_M, `-np 8` | 23.73 GiB (file; header says 6.19 bits/weight) | 0.28 GiB | 8 slots used; ~6.0 GiB left for buffers |

Formula (BUILD_PLAN 6.2–6.3): KV budget = 32 GiB × 0.85 − weights − 2 GiB reserve = 5.12 GiB for NVFP4. Sequences = floor(5.12 ÷ 0.281) = 18. The working config uses a BF16 KV cache because FP8 KV is the flag reported to break on SM12x. For hybrids, vLLM's real concurrency can be far below this projection: granite-4.0-h-tiny came in at 50.6 against 118 projected in Phase 3. The 5090 session measures it, using vLLM's `GPU KV cache size` and `Maximum concurrency` lines in `startup/*.json`.

## What the session runs

| Step | What | Settings |
|---|---|---|
| Startup checks | the card's command; then the card's command without `--kv-cache-dtype fp8` | each: start, health, a fixed 8-request sequence; result + log tail recorded |
| vLLM NVFP4 | chat (512 in / 256 out) at 1, 8 and 32 users × 3 reps; one 32,768-in / 512-out point | BF16 KV, prefix caching **off**, thinking off, `ignore_eos` |
| vLLM NVFP4 + DSpark | chat at 1 user × 3 reps | 3 speculative tokens (card value) |
| llama-server GGUF | chat at 1, 8 and 32 users × 3 reps; one 32K point | `-np 8`, `cache_prompt: false`; at 32 users, 24 requests queue |
| Ollama GGUF | chat at 1 user × 3 reps | one slot is forced; thinking on (Ollama default); output length as written by the model |
| Quality | GSM8K (250, 5-shot) + IFEval (200) on vLLM NVFP4 and llama-server GGUF | greedy, thinking off, chat template applied |

Every prompt in a run is unique, so no engine gets prompt-cache hits. Every run samples `nvidia-smi` every 500 ms and writes one JSON as soon as it finishes. A run that lost requests is marked `partial`, with the count in its note. `$ per 1M output tokens = hourly price ÷ (total output tokens/s × 3,600) × 1,000,000`. Headline numbers are the median of 3 reps (min and max kept).

## How long the 5090 session takes (estimate)

| Stage | Estimate | Notes |
|---|---|---|
| Pod start, `uv sync`, tool installs | 10–15 min | |
| Downloads (~74 GB: NVFP4 21.6 GB, DSpark 1.3 GB, GGUF 25.5 GB, Ollama's own copy of the GGUF 25.5 GB) | 10–25 min | depends on the pod's network |
| llama.cpp CUDA build | 20–35 min | **Took 27.5 min on Kaggle** (dry run, `prepare-3`: 1,654 s). Depends on the pod's CPU cores; longer if the image needs `nvcc` installed |
| vLLM startups (5 of them: 2 card checks, benchmark, DSpark, any fallback) | 30–90 min | **vLLM starts slowly on Mamba hybrids:** model load + kernel warmup + torch.compile + CUDA graphs. granite-4.0-h-tiny took 16 min on a T4. The card's 1M-token default context adds to it |
| Benchmarks (~35 AIPerf runs) | 45–75 min | llama.cpp at 32 users is the slowest |
| Quality (2 engines × 450 prompts) | 30–60 min | IFEval allows 1,280 output tokens |
| Pack + copy off | 10 min | |
| **Total** | **~2.5–5.5 h** | `MAX_HOURS=8` hard stop. At ~$0.90/h that's ~$2.50–5; the hard stop caps it at ~$7.20 |

## Rent the GPU and run the session (click by click)

Do this only after the Kaggle dry run passes ([`notebooks/04_bench_dryrun_kaggle.ipynb`](../../notebooks/04_bench_dryrun_kaggle.ipynb)). These steps follow Runpod's docs as of 2026-11-03 and haven't been tried yet; if a screen looks different, the field names are the thing to match.

1. **Account and spending cap.** Sign up at [runpod.io](https://www.runpod.io). Under **Billing**, add **$15** of credit and leave auto-pay **off**. Prepaid credit is the hard cap: when it runs out, the pod stops.
2. **Hugging Face token (optional).** None of these models is gated. A token only speeds up downloads. If you have one, keep it handy for step 5; never paste it into a file in the repo.
3. **Pick the GPU.** Go to **Pods → Deploy**. Choose **RTX 5090**, **1 GPU**, and **On-Demand** (not Spot/Interruptible: a spot pod can vanish mid-run). Note the **$/hr** shown; you'll type it in step 7.
4. **Set the container.** Click **Change Template → New Template** (or **Edit** the pod's template):
   - **Container Image:** `vllm/vllm-openai:v0.27.1`
   - **Container Start Command:** `{"entrypoint": ["bash", "-c"], "cmd": ["sleep infinity"]}`. The image would otherwise start `vllm serve` with no model and exit.
   - **Container Disk:** 30 GB · **Volume Disk:** 100 GB · **Volume Mount Path:** `/workspace`
   - **Environment variables:** `BENCH_PROVIDER=runpod`, `BENCH_CONTAINER_IMAGE=vllm/vllm-openai:v0.27.1` (and `HF_TOKEN` if you have one)
5. **Check CUDA.** Pick a machine whose filter or details show **CUDA 12.8 or newer** (the RTX 5090 needs it). Click **Deploy On-Demand**. The pod is billed from now on.
6. **Open a terminal.** On the pod, click **Connect → Start Web Terminal → Connect to Web Terminal**. If that's not offered, use the **SSH** command shown under Connect from your laptop's terminal.
7. **Start the session.** Paste these lines, replacing `0.89` with the $/hr from step 3:
   ```bash
   cd /workspace && git clone https://github.com/Tanmaypathak18/launchready && cd launchready
   HOURLY_PRICE=0.89 MAX_HOURS=8 setsid nohup bash bench/run_session_5090.sh > session.log 2>&1 &
   tail -f session.log
   ```
   `nohup` keeps it running if the browser tab closes. Reopen the terminal and run `tail -f /workspace/launchready/session.log` to watch again. Each stage prints elapsed time and estimated cost. Press **Ctrl+C** to stop watching; the session keeps going.
8. **Wait for the banner.** At the end it prints the results folders and a bold **COPY RESULTS OFF THE POD BEFORE YOU STOP IT** box with the path to `results/<session>.tar.gz`.
9. **Copy the results to your laptop.** Easiest: in the pod terminal run
   ```bash
   wget -qO /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl
   runpodctl send /workspace/launchready/results/<session>.tar.gz
   ```
   It prints a one-time code. On your laptop (install with `brew install runpod/runpodctl/runpodctl`), run `runpodctl receive <code>`. Check that the `.tar.gz` opens on your laptop **before** the next step.
10. **Stop and delete the pod.** On the pod page, click **Stop**, then **Terminate** (the trash icon). The volume disk still bills while it exists, so terminate the pod; don't just stop it. Check **Billing** a few minutes later to confirm nothing is still running.

## Troubleshooting (top 5)

1. **vLLM crashes at startup with an illegal memory access or a cuDNN FP8 GEMM error.** Remove `--kv-cache-dtype fp8` (the default in `vllm.sh`). Reported on RTX 5090 laptop and DGX Spark ([NVFP4 #7](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/discussions/7), [#10](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/discussions/10)). If it still crashes on the 5th request or so, add `--enforce-eager` ([vllm#55406](https://github.com/vllm-project/vllm/issues/55406)).
2. **`llama-server -hf ggml-org/...:Q4_K_M` says "no GGUF files found".** ggml-org publishes Q4_0, Q8_0 and BF16 only. Use `:Q4_0`, or bartowski's or unsloth's repo for Q4_K_M.
3. **Ollama is slow with several users.** It serves one request at a time for `nemotron_h_moe` ([ollama#18938](https://github.com/ollama/ollama/issues/18938)). Use `llama-server -np 8` (see [`ollama.md`](ollama.md)).
4. **Throughput is 13–16% lower than expected and TTFT is higher.** Prefix caching costs that even with no cache hits ([vllm#60008](https://github.com/vllm-project/vllm/issues/60008)), and it only hits for shared prefixes longer than ~2,096 tokens ([vllm#53749](https://github.com/vllm-project/vllm/issues/53749)). Leave it off unless your prompts share long prefixes.
5. **GGUF answers differ from vLLM's in Hindi, Thai and other non-Latin scripts.** The GGUF says `tokenizer.ggml.pre = pixtral`. Use llama.cpp v0.8.0+ (maps it to Tekken). For other tools, check that they include llama.cpp PR #30960, or rewrite the tag with `gguf_new_metadata.py --pre-tokenizer tekken`.

## Reproduce

```bash
uv run python -m bench.gguf_check hf://bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF@f0eec2267ae843d9eb21ea3926ab0046da0a8628/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf
uv run python -m bench.run_matrix --matrix bench/matrices/rtx5090_lightning.yaml --hourly-price 0.89 --dry-print
HOURLY_PRICE=0.89 bash bench/run_session_5090.sh
```

Results land in `results/bench/<session>/` (one JSON per run, startup checks, logs, GPU samples, `headline.json`), `results/summary.csv` (one row per run, environment included), `results/quality/<session>/` and `results/quality_summary.csv`.

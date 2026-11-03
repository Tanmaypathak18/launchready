# Dry run, Qwen3-4B, T4, not headline

**These are not published results.** This is the raw output of `bench/run_session_5090.sh` from a Kaggle dry run (roadmap Phase 5 step 5), kept to show the scripts ran end to end. The files say `"label": "measured"` because the scripts wrote them that way; here that means "measured in a dry run". Don't quote these numbers as Lightning or RTX 5090 performance.

| | |
|---|---|
| Session | `dryrun_kaggle_qwen3_4b_20261103T2043Z`, 2026-11-03 |
| GPU | Tesla T4 (2 in the machine, 1 used), 15,360 MiB, driver 580.178.04, CUDA 13.0 |
| Model | Qwen/Qwen3-4B @ `1cfa9a7`; GGUF Qwen/Qwen3-4B-GGUF @ `bc64014` Q4_K_M |
| Engines | vLLM 0.32.0 (pip), llama.cpp v0.8.0, Ollama 0.40.14, AIPerf 0.13.0, lm-eval 0.4.13 |
| Code | launchready `88a53fd` |
| Scale | 1 rep, small request counts, quality limit 20 (+20) |
| Price | $0 |

The session's own `.tar.gz` isn't kept here; it duplicated these files.

## Known problems in these results

All three were found by reading this output and are written up in [`docs/friction_log.md`](../../docs/friction_log.md). They're fixed in the code that came after `88a53fd`, but **that code has not been re-run**, so this folder still shows the problems.

1. **vLLM quality is invalid.** `vllm-bench` shows gsm8k 0.0 and ifeval 0.0, with status `ok`. Those aren't scores: all 60 answers were empty, because the `deepseek_r1` parser put them in `reasoning` and left `content` null. llama-server's 0.65 and 0.75 are real but come from only 20 examples.
2. **GGUF TTFT is flattered.** llama-server and Ollama hit their prompt cache on repeated prompts (long: 5,075 → 957 ms; chat: ~430 → ~140 ms), while vLLM ran with prefix caching off. Don't compare TTFT across engines here.
3. **`llamacpp-gguf__np8__chat__c32__r1` lost 2 of 64 requests** (`ServerDisconnectedError`), though it says `ok` and 64 completed. The cause isn't diagnosed.

## What the dry run did show

- All 5 servers started, and all 10 runs finished. The two expected failures failed as expected: the card's `--enable-reasoning` on vLLM 0.32.0, and FP8 KV on T4.
- The llama.cpp CUDA build works (1,654 s, 27.5 min), linking the driver's `libcuda.so.1`.
- The GPU sampler, `summary.csv`, the startup JSONs, quality output and packing all work.

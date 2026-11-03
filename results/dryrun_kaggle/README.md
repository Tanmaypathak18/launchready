# Kaggle dry runs: Qwen3-4B on T4

**Dry run, not headline.** Everything here tests the benchmark scripts, using Qwen3-4B on Tesla T4. None of it is Lightning, and none of it is an RTX 5090 number. The CSV and JSON files say `"label": "measured"` because the scripts wrote them that way; here that means "measured in a dry run". Don't quote these numbers as model or GPU performance.

| Folder | Session | Code | Result |
|---|---|---|---|
| [`run2_20261103/`](run2_20261103/README.md) | `dryrun_kaggle_qwen3_4b_20261103T2043Z` | `88a53fd` | All servers ran; exposed the problems below |
| [`run3_20261103/`](run3_20261103/) | `dryrun_kaggle_qwen3_4b_20261103T2220Z` | `8b4ca32` | All stages exited 0; 10 of 10 runs `ok`; fixes confirmed |

Run 1 crashed early and its outputs aren't kept; see the [friction log](../../docs/friction_log.md). Each session's own `.tar.gz` is left out because it duplicates these files.

The files are raw, with one edit: in run 3's vLLM IFEval samples, an address the model invented in a sample advert is written `info[at]softguard.com`, because the repo's tree guard bans email addresses. Scores aren't affected.

## Environment (run 3)

| | |
|---|---|
| Date | 2026-11-03 |
| Commit | launchready `8b4ca32` |
| GPU | Kaggle 2× Tesla T4; benchmarks used one (`CUDA_VISIBLE_DEVICES=0`). 15,360 MiB, compute capability 7.5, driver 580.178.04, CUDA 13.0 |
| Model | Qwen/Qwen3-4B @ `1cfa9a7`; GGUF Qwen/Qwen3-4B-GGUF @ `bc64014`, Q4_K_M (`tokenizer.ggml.pre = qwen2`) |
| Engines | vLLM 0.32.0 (pip, `--dtype half`), llama.cpp v0.8.0 (`-np 8`), Ollama 0.40.14 |
| Tools | AIPerf 0.13.0, lm-eval 0.4.13, Python 3.11.18 |
| Scale | Chat 512 in / 256 out; 8, 32 and 64 requests at c1, c8 and c32; 1 rep; quality n=20 per task |
| Price | $0 |

Run 2 used the same setup on a different Kaggle machine, at commit `88a53fd`. The differences that matter are in [what each fix changed](#what-each-fix-changed).

## Run 2 vs run 3: speed (dry run, not headline)

c = concurrent requests. One rep per point.

| Point | Run 2 tok/s | Run 3 tok/s | Change | Run 2 TTFT p50 | Run 3 TTFT p50 |
|---|---:|---:|---:|---:|---:|
| vLLM, c1 | 19.6 | 17.2 | -12% | 357 ms | 380 ms |
| vLLM, c8 | 107.2 | 102.9 | -4% | 2.2 s | 2.3 s |
| vLLM, c32 | 186.4 | 176.4 | -5% | 4.1 s | 4.4 s |
| vLLM + ngram, c1 | 22.1 | 21.2 | -4% | 346 ms | 372 ms |
| llama-server, c1 | 45.4 | 41.4 | -9% | 427 ms | 462 ms |
| llama-server, c8 | 124.9 | 111.2 | -11% | 2.1 s | 2.8 s |
| llama-server, c32 | 127.4 | 109.9 | -14% | 47.5 s | 56.5 s |
| Ollama, c1 | 46.3 | 45.4 | -2% | 432 ms | 459 ms |

**Don't read the tok/s changes as a regression.** Every point dropped, including vLLM, which had no prompt-cache change. With one rep on a different shared machine, a change this size can't be attributed to the fixes.

**llama-server c32 TTFT p50 of 56 s is limited by 8 slots.** The server has `-np 8`, so 24 of the 32 concurrent requests wait in a queue. Throughput shows the same limit: c32 (109.9 tok/s) matches c8 (111.2 tok/s). This point measures queueing, not 32 users being served at once.

## Run 2 vs run 3: quality (dry run, not headline)

lm-eval through each server, n=20 per task, greedy, thinking off, 0 empty answers in run 3. Stderr comes from lm-eval; it reports none for the instruction-level IFEval metrics.

| Metric (n=20) | vLLM run 2 | vLLM run 3 | llama-server run 2 | llama-server run 3 |
|---|---:|---:|---:|---:|
| GSM8K, flexible | invalid | 0.70 ± 0.11 | 0.65 ± 0.11 | 0.65 ± 0.11 |
| GSM8K, strict | invalid | 0.40 ± 0.11 | 0.65 ± 0.11 | 0.65 ± 0.11 |
| IFEval, prompt strict | invalid | 0.75 ± 0.10 | 0.75 ± 0.10 | 0.75 ± 0.10 |
| IFEval, prompt loose | invalid | 0.85 ± 0.08 | 0.75 ± 0.10 | 0.80 ± 0.09 |
| IFEval, instruction strict | invalid | 0.83 (no stderr) | 0.83 (no stderr) | 0.83 (no stderr) |
| IFEval, instruction loose | invalid | 0.90 (no stderr) | 0.83 (no stderr) | 0.87 (no stderr) |

- Run 2's vLLM scores are marked invalid because every answer was empty. Its CSV shows 0.00, which isn't a score.
- **At n=20, with stderr around 0.1, these can't rank the engines.** They use different weights too: FP16 on vLLM, Q4_K_M on llama-server.
- **vLLM's GSM8K strict score (0.40) is about answer format.** The strict filter couldn't read 11 of its 20 answers: 7 had no `####`, and 4 wrote something like `#### Answer: $18` or `#### \$460`. It couldn't read 2 of llama-server's. Flexible extraction scores 0.70 and 0.65, so report both on the 5090.

## What each fix changed

| Fix | Run 2 | Run 3 | Status |
|---|---|---|---|
| **Quality parser:** `--reasoning-parser qwen3`, not `deepseek_r1` | All 60 vLLM answers empty; false 0.0 | 0 empty; real scores. The saved `first_response` shows `content: "42"`, `reasoning: null` | Confirmed on Qwen3-4B. Lightning uses `nemotron_v3`, not tested |
| **Prompt cache:** a unique prompt per request, plus `cache_prompt: false` | llama-server long TTFT fell from 5,075 to 957 ms; chat from ~430 to ~140 ms, as repeats hit the cache | Flat: long 5,161–5,480 ms; chat 378–490 ms; Ollama 443–472 ms | Confirmed. TTFT p50 rose as a result (long: 2.9 s to 5.3 s), which is the honest number |
| **Partial status:** runs that lose requests are marked `partial` | llama-server c32 lost 2 of 64 to `ServerDisconnectedError`, marked `ok` | No request failed (64 of 64), so `partial` never triggered | **Not exercised on a GPU**; tested offline only. The disconnect didn't recur and is still undiagnosed |

## Failures that were expected

Both are findings about the configurations, not bugs in the scripts. They matched in runs 2 and 3.

**`vllm-card`, `card` variant:** Qwen3-4B's card command fails on vLLM 0.32.0 with `vllm: error: unrecognized arguments: --enable-reasoning` (exit 2). Without that flag it passes.

**`vllm-bench`, `fp8-kv` variant:** from `logs/vllm-bench__fp8-kv.log`, raised in vLLM's `v1/attention/backends/triton_attn.py`, line 496, while the engine built the model:

> `ValueError: FP8 KV cache is not supported by the Triton attention backend on Tesla T4 (compute capability 7.5); native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype float16.`

So the attention backend refused the T4 because its compute capability is below SM89. The engine then failed to initialize and the server exited with code 1. The fallback, `no-fp8-kv`, passed. This says nothing about the RTX 5090 (SM120), which clears that SM89 limit; the FP8 KV crash reported on SM12x is a separate problem that only the 5090 session can test.

## What this doesn't show

The dry run doesn't touch NVFP4, the `nemotron_v3` parser, the DSpark drafter, vLLM v0.27.1, the Runpod image, or the recipes as written. See the [readiness checklist](../../docs/readiness_checklist.md).

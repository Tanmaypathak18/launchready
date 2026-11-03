# Nemotron 3.5 Lightning: readiness checklist

**Not ready on an RTX 5090 yet: 1 ready, 5 not ready, 21 unknown, out of 27 checks.** Nothing has been measured on a 5090, so no claim about how Lightning runs can be marked ready yet. The Phase 5 bar is that each check is green or its exception is explained.

Template: BUILD_PLAN 7.2. Checked 2026-11-03 using only evidence committed in this repo, including [NVIDIA's card at `bee7596`](../results/evidence/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__README__bee7596.md). Later Kaggle runs aren't included until their results are in the repo.

| Status | Meaning |
|---|---|
| ready | Satisfied now. Only a 5090 measurement can earn this for a claim about running the model; a check that a document exists doesn't need one. |
| not ready | This project confirmed the defect directly in the published file, card or code, or the thing doesn't exist yet. These don't depend on which GPU you use. |
| unknown | Everything else, including anything that looks fine in a header, on T4 or in source, and anything only third parties have reported. |

## 1. Loads in vLLM and Ollama/llama.cpp releases

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 1.1 | Architecture registered in the engines | unknown | `NemotronHForCausalLM` is in vLLM **v0.27.1**, the recipe's pinned version ([`registry.py:182`](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/registry.py#L182), [sparse-clone check](../results/evidence/vllm_registry_nemotron_h.json); same name in the BF16 and NVFP4 configs). [Brief](../briefs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md#engine-readiness): `release` in vLLM v0.32.0, SGLang v0.5.23, llama.cpp v0.8.0, Transformers v5.22.1. Registered isn't the same as loads. | 5090 session |
| 1.2 | NVIDIA's vLLM command starts on a 5090 | unknown | The card lists the 5090 as supported, but its [hardware matrix](../results/evidence/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__README__bee7596.md#L67) says `_no recipe published_`, so the closest command is the DGX Spark one. Third-party reports, not reproduced: FP8 KV crashes at startup on a 5090 laptop on v0.27.1, and works without it ([requirement 2](requirements.md#requirements), [what this changes](requirements.md#what-this-changes-in-launchready)). | 5090 session (`vllm-card`) |
| 1.3 | Benchmark config (BF16 KV) starts and stays up on a 5090 | unknown | Never run. [Matrix](../bench/matrices/rtx5090_lightning.yaml) has fallbacks. The 8-hour soak in [requirement 2(b)](requirements.md#requirements) isn't in the session. | 5090 session; soak **not tested** |
| 1.4 | NVIDIA's llama.cpp recipe works as written | **not ready** | Confirmed: the [full server recipe](../results/evidence/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__README__bee7596.md#L586) asks for `:Q4_K_M`, which doesn't exist on ggml-org, so it exits in 6 s ([audit 3b](friction_log.md#lightnings-llamacpp-recipe-asks-for-a-q4_k_m-file-that-doesnt-exist-quickstart-audit-3b), [log](../results/quickstart_audit/3b_lightning_llama_server.log); Mac, reached the Hub, no model loaded). | NVIDIA fixes the card (issue not filed yet) |
| 1.5 | llama-server loads and serves a Lightning GGUF on a 5090 | unknown | No Lightning GGUF has been loaded by this project. llama.cpp v0.8.0 built and served Qwen3-4B on T4 ([dry run](../results/dryrun_kaggle/README.md)). | 5090 session |
| 1.6 | Ollama loads and serves it on a 5090 | unknown | Not run on Lightning. Reported single-slot limit ([ollama#18938](https://github.com/ollama/ollama/issues/18938)). Ollama 0.40.14 served Qwen3-4B on T4 ([dry run](../results/dryrun_kaggle/README.md)). | 5090 session |
| 1.7 | DSpark drafter runs on a 5090 | unknown | Not run. Reported on BF16 + MTP: 0.2–1% of requests turn to word salad ([vllm#55894](https://github.com/vllm-project/vllm/issues/55894)). The session measures speed only, not that output. | 5090 session (speed); output **not tested** |

## 2. Chat template and tool calling

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 2.1 | System prompt and multi-turn are correct | unknown | Never tested. All four GGUFs embed a chat template (headers: [bartowski](../results/bench/gguf/bartowski__Lightning-Q4_K_M.json), [unsloth](../results/bench/gguf/unsloth__Lightning-UD-Q4_K_M.json), [AtomicChat](../results/bench/gguf/AtomicChat__Lightning-Q4_K_M.json), [ggml-org](../results/bench/gguf/ggml-org__Lightning-Q4_0.json)), but that only shows it exists. | **Not tested** |
| 2.2 | With thinking off, the answer arrives in `content` | unknown | `nemotron_v3` looks right from reading vLLM v0.27.1 source; not run ([friction log](friction_log.md#dry-run-2-vllm-quality-scored-exactly-00-because-every-answer-came-back-in-reasoning-not-content)). The Qwen3-4B null-content fix hasn't been re-run in the repo's results either. | 5090 session (startup check fails if not) |
| 2.3 | Tool calling works | unknown | Never tested. Reported on Lightning: `qwen3_coder` drops trailing newlines in string arguments, and multi-turn harnesses truncate ([below the cut](requirements.md#below-the-cut)). | **Not tested** |

## 3. Each precision verified on its target GPU; memory within tolerance

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 3.1 | NVFP4 on a 5090, memory matches projection | unknown | Projected only: 20.08 GiB weights, 0.28 GiB per 40,960-token sequence, ~18 sequences ([recipe README](../recipes/nemotron-3.5-lightning/README.md#memory-on-a-32-gib-rtx-5090-projected)). Hybrids can land far below projection: granite-4.0-h-tiny measured 50.6 against 118 ([friction log](friction_log.md#granite-40-h-tiny-serves-23-fewer-users-in-vllm-than-projected)). | 5090 session |
| 3.2 | FP8 KV cache on SM120 | unknown | Reported to crash ([requirement 2(c)](requirements.md#requirements)). T4's failure is a different cause: Turing has no FP8 ([friction log](friction_log.md#dry-run-fp8-kv-cache-fails-on-t4-expected-turing-has-no-fp8)). | 5090 session (`vllm-card`) |
| 3.3 | GGUF on a 5090, memory matches projection | unknown | Projected only: 23.73 GiB file, ~6.0 GiB left with 8 slots ([ollama.md](../recipes/nemotron-3.5-lightning/ollama.md#memory-projected)). | 5090 session |
| 3.4 | GGUF file names match their contents | **not ready** | Confirmed from headers: [bartowski](../results/bench/gguf/bartowski__Lightning-Q4_K_M.json), [unsloth](../results/bench/gguf/unsloth__Lightning-UD-Q4_K_M.json), [AtomicChat](../results/bench/gguf/AtomicChat__Lightning-Q4_K_M.json), [ggml-org](../results/bench/gguf/ggml-org__Lightning-Q4_0.json): "Q4_K_M" files store 6.08–6.19 bits/weight, not ~4.85. Bartowski's bytes are 63% Q5_0 and 33% Q8_0 ([friction log](friction_log.md#every-lightning-q4_k_m-gguf-is-62-bitsweight-and-tagged-with-the-wrong-pre-tokenizer), [requirement 1(c)](requirements.md#requirements)). | GGUF publishers; llama.cpp |
| 3.5 | `launchready fit` GGUF projection is right | **not ready** | Confirmed: still projects 18.58 GiB at 4.85 bits/weight, 22% under the 23.73 GiB file ([what this changes, point 1](requirements.md#what-this-changes-in-launchready), `launchready/memory.py`). The brief and recipes use real file sizes instead. | Read GGUF headers in `fit` (not scheduled) |
| 3.6 | `tokenizer.ggml.pre` is correct | **not ready** | Confirmed: all four files say `pixtral`, not `tekken` ([friction log](friction_log.md#every-lightning-q4_k_m-gguf-is-62-bitsweight-and-tagged-with-the-wrong-pre-tokenizer)). llama.cpp v0.8.0 should map it correctly (PR #30960), but `llama-tokenize` against Hugging Face ([requirement 1(b)](requirements.md#requirements)) hasn't been run, and Ollama's bundled llama.cpp is unchecked. | Re-tagged GGUFs; tokenizer comparison **not tested** |
| 3.7 | Per-layer metadata agrees with tensors | unknown | Passes in all four headers, with no zeroed-block defect ([friction log](friction_log.md#every-lightning-q4_k_m-gguf-is-62-bitsweight-and-tagged-with-the-wrong-pre-tokenizer), [requirement 1(a)](requirements.md#requirements)). A header check doesn't prove it loads. | 5090 session |
| 3.8 | Which compute path NVFP4 uses on a 5090 | unknown | The [card](../results/evidence/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__README__bee7596.md#L67) says `_confirm_` for the 5090. On DGX Spark, `marlin` runs W4A16, not FP4 tensor cores, and our config copies `marlin`. So "NVFP4 native on Blackwell" may not describe this run. | 5090 session (if vLLM's log shows it) |

## 4. Quantized quality within an agreed gap to BF16

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 4.1 | NVFP4 quality measured | unknown | Not run. BF16 at ~61 GiB [doesn't fit a 5090](../briefs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md#hardware-requirements), and [requirement 4](requirements.md#requirements) reports kernels that give fluent wrong output. The [dry run](../results/dryrun_kaggle/README.md)'s vLLM 0.0 is invalid ([friction log](friction_log.md#dry-run-2-vllm-quality-scored-exactly-00-because-every-answer-came-back-in-reasoning-not-content)). | 5090 session |
| 4.2 | GGUF quality measured | unknown | Not run. Dry-run scores are Qwen3-4B on 20 examples. | 5090 session |
| 4.3 | The gap is agreed | **not ready** | No threshold exists. [NVIDIA's accuracy table](../results/evidence/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__README__bee7596.md#L138) has neither GSM8K nor IFEval, so the session compares NVFP4 with GGUF, not with BF16. | Your decision: set a threshold and baseline |

## 5. Benchmarks reproducible, environment recorded

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 5.1 | Lightning benchmarks exist | unknown | None run. | 5090 session |
| 5.2 | Scripts reproduce a run with its environment | unknown | Worked end to end for Qwen3-4B on T4 ([dry run](../results/dryrun_kaggle/README.md)), with the environment on every row. Fixes since then ([unique prompts](friction_log.md#dry-run-2-prefix-caching-off-wasnt-true-for-llama-server-or-ollama), [failed-request labels](friction_log.md#dry-run-2-llama-server-at-32-users-lost-2-of-64-requests-but-the-run-said-ok)) are tested offline only. | 5090 session |

## 6. Recipes copy-paste runnable on a fresh machine

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 6.1 | `vllm.sh` runs as written | unknown | Never run anywhere. The session's `recipe-vllm` step will run it unedited, with `RUNNER=local` because a pod can't run Docker, so the Docker line stays untested. | 5090 session (`recipe-vllm`) |
| 6.2 | `ollama.md` runs as written | unknown | Never run on Lightning. The session's `recipe-ollama` and `recipe-llama-server` steps will run the Ollama and `llama-server` commands; llama.cpp itself comes from `bench/build_llamacpp.sh`, not the build lines in the recipe. | 5090 session (`recipe-*`) |
| 6.3 | Rental checklist works | unknown | [Written from Runpod's docs](../recipes/nemotron-3.5-lightning/README.md#rent-the-gpu-and-run-the-session-click-by-click), not tried. Start command, web terminal and the llama.cpp build on the vLLM image are unverified. | 5090 session |

## 7. Recipe README includes top-5 troubleshooting

| # | Check | Status | Evidence | What closes it |
|---|---|---|---|---|
| 7.1 | Top-5 section exists | **ready** | [Troubleshooting (top 5)](../recipes/nemotron-3.5-lightning/README.md#troubleshooting-top-5). This is a document check; the fixes in it come from third-party reports until the 5090 session tests them. | Done |

## Not tested by the 5090 session

These stay unknown after the session, as explained exceptions:

- **Chat template (2.1):** startup checks send single user messages, with no system prompt.
- **Tool calling (2.3):** no tools are sent.
- **Soak (1.3):** no 8-hour run.
- **Tokenizer (3.6):** no `llama-tokenize` comparison with Hugging Face.
- **Drafter output (1.7):** speed is measured; the text isn't checked.

Separately, the quality check (4.3) can't pass until someone sets a threshold.

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

### Dry run 3: nothing broke; the fixes held, and `partial` never triggered
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 (1 used) · driver 580.178.04 · CUDA 13.0 · launchready 8b4ca32 · session dryrun_kaggle_qwen3_4b_20261103T2220Z · vLLM 0.32.0 · llama.cpp v0.8.0 · Ollama 0.40.14 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `MATRIX=bench/matrices/kaggle_qwen3_4b.yaml HOURLY_PRICE=0 MAX_HOURS=3 bash bench/run_session_5090.sh` (notebook 04)
- What happened: all 6 stages exited 0; 10 of 10 runs `ok`, with no `errors/` folder. vLLM quality is real (0 empty answers; the saved `first_response` shows `content: "42"`). TTFT is flat per request on llama-server and Ollama. The 2 lost requests at llama-server c32 didn't recur (64 of 64). The llama.cpp build took 1,843 s, against 1,654 s in run 2.
- Root cause: n/a
- Fix or workaround: none needed. `partial` hasn't run on a GPU, since no request failed; it's tested offline only. The disconnect from run 2 is still undiagnosed.
- Severity: cosmetic
- Filed: not filed
- Requirement this implies: none.

### Dry run 3 finding: `vllm-bench-spec` log has an `EngineDeadError` traceback at shutdown
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 (1 used) · driver 580.178.04 · CUDA 13.0 · launchready 8b4ca32 · session dryrun_kaggle_qwen3_4b_20261103T2220Z · vLLM 0.32.0
- Exact command: `vllm serve Qwen/Qwen3-4B ... --speculative-config '{"method": "ngram", ...}' ...`, stopped by `Server.stop()`
- What happened (error excerpt): `AsyncLLM output_handler failed.` · `vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.` at 23:18:39, right after `[shutdown] EngineCore: trigger received signal=SIGTERM` and `force killing remaining process EngineCore`. Run 2's log for this server had no traceback.
- Root cause: our SIGTERM. The run ended at the same second (started 23:16:25, 133.7 s), with 8 of 8 requests and 0 errors, so it's shutdown noise, not a failed run. Why it appeared in only one of the two runs isn't known.
- Fix or workaround: none needed
- Severity: cosmetic
- Filed: not filed: happens only as we stop the server
- Requirement this implies: when scanning logs, compare a traceback's time with the run's end before calling it a failure.

### Dry run 3 finding: GSM8K strict-match measures answer format as much as the answer
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 (1 used) · driver 580.178.04 · CUDA 13.0 · launchready 8b4ca32 · session dryrun_kaggle_qwen3_4b_20261103T2220Z · lm-eval 0.4.13
- Exact command: `lm_eval --model local-chat-completions ... --tasks gsm8k --limit 20`
- What happened: vLLM scored 0.40 strict and 0.70 flexible; llama-server scored 0.65 on both. The strict filter couldn't read 11 of vLLM's 20 answers (7 with no `####`, 4 like `#### Answer: $18` or `#### \$460`) and 2 of llama-server's.
- Root cause: `strict-match` needs a bare number right after `####`. Why the two engines' answers differ in format isn't established: they use different weights (FP16 and Q4_K_M), and n=20.
- Fix or workaround: none in code; `quality_summary.csv` already keeps both filters. Report both, and don't compare engines on strict alone.
- Severity: slows (a misleading headline if strict is quoted alone)
- Filed: not filed: expected filter behavior
- Requirement this implies: name the GSM8K filter next to every score.

### Dry run 3 finding: FP8 KV cache fails on T4, same as run 2
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 (1 used) · driver 580.178.04 · CUDA 13.0 · launchready 8b4ca32 · session dryrun_kaggle_qwen3_4b_20261103T2220Z · vLLM 0.32.0 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `vllm serve Qwen/Qwen3-4B --revision 1cfa9a7... --served-model-name qwen3-4b --dtype half --kv-cache-dtype fp8 --max-model-len 8192 --no-enable-prefix-caching --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`
- What happened (error excerpt): raised at `vllm/v1/attention/backends/triton_attn.py`, line 496, while building the model: `ValueError: FP8 KV cache is not supported by the Triton attention backend on Tesla T4 (compute capability 7.5); native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype float16.` Then `RuntimeError: Engine core initialization failed`, and exit 1 within 50 s.
- Root cause: this is a finding, not a bug. vLLM's Triton attention backend needs SM89+ for FP8; T4 is SM75.
- Fix or workaround: the matrix fell back to `no-fp8-kv`, which passed.
- Severity: cosmetic (expected)
- Filed: not filed: expected behavior, with a clear error
- Requirement this implies: none. This limit doesn't apply to the 5090 (SM120), so it says nothing about the FP8 KV crash reported on SM12x.

### Dry run 3 finding: Qwen3-4B's card command still fails on `--enable-reasoning`
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 (1 used) · driver 580.178.04 · CUDA 13.0 · launchready 8b4ca32 · session dryrun_kaggle_qwen3_4b_20261103T2220Z · vLLM 0.32.0 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `vllm serve Qwen/Qwen3-4B --revision 1cfa9a7... --enable-reasoning --reasoning-parser deepseek_r1 --dtype half --served-model-name qwen3-4b --max-model-len 8192`
- What happened (error excerpt): `vllm: error: unrecognized arguments: --enable-reasoning`, exit 2, third reproduction after audit 1b and run 2. Without the flag, the same line passed in 195 s.
- Root cause: this is a finding, not a bug in the kit. The card's flag no longer exists in vLLM 0.32.0.
- Fix or workaround: the matrix records the failure and goes on to `card-no-enable-reasoning`.
- Severity: blocks (card command as written)
- Filed: not filed: still the owner's call (see audit 1b)
- Requirement this implies: model-card commands name a tested engine version, not just a minimum.

### `hub_cache` never blocked the network, despite its docstring
- When / where: 2026-11-03 · no GPU · `tests/conftest.py` on branch `space-app`
- Exact command: `uv run pytest`
- What happened (error excerpt): nothing failed, and that was the problem. `hub_cache` set `HF_HUB_OFFLINE` after import, so a test that missed the cache called the Hub in silence (see the next entry).
- Root cause: huggingface_hub reads the flag at import, and CI's "no network" rule was only a convention.
- Fix or workaround: an autouse `no_network` fixture refuses sockets, DNS and `git`, loopback included, and fails the test at teardown even if the code swallowed the error. Removing the Space's variants stub now fails those tests instead of calling the Hub. The next entry's warning about `hub_cache` no longer applies.
- Severity: slows (a hidden dependency on the network and on live Hub data)
- Filed: not filed: our test setup
- Requirement this implies: rules the suite depends on are enforced by the suite.

### Space tests reached the live Hub, and the browser check found bugs the tests missed
- When / where: 2026-11-03 · no GPU · gradio 6.31.0 · `tests/test_space.py` and `space/` on branch `space-app`. Written after the fixes, not before: the rule was missed here.
- Exact command: `uv run pytest`; `python space/app.py`, then the browser pane at phone width
- What happened (error excerpt): the `hub_cache` fixture sets `HF_HUB_OFFLINE` after huggingface_hub is imported, where it has no effect, so `load_variants` called the Hub and wrote three `__variants.json` files into `tests/fixtures/hub`. In the browser: `vLLM&#x27;s` shown literally, a page-tall Model box, table cells broken mid-word on a phone.
- Root cause: the offline flag is read at import. `esc()` escaped the `#` inside the entity `html.escape` had made. Gradio defaults to a tall textbox and `overflow-wrap: anywhere` in tables.
- Fix or workaround: Space tests stub `load_variants`, and the suite passes with sockets blocked; `html.escape(quote=False)`; one-line textbox; CSS that keeps words whole and scrolls tables. `hub_cache` still doesn't block the network, so any new test that misses the cache can reach the Hub.
- Severity: slows (the test network use would fail in CI; the layout made the phone page unreadable)
- Filed: not filed: our code
- Requirement this implies: offline tests need a real network block, and front ends get checked at phone width.

### Hub errors other than gated or not-found reach the user as raw exceptions
- When / where: 2026-11-03 · no GPU · huggingface_hub 2.4.0 · launchready 0.0.1 `hub._fetch` · bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
- Exact command: `hub._fetch("bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", None)`, the call behind `launchready fit`
- What happened (error excerpt): `RemoteEntryNotFoundError 404 Client Error`: a GGUF-only repo has no config.json, and only `GatedRepoError` and `RepositoryNotFoundError` are caught. A 429 or other Hub HTTP error would escape the same way.
- Root cause: `HubError` promises "gated, missing, or unreachable", but `_fetch` only wraps the first two cases.
- Fix or workaround: wrap a missing config.json and Hub HTTP errors in `HubError`; add `GatedError(HubError)` so a front end without a token can word that case itself. The CLI's gated text doesn't change.
- Severity: slows (a traceback instead of a message, on repos people will paste)
- Filed: not filed: our code
- Requirement this implies: every Hub failure gives a message that says what to do next.

### Variants lookup crashes on repos with no org
- When / where: 2026-11-03 · no GPU · launchready 0.0.1 `variants._fetch` · gpt2
- Exact command: `load_variants("gpt2")`
- What happened (error excerpt): `ValueError: not enough values to unpack`: `builder, name = repo_id.split("/")` needs an org, and canonical repos such as gpt2 have none.
- Root cause: `_fetch` assumes every repo id is `org/name`.
- Fix or workaround: the Space treats a failed variants lookup as non-fatal, warns, and uses projected sizes. `launchready brief` keeps its current behavior.
- Severity: cosmetic for the CLI; blocks the Space without the workaround
- Filed: not filed: our code
- Requirement this implies: optional lookups never block the main answer.

### Tesla T4 catalog capacity is 1 GiB above what we measured
- When / where: 2026-11-03 · Tesla T4 · driver 580.178.04 · CUDA 13.0 · `launchready/data/gpus.yaml`
- Exact command: compare `gpus.yaml` with `vram_mib` in `results/environment.json`
- What happened (error excerpt): the catalog says 16 GiB; the Kaggle T4 reports 15,360 MiB, which is 15 GiB. `results/memory_validation.csv` used the real 15 GiB (`vram_bytes` 16106127360), so the published comparison is not affected, but `fit --gpu tesla-t4` overstates room by 0.9 GiB.
- Root cause: the marketed "16 GB" was read as GiB, as for every entry, and Phase 3's calibration note was never applied to the catalog.
- Fix or workaround: set the T4 to 15 GiB, from the measurement. Other entries stay marketed capacity read as GiB.
- Severity: cosmetic for the validation; slows anyone sizing tightly on a T4
- Filed: not filed: our data
- Requirement this implies: catalog values that have a measurement use the measurement.

### Gap check said the card has "no RTX 5090 row"; it has one
- When / where: 2026-11-03 · no GPU · NVFP4 card @ bee7596, saved verbatim in `results/evidence/`
- Exact command: `curl -sfL https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/resolve/bee7596271d1495f6992ae224aefde4410e816b8/README.md`
- What happened: the recipe README and checklist row 1.2 said the card has no 5090 row. Line 67 has one: `_confirm_` compute path, `_no recipe published_`, `Hardware: yes`. The card also lists the 5090 as supported hardware and under Test Hardware, and shows `marlin` on DGX Spark running W4A16, not FP4 tensor cores. The other gap-check claims matched the saved card.
- Root cause: ours. The gap check came from a summarized web fetch, and the card wasn't saved in the repo. `results/quickstart_audit/card_README.md` is Qwen3-4B's card.
- Fix or workaround: card saved byte for byte; the claim is corrected and linked to the line; new checklist row 3.8 tracks the 5090's compute path, since our config copies `marlin`.
- Severity: cosmetic (no number depended on it; the conclusion, no 5090 vLLM recipe, still holds)
- Filed: not filed: our doc
- Requirement this implies: quote primary sources from a saved copy, not a summary.

### Dry run 2: vLLM quality scored exactly 0.0 because every answer came back in `reasoning`, not `content`
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 88a53fd · session dryrun_kaggle_qwen3_4b_20261103T2043Z · vLLM 0.32.0 · lm-eval 0.4.13 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `vllm serve Qwen/Qwen3-4B ... --reasoning-parser deepseek_r1 --default-chat-template-kwargs '{"enable_thinking": false}'`, then `lm_eval --model local-chat-completions ... --tasks gsm8k --limit 20` (same for ifeval)
- What happened (error excerpt): vLLM gsm8k 0.0, ifeval 0.0; llama-server 0.65 and 0.75 on identical prompts. All 60 vLLM samples have `resps: [[""]]` (`filtered_resps: "[invalid]"`), and lm-eval logged, for every request, `API returned null content. Content filled with LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = . Check reasoning_content field or generation limits.`
- Root cause: `deepseek_r1` doesn't handle thinking off. With `enable_thinking=false`, Qwen3's template ends the prompt with `<think>\n\n</think>\n\n`, so the model's output has no `</think>`. vLLM's `BaseThinkingReasoningParser.extract_reasoning` (`vllm/reasoning/basic_parsers.py`, read at v0.32.0) then does `if self.end_token not in model_output: return model_output, None`: the whole answer becomes `reasoning` and `content` is null. Streaming isn't affected, because the serving layer sets `reasoning_ended` from the prompt tokens; that's why AIPerf (streaming) saw 256 normal output tokens. The raw HTTP response wasn't recorded; this rests on the samples, the lm-eval warning and the source.
- Why we missed it: two bugs of ours. `Server.chat()` joined `content` + `reasoning`, so the startup check passed on "42" and "Paris." with content actually null. `quality.py` reported 60 empty answers as status `ok`, score 0.0.
- Fix or workaround: Kaggle matrix uses `--reasoning-parser qwen3`, which returns `(None, model_output)` when `enable_thinking` is false. The startup check now requires non-empty `content` when thinking is off, and saves the first raw response. `quality.py` counts empty answers; above 5% the task is `invalid`, with no score in the CSV. Confirmed by dry run 3: 0 empty answers and real scores. The startup and `quality.py` guards didn't need to fire there, so they're tested offline only.
- Lightning: the 5090 matrix uses `nemotron_v3`, not `deepseek_r1`. By reading v0.27.1 (not running it): the server default reaches `NemotronV3Parser` through `chat_template_kwargs`, `Qwen3Parser.extract_reasoning` returns `(None, model_output)` with thinking off, and `--default-chat-template-kwargs` exists there. An attempt to run vLLM's parsers locally was blocked (it would execute downloaded source). If this reading is wrong, the new startup check fails in the first 8 requests, before any benchmark.
- Severity: wrong output (a false 0.0 with status ok)
- Filed: not filed: `deepseek_r1` isn't Qwen3's parser; our matrix copied it from the card line
- Requirement this implies: check that the answer is in `content`, not just that the server answered; a quality score from empty answers is not a score.

### Dry run 2: "prefix caching off" wasn't true for llama-server or Ollama
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 88a53fd · session dryrun_kaggle_qwen3_4b_20261103T2043Z · llama.cpp v0.8.0 · Ollama 0.40.14
- Exact command: the AIPerf runs in `bench/matrices/kaggle_qwen3_4b.yaml`
- What happened: per-request TTFT is bimodal. llama-server long: 4,871, 5,075, then 991, 957 ms. llama-server chat: ~430 ms, last two 142, 140. Ollama chat: ~430, last two 188, 186. vLLM long stayed flat at ~16,350 ms. The llama-server log shows `selected slot by LCP similarity, f_sim_best = 1.000` and `n_past = 524`.
- Root cause: AIPerf generated as many prompts as `--request-count`, used the first ones for warmup, then repeated them at the end of the run; llama-server's `cache_prompt` (on by default) reused them. Our label "off (no prompt reuse...; synthetic prompts)" was wrong. It flatters TTFT p50 and p99 for GGUF against vLLM.
- Fix or workaround: `--num-dataset-entries` = requests + warmup, so every prompt is unique; llama-server also gets `cache_prompt: false` per request. Ollama has no equivalent setting here, so unique prompts are what protect it. Confirmed by dry run 3: TTFT is flat per request.
- Severity: wrong output (comparison skewed in GGUF's favor)
- Filed: not filed: our setup
- Requirement this implies: "cache off" has to be shown per engine; look at per-request TTFT, not only percentiles.

### Dry run 2: llama-server at 32 users lost 2 of 64 requests but the run said `ok`
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 88a53fd · session dryrun_kaggle_qwen3_4b_20261103T2043Z · llama.cpp v0.8.0
- Exact command: AIPerf `--concurrency 32 --request-count 64` against `llama-server -np 8`
- What happened (error excerpt): `ServerDisconnectedError('Server disconnected')` × 2; error rate 3.125%; `status: ok`; `requests_completed` said 64 while only 62 succeeded.
- Root cause: ours for the labeling (status ignored AIPerf's errors, and `completed_request_count` includes failures). The disconnect itself is **not diagnosed**: nothing in the server log explains it, and this point deliberately queues 24 requests past 8 slots.
- Fix or workaround: runs with any failed request get status `partial` and a note naming the error and count, and `requests_completed` counts successes only. Headline rows keep them, and the note stays on the run. Watch for it at 32 users on the 5090.
- Severity: slows
- Filed: not filed: cause unknown
- Requirement this implies: throughput numbers carry their error count.

### Dry run 2: fixes from dry run 1 confirmed, and Ollama's output was all reasoning
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 88a53fd · session dryrun_kaggle_qwen3_4b_20261103T2043Z · llama.cpp v0.8.0 · Ollama 0.40.14
- What happened: all servers ran, 10 of 10 runs finished, the GPU sampler recorded memory and power, and the two expected failures failed as expected. The llama.cpp build linked `/usr/local/nvidia/lib64/libcuda.so.1`, confirming no driver stub is on Kaggle, and took 1,654 s (27.5 min). `ollama serve` started under the matrix with 37/37 layers on the GPU.
- Note for the 5090: Ollama ran with thinking on, so all 256 tokens per request were reasoning (AIPerf `reasoning_token_count` 256, no `output_token_count`). Throughput is still valid; the text isn't comparable to the other engines.
- Severity: cosmetic
- Filed: not filed
- Requirement this implies: none.

### Dry run: llama.cpp v0.8.0 CUDA build fails with "CUDA::cuda_driver ... target was not found"
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z · llama.cpp v0.8.0 (8642541) · nvcc 13.0.88
- Exact command: `bash bench/build_llamacpp.sh v0.8.0 /tmp/llama.cpp` (prepare-3)
- What happened (error excerpt): `CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:181 (target_link_libraries): Target "ggml-cuda" links to: CUDA::cuda_driver but the target was not found.` CMake did find the toolkit (`Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include ... 13.0.88`).
- Root cause: llama.cpp links `CUDA::cuda_driver` (libcuda.so, for virtual-memory pools) unless `GGML_CUDA_NO_VMM` is on. CMake's FindCUDAToolkit only creates that target when it finds a library named `libcuda.so` in the toolkit's lib dirs or their `stubs/` subdirectory. Kaggle's image has the toolkit but no `libcuda.so` there: the stub ships in the `cuda-driver-dev` package, and the driver itself provides only `libcuda.so.1`. Confirmed by dry run 2, which found only the driver's `/usr/local/nvidia/lib64/libcuda.so.1`.
- Fix or workaround: `build_llamacpp.sh` now looks for libcuda (toolkit `stubs/`, then the driver's `libcuda.so`/`libcuda.so.1` via ldconfig), installs `cuda-driver-dev-<ver>` if none is found, passes the path as `-DCUDA_cuda_driver_LIBRARY=...`, and only as a last resort builds with `-DGGML_CUDA_NO_VMM=ON` (recorded in `build/LAUNCHREADY_BUILD.txt`). It logs `cmake --version` and the libcuda it used. Verified by dry run 2 on Kaggle (it linked the driver's `libcuda.so.1`); still unverified on the Runpod vLLM image.
- Severity: blocks (no llama-server, so no concurrent GGUF points)
- Filed: not filed: environment packaging, not a llama.cpp bug
- Requirement this implies: build scripts must not assume a full CUDA devel image; check for the driver stub and say which libcuda was linked.

### Dry run: one missing directory aborted the whole matrix
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z · vLLM 0.32.0
- Exact command: `uv run python -m bench.run_matrix --matrix bench/matrices/kaggle_qwen3_4b.yaml --hourly-price 0 --with-quality --deadline ...`
- What happened (error excerpt): `FileNotFoundError: [Errno 2] No such file or directory: 'results_dryrun/bench/dryrun_kaggle_qwen3_4b_20261103T1948Z/gpu/vllm-bench__no-fp8-kv__chat__c1__r1.csv'` in `GpuSampler.__enter__`, on the first benchmark run. The matrix exited 1, so no run, no quality task, and no llama.cpp, spec-decode or Ollama server was tried.
- Root cause: our bug. `GpuSampler` opened `<session>/gpu/<run>.csv` without creating `gpu/`. The mock smoke test never reached that line because the Mac has no `nvidia-smi`, and nothing caught per-run errors.
- Fix or workaround: the sampler creates parent directories; each run is wrapped so an unexpected error becomes a `failed` row (JSON + summary.csv) and the matrix continues; each server is wrapped the same way. New offline test puts a fake `nvidia-smi` on PATH so the sampler runs in CI.
- Severity: blocks
- Filed: not filed: our code
- Requirement this implies: test the GPU-only code paths offline with fakes; one failure must never cost the rest of a paid session.

### Dry run: notebook 04's check cell crashed on a missing summary.csv, so the zip cell never ran
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z
- Exact command: notebook 04, "Check the outputs" cell (Run all)
- What happened: `pd.read_csv("results_dryrun/summary.csv")` raised because no run finished; "Run all" stopped there, so `launchready_dryrun.zip` was never made and the whole `/kaggle/working` (repo, `.git`, `.venv`) was downloaded instead.
- Root cause: our notebook assumed success.
- Fix or workaround: zip first, then check; the check cell lists what exists and reads each file only if present.
- Severity: slows
- Filed: not filed: our notebook
- Requirement this implies: result notebooks report partial results instead of failing on them.

### Dry run: Ollama install says "systemd is not running" and "Unable to detect NVIDIA/AMD GPU"
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z · Ollama 0.40.14
- Exact command: `bash bench/install_ollama.sh 0.40.14` (prepare-4)
- What happened: `WARNING: systemd is not running` · `WARNING: Unable to detect NVIDIA/AMD GPU. Install lspci or lshw to automatically detect and install GPU dependencies.` · `Warning: could not connect to a running Ollama instance`. The install itself succeeded (exit 0).
- Root cause: containers (Kaggle, Runpod) have no systemd, so the installer can't start its service; it looks for GPUs with lspci/lshw, which aren't installed. The Ollama tarball carries its CUDA libraries, so the GPU warning only skips driver installation.
- Fix or workaround: none needed for systemd: the matrix starts `ollama serve` itself as the server process (and stops it). `install_ollama.sh` now installs `pciutils` first, and, if systemd *is* running, stops and disables the `ollama` service so it can't hold port 11434 in place of the matrix's server. Ollama didn't run in this dry run (the matrix crashed first); dry run 2 confirmed it.
- Severity: cosmetic
- Filed: not filed
- Requirement this implies: none.

### Dry run: FP8 KV cache fails on T4 (expected; Turing has no FP8)
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z · vLLM 0.32.0 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `vllm serve Qwen/Qwen3-4B --revision 1cfa9a7... --served-model-name qwen3-4b --dtype half --kv-cache-dtype fp8 --max-model-len 8192 --no-enable-prefix-caching --reasoning-parser deepseek_r1 --default-chat-template-kwargs '{"enable_thinking": false}'`
- What happened (error excerpt): `ValueError: FP8 KV cache is not supported by the Triton attention backend on Tesla T4 (compute capability 7.5); native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype float16.` (55 s, exit 1)
- Root cause: hardware. Turing (SM75) has no FP8; vLLM's error is clear and names the fix.
- Fix or workaround: none needed. This variant exists to exercise the fallback, and it worked: `no-fp8-kv` started in 90 s, passed the 8-request check, answered without thinking ("42", "Paris."), so `--default-chat-template-kwargs` works on 0.32.0, and logged KV capacity 36,688 tokens / 4.48× concurrency at 8,192 tokens, weights 7.56 GiB.
- Severity: cosmetic (expected)
- Filed: not filed: expected behavior
- Requirement this implies: matches requirement 2(d): failures on unsupported GPUs should name the GPU and the alternative, as this one does.

### Dry run reproduced audit 1b: Qwen3-4B's card line fails on `--enable-reasoning` in vLLM 0.32.0
- When / where: 2026-11-03 · Kaggle 2× Tesla T4 · driver 580.178.04 · CUDA 13.0 · launchready 654f459 · session dryrun_kaggle_qwen3_4b_20261103T1948Z · vLLM 0.32.0 · Qwen/Qwen3-4B @ 1cfa9a7
- Exact command: `vllm serve Qwen/Qwen3-4B --revision 1cfa9a7... --enable-reasoning --reasoning-parser deepseek_r1 --dtype half --served-model-name qwen3-4b --max-model-len 8192`
- What happened (error excerpt): `vllm: error: unrecognized arguments: --enable-reasoning` (exit 2 within 35 s). Same failure as the Phase 3 audit 1b entry below. Without the flag, the same line started in 200 s and passed the 8-request check (thinking on, as the card intends).
- Root cause: vLLM removed `--enable-reasoning`; `--reasoning-parser` alone enables it. The card still says `vllm>=0.8.5`.
- Fix or workaround: drop `--enable-reasoning`.
- Severity: blocks (card command as written)
- Filed: not filed: still the owner's call (see audit 1b)
- Requirement this implies: model-card commands name a tested engine version, not just a minimum.

### Every Lightning "Q4_K_M" GGUF is ~6.2 bits/weight and tagged with the wrong pre-tokenizer
- When / where: 2026-11-03 · no GPU (GGUF headers read over HTTP range requests; no weights downloaded) · `bench/gguf_check.py` · bartowski @ f0eec22, unsloth @ f2d3fe3 (UD-Q4_K_M), AtomicChat @ 14571e1, ggml-org @ 8cd0821 (Q4_0)
- Exact command: `uv run python -m bench.gguf_check hf://bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF@f0eec22…/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf` (and the same for the other three)
- What happened: bartowski Q4_K_M = 6.19 bits/weight (Q5_0 63% of bytes, Q8_0 33%, Q4_K 0.3%); unsloth UD-Q4_K_M = 6.14 (Q5_0/Q5_1/Q8_0); AtomicChat Q4_K_M = 6.08; ggml-org Q4_0 = 4.79. All four: `tokenizer.ggml.pre = pixtral`. Per-layer `head_count_kv` matched the attn/ssm tensors in all four (no llama.cpp#27718-style zeroed blocks).
- Root cause: expert rows (1856, 2688) aren't divisible by 256, so K-quants fall back to Q5_0 (unsloth GGUF #1); the converter's tokenizer hash matches Pixtral's (llama.cpp#30894). `launchready fit` assumes 4.85 bits/weight for Q4_K_M and projects 18.58 GiB, 22% under the 23.73 GiB file.
- Fix or workaround: GGUF memory in the recipes comes from the file size and the header, not the 6.1 table; llama.cpp pinned to v0.8.0, which includes PR #30960 (pixtral → Tekken regex); `tokenizer.ggml.pre` is recorded with every GGUF result.
- Severity: wrong output (tokenizer, non-Latin scripts) + slows (memory projection)
- Filed: not filed: both defects are already tracked (unsloth GGUF #1, llama.cpp#30894, fixed in #30960)
- Requirement this implies: requirement 1(b)–(c): GGUF names and cards state actual tensor types and bits/weight; the calculator should read GGUF headers before projecting GGUF memory.

### The roadmap's AIPerf example uses a flag AIPerf 0.13.0 doesn't have
- When / where: 2026-11-03 · MacBook Air (no GPU) · aiperf 0.13.0 via `uvx --python 3.12` · against `bench/mock_server.py`
- Exact command: `aiperf profile --help` (roadmap 6.9 example uses `--warmup-requests 10`)
- What happened: 0.13.0 has `--warmup-request-count` (alias `--num-warmup-requests`); `--isl`/`--osl` exist as aliases of `--prompt-input-tokens-mean`/`--prompt-output-tokens-mean`; `--osl` only caps `max_completion_tokens`, so outputs can be shorter unless the server honors `ignore_eos`.
- Root cause: the plan's example predates the pinned version.
- Fix or workaround: `bench/run_matrix.py` builds the command from flags checked against `aiperf profile --help` for 0.13.0, adds `--isl-stddev 0 --osl-stddev 0`, sends `ignore_eos` to vLLM and llama-server, and flags any run whose mean output length is more than 10% off the request (Ollama ignores `ignore_eos`).
- Severity: cosmetic (caught before any GPU time)
- Filed: not filed: our plan
- Requirement this implies: pin the benchmark tool and check its flags at that version; record actual output lengths with every throughput number.

### `run_session_5090.sh` needs GNU `timeout`, which macOS doesn't ship
- When / where: 2026-11-03 · MacBook Air · testing the session script on the mock matrix
- Exact command: `BENCH_ALLOW_NO_GPU=1 MATRIX=bench/matrices/mock_smoke.yaml HOURLY_PRICE=0.89 bash bench/run_session_5090.sh`
- What happened: `timeout: command not found`
- Root cause: the hard `MAX_HOURS` limit uses coreutils `timeout`; Linux GPU pods and Kaggle have it, macOS doesn't.
- Fix or workaround: the script uses `timeout` or `gtimeout` and stops with a clear message if neither exists; the Mac test used a small stand-in in the scratch directory.
- Severity: cosmetic (the script targets Linux GPU machines)
- Filed: not filed: our script
- Requirement this implies: none beyond documenting that the session script runs on Linux.

### CI fails because ruff format checks verbatim card snippets in results/
- When / where: 2026-11-03 · GitHub Actions ubuntu-latest · ruff 0.17.3 · launchready main @ 4295256
- Exact command: `uv run ruff format --check .` (CI step "Ruff format")
- What happened (error excerpt): `results/quickstart_audit/1a_mac_snippet.py:7:46: unformatted` · `results/quickstart_audit/1a_snippet.py:7:46: unformatted` · `results/quickstart_audit/card_README.md:58:46: unformatted` · `3 files would be reformatted` → exit 1
- Root cause: our process. The quickstart audits commit the model card and its snippets byte-for-byte as evidence, and ruff's default file discovery includes them (Python files and code blocks in Markdown). Before pushing I ran `ruff check .` but not `ruff format --check .`.
- Fix or workaround: exclude `results/` from ruff in pyproject.toml (it holds raw data and third-party text that must stay verbatim), and run both ruff commands before every push.
- Severity: blocks (CI red on main)
- Filed: not filed: our config
- Requirement this implies: raw evidence directories stay out of linters and formatters; pre-push checks mirror CI exactly.

### Qwen3-4B's vLLM one-liner fails on `--enable-reasoning` in vLLM 0.32.0 (quickstart audit 1b, Kaggle)
- When / where: 2026-11-03 · 2× Tesla T4 (Kaggle) · driver 580.178.04 · CUDA 13.0 · Python 3.13.16, vllm 0.32.0, torch 2.13.0, transformers 5.19.0 · card: Qwen/Qwen3-4B @ 1cfa9a7 ("## Quickstart", vLLM line)
- Exact command (verbatim from the card; `notebooks/03_quickstart_audit.ipynb`, log in `results/quickstart_audit/1b_vllm.log`): `vllm serve Qwen/Qwen3-4B --enable-reasoning --reasoning-parser deepseek_r1`
- What happened (error excerpt): exits with code 2 after 30 s, before loading anything: `usage: vllm [-h] [-v] {chat,complete,serve,launch,bench,collect-env,download-kernels,preload,run-batch,snapshot} ...` · `vllm: error: unrecognized arguments: --enable-reasoning`
- Root cause: vLLM 0.32.0 no longer accepts `--enable-reasoning`. The card says "`vllm>=0.8.5`", so every current vLLM satisfies the stated requirement and still rejects the command.
- Fix or workaround: not run. Dropping the flag (`vllm serve Qwen/Qwen3-4B --reasoning-parser deepseek_r1`) is the likely fix; whether `deepseek_r1` is still the right parser for Qwen3 is also untested.
- Not tested: whether the default context (no `--max-model-len`, so 40,960 tokens) fits a single T4. The command failed before vLLM got that far. Notebook 01 measured room for 30,336 tokens of KV cache for Qwen3-4B on one T4 (vLLM 0.32.0, `--max-model-len 4096`), so the fixed command may hit a second break; that's a projection, not a result.
- Severity: blocks (the card's deployment command fails on any current vLLM)
- Filed: not filed: candidate card fix (drop `--enable-reasoning`, give a tested vLLM version instead of an open `>=` range); not filing upstream without the owner's OK
- Requirement this implies: a quickstart's engine command needs a tested upper version, not just a minimum, and should be re-run when the engine removes flags.

### Worked as written: Qwen3-4B Transformers quickstart on Kaggle 2× T4 (quickstart audit 1a, Kaggle)
- 2026-11-03 · 2× Tesla T4 (Kaggle) · driver 580.178.04 · CUDA 13.0 · Kaggle's preinstalled Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, accelerate 1.15.0 · card Qwen/Qwen3-4B @ 1cfa9a7, Quickstart Python block byte-for-byte · finished in 115 s with thinking (~2,050 characters) and an answer (678 characters). Cosmetic: the printed thinking still contains literal `<think>`/`</think>` tags. Unlike the Mac (entries below), Kaggle ships `accelerate`, and the two T4s have room for the weights, so `device_map="auto"` needed no offload.

### Qwen3-4B quickstart offloads to disk on a 16 GB Mac and doesn't finish in 45 minutes (quickstart audit 1a, Mac)
- When / where: 2026-11-03 · MacBook Air M4, 16 GB, macOS 26.6.2 · MPS · Python 3.12.16, torch 2.15.0, transformers 5.22.1, accelerate 1.15.0 · card: Qwen/Qwen3-4B @ 1cfa9a7 (same Quickstart block, unchanged)
- Exact command: `python 1a_mac_snippet.py`, after `pip install accelerate` (log: `results/quickstart_audit/1a_mac_run2_with_accelerate.log`)
- What happened (error excerpt): download (54 s) and weight loading (4 s) finish, then `Some parameters are on the meta device because they were offloaded to the disk.` No output after 2,700 s; stopped by the audit's 45-minute cap (no answer printed, no thinking content).
- Root cause: `device_map="auto"` asks accelerate for a memory budget. On this Mac `accelerate.utils.get_max_memory()` returns `{'mps': 6.51 GiB}` with no `cpu` entry (MPS recommended maximum: 11.84 GiB). The BF16 weights are 7.49 GiB, so the remainder goes straight to disk, and every generated token reads those layers back from disk. Combined with `max_new_tokens=32768` and thinking on, the run doesn't finish in any reasonable time. (Budget measured with `get_max_memory()` right after the run; the exact device map wasn't captured.)
- Fix or workaround: not run yet. Candidates: `device_map="mps"` (whole model on the GPU; 7.49 GiB fits under the 11.84 GiB MPS maximum) or a smaller `max_new_tokens`. On a 16 GB Mac, Ollama, MLX-LM or llama.cpp (which the card lists for local use) are likelier first choices.
- Severity: blocks (the card's snippet never produces an answer on a 16 GB Apple-silicon laptop)
- Filed: not filed: candidate card note ("on Apple silicon use device_map='mps'"), and possibly an accelerate issue about MPS budgets with no CPU fallback; waiting for the owner's OK
- Requirement this implies: quickstarts should say which hardware they were tested on, and `device_map="auto"` should never silently fall back to disk on a machine with enough unified memory.

### Qwen3-4B quickstart needs `accelerate`, which the card never mentions (quickstart audit 1a, Mac)
- When / where: 2026-11-03 · MacBook Air M4, 16 GB, macOS 26.6.2 · MPS · Python 3.12.16, torch 2.15.0, transformers 5.22.1 (latest on PyPI, as the card advises), no accelerate · card: Qwen/Qwen3-4B @ 1cfa9a7 ("## Quickstart", Python block, lines 52–95)
- Exact command: `python 1a_mac_snippet.py` (the card's block, byte-for-byte; saved in `results/quickstart_audit/`)
- What happened (error excerpt): exits after 10 s, before any weights download: `File "1a_mac_snippet.py", line 7 … model = AutoModelForCausalLM.from_pretrained(` → `ValueError: Using a \`device_map\`, \`tp_plan\`, \`torch.device\` context manager or setting \`torch.set_default_device(device)\` requires \`accelerate\`. You can install it with \`pip install accelerate\``
- Root cause: the snippet passes `device_map="auto"`, which needs the separate `accelerate` package; the card's setup text only says to use the latest `transformers`.
- Fix or workaround: `pip install accelerate`, then rerun the same snippet unchanged. That rerun hit the next break (entry above).
- Severity: blocks (first run of the card's snippet fails), though the error message names the fix
- Filed: not filed: candidate one-line card fix ("pip install transformers accelerate"); waiting for the owner's OK
- Requirement this implies: a quickstart's install line must list every package its snippet imports or triggers (`device_map` → accelerate), tested in a clean environment.

### Lightning's llama.cpp recipe asks for a Q4_K_M file that doesn't exist (quickstart audit 3b)
- When / where: 2026-11-03 · MacBook Air M4, 16 GB, macOS 26.6.2 · Metal · llama.cpp 0.8.0 (build 11960, 8642541b2, Homebrew) · card: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @ bee7596 ("Local AI → llama.cpp → Full server recipe"); weights repo ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF @ 8cd0821
- Exact command (verbatim from the card; saved in `results/quickstart_audit/`): `llama-server -hf ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_K_M --temp 1.0 --top-p 0.95 -np 1 -c 40960 --port 8000 -ngl 99 -fa on --jinja --no-webui --fit off`
- What happened (error excerpt): exits after 6 s, nothing downloaded:
  `E common_download_get_hf_plan: no GGUF files found in repository ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` · `I Available GGUF files: … BF16.gguf, … Q4_0.gguf, … Q8_0.gguf, mtp-…` · `E llama_model_load_from_file_impl: exactly one out metadata, path_model, and file must be defined` · `E srv load_model: failed to load model, ''`. Also `Use --ui/--no-ui (or deprecated --webui/--no-webui)`.
- Root cause: the card's tag `:Q4_K_M` doesn't match any file; the official GGUF repo ships `Q4_0` (18.9 GB), `Q8_0` (33.6 GB) and `BF16` (63.2 GB). llama.cpp's first error line says "no GGUF files found" even though it lists six, and the last lines blame an empty model path, so the real cause (wrong quant tag) is easy to miss.
- Fix or workaround: use `:Q4_0`, the only 4-bit file in the repo (not run: an 18.9 GB download that also exceeds this Mac's 16 GB). Replace the deprecated `--no-webui` with `--no-ui`.
- Severity: blocks (the card's recommended local recipe fails on its first line)
- Filed: not filed yet: candidate fix to the NVFP4 card (`:Q4_K_M` → `:Q4_0`, or publish a Q4_K_M) and a candidate llama.cpp issue for the misleading message; waiting for the owner's OK before filing on someone else's repo
- Requirement this implies: every quant tag in a model card's recipe must resolve against the linked weights repo at publish time (a tag-exists check is cheap and needs no download).

### granite-4.0-h-tiny serves 2.3× fewer users in vLLM than projected
- When / where: 2026-11-03 · 2× Tesla T4 (Kaggle) · driver 580.178.04 · CUDA 13.0 · vLLM 0.32.0, torch 2.13.0+cu130 · ibm-granite/granite-4.0-h-tiny @ 791e0d3
- Exact command: `vllm serve ibm-granite/granite-4.0-h-tiny --revision 791e0d3d28c86e106c9b6e0b4cecdee0375b6124 --dtype half --max-model-len 4096 --gpu-memory-utilization 0.9 --tensor-parallel-size 2`
- What happened (error excerpt): `GPU KV cache size: 207,171 tokens, Maximum concurrency for 4,096 tokens per request: 50.58x`. launchready projected 118.08 (+133%). Qwen3-4B (−4%) and OLMoE (−6%) were within target on the same session.
- Root cause: two errors that multiply (118.08 ÷ 50.58 = 1.719 × 1.358):
  - **KV budget, ×1.72 (64% of the miss in log terms):** measured 2.93 GiB (3.15 GB) per GPU vs 5.04 GiB (5.41 GB) projected. vLLM: `Desired … (0.9, 13.11 GiB). Actual usage is 7.93 GiB for consumed memory (weights + non-torch), 2.24 GiB for peak activation`, so KV = 13.11 − 7.93 − 2.24 = 2.94 GiB. Overhead beyond the 6.48 GiB of weights is 3.70 GiB per GPU (1.45 non-torch + 2.24 activation), not the 2 GiB the fit model assumes; for comparison Qwen3-4B was 1.38 and OLMoE 1.29. Also, vLLM's 13.11 GiB is 0.9 × 14.56 GiB (CUDA's total), while the projection used nvidia-smi's 15 GiB: 0.40 GiB per GPU on every model.
  - **Per-sequence cost, ×1.36 (36%):** measured 124 MB per 4,096-token sequence (2 × 2.93 GiB ÷ 50.58) vs 92 MB projected (34 MB KV + 58 MB Mamba state at FP32). vLLM keeps Mamba state in FP16 (`cache_dtype=float16`), which alone would make it *smaller* (~65 MB), so something else costs more. The log shows `Setting attention block size to 400 tokens to ensure that attention page size is >= mamba page size`, `Padding mamba page size by 1.39%`, and `Mamba cache mode is set to 'align' … when prefix caching is enabled`; page unification and per-block state copies are the likely cause. Not confirmed.
  - With only the budget corrected, the projection would be 68.7; with only per-sequence corrected, 87.0.
- Fix or workaround: no retune from one model. `launchready fit` now warns that vLLM concurrency for hybrid Mamba models can be far below the projection (with this measurement) and marks the KV-token projection as not comparable for hybrids. The notebook footer carries the same decomposition.
- Severity: slows (a developer sizing granite on 2× T4 from our numbers would expect 118 users and get 51)
- Filed: not filed: vLLM behaves as designed; the gap is in our fit model
- Requirement this implies: the fit model needs a per-architecture overhead and a hybrid per-sequence rule, calibrated on more than one hybrid model before it changes.

### vLLM startup for granite-4.0-h-tiny took 16 minutes on T4
- When / where: 2026-11-03 · 2× Tesla T4 (Kaggle) · driver 580.178.04 · CUDA 13.0 · vLLM 0.32.0 · ibm-granite/granite-4.0-h-tiny @ 791e0d3
- Exact command: same `vllm serve` as above
- What happened (error excerpt): `init engine (profile, create kv cache, warmup model) took 951.29 s`. The bulk was `Warming up Mamba2 SSD Triton kernels...` → `Initial profiling/warmup run took 666.77 s`; CUDA-graph capture took ~220 s (`Graph capturing finished in 120 secs` + `in 100 secs`) and torch.compile 21.9 s. Qwen3-4B's warmup run took 0.14 s and OLMoE's 7.7 s. The notebook's vLLM timeout was 1,200 s, so this passed with ~4 min to spare.
- Root cause: Mamba2 SSD Triton kernels are JIT-compiled and autotuned on first use; there are no tuned configs for Tesla T4 (`Using default Mamba SSU config … Config file not found … device_name=Tesla_T4`).
- Fix or workaround: raised the notebook's vLLM timeout to 1,800 s.
- Severity: slows
- Filed: not filed: expected first-run JIT cost on an old GPU; worth a line in any T4 recipe
- Requirement this implies: recipes for hybrid Mamba models on older GPUs should warn that first startup can take 15+ minutes.

### Transformers weight measurement crashed for every model
- When / where: 2026-11-03 · 2× Tesla T4 (Kaggle) · driver 580.178.04 · CUDA 13.0 · torch 2.13.0+cu130, transformers 5.19.0 · all three Phase 3 models
- Exact command: `python measure_hf.py Qwen/Qwen3-4B 1cfa9a7… 1` (notebooks/01_memory_validation.ipynb, commit cc0090a)
- What happened (error excerpt): `torch.cuda.reset_peak_memory_stats(d)` → `RuntimeError: Invalid device argument`. Every `weights_bytes` measured row is empty; the chart's weights panel says "not measured".
- Root cause: our bug. The script reset the peak-memory counter before anything had touched the GPU; torch 2.13 rejects the call while the caching allocator for that device is uninitialized.
- Fix or workaround: initialize each device first (`torch.zeros(1, device=d)`) and then reset. The Transformers stage is now optional and off by default; the notebook uses vLLM's `Model loading took X GiB` lines as the measured weights (within 1% of projected for all three models).
- Severity: blocks (no Transformers weights measurement)
- Filed: not filed: our bug
- Requirement this implies: run each GPU helper once on a real GPU before relying on it in a long notebook run.

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
- Root cause: the roadmap says "check main", but llama.cpp's development branch is `master`.
- Fix or workaround: clone the remote's default branch (no `--branch`) and label it with its real name, e.g. `master @ abc1234`.
- Severity: blocks (no llama.cpp/Ollama readiness)
- Filed: not filed: our assumption, not an upstream bug
- Requirement this implies: never hard-code branch names; read the default branch from the remote.

### Two engine registry paths in roadmap 6.6 have moved
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
- What happened (error excerpt): safetensors headers count 32,913,266,240 parameters (BF16 32,913,263,168 + F32 3,072). roadmap 6.5 used the nominal ~30B (BF16 ~60 GB, NVFP4 ~17 GB); real figures are BF16 ~65.8 GB and NVFP4 ~18.5 GB before the BF16 share.
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

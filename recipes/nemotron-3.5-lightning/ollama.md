# Nemotron 3.5 Lightning GGUF with Ollama (and llama-server for more than one user)

| | |
|---|---|
| GPU | 1× NVIDIA GeForce RTX 5090 (32 GB) |
| Engine | Ollama 0.40.14 (NVIDIA's card was written for 0.32.9); llama.cpp v0.8.0 for `llama-server` |
| Model | [`bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) @ `f0eec2267ae843d9eb21ea3926ab0046da0a8628`, file `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf` (25,477,403,616 bytes = 23.73 GiB, sha256 `b520a899…5787a9158`) |
| What's in the file | Measured from the GGUF header on 2026-11-03 ([`results/bench/gguf/bartowski__Lightning-Q4_K_M.json`](../../results/bench/gguf/bartowski__Lightning-Q4_K_M.json)): **6.19 bits per weight**, not ~4.85. The expert tensors fell back to `Q5_0` (63% of bytes) and `Q8_0` (33%); only 0.3% is `Q4_K`. `tokenizer.ggml.pre = pixtral` |
| Verified | Not run on a 5090 |

## Ollama: one user

```bash
OLLAMA_CONTEXT_LENGTH=40960 ollama serve &
ollama run hf.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_K_M --verbose
```

- `--verbose` prints the eval rate (tokens/s) after each answer; `ollama ps` shows how much is on the GPU.
- `OLLAMA_CONTEXT_LENGTH=40960` matches llama.cpp's card recipe and the vLLM recipe. Ollama picks 32K by default on a 24–48 GB GPU.
- `hf.co/...` pulls whatever is on the repo's main branch at pull time. Compare the digest in `ollama show` with the sha256 above.
- NVIDIA's card also lists the Ollama library tag `ollama run nemotron-3.5-lightning`. That's a different file; check what it contains before comparing numbers.

**Ollama serves one request at a time for this model.** It forces `OLLAMA_NUM_PARALLEL=1` for the `nemotron_h_moe` architecture ([ollama#18938](https://github.com/ollama/ollama/issues/18938), closed as a duplicate, restriction still in place). With 8 or 32 users, requests wait in a queue, so Ollama is benchmarked at concurrency 1 only.

## llama-server: up to 8 users at once

```bash
git clone --depth 1 --branch v0.8.0 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j --target llama-server
hf download bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf \
  --revision f0eec2267ae843d9eb21ea3926ab0046da0a8628 --local-dir models
./build/bin/llama-server -m models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf \
  --port 8001 -np 8 -c 327680 -ngl 99 -fa on --jinja --no-webui --fit off
```

- `-np 8 -c 327680` gives 8 slots of 40,960 tokens each (`-c` is the total across slots).
- `-fa on --jinja --fit off` come from NVIDIA's llama.cpp recipe. The benchmark adds `--chat-template-kwargs '{"enable_thinking": false}'` so vLLM and llama.cpp answer the same way.
- **Use llama.cpp v0.8.0 or newer.** Every Lightning GGUF checked (bartowski, unsloth, ggml-org, AtomicChat; 2026-11-03) is tagged `tokenizer.ggml.pre = pixtral`. Older llama.cpp maps that to the Llama 3 regex instead of Tekken, so token splits differ from Hugging Face's, most visibly in Hindi, Tamil, Thai and Bengali ([llama.cpp#30894](https://github.com/ggml-org/llama.cpp/issues/30894)). [PR #30960](https://github.com/ggml-org/llama.cpp/pull/30960) (merged 2026-10-22, in v0.8.0) fixes it for existing files. Ollama bundles its own llama.cpp, so whether Ollama has the fix depends on its version: not verified here.
- NVIDIA's card command `-hf ggml-org/...:Q4_K_M` fails: ggml-org publishes only Q4_0, Q8_0 and BF16 (Phase 3 audit, `results/quickstart_audit/3b_lightning_llama_server.log`).

## Memory (projected)

| | Weights | Per 40,960-token sequence | 8 sequences | Left on a 32 GiB card |
|---|---|---|---|---|
| This GGUF | 23.73 GiB (file size; header: 6.19 bits/weight) | 0.28 GiB (0.23 GiB attention KV at FP16 + 0.05 GiB Mamba state) | 2.25 GiB | ~6.0 GiB for compute buffers |

Formula: per sequence = 6 attention layers × 2 × 2 KV heads × 128 × 2 bytes × 40,960 tokens + Mamba state (`launchready fit`, roadmap 6.2). `launchready fit` projects 18.58 GiB for "Q4_K_M" at 4.85 bits/weight. That's 22% below the real file, because the experts fall back to Q5_0. Use the file size. Measured numbers will replace this table after the 5090 session.

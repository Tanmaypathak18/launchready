# LaunchReady

## Start here

**[Read the one-page summary](https://tanmaypathak18.github.io/launchready/)** for what LaunchReady does, what it found and what isn't done yet, in plain English. Source: [docs/index.html](docs/index.html). Each number is labeled measured or projected and links to its file.

- Live tool: [https://huggingface.co/spaces/tanmaypathak4/launchready](https://huggingface.co/spaces/tanmaypathak4/launchready) (free CPU Basic, no login)
- Summary page: [tanmaypathak18.github.io/launchready](https://tanmaypathak18.github.io/launchready/)

## What it is

LaunchReady is a small, open-source launch-readiness kit for open-weight LLMs on NVIDIA GPUs, written for developers, researchers and ML engineers who want to know whether a new model will run on their hardware before they download it. It reads a model's config and safetensors headers from Hugging Face (never the weights) to project memory at BF16, FP8, NVFP4 and GGUF and which NVIDIA GPUs fit it; checks whether vLLM, SGLang, llama.cpp and Transformers support the architecture; writes a one-page model brief; and backs it all with measured validation and benchmarks, starting with NVIDIA Nemotron 3.5 Lightning on an RTX 5090. Every number is labeled *projected* (with its formula) or *measured* (with its full environment).

## Status

- [x] Phase 0: Setup (repo, CI, smoke test)
- [x] Phase 1: Memory calculator (`launchready fit`)
- [x] Phase 2: Engine readiness + Nemotron brief (`launchready brief`, `launchready radar`)
- [x] Phase 3: Validation on free GPUs + first-user audits
- [x] Phase 4: Developer feedback → requirements
- [ ] Phase 5: Nemotron 3.5 Lightning launch kit (recipes, benchmarks, quality check)
  - Steps 1–5 done: [readiness checklist](docs/readiness_checklist.md) (1 ready, 5 not ready, 21 unknown), gap check, [recipes](recipes/nemotron-3.5-lightning/README.md), bench scripts, Kaggle dry run
  - Left: rented RTX 5090 session (nothing is measured on a 5090 yet), results notebook
- [ ] Phase 6: Publish (blog post, README quickstart and results)

## Development

```bash
uv sync
uv run launchready fit Qwen/Qwen3-4B --context 32768 --users 8
uv run launchready fit nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --gpu rtx-5090 --json
uv run launchready brief nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16   # → briefs/
uv run launchready radar --since 30d                                        # → radar/<date>.md
uv run ruff check .
uv run pytest
```

`launchready radar`: dates are Hub repo-creation dates; a repo published later than it was created can look older than its public release. Use `--since 90d` for a wider look.

## License

Apache-2.0. See [LICENSE](LICENSE).

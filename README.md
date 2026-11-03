# LaunchReady

LaunchReady is a small, open-source launch-readiness kit for open-weight LLMs on NVIDIA GPUs, written for developers, researchers and ML engineers who want to know whether a new model will run on their hardware before they download it. It reads a model's config and safetensors headers from Hugging Face (never the weights) to project memory at BF16, FP8, NVFP4 and GGUF and which NVIDIA GPUs fit it; checks whether vLLM, SGLang, llama.cpp and Transformers support the architecture; writes a one-page model brief; and backs it all with measured validation and benchmarks, starting with NVIDIA Nemotron 3.5 Lightning on an RTX 5090. Every number is labeled *projected* (with its formula) or *measured* (with its full environment).

## Status

- [x] Phase 0: Setup (repo, CI, smoke test)
- [ ] Phase 1: Memory calculator (`launchready fit`)
- [ ] Phase 2: Engine readiness + Nemotron brief (`launchready brief`, `launchready radar`)
- [ ] Phase 3: Validation on free GPUs + first-user audits
- [ ] Phase 4: Developer feedback → requirements
- [ ] Phase 5: Nemotron 3.5 Lightning launch kit (recipes, benchmarks, quality check)
- [ ] Phase 6: Publish (blog post, README quickstart and results)

## Development

```bash
uv sync
uv run launchready --help
uv run ruff check .
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).

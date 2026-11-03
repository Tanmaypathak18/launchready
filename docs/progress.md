# Progress

One "Done so far" entry per phase, newest first: what shipped, decisions made, what's next.

## Phase 1: Memory calculator (2026-11-02)

- **Shipped:** `launchready fit` (weights per precision, KV + Mamba state per sequence, max users per GPU) from config.json and safetensors headers only; hub.py with a JSON cache, arch.py for dense GQA, MoE, MLA, sliding-window and hybrid Mamba configs, memory.py, gpus.yaml; 34 offline tests incl. exact 6.5 KV values.
- **Decisions:** GPU capacity read as GiB; Mamba state FP32 unless the config names a dtype, labeled "assumed"; quantized weights = total params × bits (no BF16 share yet); MoE active params estimated as total minus idle routed experts, labeled "~".
- **Next:** Phase 2. Known gaps: projected NVFP4 Lightning 17.24 GiB vs 20.08 GiB real NVFP4 files (BF16 share); active params run high (Lightning ~3.7B, granite-h-tiny ~1.5B, DeepSeek-V3 ~40B); no tensor parallel, so DeepSeek-V3 fits no single GPU; calibrate reserve and GiB reading in Phase 3.

## Phase 0: Setup (2026-11-02)

- **Shipped:** uv-managed Python 3.11 scaffold with a Typer CLI (`fit`, `brief`, `radar` stubs that print "not implemented"), docstring-only modules, ruff + pytest smoke tests, GitHub Actions CI, Apache-2.0 LICENSE, README Status checklist, friction log template.
- **Decisions:** `typer` is the only runtime dependency; huggingface_hub, Jinja2 and PyYAML get added in the phase that needs them. Stub commands exit with code 1 so nobody mistakes them for working.
- **Next:** Phase 1 (memory calculator: `launchready fit`). Repo is public at Tanmaypathak18/launchready with CI green.

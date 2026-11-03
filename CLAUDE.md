# LaunchReady

Small open-source launch-readiness kit for open-weight LLMs on NVIDIA GPUs: sizes hardware needs, checks engine support, writes model briefs, and benchmarks NVIDIA Nemotron 3.5 Lightning. Full plan: BUILD_PLAN.md (read the current phase before starting). Read docs/progress.md at session start to see what's done and what's next.

## Audience
Developers and researchers: Hugging Face users, local-model users, ML engineers serving models. Write every output for them.

## Rules
- Never download model weights in the calculator; use config.json and safetensors headers only.
- Label every number "measured" (GPU, driver, CUDA, engine version, model revision, date) or "projected" (with the formula).
- When a config field can't be interpreted, warn and name it. Never guess silently.
- Pin versions and record the environment with every result.
- No secrets in git: read HF_TOKEN and GITHUB_TOKEN from the environment.
- Keep scope lean: don't add features beyond the current phase.
- When something breaks, add an entry to docs/friction_log.md before fixing it.

## Stack
Python 3.11+, uv, Typer, huggingface_hub, Jinja2, PyYAML, pytest, ruff. Benchmarks: AIPerf, lm-evaluation-harness. Tests use saved fixtures; CI has no network.

## Commands
- `uv run launchready fit <repo_id> --context 32768 --users 8`
- `uv run launchready brief <repo_id>`
- `uv run launchready radar --since 30d`
- `uv run pytest` and `uv run ruff check .`

## Conventions
- Small commits.
- At the end of each phase, update README Status and add a 3-line "Done so far" entry to docs/progress.md (what shipped, decisions made, what's next).
- Compute in bytes, display GiB.

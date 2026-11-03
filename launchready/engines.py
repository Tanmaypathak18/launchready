"""Engine readiness checks (Phase 2; BUILD_PLAN.md section 6.6).

Uses shallow sparse clones of vLLM, SGLang, llama.cpp and Transformers at `main` and the latest
release tag, cached under ~/.cache/launchready, to report each architecture as `release`,
`main-only`, `not-found` or `unknown` with the matching file and line. Flags `auto_map` configs
as needing trust_remote_code.
"""

"""Hugging Face Hub access without downloading weights (Phase 1).

Fetches config.json (including nested `text_config`), model info (release date, license,
gated status), safetensors parameter counts from headers via
`huggingface_hub.get_safetensors_metadata`, and file sizes. Results are cached on disk.
Reads HF_TOKEN from the environment for gated repos and reports "gated" instead of crashing.
"""

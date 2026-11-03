"""Projected memory math (Phase 1; BUILD_PLAN.md sections 6.1-6.3).

Weights per precision (BF16, FP8, NVFP4, GGUF), KV cache and Mamba state per sequence
(FP16 or FP8 KV), and a vLLM-style fit model giving max users at a context length per GPU.
Computes in bytes and displays GiB; every result is labeled "projected".
"""

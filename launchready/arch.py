"""Normalize a model config into an `Architecture` dataclass (Phase 1).

Covers layers and layer types (full attention, sliding window, Mamba), attention and KV heads,
head_dim, context length, MoE fields (total vs active parameters), MLA fields and Mamba fields.
Any config field that can't be interpreted produces a warning naming the field; nothing is
guessed silently.
"""

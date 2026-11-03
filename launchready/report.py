"""The projected fit for one model, shared by `launchready fit` and the Space.

Collects total parameters, weights per precision, per-sequence memory, the fit per GPU and every
warning and note, so front ends only render. Calls launchready.arch and launchready.memory; no
math lives here. Every number is projected, not measured.
"""

from dataclasses import dataclass

from launchready.arch import Architecture, parse_config
from launchready.hub import HubModel
from launchready.memory import (
    BITS_PER_WEIGHT,
    Gpu,
    SequenceMemory,
    fit,
    sequence_bytes,
    total_parameters,
    weight_bytes,
)
from launchready.variants import Variant, real_sizes

PRECISION_LABELS = {
    "bf16": "BF16",
    "fp8": "FP8",
    "nvfp4": "NVFP4",
    "gguf-q8_0": "GGUF Q8_0",
    "gguf-q4_k_m": "GGUF Q4_K_M",
}
# Measured in notebooks/01_memory_validation.ipynb (results/memory_validation.csv). One model,
# so the fit model is not retuned; we warn instead (docs/friction_log.md has the breakdown).
HYBRID_VLLM_WARNING = (
    "hybrid Mamba: vLLM concurrency can be far below this projection (measured: "
    "ibm-granite/granite-4.0-h-tiny at 4,096 tokens on 2× Tesla T4, vLLM 0.32.0, 2026-11-03: "
    "projected 118, measured 50.6); KV-cache token counts are not comparable to vLLM's for "
    "hybrids, since vLLM shares pages between attention KV and Mamba state"
)
# A BF16 repo whose files are this far from the projection isn't plain BF16 (FP32 weights, or
# consolidated.safetensors next to the shards), so its size says nothing about BF16.
_BF16_FILE_TOLERANCE = 0.25


@dataclass
class FitReport:
    arch: Architecture
    total: int | None
    active: int | None
    projected_weights: dict[str, float]  # params × bits ÷ 8, by precision
    weights: dict[str, float]  # what the fit uses: Hub file size if given, else projected
    seq: SequenceMemory
    fits: dict[str, dict[str, dict]]  # gpu id → precision → kv_budget_bytes, max_users, native
    warnings: list[str]
    notes: list[str]


def hub_file_sizes(model: HubModel, variants: list[Variant]) -> dict[str, tuple[str, int]]:
    """precision → (repo_id, bytes): real files on the Hub, the base repo's own for BF16."""
    real = real_sizes(variants, model.repo_id.split("/")[0])
    if model.parameters and model.safetensors_bytes and not model.config.get("quantization_config"):
        total, _ = total_parameters(model.parameters, model.config)
        drift = model.safetensors_bytes / weight_bytes(total, "bf16") - 1
        if abs(drift) < _BF16_FILE_TOLERANCE:
            real["bf16"] = (model.repo_id, model.safetensors_bytes)
    return real


def effective_weights(
    projected: dict[str, float], hub_files: dict[str, tuple[str, int]]
) -> dict[str, float]:
    """The Hub file size where one exists, else the projection."""
    return {p: hub_files[p][1] if p in hub_files else w for p, w in projected.items()}


def project(
    model: HubModel,
    context: int,
    kv_dtype: str,
    gpus: list[Gpu],
    hub_files: dict[str, tuple[str, int]] | None = None,
) -> FitReport:
    """Fit `model` on `gpus` at `context` tokens; `hub_files` swaps in real sizes for the fit."""
    arch = parse_config(model.config)
    warnings = list(model.warnings) + list(arch.warnings)
    notes = list(arch.notes)
    total = active = None
    if model.parameters:
        total, param_warnings = total_parameters(model.parameters, model.config)
        warnings += param_warnings
        active = arch.active_parameters(total)
    if model.license is None:
        warnings.append("model card declares no license: check the repo before using it")
    if arch.max_context and context > arch.max_context:
        warnings.append(f"--context {context:,} exceeds max_position_embeddings")

    projected = {p: weight_bytes(total, p) for p in BITS_PER_WEIGHT} if total else {}
    weights = effective_weights(projected, hub_files or {})
    seq = sequence_bytes(arch, context, kv_dtype)
    if seq.mamba.dtype is not None:
        warnings.append(HYBRID_VLLM_WARNING)
    if seq.mamba.assumed:
        notes.append(
            f"Mamba state dtype not in config: assumed {seq.mamba.dtype} "
            "(engines may keep it smaller)"
        )
    if weights:
        notes.append(
            "Quantized sizes assume every tensor is quantized; real files keep embeddings, "
            "lm_head, norms and routers in BF16, so expect somewhat more"
        )
    fits = {
        g.id: {
            p: {**vars(fit(g.memory_bytes, w, seq.total)), "native": g.is_native(p)}
            for p, w in weights.items()
        }
        for g in gpus
    }
    return FitReport(
        arch=arch,
        total=total,
        active=active,
        projected_weights=projected,
        weights=weights,
        seq=seq,
        fits=fits,
        warnings=warnings,
        notes=notes,
    )

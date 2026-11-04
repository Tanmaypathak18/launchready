"""Model brief rendering (Phase 2; docs/roadmap.md template 7.1).

Renders templates/brief.md.j2 with Jinja2 to briefs/<org>__<model>.md, combining hub metadata,
projected hardware requirements, engine readiness and available quantized variants. The
"Ecosystem impact" and "Open questions and risks" sections are left for hand-writing.
"""

import re
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from launchready import __version__
from launchready.arch import Architecture, parse_config
from launchready.engines import EngineStatus
from launchready.engines import check as _check_engines
from launchready.hub import HubModel, load_model
from launchready.memory import (
    BITS_PER_WEIGHT,
    GIB,
    GPU_MEMORY_UTILIZATION,
    RUNTIME_RESERVE_BYTES,
    Gpu,
    fit,
    kv_bytes_per_token,
    load_gpus,
    sequence_bytes,
    total_parameters,
    weight_bytes,
)
from launchready.report import PRECISION_LABELS, effective_weights, hub_file_sizes
from launchready.variants import Variant, load_variants

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
CONTEXTS = (8_192, 32_768, 131_072)
USER_COUNTS = (1, 8)
HEADLINE_CONTEXT = 32_768
LOCAL_GPU = "rtx-5090"
# Format → precision used to sanity-check a variant's size; 4-bit-ish formats compare to NVFP4.
_FORMAT_PRECISION = {"FP8": "fp8", "NVFP4": "nvfp4", "MLX": "nvfp4", "other": "nvfp4"}
_FORMAT_PRECISION.update({"AWQ": "nvfp4", "GPTQ": "nvfp4", "bitsandbytes": "nvfp4"})


def check_engines(config: dict, refresh: bool = False) -> list[EngineStatus]:
    return _check_engines(config, refresh=refresh)


def _gib(n: float) -> str:
    return f"{n / GIB:,.2f}"


def _billions(n: int) -> str:
    return f"{n / 1e9:,.2f}B"


def _tokens(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else f"{n:,}"


def _smallest_gpu(gpus: list[Gpu], precision: str, weights: float, seq: int, users: int):
    for g in sorted(gpus, key=lambda g: g.memory_gib):
        if g.is_native(precision) and fit(g.memory_bytes, weights, seq).max_users >= users:
            return g
    return None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_context(
    model: HubModel,
    variants: list[Variant],
    engines: list[EngineStatus],
    today: date | None = None,
) -> dict:
    """Everything template 7.1 needs, computed from hub metadata, variants and engine checks."""
    arch: Architecture = parse_config(model.config)
    builder, model_name = model.repo_id.split("/")
    gpus = load_gpus()
    total = active = None
    hardware_notes: list[str] = []
    if model.parameters:
        total, param_warnings = total_parameters(model.parameters, model.config)
        hardware_notes += param_warnings
        active = arch.active_parameters(total)

    # Weights: projected for every precision, plus Hub file sizes where a repo exists.
    real = hub_file_sizes(model, variants)
    projected = {p: weight_bytes(total, p) for p in BITS_PER_WEIGHT} if total else {}
    effective = effective_weights(projected, real)
    weights = []
    for p, w in projected.items():
        row = {"label": PRECISION_LABELS[p], "bits": f"{BITS_PER_WEIGHT[p]:g}"}
        row.update(projected=_gib(w), real=None, repo_link=None)
        if p in real:
            repo, size = real[p]
            row["real"] = f"{_gib(size)} ({size / w - 1:+.0%})"
            row["repo_link"] = f"[{repo}](https://huggingface.co/{repo})"
        weights.append(row)

    # Per-sequence memory and the smallest GPU per precision × context × users.
    contexts = [c for c in CONTEXTS if not arch.max_context or c <= arch.max_context]
    seqs = {c: sequence_bytes(arch, c) for c in contexts}
    sequences = [
        {
            "context": f"{c:,}",
            "kv": _gib(s.attention_bytes),
            "state": _gib(s.mamba_bytes),
            "total": _gib(s.total),
        }
        for c, s in seqs.items()
    ]
    state_dtype = seqs[contexts[0]].mamba.dtype if contexts else None
    if contexts and seqs[contexts[0]].mamba.assumed:
        hardware_notes.append(f"Mamba state dtype not in config: assumed {state_dtype}")
    gpu_columns = [f"{_tokens(c)} × {u}" for c in contexts for u in USER_COUNTS]
    gpu_rows = []
    for p, w in effective.items():
        cells = []
        for c in contexts:
            for u in USER_COUNTS:
                g = _smallest_gpu(gpus, p, w, seqs[c].total, u)
                cells.append(g.name if g else "—")
        gpu_rows.append({"label": PRECISION_LABELS[p], "cells": cells})
    if any(p.startswith("gguf") for p in effective):
        hardware_notes.append(
            "GGUF rows assume the whole model on the GPU; llama.cpp/Ollama can offload layers to "
            "CPU RAM at lower speed"
        )

    # Architecture summary.
    layer_summary = ", ".join(
        f"{arch.count(k)} {label}"
        for k, label in (
            ("full_attention", "attention"),
            ("sliding_attention", "sliding-window attention"),
            ("mla", "MLA"),
            ("mamba", "Mamba"),
            ("none", "MLP/MoE-only"),
        )
        if arch.count(k)
    )
    if total is None:
        params = "unknown (no safetensors headers)"
    elif arch.moe:
        shown = f"~{_billions(active)}" if active is not None else "unknown"
        params = f"{_billions(total)} total / {shown} active per token (estimated)"
        params += f" · {arch.moe.num_experts} experts, {arch.moe.experts_per_token} per token"
    else:
        params = _billions(total)
    if arch.count("mla"):
        attention = f"MLA (kv_lora_rank {arch.kv_lora_rank})"
    elif arch.num_kv_heads:
        attention = f"GQA, {arch.num_attention_heads} query / {arch.num_kv_heads} KV heads"
        attention += f", head_dim {arch.head_dim}"
    else:
        attention = "unknown"
    if arch.sliding_window:
        attention += f", sliding window {arch.sliding_window:,}"
    drafters = [v for v in variants if v.format == "drafter"]
    extras = []
    if arch.mtp_layers:
        extras.append(f"{arch.mtp_layers} multi-token-prediction (MTP) layer(s) in the checkpoint")
    if drafters:
        names = ", ".join(f"[{v.repo_id}](https://huggingface.co/{v.repo_id})" for v in drafters)
        extras.append(f"drafter models for speculative decoding: {names}")
    if arch.trust_remote_code or model.config.get("auto_map"):
        extras.append("custom code (`auto_map`): needs `trust_remote_code`")
    arch_ctx = {
        "family": arch.family,
        "model_type": arch.model_type,
        "layers": arch.num_layers,
        "layer_summary": layer_summary,
        "params": params,
        "attention": attention,
        "context": f"{arch.max_context:,} tokens" if arch.max_context else "unknown",
        "extras": extras,
    }

    # Variants table (drafters are listed under Architecture → Extras).
    variant_rows = []
    for v in variants:
        if v.format == "drafter":
            continue
        if v.gguf:
            shown = [f"{q} {_gib(v.gguf[q])} GiB" for q in ("Q4_K_M", "Q8_0") if q in v.gguf]
            size = " · ".join(shown) or f"{len(v.gguf)} quant types"
            if len(v.gguf) > len(shown):
                size += f" (+{len(v.gguf) - len(shown)} more types)"
        else:
            size = f"{_gib(v.total_bytes)} GiB" if v.total_bytes else "—"
            check = _FORMAT_PRECISION.get(v.format)
            if v.total_bytes and check and total and v.total_bytes < 0.5 * projected[check]:
                size += " ⚠ far below the projected size: probably not the full model"
        variant_rows.append(
            {"format": v.format, "repo_id": v.repo_id, "size": size, "downloads": v.downloads}
        )

    # Generated prose.
    seq_head = seqs.get(HEADLINE_CONTEXT)
    head_tokens = _tokens(HEADLINE_CONTEXT)
    nominal = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    developer = []
    if total and arch.moe and active:
        named = f"; the name says {nominal.group(0)}" if nominal else ""
        developer.append(
            f"It stores {_billions(total)} parameters (counted from safetensors headers{named}) "
            f"but runs only ~{_billions(active)} per token: plan memory for the full "
            f"{_billions(total)} and expect the speed of a much smaller model."
        )
    elif total:
        developer.append(f"It has {_billions(total)} parameters (from safetensors headers).")
    if seq_head:
        kv = kv_bytes_per_token(arch)
        if arch.count("mamba"):
            attn = arch.count("full_attention") + arch.count("sliding_attention")
            developer.append(
                f"Only {attn} of its {arch.num_layers} layers keep a KV cache that grows with "
                f"the conversation ({kv:,} bytes per token); its {arch.count('mamba')} Mamba "
                f"layers keep a fixed {_gib(seq_head.mamba_bytes)} GiB state, so one "
                f"{head_tokens}-token conversation needs just {_gib(seq_head.total)} GiB."
            )
        else:
            developer.append(
                f"Each token adds {kv:,} bytes of FP16 KV cache, so one {head_tokens}-token "
                f"conversation needs {_gib(seq_head.total)} GiB on top of the weights."
            )
    local = next((g for g in gpus if g.id == LOCAL_GPU), None)
    bf16 = effective.get("bf16")
    if bf16 and seq_head:
        g = _smallest_gpu(gpus, "bf16", bf16, seq_head.total, 1)
        where = (
            f"the smallest single GPU that fits them is the {g.name}"
            if g
            else ("no single GPU in the catalog fits them")
        )
        missing = (
            f", not an {local.name}"
            if local and fit(local.memory_bytes, bf16, seq_head.total).max_users < 1
            else ""
        )
        developer.append(f"BF16 weights are {_gib(bf16)} GiB: {where}{missing}.")
    local_fits = []
    if local and seq_head:
        for p in ("nvfp4", "gguf-q4_k_m", "fp8"):
            if p not in effective or not local.is_native(p):
                continue
            users = fit(local.memory_bytes, effective[p], seq_head.total).max_users
            if users >= 1:
                source = "Hub files" if p in real else "projected"
                local_fits.append(
                    f"{PRECISION_LABELS[p]} ({_gib(effective[p])} GiB {source}, "
                    f"~{users} × {head_tokens} sequences)"
                )
        if local_fits:
            developer.append(
                f"On a single {local.name} ({local.memory_gib:g} GB) it fits at "
                f"{_join(local_fits)}: projected, not measured."
            )
        else:
            developer.append(f"It doesn't fit a single {local.name} at any precision checked.")
    by_status: dict[str, list[str]] = {}
    for e in engines:
        by_status.setdefault(e.status, []).append(e.engine)
    engine_parts = []
    for status, label in (
        ("release", "registered in the latest release of"),
        ("main-only", "only on the default branch of"),
        ("not-found", "not found in"),
        ("unknown", "couldn't check"),
    ):
        if status in by_status:
            engine_parts.append(f"{label} {_join(by_status[status])}")
    if engine_parts:
        developer.append(f"The architecture is {'; '.join(engine_parts)}.")

    released = by_status.get("release", [])
    tldr = [
        f"**What it is:** {arch.family} model from {builder}, {arch_ctx['params'].split(' ·')[0]}, "
        f"{arch_ctx['context']} context, license {model.license or 'not declared'}.",
    ]
    needs = []
    for p in ("bf16", "nvfp4", "gguf-q4_k_m"):
        if p in effective:
            needs.append(f"{_gib(effective[p])} GiB at {PRECISION_LABELS[p]}")
    if needs and seq_head:
        tldr.append(
            f"**What it needs:** weights {_join(needs)}, plus {_gib(seq_head.total)} GiB per "
            f"{head_tokens}-token sequence (projected)."
        )
    run = []
    if released:
        run.append(f"supported in released {_join(released)}")
    if local_fits:
        run.append(f"fits one {local.name} at {_join([f.split(' (')[0] for f in local_fits])}")
    if bf16 and seq_head and (g := _smallest_gpu(gpus, "bf16", bf16, seq_head.total, 8)):
        run.append(f"BF16 for 8 users at {head_tokens} needs an {g.name}")
    if run:
        tldr.append(f"**Can you run it today:** {'; '.join(run)}.")

    tree_url = f"https://huggingface.co/{model.repo_id}/tree/{model.revision}"
    sources = [
        {"label": "Hugging Face repo and model info at this revision", "url": tree_url},
        {
            "label": "config.json",
            "url": f"https://huggingface.co/{model.repo_id}/blob/{model.revision}/config.json",
        },
        {
            "label": "safetensors headers (parameter counts by dtype, via "
            "huggingface_hub.get_safetensors_metadata)",
            "url": tree_url,
        },
        {
            "label": "launchready memory formulas (docs/roadmap.md sections 6.1-6.3)",
            "url": "https://github.com/Tanmaypathak18/launchready/blob/main/docs/roadmap.md",
        },
    ]
    sources += [
        {"label": f"{e.engine} registry at {e.version}", "url": e.url} for e in engines if e.url
    ]
    sources += [
        {"label": f"{v.format} variant", "url": f"https://huggingface.co/{v.repo_id}"}
        for v in variants
    ]

    return {
        "model_name": model_name,
        "repo_id": model.repo_id,
        "builder": builder,
        "revision": model.revision,
        "hf_url": f"https://huggingface.co/{model.repo_id}",
        "created": (model.created_at or "")[:10] or None,
        "last_modified": (model.last_modified or "")[:10] or None,
        "license": model.license,
        "gated": model.gated,
        "generated": (today or date.today()).isoformat(),
        "version": __version__,
        "tldr": tldr,
        "arch": arch_ctx,
        "developer": developer,
        "weights": weights,
        "sequences": sequences,
        "state_dtype": state_dtype,
        "gpu_columns": gpu_columns,
        "gpu_rows": gpu_rows,
        "utilization": f"{GPU_MEMORY_UTILIZATION:.2f}",
        "reserve_gib": RUNTIME_RESERVE_BYTES // GIB,
        "hardware_notes": hardware_notes + list(arch.warnings) + list(model.warnings),
        "engines": engines,
        "variants": variant_rows,
        "sources": sources,
    }


def render(context: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    return env.get_template("brief.md.j2").render(**context)


def write_brief(repo_id: str, out_dir: Path, refresh: bool = False) -> Path:
    """Fetch everything for `repo_id`, render the brief and write briefs/<org>__<model>.md."""
    model = load_model(repo_id, refresh=refresh)
    variants = load_variants(repo_id, refresh=refresh)
    engines = check_engines(model.config, refresh=refresh)
    text = render(build_context(model, variants, engines))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{repo_id.replace('/', '__')}.md"
    path.write_text(text)
    return path

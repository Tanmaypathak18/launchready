"""Command-line interface: `launchready fit`, `launchready brief` and `launchready radar`."""

import json

import typer
from rich.console import Console
from rich.table import Table

from launchready.arch import parse_config
from launchready.hub import HubError, load_model
from launchready.memory import (
    BITS_PER_WEIGHT,
    GIB,
    GPU_MEMORY_UTILIZATION,
    RUNTIME_RESERVE_BYTES,
    kv_bytes_per_token,
    load_gpus,
    sequence_bytes,
    total_parameters,
    weight_bytes,
)
from launchready.memory import (
    fit as fit_users,
)

app = typer.Typer(
    help="Launch-readiness kit for open-weight LLMs on NVIDIA GPUs.",
    no_args_is_help=True,
)

PRECISION_LABELS = {
    "bf16": "BF16",
    "fp8": "FP8",
    "nvfp4": "NVFP4",
    "gguf-q8_0": "GGUF Q8_0",
    "gguf-q4_k_m": "GGUF Q4_K_M",
}
REFERENCE_CONTEXTS = (8_192, 32_768, 131_072)
KV_DTYPES = ("fp16", "fp8")


def _not_implemented() -> None:
    typer.echo("not implemented")
    raise typer.Exit(code=1)


def _fail(message: str, code: int) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


def _gib(n: float) -> str:
    return f"{n / GIB:,.2f}"


def _billions(n: int) -> str:
    return f"{n / 1e9:,.2f}B"


@app.command()
def fit(
    repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B."),
    context: int = typer.Option(32_768, help="Tokens per sequence (prompt + output)."),
    users: int = typer.Option(8, help="Concurrent sequences you want to serve."),
    gpu: str | None = typer.Option(None, help="Only this GPU id, e.g. rtx-5090."),
    kv_dtype: str = typer.Option("fp16", help="KV cache dtype: fp16 or fp8."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON instead of tables."),
    refresh: bool = typer.Option(False, help="Ignore the cache and re-read the Hub."),
):
    """Project memory at each precision and which NVIDIA GPUs fit the model. No weights are
    downloaded: only config.json and safetensors headers."""
    if kv_dtype not in KV_DTYPES:
        _fail(f"--kv-dtype must be one of {', '.join(KV_DTYPES)}, not {kv_dtype!r}", 2)
    gpus = load_gpus()
    if gpu:
        gpus = [g for g in gpus if g.id == gpu]
        if not gpus:
            known = ", ".join(g.id for g in load_gpus())
            _fail(f"unknown GPU {gpu!r}; choose from {known}", 2)

    try:
        model = load_model(repo_id, refresh=refresh)
        arch = parse_config(model.config)
    except (HubError, ValueError) as e:
        _fail(str(e), 1)

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

    weights = {p: weight_bytes(total, p) for p in BITS_PER_WEIGHT} if total else {}
    seq = sequence_bytes(arch, context, kv_dtype)
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
            p: {
                **vars(fit_users(g.memory_bytes, w, seq.total)),
                "native": g.is_native(p),
            }
            for p, w in weights.items()
        }
        for g in gpus
    }

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "label": "projected",
                    "repo_id": repo_id,
                    "revision": model.revision,
                    "license": model.license,
                    "gated": model.gated,
                    "created_at": model.created_at,
                    "architecture": {
                        "family": arch.family,
                        "model_type": arch.model_type,
                        "layers": arch.num_layers,
                        "layer_counts": {k: arch.count(k) for k in sorted(set(arch.layer_kinds))},
                        "num_kv_heads": arch.num_kv_heads,
                        "head_dim": arch.head_dim,
                        "max_context": arch.max_context,
                        "sliding_window": arch.sliding_window,
                    },
                    "parameters": {
                        "total": total,
                        "active": active,
                        "source": "safetensors headers",
                    },
                    "assumptions": {
                        "bits_per_weight": BITS_PER_WEIGHT,
                        "kv_dtype": kv_dtype,
                        "mamba_state_dtype": seq.mamba.dtype,
                        "mamba_state_dtype_assumed": seq.mamba.assumed,
                        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                        "runtime_reserve_bytes": RUNTIME_RESERVE_BYTES,
                        "gpu_memory": "marketed capacity read as GiB",
                    },
                    "context": context,
                    "users": users,
                    "weights_bytes": weights,
                    "sequence": {
                        "kv_bytes_per_token": kv_bytes_per_token(arch, kv_dtype),
                        "attention_bytes": seq.attention_bytes,
                        "mamba_state_bytes": seq.mamba_bytes,
                        "total_bytes": seq.total,
                    },
                    "fit": fits,
                    "warnings": warnings,
                    "notes": notes,
                },
                indent=2,
            )
        )
        return

    console = Console()
    console.print(
        f"[bold]{repo_id}[/] @ {model.revision[:7]} · license {model.license or 'not declared'}"
        " · [bold]all numbers projected[/], not measured"
    )
    layer_summary = ", ".join(
        f"{arch.count(k)} {k.replace('_', ' ')}"
        for k in ("full_attention", "sliding_attention", "mla", "mamba", "none")
        if arch.count(k)
    ).replace(" none", " MLP/MoE-only")
    if total is None:
        params = "parameters unknown (no safetensors)"
    elif arch.moe:
        shown_active = f"~{_billions(active)}" if active is not None else "unknown"
        params = f"{_billions(total)} total / {shown_active} active params"
    else:
        params = f"{_billions(total)} params"
    context_text = f"context {arch.max_context:,}" if arch.max_context else "context unknown"
    console.print(f"{arch.family} · {arch.num_layers} layers ({layer_summary}) · {params}")
    console.print(f"{context_text} · from safetensors headers and config.json\n")

    if weights:
        console.print("[bold]Weights[/] (projected: params × bits/weight ÷ 8)")
        table = Table()
        table.add_column("Precision")
        table.add_column("Bits/weight", justify="right")
        table.add_column("GiB", justify="right")
        for p, w in weights.items():
            table.add_row(PRECISION_LABELS[p], f"{BITS_PER_WEIGHT[p]:g}", _gib(w))
        console.print(table)

    has_mamba = seq.mamba.dtype is not None
    state_text = (
        f", Mamba state {seq.mamba.dtype}{' (assumed)' if seq.mamba.assumed else ''}"
        if has_mamba
        else ""
    )
    console.print(f"[bold]Per-sequence memory[/] (projected, KV {kv_dtype.upper()}{state_text})")
    table = Table()
    table.add_column("Context", justify="right")
    table.add_column("KV cache GiB", justify="right")
    if has_mamba:
        table.add_column("Mamba state GiB", justify="right")
    table.add_column("Total GiB", justify="right")
    contexts = sorted({c for c in REFERENCE_CONTEXTS if c <= (arch.max_context or c)} | {context})
    for c in contexts:
        s = sequence_bytes(arch, c, kv_dtype)
        state = [_gib(s.mamba_bytes)] if has_mamba else []
        style = "bold" if c == context else None
        table.add_row(f"{c:,}", _gib(s.attention_bytes), *state, _gib(s.total), style=style)
    console.print(table)
    console.print(f"KV per token: {kv_bytes_per_token(arch, kv_dtype):,} bytes\n")

    if weights:
        console.print(
            f"[bold]Max concurrent users at {context:,} tokens[/] (projected: "
            f"VRAM × {GPU_MEMORY_UTILIZATION:.2f} − weights − {RUNTIME_RESERVE_BYTES // GIB} GiB, "
            "÷ per-sequence)"
        )
        table = Table()
        table.add_column("GPU")
        table.add_column("GiB", justify="right")
        for p in weights:
            table.add_column(PRECISION_LABELS[p], justify="right")
        for g in gpus:
            cells = []
            for p in weights:
                r = fits[g.id][p]
                text = "—" if r["max_users"] == 0 else f"{r['max_users']:,}"
                text += "" if r["native"] else "*"
                if r["max_users"] >= users:
                    text = f"[green]{text}[/]"
                cells.append(text)
            table.add_row(g.name, f"{g.memory_gib:g}", *cells)
        console.print(table)
        console.print(
            f"[green]green[/] = at least {users} users · — = weights don't fit · "
            "* = not native on this GPU (emulated or upcast)\n"
        )

    for w in warnings:
        console.print(f"[yellow]warning:[/] {w}")
    for n in notes:
        console.print(f"note: {n}")


@app.command()
def brief(repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B.")):
    """Write a one-page LaunchReady brief for the model to briefs/ (Phase 2)."""
    _not_implemented()


@app.command()
def radar(since: str = typer.Option("30d", help="Look-back window, e.g. 30d.")):
    """List recent root-model releases from the watchlist (Phase 2)."""
    _not_implemented()

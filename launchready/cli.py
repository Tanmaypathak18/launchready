"""Command-line interface: `launchready fit`, `brief`, `radar`, and `run` for all three."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from launchready.brief import HAND_WRITTEN, unwritten_sections, write_brief
from launchready.hub import HubError, load_model
from launchready.memory import (
    BITS_PER_WEIGHT,
    GIB,
    GPU_MEMORY_UTILIZATION,
    RUNTIME_RESERVE_BYTES,
    kv_bytes_per_token,
    load_gpus,
    sequence_bytes,
)
from launchready.radar import parse_since
from launchready.radar import run as run_radar
from launchready.report import PRECISION_LABELS, fit_payload, project

app = typer.Typer(
    help="Launch-readiness kit for open-weight LLMs on NVIDIA GPUs.",
    no_args_is_help=True,
)

REFERENCE_CONTEXTS = (8_192, 32_768, 131_072)
KV_DTYPES = ("fp16", "fp8")


def _fail(message: str, code: int) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


def _gib(n: float) -> str:
    return f"{n / GIB:,.2f}"


def _billions(n: int) -> str:
    return f"{n / 1e9:,.2f}B"


def _echo_brief(path: Path) -> None:
    typer.echo(f"wrote {path}")
    missing = unwritten_sections(path.read_text())
    if len(missing) < len(HAND_WRITTEN):
        typer.echo("kept the hand-written sections: check them against the new numbers")
    if missing:
        typer.echo("TODO: hand-write " + " and ".join(f'"{title}"' for title in missing))


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
        report = project(model, context, kv_dtype, gpus)
    except (HubError, ValueError) as e:
        _fail(str(e), 1)
    arch, total, active, seq = report.arch, report.total, report.active, report.seq
    weights, fits, warnings, notes = report.weights, report.fits, report.warnings, report.notes

    if json_output:
        typer.echo(json.dumps(fit_payload(model, report, context, users, kv_dtype), indent=2))
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
def brief(
    repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B."),
    out: str = typer.Option("briefs", help="Directory to write the brief to."),
    refresh: bool = typer.Option(False, help="Ignore caches: re-read the Hub and re-clone."),
):
    """Write a one-page LaunchReady brief to briefs/<org>__<model>.md: identity, architecture,
    projected hardware needs, engine readiness and quantized variants."""
    try:
        path = write_brief(repo_id, Path(out), refresh=refresh)
    except (HubError, ValueError) as e:
        _fail(str(e), 1)
    _echo_brief(path)


@app.command()
def radar(
    since: str = typer.Option("30d", help="Look-back window, e.g. 30d or 2w."),
    out: str = typer.Option("radar", help="Directory to write radar/<date>.md to."),
):
    """List recent root-model releases from the watchlist orgs (roadmap 6.7). Run by hand."""
    try:
        parse_since(since)
    except ValueError as e:
        _fail(str(e), 2)
    path, kept = run_radar(since, Path(out))
    for org, releases in kept.items():
        typer.echo(f"{org:<12} {len(releases):>3} root-model repos")
    typer.echo(f"wrote {path}")


@app.command()
def run(
    repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B."),
    context: int = typer.Option(32_768, help="Tokens per sequence (prompt + output)."),
    users: int = typer.Option(8, help="Concurrent sequences you want to serve."),
    since: str = typer.Option("30d", help="Radar look-back window, e.g. 30d or 2w."),
    out: str = typer.Option(".", help="Base directory for fit/, briefs/ and radar/."),
    refresh: bool = typer.Option(False, help="Ignore caches: re-read the Hub and re-clone."),
):
    """Generate everything that can be generated for a model: the fit as JSON in fit/, the brief
    in briefs/ and a release radar in radar/. Hand-written brief sections are kept."""
    try:
        parse_since(since)
    except ValueError as e:
        _fail(str(e), 2)
    base = Path(out)
    try:
        model = load_model(repo_id, refresh=refresh)
        report = project(model, context, "fp16", load_gpus())
        fit_path = base / "fit" / f"{repo_id.replace('/', '__')}.json"
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = fit_payload(model, report, context, users, "fp16")
        fit_path.write_text(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"wrote {fit_path}")
        _echo_brief(write_brief(repo_id, base / "briefs", refresh=refresh))
    except (HubError, ValueError) as e:
        _fail(str(e), 1)
    radar_path, _ = run_radar(since, base / "radar")
    typer.echo(f"wrote {radar_path}")

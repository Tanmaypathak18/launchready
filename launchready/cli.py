"""Command-line interface: `launchready fit`, `launchready brief` and `launchready radar`."""

import typer

app = typer.Typer(
    help="Launch-readiness kit for open-weight LLMs on NVIDIA GPUs.",
    no_args_is_help=True,
)


def _not_implemented() -> None:
    typer.echo("not implemented")
    raise typer.Exit(code=1)


@app.command()
def fit(repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B.")):
    """Project memory at each precision and which NVIDIA GPUs fit the model (Phase 1)."""
    _not_implemented()


@app.command()
def brief(repo_id: str = typer.Argument(..., help="Hugging Face repo ID, e.g. Qwen/Qwen3-4B.")):
    """Write a one-page LaunchReady brief for the model to briefs/ (Phase 2)."""
    _not_implemented()


@app.command()
def radar(since: str = typer.Option("30d", help="Look-back window, e.g. 30d.")):
    """List recent root-model releases from the watchlist (Phase 2)."""
    _not_implemented()

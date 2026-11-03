"""Smoke tests: the CLI loads and every subcommand is wired up."""

from typer.testing import CliRunner

from launchready.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fit", "brief", "radar"):
        assert command in result.output


def test_brief_not_implemented():
    result = runner.invoke(app, ["brief", "Qwen/Qwen3-4B"])
    assert result.exit_code == 1
    assert "not implemented" in result.output

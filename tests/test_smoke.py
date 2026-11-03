"""Smoke tests: the CLI loads and every subcommand is wired up."""

from typer.testing import CliRunner

from launchready.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fit", "brief", "radar"):
        assert command in result.output


def test_radar_rejects_bad_since():
    result = runner.invoke(app, ["radar", "--since", "30"])
    assert result.exit_code == 2
    assert "30d" in result.output

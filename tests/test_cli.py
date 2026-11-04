"""`launchready fit` end to end against saved hub fixtures (no network)."""

import json

from typer.testing import CliRunner

from launchready import brief as brief_module
from launchready import cli as cli_module
from launchready.cli import app
from launchready.hub import HubError

runner = CliRunner()


def test_fit_json_qwen3_4b(hub_cache):
    result = runner.invoke(app, ["fit", "Qwen/Qwen3-4B", "--gpu", "rtx-5090", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["label"] == "projected"
    assert data["revision"].startswith("1cfa9a7")
    assert data["parameters"]["total"] == 4_022_468_096
    assert data["weights_bytes"]["bf16"] == 8_044_936_192
    # 36 layers x 2 x 8 KV heads x 128 x 2 bytes = 147,456 B/token; x 32,768 = 4.5 GiB.
    assert data["sequence"]["total_bytes"] == 147_456 * 32_768
    gpu = data["fit"]["rtx-5090"]
    assert gpu["bf16"]["max_users"] == 4
    assert gpu["bf16"]["native"] is True


def test_fit_table_labels_projected_and_assumed_state(hub_cache):
    result = runner.invoke(
        app, ["fit", "ibm-granite/granite-4.0-h-tiny", "--context", "8192", "--users", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "projected" in result.output.lower()
    assert "assumed" in result.output.lower()


def test_fit_warns_hybrid_mamba_vllm_gap(hub_cache):
    result = runner.invoke(app, ["fit", "ibm-granite/granite-4.0-h-tiny", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    hybrid_warnings = [w for w in data["warnings"] if w.startswith("hybrid Mamba")]
    assert len(hybrid_warnings) == 1
    assert "projected 118, measured 50.6" in hybrid_warnings[0]
    assert data["sequence"]["kv_tokens_comparable_to_vllm"] is False


def test_fit_no_hybrid_warning_for_dense(hub_cache):
    result = runner.invoke(app, ["fit", "Qwen/Qwen3-4B", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert not any(w.startswith("hybrid Mamba") for w in data["warnings"])
    assert data["sequence"]["kv_tokens_comparable_to_vllm"] is True


def test_fit_unknown_gpu(hub_cache):
    result = runner.invoke(app, ["fit", "Qwen/Qwen3-4B", "--gpu", "rtx-9999"])
    assert result.exit_code == 2
    assert "rtx-9999" in result.output


def _stub_run(monkeypatch, tmp_path):
    def fake_radar(since, out_dir):
        out_dir.mkdir(parents=True)
        path = out_dir / "2026-11-03.md"
        path.write_text(f"radar {since}\n")
        return path, {}

    monkeypatch.setattr(cli_module, "run_radar", fake_radar)
    monkeypatch.setattr(brief_module, "check_engines", lambda config, refresh=False: [])
    monkeypatch.setattr(brief_module, "load_variants", lambda repo_id, refresh=False: [])
    return ["--out", str(tmp_path)]


def test_run_generates_fit_brief_and_radar(hub_cache, monkeypatch, tmp_path):
    args = _stub_run(monkeypatch, tmp_path)
    result = runner.invoke(app, ["run", "Qwen/Qwen3-4B", "--since", "2w", *args])
    assert result.exit_code == 0, result.output

    data = json.loads((tmp_path / "fit" / "Qwen__Qwen3-4B.json").read_text())
    direct = json.loads(runner.invoke(app, ["fit", "Qwen/Qwen3-4B", "--json"]).output)
    assert data == direct
    assert data["label"] == "projected"
    assert (tmp_path / "briefs" / "Qwen__Qwen3-4B.md").read_text().startswith("# Qwen3-4B")
    assert (tmp_path / "radar" / "2026-11-03.md").read_text() == "radar 2w\n"


def test_run_rejects_a_bad_window_before_writing(hub_cache, monkeypatch, tmp_path):
    args = _stub_run(monkeypatch, tmp_path)
    result = runner.invoke(app, ["run", "Qwen/Qwen3-4B", "--since", "soon", *args])
    assert result.exit_code == 2
    assert list(tmp_path.iterdir()) == []


def test_run_stops_cleanly_on_a_hub_error(monkeypatch, tmp_path):
    def gated(repo_id, refresh=False):
        raise HubError("org/name is gated: accept its license on huggingface.co and set HF_TOKEN")

    monkeypatch.setattr(cli_module, "load_model", gated)
    args = _stub_run(monkeypatch, tmp_path)
    result = runner.invoke(app, ["run", "org/name", *args])
    assert result.exit_code == 1
    assert "gated" in result.output
    assert list(tmp_path.iterdir()) == []

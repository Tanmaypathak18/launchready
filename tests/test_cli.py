"""`launchready fit` end to end against saved hub fixtures (no network)."""

import json

from typer.testing import CliRunner

from launchready.cli import app

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

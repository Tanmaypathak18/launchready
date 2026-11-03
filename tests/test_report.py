"""launchready.report: the fit shared by the CLI and the Space, on saved hub fixtures."""

import json

import pytest

from launchready.hub import load_model
from launchready.memory import GIB, load_gpus, weight_bytes
from launchready.report import HYBRID_VLLM_WARNING, hub_file_sizes, project
from launchready.variants import load_variants

LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def test_project_without_hub_files_uses_projected_weights(hub_cache):
    report = project(load_model(LIGHTNING), 32_768, "fp16", load_gpus())
    assert report.weights == report.projected_weights
    assert report.fits["rtx-5090"]["nvfp4"]["max_users"] == 40
    assert HYBRID_VLLM_WARNING in report.warnings


def test_hub_file_sizes_change_the_fit(hub_cache):
    model = load_model(LIGHTNING)
    hub_files = hub_file_sizes(model, load_variants(LIGHTNING))
    report = project(model, 32_768, "fp16", load_gpus(), hub_files)
    assert report.fits["rtx-5090"]["nvfp4"]["max_users"] == 28
    assert report.fits["rtx-5090"]["gguf-q4_k_m"]["max_users"] == 13
    # The projection alone says 4 users; the real file doesn't load.
    assert report.fits["rtx-4090"]["gguf-q4_k_m"]["max_users"] == 0
    assert report.projected_weights["nvfp4"] < report.weights["nvfp4"]


def test_hub_file_sizes_ignores_files_that_are_not_plain_bf16(hub_cache):
    model = load_model(LIGHTNING)
    assert hub_file_sizes(model, [])["bf16"] == (LIGHTNING, model.safetensors_bytes)
    model.safetensors_bytes *= 2  # FP32 weights, or consolidated.safetensors beside the shards
    assert "bf16" not in hub_file_sizes(model, [])


@pytest.mark.parametrize(
    ("scale", "kept"),
    [(1.0, True), (1.24, True), (0.76, True), (1.26, False), (0.74, False)],
)
def test_hub_file_sizes_keeps_base_files_within_25_percent(hub_cache, scale, kept):
    model = load_model(LIGHTNING)
    model.safetensors_bytes = int(weight_bytes(32_913_266_240, "bf16") * scale)
    assert ("bf16" in hub_file_sizes(model, [])) is kept


def test_hub_file_sizes_filter_leaves_quantized_variants_alone(hub_cache):
    model = load_model(LIGHTNING)
    model.safetensors_bytes *= 2
    hub_files = hub_file_sizes(model, load_variants(LIGHTNING))
    assert "bf16" not in hub_files and "nvfp4" in hub_files


def test_project_has_same_warnings_as_the_cli_json(hub_cache):
    from typer.testing import CliRunner

    from launchready.cli import app

    result = CliRunner().invoke(app, ["fit", LIGHTNING, "--json"])
    data = json.loads(result.output)
    report = project(load_model(LIGHTNING), 32_768, "fp16", load_gpus())
    assert data["warnings"] == report.warnings
    assert data["notes"] == report.notes
    assert data["sequence"]["total_bytes"] == report.seq.total
    assert report.seq.total < GIB

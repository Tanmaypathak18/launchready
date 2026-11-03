"""`launchready brief` end to end on saved fixtures: hub and variants caches, fake engines."""

from typer.testing import CliRunner

from launchready import brief as brief_module
from launchready.brief import build_context
from launchready.cli import app
from launchready.engines import EngineStatus
from launchready.hub import load_model
from launchready.variants import load_variants

LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
runner = CliRunner()

FAKE_ENGINES = [
    EngineStatus(
        engine="vLLM",
        status="release",
        version="v0.32.0",
        path="vllm/model_executor/models/registry.py",
        line=186,
        text='"NemotronHForCausalLM": ("nemotron_h", "NemotronHForCausalLM"),',
        url="https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/registry.py#L186",
        notes=[],
    ),
    EngineStatus(
        engine="SGLang",
        status="main-only",
        version="main @ abcdef1",
        path="python/sglang/srt/models/nemotron_h.py",
        line=1357,
        text="EntryClass = [NemotronHForCausalLM, NemotronHPuzzleForCausalLM]",
        url="https://github.com/sgl-project/sglang/blob/abcdef1/python/sglang/srt/models/nemotron_h.py#L1357",
        notes=[],
    ),
    EngineStatus("llama.cpp", "not-found", "v0.8.0", None, None, None, None, ["→ Ollama via GGUF"]),
    EngineStatus("Transformers", "unknown", None, None, None, None, None, ["clone failed"]),
]


def _bf16_row(model):
    variants = load_variants(LIGHTNING)
    return build_context(model, variants, FAKE_ENGINES)["weights"][0]


def test_brief_drops_a_base_file_far_from_the_bf16_projection(hub_cache):
    """The 25% filter reaches the brief: no Hub size, and the fit uses the projection."""
    model = load_model(LIGHTNING)
    assert _bf16_row(model)["real"] is not None
    model.safetensors_bytes *= 2
    row = _bf16_row(model)
    assert row["real"] is None and row["repo_link"] is None
    assert row["projected"] == "61.31"


def _render(hub_cache, monkeypatch, tmp_path):
    monkeypatch.setattr(brief_module, "check_engines", lambda config, refresh=False: FAKE_ENGINES)
    result = runner.invoke(app, ["brief", LIGHTNING, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    path = tmp_path / "nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md"
    assert str(path.name) in result.output
    return path.read_text()


def test_brief_has_every_template_section(hub_cache, monkeypatch, tmp_path):
    text = _render(hub_cache, monkeypatch, tmp_path)
    for heading in (
        "# NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16: LaunchReady brief",
        "## TL;DR",
        "## Identity",
        "## Architecture",
        "## What this means for developers",
        "## Hardware requirements",
        "## Engine readiness",
        "## Available variants",
        "## Ecosystem impact",
        "## Open questions and risks",
        "## Sources",
    ):
        assert heading in text, heading
    assert "Projected unless marked measured" in text


def test_brief_hand_written_sections_are_todo(hub_cache, monkeypatch, tmp_path):
    text = _render(hub_cache, monkeypatch, tmp_path)
    ecosystem = text.split("## Ecosystem impact")[1].split("## Open questions")[0]
    questions = text.split("## Open questions and risks")[1].split("## Sources")[0]
    assert "TODO" in ecosystem
    assert "TODO" in questions


def test_brief_numbers_and_sources(hub_cache, monkeypatch, tmp_path):
    text = _render(hub_cache, monkeypatch, tmp_path)
    assert "32.91B" in text  # header count, not the nominal 30B
    assert "openmdw-1.1" in text
    assert "17.24" in text  # projected NVFP4 GiB
    assert "20.08" in text  # real NVFP4 repo GiB from the variants cache
    assert "61.31" in text  # real BF16 repo GiB (65,827,374,264 bytes)
    assert "| vLLM | release | v0.32.0 |" in text
    assert "registry.py#L186" in text
    assert "main-only" in text
    sources = text.split("## Sources")[1]
    assert (
        "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/"
        "a9904d24bcc1d289a1950fa9d2b978c47cf903b9" in sources
    )
    assert "https://github.com/vllm-project/vllm/blob/v0.32.0/" in sources
    assert "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" in sources

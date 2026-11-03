"""The Space's verdicts and wording, on saved hub fixtures (no network, no Gradio)."""

import csv
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from launchready import hub
from launchready.memory import GIB, load_gpus
from launchready.report import HYBRID_VLLM_WARNING
from space import logic
from space.logic import LIGHTNING, BadInput, check, normalize_repo_id, whole_number

ROOT = Path(__file__).parent.parent
RESPONSE_ERRORS = {
    "gated": hub.GatedError("x is gated: accept its license on huggingface.co and set HF_TOKEN"),
    "no-config": hub.NoConfigError("x has no config.json"),
    "unavailable": hub.HubUnavailableError("429"),
    "missing": hub.HubError("x not found (or private without HF_TOKEN)"),
}


@pytest.fixture(autouse=True)
def no_live_variants(monkeypatch):
    """Only Lightning has saved variants; any other lookup fails instead of reaching the Hub."""
    real = logic.load_variants

    def saved_only(repo_id):
        if repo_id != LIGHTNING:
            raise OSError("no saved variants")
        return real(repo_id)

    monkeypatch.setattr(logic, "load_variants", saved_only)


@pytest.mark.parametrize(
    "text",
    [
        "Qwen/Qwen3-4B",
        " Qwen/Qwen3-4B/ ",
        "https://huggingface.co/Qwen/Qwen3-4B",
        "huggingface.co/Qwen/Qwen3-4B/tree/main",
        "https://hf.co/Qwen/Qwen3-4B?library=transformers",
        "https://huggingface.co/models/Qwen/Qwen3-4B",
    ],
)
def test_normalize_repo_id_accepts_ids_and_links(text):
    assert normalize_repo_id(text) == "Qwen/Qwen3-4B"


@pytest.mark.parametrize(
    "text", ["", None, "a b", "not/a/model", "../etc", "https://example.com/a/b", "<script>"]
)
def test_normalize_repo_id_rejects_bad_ids(text):
    with pytest.raises(BadInput):
        normalize_repo_id(text)


def test_normalize_repo_id_rejects_links_to_datasets():
    with pytest.raises(BadInput):
        normalize_repo_id("https://huggingface.co/datasets/org/name")


@pytest.mark.parametrize("value", [None, 0, -1, 1.5])
def test_whole_number_rejects(value):
    with pytest.raises(BadInput):
        whole_number(value, "Users")


def test_lightning_fits_a_5090_at_nvfp4(hub_cache):
    page = check(LIGHTNING, "rtx-5090", 32_768, 8)
    assert "## Fits (projected)" in page.verdict
    assert "NVFP4" in page.verdict
    assert "room for up to 28 users" in page.verdict
    assert logic.HYBRID_CAUTION in page.verdict
    assert logic.MEMORY_ONLY in page.verdict
    assert "× 0.90" in page.verdict


def test_lightning_does_not_fit_a_4090(hub_cache):
    """The projection alone gives 4 users at Q4_K_M; the real 23.53 GiB file doesn't load."""
    page = check(LIGHTNING, "rtx-4090", 32_768, 8)
    assert "## Does not fit (projected)" in page.verdict
    assert "23.53 GiB" in page.verdict
    assert "19.60 GiB" in page.verdict
    assert logic.HYBRID_CAUTION not in page.verdict  # no user count to caution about


def test_too_few_users(hub_cache):
    page = check(LIGHTNING, "rtx-5090", 32_768, 50)
    assert "## Does not fit 50 users (projected)" in page.verdict
    assert "only room for 28" in page.verdict


def test_weights_too_large(hub_cache):
    assert (
        "## Does not fit (projected)"
        in check("deepseek-ai/DeepSeek-V3", "rtx-5090", 32_768, 8).verdict
    )


def test_unsupported_precisions_are_not_counted(hub_cache):
    """FP8 and NVFP4 would fit an A100 40GB on paper, but it supports neither."""
    page = check(LIGHTNING, "a100-40gb", 8_192, 1)
    assert "using the **GGUF Q8_0**" in page.verdict
    assert "No, not counted" in page.memory


def _supported_cell(page, label):
    row = next(line for line in page.memory.splitlines() if line.startswith(f"| {label} |"))
    return row.strip("| ").split(" | ")[-1]


def test_qwen_on_a_4090_uses_fp8_which_ada_supports(hub_cache):
    gpu = next(g for g in load_gpus() if g.id == "rtx-4090")
    assert gpu.architecture == "Ada" and gpu.is_native("fp8") and not gpu.is_native("nvfp4")
    page = check("Qwen/Qwen3-4B", "rtx-4090", 32_768, 8)
    assert "only room for 3" in page.verdict
    assert "using the **FP8** version" in page.verdict
    assert _supported_cell(page, "FP8") == "Yes"
    assert _supported_cell(page, "NVFP4") == "No, not counted"


@pytest.mark.parametrize("gpu_id", [g.id for g in load_gpus()])
@pytest.mark.parametrize("users", [1, 8, 50])
def test_verdict_never_names_a_version_the_table_says_is_unsupported(hub_cache, gpu_id, users):
    page = check(LIGHTNING, gpu_id, 32_768, users)
    named = re.search(r"using the \*\*(.+?)\*\* version", page.verdict)
    if named:
        assert _supported_cell(page, named.group(1)) == "Yes"


def test_words_we_must_not_use(hub_cache):
    page = check(LIGHTNING, "rtx-5090", 32_768, 8)
    text = "\n".join(vars(page).values()).lower()
    assert "native" not in text
    assert "measured on an rtx" not in text
    assert "HF_TOKEN".lower() not in text


def test_dense_model_has_no_hybrid_caution(hub_cache):
    page = check("Qwen/Qwen3-4B", "rtx-4090", 32_768, 1)
    assert logic.HYBRID_CAUTION not in page.verdict


def test_failed_variants_lookup_is_not_fatal(hub_cache):
    """As for a no-org repo such as gpt2, where the variants lookup raises."""
    page = check("Qwen/Qwen3-4B", "rtx-4090", 32_768, 1)
    assert "couldn" in page.warnings
    assert "(projected)" in page.verdict


def test_same_warnings_as_the_cli(hub_cache):
    page = check(LIGHTNING, "rtx-5090", 32_768, 8)
    assert logic.esc(HYBRID_VLLM_WARNING) in page.warnings


def test_no_safetensors_is_cant_tell(hub_cache, monkeypatch):
    real = logic.load_model

    def no_params(repo_id):
        model = real(repo_id)
        model.parameters = {}
        return model

    monkeypatch.setattr(logic, "load_model", no_params)
    assert "## Can't tell" in check(LIGHTNING, "rtx-5090", 32_768, 8).verdict


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("gated", "License required"),
        ("no-config", "original model"),
        ("unavailable", "Try again"),
        ("missing", "Model not found"),
    ],
)
def test_hub_errors_are_worded_for_this_page(monkeypatch, key, expected):
    def fail(repo_id):
        raise RESPONSE_ERRORS[key]

    monkeypatch.setattr(logic, "load_model", fail)
    page = check("org/name", "rtx-5090", 32_768, 8)
    assert expected in page.verdict
    assert "HF_TOKEN" not in page.verdict


def test_non_text_model_gets_a_message(hub_cache, monkeypatch):
    model = SimpleNamespace(config={}, warnings=[], parameters={}, repo_id="org/name")
    model.safetensors_bytes, model.license = 0, "mit"
    monkeypatch.setattr(logic, "load_model", lambda repo_id: model)
    assert "## Can't size this repo" in check("org/name", "rtx-5090", 32_768, 8).verdict


def test_unexpected_failure_gets_a_message(monkeypatch):
    def fail(repo_id):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(logic, "load_model", fail)
    page = check("org/name", "rtx-5090", 32_768, 8)
    assert "Something went wrong" in page.verdict
    assert "secret detail" not in page.verdict


def test_repo_text_is_escaped(hub_cache):
    report = SimpleNamespace(warnings=["config field x='<img src=x onerror=alert(1)>'"], notes=[])
    text = logic.warnings_text(report, [])
    assert "<img" not in text
    assert "&lt;img" in text


def test_apostrophes_survive_escaping():
    assert logic.esc("vLLM's") == "vLLM's"


def test_how_calculated_shows_formula_and_unverified_gpus():
    text = logic.how_calculated_text(load_gpus())
    assert "× 0.90" in text and "2 GiB" in text
    assert "B200" in text and "DGX Spark" in text


def test_examples_use_catalog_gpus():
    ids = {g.id for g in load_gpus()}
    assert len(logic.EXAMPLES) == 6
    assert all(example[1] in ids for example in logic.EXAMPLES)


def _validation():
    rows = list(csv.DictReader((ROOT / "results" / "memory_validation.csv").open()))
    return {(r["repo_id"], r["metric"], r["label"]): r for r in rows}


@pytest.mark.parametrize(
    "repo_id",
    ["Qwen/Qwen3-4B", "allenai/OLMoE-1B-7B-0924-Instruct", "ibm-granite/granite-4.0-h-tiny"],
)
def test_measured_section_matches_the_csv(repo_id):
    rows, text = _validation(), logic.measured_text()
    weights = [
        float(rows[(repo_id, m, label)]["value"]) / GIB
        for m, label in (("weights_bytes", "projected"), ("vllm_weights_bytes", "measured"))
    ]
    users = [
        float(rows[(repo_id, "max_concurrency", label)]["value"])
        for label in ("projected", "measured")
    ]
    row = next(line for line in text.splitlines() if line.startswith(f"| {repo_id} |"))
    assert f"{weights[0]:.2f} / {weights[1]:.2f}" in row
    assert f"{users[0]:.1f} / {users[1]:.1f}" in row
    assert rows[(repo_id, "max_concurrency", "measured")]["revision"].startswith(row.split("`")[1])


def test_measured_section_environment_comes_from_the_csv():
    rows, text = _validation(), logic.measured_text()
    row = rows[("Qwen/Qwen3-4B", "max_concurrency", "measured")]
    assert f"Measured on {row['date_utc'][:10]}" in text
    assert row["driver"] in text and f"vLLM {row['vllm']}" in text


def test_measured_links_are_pinned():
    text = logic.measured_text()
    assert f"{logic.RESULTS_COMMIT}/results/memory_validation.png" in text
    assert "{" not in text

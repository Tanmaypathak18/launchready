"""Radar: look-back parsing, root-model filtering and the digest (no network)."""

from datetime import UTC, datetime, timedelta

import pytest

from launchready.radar import CAVEAT, Release, filter_releases, parse_since, render


def _r(repo_id, tags=(), days_ago=1, pipeline="text-generation", library=None, params=None):
    return Release(
        repo_id=repo_id,
        created_at=datetime(2026, 11, 2, tzinfo=UTC) - timedelta(days=days_ago),
        tags=list(tags),
        pipeline_tag=pipeline,
        library_name=library,
        license="apache-2.0",
        parameters=params,
    )


@pytest.mark.parametrize(
    ("text", "expected"), [("30d", timedelta(days=30)), ("2w", timedelta(weeks=2))]
)
def test_parse_since(text, expected):
    assert parse_since(text) == expected


@pytest.mark.parametrize("text", ["30", "d", "-3d", "3m"])
def test_parse_since_rejects(text):
    with pytest.raises(ValueError):
        parse_since(text)


def test_filter_keeps_roots_and_same_org_finetunes():
    releases = [
        _r("Qwen/Qwen4-8B-Base"),
        _r("Qwen/Qwen4-8B", ["base_model:finetune:Qwen/Qwen4-8B-Base"]),
        _r("Qwen/Qwen4-8B-FP8", ["base_model:quantized:Qwen/Qwen4-8B"]),
        _r("Qwen/Qwen4-8B-GGUF", ["gguf"]),
        _r("Qwen/Qwen4-8B-AWQ"),
        _r("Qwen/Qwen4-8B-lora", ["base_model:adapter:Qwen/Qwen4-8B"]),
        _r("Qwen/Qwen4-8B-peft", library="peft"),
        _r("Qwen/Llama-tune", ["base_model:finetune:meta-llama/Llama-4"]),
        _r("Qwen/Qwen4-merge", ["base_model:merge:other/thing"]),
    ]
    kept, skipped = filter_releases(releases)
    assert [r.repo_id for r in kept] == ["Qwen/Qwen4-8B-Base", "Qwen/Qwen4-8B"]
    assert kept[1].parent == "Qwen/Qwen4-8B-Base"
    assert skipped == {"quantized": 3, "adapter": 2, "fine-tune of another org": 2}


def test_render_nests_children_and_counts_skips():
    kept, skipped = filter_releases(
        [
            _r("Qwen/Qwen4-8B-Base", days_ago=3, params=8_190_000_000),
            _r("Qwen/Qwen4-8B", ["base_model:finetune:Qwen/Qwen4-8B-Base"], days_ago=2),
            _r("Qwen/Qwen4-8B-FP8", ["base_model:quantized:Qwen/Qwen4-8B"]),
        ]
    )
    text = render(
        {"Qwen": kept, "nvidia": []},
        {"Qwen": skipped, "nvidia": {}},
        missing=["not-an-org"],
        since="30d",
        today=datetime(2026, 11, 2, tzinfo=UTC).date(),
    )
    assert "# Release radar: 2026-11-02" in text
    assert "[Qwen/Qwen4-8B-Base](https://huggingface.co/Qwen/Qwen4-8B-Base)" in text
    assert "↳ [Qwen/Qwen4-8B]" in text
    assert "8.19B" in text
    assert "1 quantized" in text
    assert "not-an-org" in text
    assert "No root-model releases" in text  # nvidia section
    assert text.rstrip().endswith(CAVEAT)

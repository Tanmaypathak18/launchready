"""The one-page site: links resolve, and its numbers still match the committed data."""

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts import check_site as site
from scripts.check_site import (
    ROOT,
    check_site,
    check_tree,
    heading_anchors,
    parse_page,
    text_problems,
)


def page_text() -> str:
    return " ".join(" ".join(parse_page().text).split())


def validation(repo_id: str, metric: str, label: str) -> float:
    with open(ROOT / "results" / "memory_validation.csv", newline="") as f:
        for row in csv.DictReader(f):
            if (row["repo_id"], row["metric"], row["label"]) == (repo_id, metric, label):
                return float(row["value"])
    raise KeyError((repo_id, metric, label))


def test_site_links_and_rules():
    problems, external = check_site()
    assert problems == []
    assert external


def test_heading_anchors_match_github():
    anchors = heading_anchors(
        "### Qwen3-4B's one-liner fails on `--flag` in 0.32.0 (audit, Kaggle)"
    )
    assert anchors == {"qwen3-4bs-one-liner-fails-on---flag-in-0320-audit-kaggle"}


def test_check_catches_a_broken_link(tmp_path):
    page = tmp_path / "index.html"
    repo = "https://github.com/Tanmaypathak18/launchready/blob/main"
    page.write_text(
        f'<a href="{repo}/results/nope.csv">x</a>'
        f'<a href="{repo}/docs/friction_log.md#nope">x</a>'
        f'<a href="#nope">x</a><img src="../results/memory_validation.png">'
    )
    problems, _ = check_site(page)
    assert len(problems) == 4


@pytest.mark.parametrize(
    ("repo_id", "gap"),
    [("Qwen/Qwen3-4B", "−3.9%"), ("allenai/OLMoE-1B-7B-0924-Instruct", "−5.8%")],
)
def test_validation_numbers(repo_id, gap):
    projected = validation(repo_id, "max_concurrency", "projected")
    measured = validation(repo_id, "max_concurrency", "measured")
    text = page_text()
    assert f"projected {projected:.2f}, measured {measured:.2f}, a gap of {gap}" in text
    assert f"{(projected / measured - 1) * 100:+.1f}%".replace("-", "−") == gap


def test_hybrid_miss():
    projected = validation("ibm-granite/granite-4.0-h-tiny", "max_concurrency", "projected")
    measured = validation("ibm-granite/granite-4.0-h-tiny", "max_concurrency", "measured")
    assert round((projected / measured - 1) * 100) == 133
    assert f"projected {projected:.2f}, measured {measured:.2f}" in page_text()


def test_gguf_bits_per_weight():
    text = page_text()
    gguf = ROOT / "results" / "bench" / "gguf"
    for name, publisher in [
        ("bartowski__Lightning-Q4_K_M", "bartowski"),
        ("unsloth__Lightning-UD-Q4_K_M", "unsloth"),
        ("AtomicChat__Lightning-Q4_K_M", "AtomicChat"),
    ]:
        bits = json.loads((gguf / f"{name}.json").read_text())["bits_per_weight"]
        assert f"{bits:.2f} ({publisher})" in text


def test_checklist_counts():
    checklist = Path(ROOT / "docs" / "readiness_checklist.md").read_text()
    rows = re.findall(r"^\| \d\.\d \|.*$", checklist, flags=re.M)
    assert len(rows) == 27
    assert sum("| **ready** |" in r for r in rows) == 1
    assert sum("| **not ready** |" in r for r in rows) == 5
    assert sum("| unknown |" in r for r in rows) == 21
    assert "27 checks in all" in page_text()


def test_public_tree_is_clean():
    assert check_tree() == []


def test_tree_check_catches_private_text(monkeypatch):
    word = hashlib.sha256(b"privateword").hexdigest()
    pair = hashlib.sha256(b"sample phrase").hexdigest()
    monkeypatch.setattr(site, "PRIVATE_WORD_HASHES", {word})
    monkeypatch.setattr(site, "JOB_WORD_HASHES", {pair})
    assert text_problems("PrivateWord here", authored=False) == {"private word"}
    assert text_problems("A sample  phrase", authored=True) == {"job-search wording"}
    assert text_problems("A sample phrase", authored=False) == set()


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("/" + "Users/someone/x.py", "home directory path"),
        ("/" + "home/someone/x.py", "home directory path"),
        ("someone" + "@mail.org", "email address"),
        ("host-" + "MacBook-Air", "laptop hostname"),
        ("someone@laptop." + "local", "laptop hostname"),
    ],
)
def test_tree_check_catches_local_details(text, category):
    assert category in text_problems(text, authored=False)


def test_allowed_url_passes_where_it_is_linked():
    url = site.ALLOWED_URL
    for text in (url, f"Live tool: {url} (free)", f"[{url}]({url})", f'<a href="{url}">{url}</a>'):
        assert text_problems(text, authored=True) == set()


def test_the_word_in_the_allowed_url_stays_banned_everywhere_else():
    url = site.ALLOWED_URL
    word = url.split("/")[-2]
    assert hashlib.sha256(word.encode()).hexdigest() in site.PRIVATE_WORD_HASHES
    for text in (
        word,
        f"contact {word} today",
        f"{url} and also {word}",
        f"{url}/tree/main",
        f"{url}-fork",
        f"{url}?x=1",
        f"{url}#top",
        url.removesuffix("launchready") + "other",
        url.removeprefix("https://"),
        url.replace("https://", "http://"),
        f"x{url}",
        url.upper(),
    ):
        assert text_problems(text, authored=False) == {"private word"}, text


def test_tree_check_allows_redacted_paths():
    assert text_problems("/" + "Users/<user>/x and <home>/.cache", authored=False) == set()


def test_brief_makes_no_unmeasured_5090_claim():
    brief = (ROOT / "briefs" / "nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md").read_text()
    template = (ROOT / "templates" / "brief.md.j2").read_text()
    assert "shows Blackwell" not in brief + template
    assert "projected to fit on a 32 GiB RTX 5090" in brief
    assert "has not been measured on one yet" in brief


def test_page_and_brief_agree_on_nvfp4():
    brief = (ROOT / "briefs" / "nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.md").read_text()
    assert "| NVFP4 | 4.5 | 17.24 | 20.08 (+16%) |" in brief
    assert "projected 17.24 GiB, measured 20.08 GiB" in page_text()

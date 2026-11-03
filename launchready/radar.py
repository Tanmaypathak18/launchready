"""One-off release radar (Phase 2; docs/roadmap.md section 6.7).

Lists recent root-model releases from the watchlist orgs within a look-back window, skipping
quantized, GGUF, adapter repos and fine-tunes of other orgs' models, and writes radar/<date>.md.
A builder's own fine-tunes (an Instruct model of its own Base) are kept, nested under the base.
Run by hand; no scheduling.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

WATCHLIST = (
    "nvidia", "meta-llama", "google", "Qwen", "deepseek-ai", "mistralai", "microsoft", "openai",
    "moonshotai", "zai-org", "MiniMaxAI", "ibm-granite", "allenai",
)  # fmt: skip
PAGE_LIMIT = 500  # newest repos read per org; stops earlier at the cutoff
_QUANT_NAME = re.compile(
    r"gguf|awq|gptq|fp8|nvfp4|mxfp4|int4|int8|w4a16|w8a8|mlx|bnb|[-_]\d+bit|exl2", re.IGNORECASE
)
_SINCE = re.compile(r"^(\d+)([dw])$")
CAVEAT = (
    "Dates are Hub repo-creation dates; a repo published later than it was created can look "
    "older than its public release."
)


@dataclass
class Release:
    repo_id: str
    created_at: datetime
    tags: list[str]
    pipeline_tag: str | None
    library_name: str | None
    license: str | None
    parameters: int | None
    parent: str | None = None  # same-org base model this repo was fine-tuned from


def parse_since(text: str) -> timedelta:
    match = _SINCE.match(text.strip())
    if not match or int(match.group(1)) == 0:
        raise ValueError(f"--since must look like 30d or 2w, not {text!r}")
    n, unit = int(match.group(1)), match.group(2)
    return timedelta(days=n) if unit == "d" else timedelta(weeks=n)


def _skip_reason(r: Release) -> tuple[str | None, str | None]:
    """(reason to skip or None, same-org parent or None)."""
    org = r.repo_id.split("/")[0]
    tags = [t.lower() for t in r.tags]
    if any(t.startswith("base_model:adapter:") for t in tags) or r.library_name == "peft":
        return "adapter", None
    if "peft" in tags or "lora" in tags:
        return "adapter", None
    if any(t.startswith("base_model:quantized:") for t in tags) or "gguf" in tags:
        return "quantized", None
    if r.library_name in ("gguf", "mlx") or _QUANT_NAME.search(r.repo_id.split("/")[-1]):
        return "quantized", None
    parent = None
    for prefix in ("base_model:finetune:", "base_model:merge:"):
        for t in r.tags:
            if t.lower().startswith(prefix):
                base = t[len(prefix) :]
                if base.split("/")[0].lower() != org.lower():
                    return "fine-tune of another org", None
                parent = base
    return None, parent


def filter_releases(releases: list[Release]) -> tuple[list[Release], dict[str, int]]:
    """Root models (and same-org fine-tunes) in the given order, plus skip counts by reason."""
    kept, skipped = [], Counter()
    for r in releases:
        reason, parent = _skip_reason(r)
        if reason:
            skipped[reason] += 1
        else:
            r.parent = parent
            kept.append(r)
    return kept, dict(skipped)


def fetch(org: str, cutoff: datetime) -> list[Release]:
    """An org's repos created after `cutoff`, newest first."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    releases = []
    models = api.list_models(
        author=org,
        sort="created_at",
        limit=PAGE_LIMIT,
        expand=["createdAt", "tags", "pipeline_tag", "library_name", "cardData", "safetensors"],
    )
    for m in models:
        if m.created_at and m.created_at < cutoff:
            break
        card = m.card_data.to_dict() if m.card_data else {}
        releases.append(
            Release(
                repo_id=m.id,
                created_at=m.created_at,
                tags=list(m.tags or []),
                pipeline_tag=m.pipeline_tag,
                library_name=m.library_name,
                license=card.get("license_name") or card.get("license"),
                parameters=m.safetensors.total if m.safetensors else None,
            )
        )
    return releases


def _org_exists(org: str) -> bool:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return next(iter(api.list_models(author=org, limit=1)), None) is not None


def _row(r: Release, child: bool) -> str:
    params = f"{r.parameters / 1e9:,.2f}B" if r.parameters else "—"
    link = f"[{r.repo_id}](https://huggingface.co/{r.repo_id})"
    name = f"↳ {link}" if child else link
    created = r.created_at.date().isoformat()
    return f"| {name} | {created} | {r.pipeline_tag or '—'} | {params} | {r.license or '—'} |"


def render(
    kept: dict[str, list[Release]],
    skipped: dict[str, dict[str, int]],
    missing: list[str],
    since: str,
    today: date,
) -> str:
    total = sum(len(v) for v in kept.values())
    lines = [
        f"# Release radar: {today.isoformat()}",
        "",
        f"New root-model repos on Hugging Face from the roadmap 6.7 watchlist in the last "
        f"{since} (repo creation dates; {total} repos). Skipped: quantized/GGUF repos, adapters "
        "and fine-tunes of other orgs' models. A builder's own fine-tunes (e.g. Instruct from its "
        "Base) are nested under their base with ↳. Parameters are from safetensors metadata.",
        "",
    ]
    for org, releases in kept.items():
        lines += [f"## {org}", ""]
        counts = skipped.get(org, {})
        skip_text = ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items()))
        if not releases:
            lines.append("No root-model releases in this window.")
        else:
            lines += [
                "| Repo | Created | Task | Params | License |",
                "|---|---|---|---:|---|",
            ]
            ids = {r.repo_id for r in releases}
            children: dict[str, list[Release]] = {}
            for r in releases:
                if r.parent in ids:
                    children.setdefault(r.parent, []).append(r)
            for r in releases:
                if r.parent in ids:
                    continue
                lines.append(_row(r, child=False))
                lines += [_row(c, child=True) for c in children.get(r.repo_id, [])]
        if skip_text:
            lines += ["", f"Skipped: {skip_text}."]
        lines.append("")
    if missing:
        lines += [
            "## Not checked",
            "",
            f"Handles not found on the Hub: {', '.join(missing)}. Verify them in roadmap 6.7.",
            "",
        ]
    lines += ["---", "", CAVEAT, ""]
    return "\n".join(lines)


def run(since: str, out_dir: Path, now: datetime | None = None) -> tuple[Path, dict]:
    """Fetch every watchlist org, write radar/<date>.md, return (path, kept per org)."""
    now = now or datetime.now().astimezone()  # local date names the file
    cutoff = now - parse_since(since)
    kept, skipped, missing = {}, {}, []
    for org in WATCHLIST:
        try:
            releases = fetch(org, cutoff)
            if not releases and not _org_exists(org):
                missing.append(org)
                continue
        except HfHubHTTPError:
            missing.append(org)
            continue
        kept[org], skipped[org] = filter_releases(releases)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now.date().isoformat()}.md"
    path.write_text(render(kept, skipped, missing, since, now.date()))
    return path, kept

"""Engine readiness checks (Phase 2; docs/roadmap.md section 6.6).

Uses shallow sparse clones of vLLM, SGLang, llama.cpp and Transformers at `main` and the latest
stable release tag, cached under ~/.cache/launchready/engines (override with
LAUNCHREADY_ENGINES_CACHE), to report each architecture as `release`, `main-only`, `not-found` or
`unknown` with the matching file and line. Registries are located by symbol: when the registry
isn't in the expected file, the checked-out tree is searched and the move is reported. Flags
`auto_map` configs as needing trust_remote_code.
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

MAIN = "main"  # key for the default branch (main for most, master for llama.cpp)
MAIN_MAX_AGE_SECONDS = 24 * 3600
# v0.33.0rc3, v1.2.0b1, v0.33.0.dev1, v2.0.0-alpha; llama.cpp's b6000 build tags are stable.
_PRERELEASE_TAG = re.compile(r"\d(rc|a|b|alpha|beta)\d*$|dev|-(rc|alpha|beta|pre)", re.IGNORECASE)


@dataclass(frozen=True)
class EngineSpec:
    key: str
    name: str
    repo: str
    sparse_dirs: tuple[str, ...]  # cone-mode sparse checkout; root files are always included
    expected: str  # glob (relative to the repo root) where the registry should be
    block: str  # regex for one registry block; the name must appear inside a block
    needle: str  # regex template for the looked-up name, `{name}` is re.escape'd
    lookup: str  # "architecture" (config.architectures[0]) or "model_type"
    note: str = ""


_FLAGS = re.MULTILINE | re.DOTALL

ENGINES: dict[str, EngineSpec] = {
    spec.key: spec
    for spec in (
        EngineSpec(
            key="vllm",
            name="vLLM",
            repo="vllm-project/vllm",
            sparse_dirs=("vllm/model_executor/models",),
            expected="vllm/model_executor/models/registry.py",
            # Every `_*_MODELS = {...}` dict except the list of dropped models.
            block=r"^_(?!PREVIOUSLY_)[A-Z_]*MODELS\s*=\s*\{.*?^\}",
            needle=r'"{name}"\s*:',
            lookup="architecture",
        ),
        EngineSpec(
            key="sglang",
            name="SGLang",
            repo="sgl-project/sglang",
            sparse_dirs=("python/sglang/srt/models",),
            expected="python/sglang/srt/models/*.py",
            block=r"^EntryClass\s*=\s*(?:\[.*?\]|[^\n]*)",
            needle=r"\b{name}\b",
            lookup="architecture",
        ),
        EngineSpec(
            key="llama.cpp",
            name="llama.cpp",
            repo="ggml-org/llama.cpp",
            sparse_dirs=("conversion",),
            expected="conversion/*.py",
            block=r"@\w+\.register\(.*?\)",
            needle=r'"{name}"',
            lookup="architecture",
            note="→ Ollama via GGUF",
        ),
        EngineSpec(
            key="transformers",
            name="Transformers",
            repo="huggingface/transformers",
            sparse_dirs=("src/transformers/models/auto",),
            expected="src/transformers/models/auto/auto_mappings.py",
            block=r"^CONFIG_MAPPING_NAMES\s*=\s*OrderedDict\(\s*\[.*?^\s*\]",
            needle=r'\(\s*"{name}"\s*,',
            lookup="model_type",
        ),
    )
}


@dataclass
class Checkout:
    path: Path
    ref: str
    sha: str
    branch: str | None = None  # real branch name when ref is MAIN


@dataclass
class Match:
    path: str  # relative to the repo root, POSIX style
    line: int
    text: str


@dataclass
class RefResult:
    registry_found: bool
    match: Match | None
    moved: str | None = None  # where the registry was found when not at spec.expected


@dataclass
class EngineStatus:
    engine: str
    status: str  # release | main-only | not-found | unknown
    version: str | None
    path: str | None
    line: int | None
    text: str | None
    url: str | None
    notes: list[str] = field(default_factory=list)


def cache_dir() -> Path:
    default = Path.home() / ".cache" / "launchready" / "engines"
    return Path(os.environ.get("LAUNCHREADY_ENGINES_CACHE", default))


# --- matching (pure, on a checked-out tree) ---------------------------------------------------


def _find_in_text(text: str, spec: EngineSpec, name: str) -> tuple[bool, int | None]:
    """(registry present, offset of the name inside a registry block)."""
    needle = re.compile(spec.needle.format(name=re.escape(name)))
    found = False
    for block in re.finditer(spec.block, text, _FLAGS):
        found = True
        hit = needle.search(block.group(0))
        if hit:
            return True, block.start() + hit.start()
    return found, None


def _scan(root: Path, files: list[Path], spec: EngineSpec, name: str) -> RefResult:
    registry_files = []
    for path in files:
        text = path.read_text(errors="replace")
        found, offset = _find_in_text(text, spec, name)
        if not found:
            continue
        registry_files.append(path)
        if offset is not None:
            line = text.count("\n", 0, offset) + 1
            return RefResult(
                registry_found=True,
                match=Match(
                    path=path.relative_to(root).as_posix(),
                    line=line,
                    text=text.splitlines()[line - 1].strip(),
                ),
            )
    return RefResult(registry_found=bool(registry_files), match=None)


def check_ref(root: Path, spec: EngineSpec, name: str) -> RefResult:
    """Look for `name` in the engine's registry under `root`; search the tree if it moved."""
    expected = sorted(root.glob(spec.expected))
    result = _scan(root, expected, spec, name)
    if result.registry_found:
        return result
    others = sorted(p for p in root.rglob("*.py") if p not in expected and ".git" not in p.parts)
    result = _scan(root, others, spec, name)
    if result.registry_found:
        where = result.match.path if result.match else _first_registry(root, others, spec)
        result.moved = where
    return result


def _first_registry(root: Path, files: list[Path], spec: EngineSpec) -> str | None:
    for path in files:
        if re.search(spec.block, path.read_text(errors="replace"), _FLAGS):
            return path.relative_to(root).as_posix()
    return None


def classify(release: RefResult | None, main: RefResult | None) -> str:
    if release and release.match:
        return "release"
    if main and main.match:
        return "main-only"
    if release and main and release.registry_found and main.registry_found:
        return "not-found"
    return "unknown"


def pick_release(releases: list[dict]) -> str | None:
    """First stable tag: not a draft, not flagged prerelease, and no rc/dev/alpha/beta suffix."""
    for r in releases:
        tag = r.get("tag_name", "")
        if r.get("draft") or r.get("prerelease"):
            continue
        if _PRERELEASE_TAG.search(tag):
            continue
        return tag
    return None


# --- network: GitHub releases and sparse clones -----------------------------------------------


def latest_release(spec: EngineSpec, refresh: bool = False) -> str | None:
    """Latest stable release tag from the GitHub API, cached for a day."""
    path = cache_dir() / "releases.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    entry = cached.get(spec.repo)
    if entry and not refresh and time.time() - entry["fetched_at"] < MAIN_MAX_AGE_SECONDS:
        return entry["tag"]
    # releases/latest skips flagged prereleases (llama.cpp flags every build); scan the list
    # only when it returns an unflagged rc (vLLM).
    tag = pick_release([_github(f"repos/{spec.repo}/releases/latest")])
    if tag is None:
        tag = pick_release(_github(f"repos/{spec.repo}/releases?per_page=100"))
    cached[spec.repo] = {"tag": tag, "fetched_at": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n")
    return tag


def _github(endpoint: str):
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "launchready"},
    )
    if token := os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def checkout(spec: EngineSpec, ref: str, refresh: bool = False) -> Checkout:
    """Shallow sparse clone of `spec.repo` at `ref`. Tags are reused; main expires after a day."""
    dest = cache_dir() / spec.repo.replace("/", "__") / ref
    meta_path = dest / ".launchready.json"
    if meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text())
        fresh = ref != MAIN or time.time() - meta["fetched_at"] < MAIN_MAX_AGE_SECONDS
        if fresh and meta.get("sparse_dirs") == list(spec.sparse_dirs):
            return Checkout(path=dest, ref=ref, sha=meta["sha"], branch=meta.get("branch"))
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    branch_args = [] if ref == MAIN else ["--branch", ref]  # MAIN: the remote's default branch
    _git(
        "clone", "--depth", "1", "--filter=blob:none", "--sparse", *branch_args,
        f"https://github.com/{spec.repo}.git", str(tmp),
    )  # fmt: skip
    _git("sparse-checkout", "set", *spec.sparse_dirs, cwd=tmp)
    sha = _git("rev-parse", "HEAD", cwd=tmp)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=tmp) if ref == MAIN else None
    meta = {"sha": sha, "branch": branch, "fetched_at": time.time()}
    meta["sparse_dirs"] = list(spec.sparse_dirs)
    (tmp / ".launchready.json").write_text(json.dumps(meta))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return Checkout(path=dest, ref=ref, sha=sha, branch=branch)


# --- the readiness table ----------------------------------------------------------------------


def _lookup_name(config: dict, spec: EngineSpec) -> str | None:
    cfg = config
    if spec.lookup == "model_type":
        return cfg.get("model_type")
    archs = cfg.get("architectures") or []
    return archs[0] if archs else None


def _url(spec: EngineSpec, ref: str, match: Match) -> str:
    return f"https://github.com/{spec.repo}/blob/{ref}/{match.path}#L{match.line}"


def check(
    config: dict,
    engines: list[str] | None = None,
    refresh: bool = False,
    checkout_fn: Callable[..., Checkout] | None = checkout,
    release_fn: Callable[[EngineSpec], str | None] | None = None,
) -> list[EngineStatus]:
    """Readiness of every engine (or the named ones) for this config.json."""
    if release_fn is None:
        release_fn = lambda spec: latest_release(spec, refresh=refresh)  # noqa: E731
    statuses = []
    for key in engines or list(ENGINES):
        spec = ENGINES[key]
        notes = [spec.note] if spec.note else []
        if config.get("auto_map"):
            notes.append("config has auto_map: needs trust_remote_code")
        name = _lookup_name(config, spec)
        if name is None:
            field_name = "model_type" if spec.lookup == "model_type" else "architectures"
            notes.append(f"config has no {field_name}: can't look the model up")
            statuses.append(EngineStatus(spec.name, "unknown", None, None, None, None, None, notes))
            continue
        try:
            tag = release_fn(spec)
            refs = ([tag] if tag else []) + [MAIN]
            results = {}
            for ref in refs:
                co = checkout_fn(spec, ref, refresh=refresh)
                results[ref] = (co, check_ref(co.path, spec, name))
        except (RuntimeError, OSError) as e:
            notes.append(f"check failed: {e}")
            statuses.append(EngineStatus(spec.name, "unknown", None, None, None, None, None, notes))
            continue
        if tag is None:
            notes.append("no stable release tag found: checked main only")

        release = results[tag][1] if tag else None
        main_co, main = results[MAIN]
        status = classify(release, main)
        for ref, (_, result) in results.items():
            if result.moved:
                notes.append(f"registry not at {spec.expected} at {ref}: found in {result.moved}")
            if not result.registry_found:
                notes.append(f"registry not found at {ref}: the layout may have changed")
        main_label = f"{main_co.branch or MAIN} @ {main_co.sha[:7]}"
        if status == "release":
            version, match, url_ref = tag, release.match, tag
            notes.append(f"also checked {main_label}")
        elif status == "main-only":
            version, match, url_ref = main_label, main.match, main_co.sha
            notes.append(f"not in {tag}" if tag else "")
        else:
            version, match, url_ref = tag or main_label, None, None
            if status == "not-found":
                notes.append(f"looked for {name!r}; also checked {main_label}")
        statuses.append(
            EngineStatus(
                engine=spec.name,
                status=status,
                version=version,
                path=match.path if match else None,
                line=match.line if match else None,
                text=match.text if match else None,
                url=_url(spec, url_ref, match) if match else None,
                notes=[n for n in notes if n],
            )
        )
    return statuses

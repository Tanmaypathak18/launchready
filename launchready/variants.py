"""Quantized variants of a model on the Hub, with real file sizes (Phase 2).

Finds FP8, NVFP4, GGUF and other quantized derivatives via the Hub filter
`base_model:quantized:<repo_id>` plus a search of the builder's own org (builders don't always
tag their own quantized repos), and reads file sizes from repo metadata. No weights are
downloaded. Results are cached as JSON next to the hub cache.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field

from huggingface_hub import HfApi

from launchready.hub import cache_dir

MAX_SIZED = 12  # model_info calls per lookup; most-downloaded first, builder org always included

_GGUF_QUANT = re.compile(
    r"(?<![A-Za-z0-9])(I?Q\d_K_[SMLX]{1,2}|I?Q\d_K|IQ\d_[A-Z]+|Q\d_\d|MXFP4|BF16|F16|F32)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)")
_STEM_SUFFIX = re.compile(r"(-(BF16|FP16|FP32|Base|Instruct|Chat|hf))+$", re.IGNORECASE)
_DRAFTER = re.compile(r"dflash|dspark|eagle|medusa|draft", re.I)
_NAME_FORMATS = (
    ("NVFP4", re.compile(r"nvfp4", re.I)),
    ("FP8", re.compile(r"fp8", re.I)),
    ("GGUF", re.compile(r"gguf", re.I)),
    ("AWQ", re.compile(r"awq", re.I)),
    ("GPTQ", re.compile(r"gptq", re.I)),
    ("bitsandbytes", re.compile(r"bnb|bitsandbytes", re.I)),
    ("MLX", re.compile(r"mlx", re.I)),
)
# Variant format → the precision key used by launchready.memory.
_PRECISIONS = {"FP8": "fp8", "NVFP4": "nvfp4"}
_GGUF_PRECISIONS = {"Q8_0": "gguf-q8_0", "Q4_K_M": "gguf-q4_k_m"}


@dataclass
class Variant:
    repo_id: str
    format: str
    author: str
    total_bytes: int  # safetensors bytes (0 for GGUF repos; see gguf)
    gguf: dict[str, int] = field(default_factory=dict)  # quant type → bytes (shards summed)
    downloads: int = 0


def classify_format(repo_id: str, tags: list[str]) -> str | None:
    """Quantized format from tags and repo name; None when the repo isn't quantized."""
    tag_set = {t.lower() for t in tags}
    name = repo_id.split("/")[-1]
    if _DRAFTER.search(name):
        return "drafter"  # speculative-decoding drafters often carry base_model:quantized
    if "gguf" in tag_set:
        return "GGUF"
    if "mlx" in tag_set:
        return "MLX"
    for fmt, pattern in _NAME_FORMATS:
        if pattern.search(name):
            return fmt
    if any(t.startswith("base_model:quantized:") for t in tag_set):
        return "other"
    if re.search(r"(int4|int8|4bit|8bit|w4a16|w8a8)", name, re.I):
        return "other"
    return None


def name_stem(name: str) -> str:
    """Model name without precision/stage suffixes, for searching the builder's org."""
    return _STEM_SUFFIX.sub("", name)


def gguf_sizes(files: dict[str, int]) -> dict[str, int]:
    """Bytes per GGUF quant type, summing split shards and skipping vision projectors."""
    sizes: dict[str, int] = {}
    for name, size in files.items():
        base = name.split("/")[-1]
        if not base.endswith(".gguf") or base.lower().startswith("mmproj"):
            continue
        match = _GGUF_QUANT.search(_SHARD.sub("", base))
        if match:
            quant = match.group(1).upper()
            sizes[quant] = sizes.get(quant, 0) + size
    return sizes


def real_sizes(variants: list[Variant], builder: str) -> dict[str, tuple[str, int]]:
    """precision → (repo_id, bytes) from the builder's own variant, else the most downloaded."""
    ranked = sorted(variants, key=lambda v: (v.author != builder, -v.downloads))
    sizes: dict[str, tuple[str, int]] = {}
    for v in ranked:
        if v.format in _PRECISIONS and v.total_bytes and _PRECISIONS[v.format] not in sizes:
            sizes[_PRECISIONS[v.format]] = (v.repo_id, v.total_bytes)
        for quant, precision in _GGUF_PRECISIONS.items():
            if quant in v.gguf and precision not in sizes:
                sizes[precision] = (v.repo_id, v.gguf[quant])
    return sizes


def _cache_path(repo_id: str):
    return cache_dir() / f"{repo_id.replace('/', '__')}__variants.json"


def load_variants(repo_id: str, refresh: bool = False) -> list[Variant]:
    """Quantized variants of `repo_id`, from the disk cache when present."""
    path = _cache_path(repo_id)
    if path.exists() and not refresh:
        return [Variant(**v) for v in json.loads(path.read_text())]
    variants = _fetch(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(v) for v in variants], indent=2, sort_keys=True) + "\n")
    return variants


def _fetch(repo_id: str) -> list[Variant]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    builder, name = repo_id.split("/")
    expand = ["tags", "downloads", "author"]
    found = {
        m.id: m
        for m in api.list_models(
            filter=f"base_model:quantized:{repo_id}", sort="downloads", limit=100, expand=expand
        )
    }
    for m in api.list_models(author=builder, search=name_stem(name), limit=100, expand=expand):
        found.setdefault(m.id, m)

    candidates = []
    for m in found.values():
        fmt = classify_format(m.id, m.tags or [])
        if m.id != repo_id and fmt:
            candidates.append((m, fmt))
    candidates.sort(key=lambda c: (c[0].id.split("/")[0] != builder, -(c[0].downloads or 0)))

    variants = []
    for m, fmt in candidates[:MAX_SIZED]:
        info = api.model_info(m.id, files_metadata=True)
        files = {s.rfilename: s.size or 0 for s in info.siblings or []}
        variants.append(
            Variant(
                repo_id=m.id,
                format=fmt,
                author=m.id.split("/")[0],
                total_bytes=sum(v for k, v in files.items() if k.endswith(".safetensors")),
                gguf=gguf_sizes(files) if fmt == "GGUF" else {},
                downloads=m.downloads or 0,
            )
        )
    return variants

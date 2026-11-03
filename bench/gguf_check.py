"""Read a GGUF file's header (local path or Hugging Face, via range reads) and report what's inside.

No weights are downloaded: only the header (metadata + tensor table) is read. Reports the
pre-tokenizer (`tokenizer.ggml.pre`), the tensor types that are actually stored (quantizers fall
back, e.g. Lightning's experts Q4_K → Q5_0), real bits per weight, tensor bytes vs file size,
and a per-layer check that `attention.head_count_kv` agrees with the attn_* / ssm_* tensors
present in each block (llama.cpp#27718).

    uv run python -m bench.gguf_check hf://unsloth/<repo>@<rev>/<file>.gguf
    uv run python -m bench.gguf_check /models/file.gguf --out results/bench/gguf_x.json
"""

import json
import re
import struct
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO

import typer

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# ggml type id → (name, block size, bytes per block), from ggml.h / ggml.c type traits.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

# GGUF metadata value types.
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9
# Long arrays (vocab, merges) are summarized rather than kept.
_KEEP_ARRAY_MAX = 4096

_BLOCK = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    """The file isn't a GGUF file this parser understands."""


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f
        self.pos = 0

    def take(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFError(f"unexpected end of file at byte {self.pos} (wanted {n} bytes)")
        self.pos += n
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type, n = self.unpack("<I"), self.unpack("<Q")
            if n > _KEEP_ARRAY_MAX:
                if item_type in _SCALARS:  # skip fixed-size items without decoding them
                    self.take(n * struct.calcsize(_SCALARS[item_type]))
                else:
                    for _ in range(n):
                        self.value(item_type)
                return {"array_len": n, "item_type": item_type}
            return [self.value(item_type) for _ in range(n)]
        raise GGUFError(f"unknown metadata value type {vtype} at byte {self.pos}")


def read_header(f: BinaryIO) -> dict:
    """Parse the GGUF header: version, metadata KV pairs and the tensor table."""
    r = _Reader(f)
    if r.take(4) != MAGIC:
        raise GGUFError("bad magic: not a GGUF file")
    version = r.unpack("<I")
    if version < 2:
        raise GGUFError(f"GGUF version {version} is too old for this parser (need 2+)")
    n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.unpack("<I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
        ggml_type, offset = r.unpack("<I"), r.unpack("<Q")
        tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})
    alignment = int(kv.get("general.alignment", DEFAULT_ALIGNMENT))
    data_start = (r.pos + alignment - 1) // alignment * alignment
    return {"version": version, "kv": kv, "tensors": tensors, "data_start": data_start}


def _numel(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def tensor_bytes(t: dict) -> int | None:
    """Bytes a tensor occupies from its type and shape; None for types not in GGML_TYPES."""
    if t["type"] not in GGML_TYPES:
        return None
    _, block, size = GGML_TYPES[t["type"]]
    return _numel(t["dims"]) // block * size


def layer_check(header: dict, arch: str) -> dict:
    """Per block: does head_count_kv agree with the physical attn_* / ssm_* tensors?"""
    kv_heads = header["kv"].get(f"{arch}.attention.head_count_kv")
    kinds: dict[int, set[str]] = defaultdict(set)
    for t in header["tensors"]:
        m = _BLOCK.match(t["name"])
        if not m:
            continue
        rest = t["name"][m.end() :]
        if rest.startswith(("attn_q", "attn_k", "attn_v", "attn_output")):
            kinds[int(m.group(1))].add("attn")
        elif rest.startswith("ssm_"):
            kinds[int(m.group(1))].add("ssm")
    blocks = int(header["kv"].get(f"{arch}.block_count", 0)) or (max(kinds) + 1 if kinds else 0)
    out = {"blocks": blocks, "head_count_kv": None, "mismatches": [], "note": None}
    if not isinstance(kv_heads, list):
        out["head_count_kv"] = kv_heads
        out["note"] = "head_count_kv is not per-layer; per-layer check skipped"
        return out
    out["head_count_kv"] = kv_heads
    if len(kv_heads) != blocks:
        out["mismatches"].append(
            {"block": None, "reason": f"head_count_kv has {len(kv_heads)} entries, {blocks} blocks"}
        )
    for b in range(min(blocks, len(kv_heads))):
        k = kinds.get(b, set())
        if "attn" in k and kv_heads[b] == 0:
            out["mismatches"].append(
                {"block": b, "reason": "attn tensors present but head_count_kv = 0 (loads as SSM)"}
            )
        if "ssm" in k and "attn" not in k and kv_heads[b] > 0:
            out["mismatches"].append(
                {"block": b, "reason": f"ssm tensors only but head_count_kv = {kv_heads[b]}"}
            )
    return out


def summarize(header: dict, file_bytes: int, name: str) -> dict:
    """Report tensor types, bits per weight, pre-tokenizer and the per-layer check."""
    kv = header["kv"]
    arch = kv.get("general.architecture", "")
    types: dict[str, dict] = {}
    unknown = set()
    params = total = 0
    for t in header["tensors"]:
        n = _numel(t["dims"])
        b = tensor_bytes(t)
        tname = GGML_TYPES.get(t["type"], (f"type_{t['type']}",))[0]
        if b is None:
            unknown.add(tname)
            b = 0
        entry = types.setdefault(tname, {"tensors": 0, "params": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["params"] += n
        entry["bytes"] += b
        params += n
        total += b
    for entry in types.values():
        entry["bits_per_weight"] = entry["bytes"] * 8 / entry["params"] if entry["params"] else None
        entry["share_of_bytes"] = entry["bytes"] / total if total else None
    warnings = []
    if unknown:
        warnings.append(f"unknown ggml tensor type(s) {sorted(unknown)}: bytes not counted")
    if total > file_bytes:
        warnings.append("tensor bytes exceed file size: header and file disagree")
    return {
        "label": "measured",
        "what": "GGUF header (metadata + tensor table); weights not read",
        "file": name,
        "file_bytes": file_bytes,
        "gguf_version": header["version"],
        "architecture": arch,
        "general_name": kv.get("general.name"),
        "file_type": kv.get("general.file_type"),
        "tokenizer_ggml_pre": kv.get("tokenizer.ggml.pre"),
        "tokenizer_ggml_model": kv.get("tokenizer.ggml.model"),
        "has_chat_template": "tokenizer.chat_template" in kv,
        "context_length": kv.get(f"{arch}.context_length"),
        "tensor_count": len(header["tensors"]),
        "params_total": params,
        "tensor_bytes": total,
        "header_bytes": header["data_start"],
        "bits_per_weight": total * 8 / params if params else None,
        "tensor_types": dict(sorted(types.items(), key=lambda kv_: -kv_[1]["bytes"])),
        "layer_check": layer_check(header, arch),
        "warnings": warnings,
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _open(path: str) -> tuple[BinaryIO, int, str]:
    """Open a local file or hf://<repo>[@rev]/<file> (range reads via HfFileSystem)."""
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        spec = path.removeprefix("hf://")
        size = fs.info(spec)["size"]
        # Small blocks: the header is a few MB, so read only what the parser asks for.
        return fs.open(spec, "rb", block_size=4 * 2**20), size, path
    p = Path(path)
    return p.open("rb"), p.stat().st_size, str(p)


def check(path: str) -> dict:
    f, size, name = _open(path)
    with f:
        return summarize(read_header(f), size, name)


def main(
    path: Annotated[
        str, typer.Argument(help="Local .gguf path or hf://<repo>[@<rev>]/<file>.gguf")
    ],
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
) -> None:
    report = check(path)
    text = json.dumps(report, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    typer.run(main)

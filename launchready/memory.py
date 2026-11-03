"""Projected memory math (Phase 1; docs/roadmap.md sections 6.1-6.3).

Weights per precision (BF16, FP8, NVFP4, GGUF), KV cache and Mamba state per sequence
(FP16 or FP8 KV), and a vLLM-style fit model giving max users at a context length per GPU.
Computes in bytes; callers display GiB. Every result here is projected, not measured.
"""

import json
import math
from dataclasses import dataclass
from importlib.resources import files

import yaml

from launchready.arch import Architecture

GIB = 2**30

# Effective bits per weight (6.1). GGUF Q4_K_M is ~4.8-4.9; the midpoint is used.
BITS_PER_WEIGHT = {
    "bf16": 16,
    "fp8": 8,
    "nvfp4": 4.5,
    "gguf-q8_0": 8.5,
    "gguf-q4_k_m": 4.85,
}
KV_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}
STATE_DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
DEFAULT_STATE_DTYPE = "float32"

# Fit model (6.3): vLLM's default gpu_memory_utilization and a starting runtime reserve
# (activations, CUDA graphs, allocator slack), to be calibrated in Phase 3.
GPU_MEMORY_UTILIZATION = 0.90
RUNTIME_RESERVE_BYTES = 2 * GIB

_PACKED_4BIT_MARKERS = ("fp4", "int4", "nvfp4", "mxfp4", "awq", "gptq", "w4a")


@dataclass
class MambaState:
    bytes: int
    dtype: str | None
    assumed: bool  # True when the config doesn't name the state dtype


@dataclass
class SequenceMemory:
    attention_bytes: int
    mamba: MambaState

    @property
    def mamba_bytes(self) -> int:
        return self.mamba.bytes

    @property
    def total(self) -> int:
        return self.attention_bytes + self.mamba.bytes


@dataclass
class FitResult:
    kv_budget_bytes: float
    max_users: int


@dataclass
class Gpu:
    id: str
    name: str
    memory_gib: float
    architecture: str
    native: list[str]
    unverified: str | None = None  # what the spec-page check couldn't confirm

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gib * GIB)

    def is_native(self, precision: str) -> bool:
        return precision.startswith("gguf") or precision in self.native


def weight_bytes(params: int, precision: str) -> float:
    """Weights = parameters x bits per weight / 8."""
    return params * BITS_PER_WEIGHT[precision] / 8


def kv_bytes_per_token(arch: Architecture, kv_dtype: str = "fp16") -> int:
    """KV cache bytes one token adds across all attention layers (sliding layers until full)."""
    size = KV_DTYPE_BYTES[kv_dtype]
    return sum(_layer_kv_bytes(arch, kind, size) for kind in arch.layer_kinds)


def _layer_kv_bytes(arch: Architecture, kind: str, size: int) -> int:
    if kind in ("full_attention", "sliding_attention"):
        return 2 * arch.num_kv_heads * arch.head_dim * size
    if kind == "mla":
        return (arch.kv_lora_rank + (arch.qk_rope_head_dim or 0)) * size
    return 0


def mamba_state_bytes(arch: Architecture) -> MambaState:
    """Fixed Mamba-2 state per sequence: conv state plus SSM state, for every Mamba layer."""
    spec = arch.mamba
    if spec is None:
        return MambaState(bytes=0, dtype=None, assumed=False)
    assumed = spec.state_dtype not in STATE_DTYPE_BYTES
    dtype = DEFAULT_STATE_DTYPE if assumed else spec.state_dtype
    conv_dim = spec.num_heads * spec.head_dim + 2 * spec.n_groups * spec.state_size
    elements = (spec.conv_kernel - 1) * conv_dim + spec.num_heads * spec.head_dim * spec.state_size
    return MambaState(
        bytes=arch.count("mamba") * elements * STATE_DTYPE_BYTES[dtype],
        dtype=dtype,
        assumed=assumed,
    )


def sequence_bytes(arch: Architecture, context: int, kv_dtype: str = "fp16") -> SequenceMemory:
    """Memory for one sequence at `context` tokens: KV cache plus Mamba state."""
    size = KV_DTYPE_BYTES[kv_dtype]
    window = arch.sliding_window or context
    attention = 0
    for kind in arch.layer_kinds:
        tokens = min(context, window) if kind == "sliding_attention" else context
        attention += _layer_kv_bytes(arch, kind, size) * tokens
    return SequenceMemory(attention_bytes=attention, mamba=mamba_state_bytes(arch))


def fit(
    vram_bytes: float,
    weights_bytes: float,
    per_sequence_bytes: int,
    utilization: float = GPU_MEMORY_UTILIZATION,
    reserve_bytes: float = RUNTIME_RESERVE_BYTES,
) -> FitResult:
    """KV budget = VRAM x utilization - weights - reserve; max users = floor(budget / sequence)."""
    if per_sequence_bytes <= 0:
        raise ValueError("per-sequence memory must be positive")
    budget = vram_bytes * utilization - weights_bytes - reserve_bytes
    users = math.floor(budget / per_sequence_bytes) if budget > 0 else 0
    return FitResult(kv_budget_bytes=budget, max_users=users)


def total_parameters(dtype_counts: dict[str, int], config: dict) -> tuple[int, list[str]]:
    """Parameter count from safetensors dtype counts, unpacking packed 4-bit (U8) storage."""
    warnings = []
    quant = config.get("quantization_config")
    quant_text = json.dumps(quant).lower() if quant else ""
    total = 0
    for dtype, count in dtype_counts.items():
        if dtype == "U8" and any(m in quant_text for m in _PACKED_4BIT_MARKERS):
            total += 2 * count
            warnings.append(f"U8 tensors ({count:,}) counted as packed 4-bit, two per byte")
        elif dtype == "U8":
            total += count
            warnings.append(
                f"U8 tensors ({count:,}) without a 4-bit quantization_config: counted one per byte"
            )
        else:
            total += count
    if quant:
        method = quant.get("quant_method", "unknown")
        warnings.append(
            f"repo is pre-quantized ({method}): count includes stored scale tensors "
            f"({', '.join(f'{d} {c:,}' for d, c in sorted(dtype_counts.items()))})"
        )
    return total, warnings


def load_gpus() -> list[Gpu]:
    """The GPU catalog in launchready/data/gpus.yaml."""
    text = files("launchready").joinpath("data/gpus.yaml").read_text()
    return [Gpu(**entry) for entry in yaml.safe_load(text)["gpus"]]

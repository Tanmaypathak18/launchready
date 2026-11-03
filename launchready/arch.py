"""Normalize a model config into an `Architecture` dataclass (Phase 1).

Covers layers and layer types (full attention, sliding window, MLA, Mamba), attention and KV
heads, head_dim, context length, MoE fields (total vs active parameters) and Mamba fields.
Any architecture-looking config field that isn't interpreted produces a warning naming the
field; nothing is guessed silently.
"""

from dataclasses import dataclass, field

# Per-layer cache kinds. "none" layers (MLP/MoE blocks in hybrids) hold no KV or state.
LAYER_KINDS = ("full_attention", "sliding_attention", "mla", "mamba", "none")

_LAYER_TYPE_NAMES = {
    "full_attention": "full_attention",
    "attention": "full_attention",
    "sliding_attention": "sliding_attention",
    "mamba": "mamba",
    "moe": "none",
    "mlp": "none",
}
# Nemotron-H `hybrid_override_pattern`: M = Mamba, * = attention, - = MLP, E = MoE.
_PATTERN_CHARS = {"M": "mamba", "*": "full_attention", "-": "none", "E": "none"}

_GATED_ACTIVATIONS = {"silu": True, "swiglu": True, "relu2": False}

# A non-empty field whose name contains one of these is architecture-related: if it's neither
# handled nor known to be irrelevant to memory, warn and name it.
_KEYWORDS = (
    "window", "mamba", "ssm", "expert", "lora", "kv", "layer_type", "block_type", "pattern",
    "nextn", "mtp", "conv", "state", "moe", "head", "sparse", "chunk", "global", "local",
)  # fmt: skip
_HANDLED = {
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "max_position_embeddings", "vocab_size", "layer_types", "layers_block_type",
    "hybrid_override_pattern", "sliding_window", "use_sliding_window", "max_window_layers",
    "kv_lora_rank", "qk_rope_head_dim", "mamba_num_heads", "mamba_n_heads", "mamba_head_dim",
    "mamba_d_head", "ssm_state_size", "mamba_d_state", "n_groups", "mamba_n_groups",
    "conv_kernel", "mamba_d_conv", "mamba_ssm_cache_dtype", "n_routed_experts", "num_experts",
    "num_local_experts", "num_experts_per_tok", "experts_per_token", "moe_intermediate_size",
    "intermediate_size", "first_k_dense_replace", "moe_layer_freq", "decoder_sparse_step",
    "mlp_only_layers", "num_nextn_predict_layers", "mtp_layers_block_type",
}  # fmt: skip
_IRRELEVANT = {
    "attention_bias", "attention_dropout", "attention_multiplier", "chunk_size",
    "mamba_chunk_size", "mamba_conv_bias", "use_conv_bias", "mamba_proj_bias",
    "mamba_hidden_act", "mamba_expand", "use_mamba_kernels", "router_aux_loss_coef",
    "output_router_logits", "norm_topk_prob", "ep_size", "moe_shared_expert_overlap",
    "moe_shared_expert_intermediate_size", "shared_intermediate_size", "n_shared_experts",
    "q_lora_rank", "qk_nope_head_dim", "v_head_dim", "n_group", "topk_group",
}  # fmt: skip


@dataclass
class MambaSpec:
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    state_dtype: str | None  # from mamba_ssm_cache_dtype; None means the config doesn't say


@dataclass
class MoESpec:
    num_experts: int
    experts_per_token: int | None
    expert_intermediate_size: int | None
    num_moe_layers: int
    mtp_moe_layers: int
    gated_mlp: bool | None  # SwiGLU-style (3 matrices per expert) vs plain (2); None = unknown


@dataclass
class Architecture:
    model_type: str | None
    num_layers: int
    hidden_size: int
    num_attention_heads: int | None
    num_kv_heads: int | None
    head_dim: int | None
    max_context: int | None
    layer_kinds: list[str]
    sliding_window: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    mamba: MambaSpec | None = None
    moe: MoESpec | None = None
    mtp_layers: int = 0
    trust_remote_code: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return self.layer_kinds.count(kind)

    @property
    def family(self) -> str:
        parts = [
            name
            for name, present in (
                ("hybrid Mamba", self.count("mamba")),
                ("MoE", self.moe),
                ("MLA", self.count("mla")),
                ("sliding window", self.count("sliding_attention")),
            )
            if present
        ]
        if not self.count("mamba") and not self.moe:
            parts.insert(0, "dense")
        return " ".join(parts)

    def active_parameters(self, total: int) -> int | None:
        """Parameters used per token: total minus the routed experts that don't run (~)."""
        moe = self.moe
        if moe is None:
            return total
        if None in (moe.gated_mlp, moe.experts_per_token, moe.expert_intermediate_size):
            return None
        matrices = 3 if moe.gated_mlp else 2
        per_expert = matrices * self.hidden_size * moe.expert_intermediate_size
        idle = moe.num_experts - moe.experts_per_token
        return total - (moe.num_moe_layers + moe.mtp_moe_layers) * idle * per_expert


def _first(cfg: dict, *keys):
    for key in keys:
        if cfg.get(key) is not None:
            return cfg[key]
    return None


def parse_config(config: dict) -> Architecture:
    """Normalize a Hugging Face config.json (nested `text_config` included)."""
    cfg = config.get("text_config") or config
    warnings: list[str] = []
    notes: list[str] = []

    num_layers = cfg.get("num_hidden_layers")
    hidden = cfg.get("hidden_size")
    if num_layers is None or hidden is None:
        missing = [k for k in ("num_hidden_layers", "hidden_size") if cfg.get(k) is None]
        raise ValueError(f"config is missing {', '.join(missing)}: can't size this model")
    heads = cfg.get("num_attention_heads")
    # Transformers' rules: no num_key_value_heads means one KV head per query head (no GQA),
    # and no head_dim means hidden_size / num_attention_heads.
    kv_heads = _first(cfg, "num_key_value_heads") or heads
    head_dim = _first(cfg, "head_dim") or (hidden // heads if heads else None)

    kinds = _layer_kinds(cfg, num_layers, warnings)
    window = cfg.get("sliding_window")
    if "sliding_attention" in kinds and window is None:
        warnings.append("layer_types has sliding_attention but sliding_window is not set")
    if not _explicit_layer_types(cfg) and window is not None:
        if cfg.get("use_sliding_window") is True:
            warnings.append(
                "use_sliding_window=true with max_window_layers: per-layer windows not "
                "interpreted, counting every layer as full attention (upper bound)"
            )
        elif "use_sliding_window" not in cfg:
            kinds = ["sliding_attention" if k == "full_attention" else k for k in kinds]

    kv_lora_rank = cfg.get("kv_lora_rank")
    if kv_lora_rank is not None:
        kinds = ["mla" if k == "full_attention" else k for k in kinds]
        if cfg.get("qk_rope_head_dim") is None:
            warnings.append("kv_lora_rank is set but qk_rope_head_dim is missing")

    mtp_layers = cfg.get("num_nextn_predict_layers") or 0
    if mtp_layers:
        notes.append(
            f"num_nextn_predict_layers={mtp_layers}: multi-token-prediction weights are in the "
            "parameter count; their KV cache (used only for speculative decoding) is not"
        )
    if "auto_map" in config:
        notes.append("auto_map is set: Transformers needs trust_remote_code=True")

    arch = Architecture(
        model_type=cfg.get("model_type") or config.get("model_type"),
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_context=cfg.get("max_position_embeddings"),
        layer_kinds=kinds,
        sliding_window=window,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=cfg.get("qk_rope_head_dim"),
        mamba=_mamba(cfg, warnings) if "mamba" in kinds else None,
        moe=_moe(cfg, num_layers, mtp_layers, warnings),
        mtp_layers=mtp_layers,
        trust_remote_code="auto_map" in config,
        warnings=warnings,
        notes=notes,
    )
    warnings.extend(_unknown_fields(cfg))
    return arch


def _explicit_layer_types(cfg: dict) -> bool:
    return any(cfg.get(k) for k in ("layer_types", "layers_block_type", "hybrid_override_pattern"))


def _layer_kinds(cfg: dict, num_layers: int, warnings: list[str]) -> list[str]:
    if cfg.get("layer_types") or cfg.get("layers_block_type"):
        key = "layer_types" if cfg.get("layer_types") else "layers_block_type"
        names, table = cfg[key], _LAYER_TYPE_NAMES
    elif cfg.get("hybrid_override_pattern"):
        key = "hybrid_override_pattern"
        names, table = list(cfg[key]), _PATTERN_CHARS
    else:
        return ["full_attention"] * num_layers

    unknown = sorted({n for n in names if n not in table})
    if unknown:
        warnings.append(f"{key} has unknown layer type(s) {unknown}: not counted in KV or state")
    if len(names) != num_layers:
        warnings.append(f"{key} lists {len(names)} layers but num_hidden_layers is {num_layers}")
    return [table.get(n, "none") for n in names]


def _mamba(cfg: dict, warnings: list[str]) -> MambaSpec | None:
    fields = {
        "num_heads": _first(cfg, "mamba_num_heads", "mamba_n_heads"),
        "head_dim": _first(cfg, "mamba_head_dim", "mamba_d_head"),
        "state_size": _first(cfg, "ssm_state_size", "mamba_d_state"),
        "n_groups": _first(cfg, "n_groups", "mamba_n_groups"),
        "conv_kernel": _first(cfg, "conv_kernel", "mamba_d_conv"),
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        warnings.append(f"Mamba layers present but {missing} not found: state not counted")
        return None
    return MambaSpec(**fields, state_dtype=cfg.get("mamba_ssm_cache_dtype"))


def _moe(cfg: dict, num_layers: int, mtp_layers: int, warnings: list[str]) -> MoESpec | None:
    num_experts = _first(cfg, "n_routed_experts", "num_experts", "num_local_experts")
    if not num_experts:
        return None

    if cfg.get("layers_block_type"):
        moe_layers = cfg["layers_block_type"].count("moe")
    elif cfg.get("first_k_dense_replace") is not None:
        moe_layers = num_layers - cfg["first_k_dense_replace"]
        if cfg.get("moe_layer_freq", 1) != 1:
            warnings.append(f"moe_layer_freq={cfg['moe_layer_freq']} not interpreted")
    elif cfg.get("decoder_sparse_step") is not None:
        step, dense = cfg["decoder_sparse_step"], set(cfg.get("mlp_only_layers") or [])
        moe_layers = sum(1 for i in range(num_layers) if i not in dense and (i + 1) % step == 0)
    else:
        moe_layers = num_layers

    if cfg.get("mtp_layers_block_type"):
        mtp_moe = mtp_layers * cfg["mtp_layers_block_type"].count("moe")
    else:
        mtp_moe = mtp_layers  # DeepSeek-style MTP blocks are full MoE decoder layers

    activation = _first(cfg, "mlp_hidden_act", "hidden_act")
    gated = _GATED_ACTIVATIONS.get(activation)
    if gated is None:
        warnings.append(f"MLP activation {activation!r} not interpreted: active params unknown")
    top_k = _first(cfg, "num_experts_per_tok", "experts_per_token")
    if top_k is None:
        warnings.append("experts per token not found: active params unknown")
    return MoESpec(
        num_experts=num_experts,
        experts_per_token=top_k,
        expert_intermediate_size=_first(cfg, "moe_intermediate_size", "intermediate_size"),
        num_moe_layers=moe_layers,
        mtp_moe_layers=mtp_moe,
        gated_mlp=gated,
    )


def _unknown_fields(cfg: dict) -> list[str]:
    unknown = [
        key
        for key, value in cfg.items()
        if value not in (None, False, [], {}, "")
        and key not in _HANDLED
        and key not in _IRRELEVANT
        and any(word in key.lower() for word in _KEYWORDS)
    ]
    return [f"config field {key}={cfg[key]!r} not interpreted" for key in sorted(unknown)]

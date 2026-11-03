"""Config normalization: one test per supported layer type, MoE fields and warnings."""

from conftest import load_config

from launchready.arch import parse_config


def test_dense_gqa():
    arch = parse_config(load_config("Qwen__Qwen3-4B"))
    assert arch.num_layers == 36
    assert arch.count("full_attention") == 36
    assert (arch.num_kv_heads, arch.head_dim) == (8, 128)
    assert arch.max_context == 40_960
    assert arch.moe is None and arch.mamba is None
    assert arch.warnings == []


def test_head_dim_derived_when_missing():
    arch = parse_config(load_config("NousResearch__Llama-2-7b-hf"))
    assert arch.head_dim == 4096 // 32
    assert arch.num_kv_heads == 32


def test_mla():
    arch = parse_config(load_config("deepseek-ai__DeepSeek-V3"))
    assert arch.count("mla") == 61
    assert (arch.kv_lora_rank, arch.qk_rope_head_dim) == (512, 64)
    assert arch.moe.num_experts == 256
    assert arch.moe.experts_per_token == 8
    assert arch.moe.num_moe_layers == 61 - 3
    assert arch.mtp_layers == 1
    assert arch.trust_remote_code is True


def test_sliding_window_layer_types():
    arch = parse_config(load_config("openai__gpt-oss-20b"))
    assert arch.count("sliding_attention") == 12
    assert arch.count("full_attention") == 12
    assert arch.sliding_window == 128
    assert arch.moe.num_experts == 32


def test_hybrid_mamba_layer_types():
    arch = parse_config(load_config("ibm-granite__granite-4.0-h-tiny"))
    assert arch.count("mamba") == 36
    assert arch.count("full_attention") == 4
    m = arch.mamba
    assert (m.num_heads, m.head_dim, m.state_size, m.n_groups, m.conv_kernel) == (48, 64, 128, 1, 4)
    assert m.state_dtype is None
    assert arch.moe.num_experts == 64
    assert arch.moe.num_moe_layers == 40


def test_hybrid_mamba_layers_block_type():
    arch = parse_config(load_config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
    assert arch.num_layers == 52
    assert arch.count("mamba") == 23
    assert arch.count("full_attention") == 6
    assert arch.count("none") == 23
    m = arch.mamba
    assert (m.num_heads, m.head_dim, m.state_size, m.n_groups, m.conv_kernel) == (64, 64, 128, 8, 4)
    assert m.state_dtype == "float32"
    assert arch.moe.num_moe_layers == 23
    assert arch.moe.gated_mlp is False
    assert arch.mtp_layers == 1


def test_nested_text_config():
    config = {"model_type": "wrapper", "text_config": load_config("Qwen__Qwen3-4B")}
    arch = parse_config(config)
    assert arch.count("full_attention") == 36


def test_moe_active_parameters():
    arch = parse_config(load_config("Qwen__Qwen3-30B-A3B"))
    # Qwen reports 30.5B total, 3.3B active.
    active = arch.active_parameters(30_532_122_624)
    assert 3.2e9 < active < 3.4e9


def test_dense_active_equals_total():
    arch = parse_config(load_config("Qwen__Qwen3-4B"))
    assert arch.active_parameters(4_022_468_096) == 4_022_468_096


def test_unknown_architecture_field_is_named():
    config = load_config("Qwen__Qwen3-4B") | {"global_attn_every_n_window": 4}
    arch = parse_config(config)
    assert any("global_attn_every_n_window" in w for w in arch.warnings)


def test_unknown_layer_type_is_named():
    config = load_config("Qwen__Qwen3-4B") | {"layer_types": ["full_attention"] * 35 + ["chunked"]}
    arch = parse_config(config)
    assert any("chunked" in w for w in arch.warnings)


def test_qwen_use_sliding_window_true_is_warned():
    config = load_config("Qwen__Qwen3-4B") | {"use_sliding_window": True, "sliding_window": 4096}
    arch = parse_config(config)
    assert any("use_sliding_window" in w for w in arch.warnings)
    assert arch.count("full_attention") == 36

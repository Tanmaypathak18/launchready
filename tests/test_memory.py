"""docs/roadmap.md section 6.5 reference values, plus the 6.2 state and 6.3 fit formulas.

KV values are exact: no tolerances.
"""

from conftest import load_config

from launchready.arch import parse_config
from launchready.memory import (
    GIB,
    fit,
    kv_bytes_per_token,
    mamba_state_bytes,
    sequence_bytes,
    total_parameters,
    weight_bytes,
)


def test_llama31_8b_kv_per_token():
    arch = parse_config(load_config("unsloth__Meta-Llama-3.1-8B"))
    assert kv_bytes_per_token(arch, "fp16") == 131_072
    assert sequence_bytes(arch, 8192, "fp16").total == 1 * GIB


def test_llama31_70b_kv_per_token():
    arch = parse_config(load_config("unsloth__Meta-Llama-3.1-70B"))
    assert kv_bytes_per_token(arch, "fp16") == 327_680
    assert sequence_bytes(arch, 131_072, "fp16").total == 40 * GIB


def test_llama2_7b_kv_per_token_no_gqa():
    arch = parse_config(load_config("NousResearch__Llama-2-7b-hf"))
    assert kv_bytes_per_token(arch, "fp16") == 524_288
    llama31 = parse_config(load_config("unsloth__Meta-Llama-3.1-8B"))
    assert kv_bytes_per_token(arch, "fp16") == 4 * kv_bytes_per_token(llama31, "fp16")


def test_deepseek_v3_mla_kv_per_token():
    arch = parse_config(load_config("deepseek-ai__DeepSeek-V3"))
    assert kv_bytes_per_token(arch, "bf16") == 70_272


def test_fp8_kv_halves_attention_cache():
    arch = parse_config(load_config("unsloth__Meta-Llama-3.1-8B"))
    assert kv_bytes_per_token(arch, "fp8") == 65_536


def test_70b_dense_weights():
    params = 70_000_000_000
    assert weight_bytes(params, "bf16") == 140e9
    assert weight_bytes(params, "fp8") == 70e9
    assert weight_bytes(params, "nvfp4") == 39.375e9


# Safetensors header count for Lightning @ a9904d2: BF16 32,913,263,168 + F32 3,072.
LIGHTNING_PARAMS = 32_913_266_240


def test_nemotron_lightning_parameter_count():
    total, _ = total_parameters({"BF16": 32_913_263_168, "F32": 3_072}, {})
    assert total == LIGHTNING_PARAMS


def test_nemotron_lightning_weights():
    # 6.5: BF16 ~65.8 GB, NVFP4 ~18.5 GB (before the BF16 share).
    assert weight_bytes(LIGHTNING_PARAMS, "bf16") == 65_826_532_480
    assert weight_bytes(LIGHTNING_PARAMS, "nvfp4") == 18_513_712_260


def test_nemotron_lightning_bf16_does_not_fit_rtx_5090():
    arch = parse_config(load_config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
    params = LIGHTNING_PARAMS
    per_seq = sequence_bytes(arch, 8192, "fp16").total
    assert fit(32 * GIB, weight_bytes(params, "bf16"), per_seq).max_users == 0
    assert fit(32 * GIB, weight_bytes(params, "nvfp4"), per_seq).max_users > 0


def test_sliding_window_capped_at_window():
    arch = parse_config(load_config("openai__gpt-oss-20b"))
    # 12 full + 12 sliding layers, 8 KV heads x 64 head_dim, window 128.
    per_layer_token = 2 * 8 * 64 * 2
    expected = 12 * per_layer_token * 32_768 + 12 * per_layer_token * 128
    assert sequence_bytes(arch, 32_768, "fp16").attention_bytes == expected
    # Below the window, sliding layers behave like full attention.
    assert sequence_bytes(arch, 100, "fp16").attention_bytes == 24 * per_layer_token * 100


def test_granite_mamba_state_assumed_fp32():
    arch = parse_config(load_config("ibm-granite__granite-4.0-h-tiny"))
    # conv_dim = 48*64 + 2*1*128 = 3328; per layer (4-1)*3328 + 48*64*128 = 403,200 elements.
    state = mamba_state_bytes(arch)
    assert state.dtype == "float32"
    assert state.assumed is True
    assert state.bytes == 36 * 403_200 * 4


def test_nemotron_mamba_state_dtype_from_config():
    arch = parse_config(load_config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
    # conv_dim = 64*64 + 2*8*128 = 6144; per layer 3*6144 + 64*64*128 = 542,720 elements.
    state = mamba_state_bytes(arch)
    assert state.dtype == "float32"
    assert state.assumed is False
    assert state.bytes == 23 * 542_720 * 4


def test_hybrid_sequence_is_attention_plus_fixed_state():
    arch = parse_config(load_config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"))
    # 6 attention layers x 2 KV heads x 128 head_dim.
    assert kv_bytes_per_token(arch, "fp16") == 6 * 2 * 2 * 128 * 2
    short, long = sequence_bytes(arch, 1024, "fp16"), sequence_bytes(arch, 2048, "fp16")
    assert short.mamba_bytes == long.mamba_bytes
    assert long.total - short.total == 1024 * kv_bytes_per_token(arch, "fp16")


def test_fit_model_vllm_style():
    # 32 GiB x 0.90 - 8 GiB weights - 2 GiB reserve = 18.8 GiB budget; 4.5 GiB per sequence.
    result = fit(32 * GIB, 8 * GIB, int(4.5 * GIB))
    assert result.kv_budget_bytes == 32 * GIB * 0.90 - 8 * GIB - 2 * GIB
    assert result.max_users == 4


def test_fit_negative_budget_is_zero_users():
    result = fit(16 * GIB, 20 * GIB, GIB)
    assert result.kv_budget_bytes < 0
    assert result.max_users == 0


def test_total_parameters_plain():
    total, warnings = total_parameters({"BF16": 4_022_468_096}, {})
    assert total == 4_022_468_096
    assert warnings == []


def test_total_parameters_packed_4bit_counts_two_per_byte():
    config = {"quantization_config": {"quant_method": "mxfp4"}}
    total, warnings = total_parameters({"BF16": 1_000, "U8": 500}, config)
    assert total == 2_000
    assert any("U8" in w for w in warnings)


def test_total_parameters_unexplained_u8_is_warned_not_doubled():
    total, warnings = total_parameters({"BF16": 1_000, "U8": 500}, {})
    assert total == 1_500
    assert any("U8" in w for w in warnings)

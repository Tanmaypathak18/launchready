"""Quantized-variant classification and real-size selection (no network)."""

import pytest

from launchready.variants import (
    Variant,
    classify_format,
    gguf_sizes,
    load_variants,
    name_stem,
    real_sizes,
)


@pytest.mark.parametrize(
    ("repo_id", "tags", "expected"),
    [
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", [], "NVFP4"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8", [], "FP8"),
        ("unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", ["gguf"], "GGUF"),
        ("someone/lightning-quants", ["gguf", "base_model:quantized:x/y"], "GGUF"),
        ("someone/Lightning-AWQ", [], "AWQ"),
        ("someone/Lightning-GPTQ-Int4", [], "GPTQ"),
        ("mlx-community/Lightning-4bit", ["mlx"], "MLX"),
        ("someone/Lightning-bnb-4bit", [], "bitsandbytes"),
        ("someone/Lightning-q", ["base_model:quantized:x/y"], "other"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16", [], None),
        ("nvidia/Lightning-NVFP4-DFlash", ["base_model:quantized:x"], "drafter"),
    ],
)
def test_classify_format(repo_id, tags, expected):
    assert classify_format(repo_id, tags) == expected


def test_name_stem_drops_precision_and_stage_suffixes():
    assert name_stem("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16") == (
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
    )
    assert name_stem("Qwen3-4B") == "Qwen3-4B"


def test_gguf_sizes_sums_split_shards():
    files = {
        "Lightning-Q4_K_M.gguf": 19_000_000_000,
        "Q8_0/Lightning-Q8_0-00001-of-00002.gguf": 20_000_000_000,
        "Q8_0/Lightning-Q8_0-00002-of-00002.gguf": 15_000_000_000,
        "Lightning-UD-IQ2_XXS.gguf": 9_000_000_000,
        "mmproj-F16.gguf": 1_000_000,
        "README.md": 10,
    }
    sizes = gguf_sizes(files)
    assert sizes["Q4_K_M"] == 19_000_000_000
    assert sizes["Q8_0"] == 35_000_000_000
    assert sizes["IQ2_XXS"] == 9_000_000_000
    assert "F16" not in sizes  # projector files aren't model weights


def test_real_sizes_prefers_builder_org():
    variants = [
        Variant("other/L-NVFP4", "NVFP4", "other", 30_000_000_000, {}, 500),
        Variant("nvidia/L-NVFP4", "NVFP4", "nvidia", 21_561_882_284, {}, 100),
        Variant("unsloth/L-GGUF", "GGUF", "unsloth", 0, {"Q4_K_M": 19, "Q8_0": 34}, 900),
        Variant("bob/L-GGUF", "GGUF", "bob", 0, {"Q4_K_M": 18}, 10),
    ]
    sizes = real_sizes(variants, builder="nvidia")
    assert sizes["nvfp4"] == ("nvidia/L-NVFP4", 21_561_882_284)
    assert sizes["gguf-q4_k_m"] == ("unsloth/L-GGUF", 19)  # most downloaded wins otherwise
    assert sizes["gguf-q8_0"] == ("unsloth/L-GGUF", 34)
    assert "fp8" not in sizes


def test_load_variants_reads_cache(hub_cache):
    variants = load_variants("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    nvfp4 = [v for v in variants if v.repo_id.endswith("-NVFP4") and v.author == "nvidia"]
    assert nvfp4 and nvfp4[0].total_bytes == 21_561_882_284

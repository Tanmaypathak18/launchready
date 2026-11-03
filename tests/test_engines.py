"""Engine readiness: registry matchers, move detection and status, on fake checkouts (no git)."""

from pathlib import Path

import pytest

from launchready.engines import (
    ENGINES,
    Checkout,
    check,
    check_ref,
    classify,
    pick_release,
)

ARCH = "NemotronHForCausalLM"
MODEL_TYPE = "nemotron_h"
CONFIG = {"architectures": [ARCH], "model_type": MODEL_TYPE}

VLLM_REGISTRY = """\
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "NemotronHForCausalLM": ("nemotron_h", "NemotronHForCausalLM"),
}

_PREVIOUSLY_SUPPORTED_MODELS = {
    "OldForCausalLM": "0.10.0",
}
"""

SGLANG_MODEL = """\
class NemotronHForCausalLM(nn.Module):
    pass


EntryClass = [
    NemotronHForCausalLM,
    NemotronHPuzzleForCausalLM,
]
"""

LLAMA_CPP_MODEL = """\
@ModelBase.register(
    "NemotronH_Nano_VL_V2",
    "RADIOModel",
)
class NemotronNanoV2VLModel(MmprojModel):
    pass


@ModelBase.register("NemotronHForCausalLM")
@ModelBase.example("nvidia/Nemotron-H-8B-Base-8K")
class NemotronHModel(GraniteHybridModel):
    pass
"""

TRANSFORMERS_MAPPINGS = """\
from collections import OrderedDict


CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("llama", "LlamaConfig"),
        ("nemotron_h", "NemotronHConfig"),
    ]
)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("nemotron_h_fake", "nemotron_h"),
    ]
)
"""


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_vllm_match_line(tmp_path):
    root = _tree(tmp_path, {"vllm/model_executor/models/registry.py": VLLM_REGISTRY})
    result = check_ref(root, ENGINES["vllm"], ARCH)
    assert result.registry_found
    assert result.match.path == "vllm/model_executor/models/registry.py"
    assert result.match.line == 3
    assert result.match.text.startswith('"NemotronHForCausalLM"')


def test_vllm_previously_supported_is_not_support(tmp_path):
    root = _tree(tmp_path, {"vllm/model_executor/models/registry.py": VLLM_REGISTRY})
    result = check_ref(root, ENGINES["vllm"], "OldForCausalLM")
    assert result.registry_found
    assert result.match is None


def test_sglang_multiline_entryclass(tmp_path):
    root = _tree(tmp_path, {"python/sglang/srt/models/nemotron_h.py": SGLANG_MODEL})
    result = check_ref(root, ENGINES["sglang"], ARCH)
    assert result.match.path == "python/sglang/srt/models/nemotron_h.py"
    assert result.match.line == 6  # the arch's line inside the list, not the class definition


def test_sglang_single_line_entryclass(tmp_path):
    text = "class Foo: ...\nEntryClass = Foo\n"
    root = _tree(tmp_path, {"python/sglang/srt/models/foo.py": text})
    assert check_ref(root, ENGINES["sglang"], "Foo").match.line == 2
    assert check_ref(root, ENGINES["sglang"], "FooBar").match is None


def test_llama_cpp_register_decorators(tmp_path):
    root = _tree(tmp_path, {"conversion/nemotron.py": LLAMA_CPP_MODEL})
    result = check_ref(root, ENGINES["llama.cpp"], ARCH)
    assert result.match.path == "conversion/nemotron.py"
    assert result.match.line == 9
    multi = check_ref(root, ENGINES["llama.cpp"], "RADIOModel")
    assert multi.match.line == 3


def test_transformers_mapping_by_model_type(tmp_path):
    root = _tree(tmp_path, {"src/transformers/models/auto/auto_mappings.py": TRANSFORMERS_MAPPINGS})
    result = check_ref(root, ENGINES["transformers"], MODEL_TYPE)
    assert result.match.line == 7
    # Entries in other mappings in the same file don't count.
    assert check_ref(root, ENGINES["transformers"], "nemotron_h_fake").match is None


def test_moved_registry_is_found_and_reported(tmp_path):
    # Older layout: decorators in the root converter instead of conversion/*.py.
    root = _tree(tmp_path, {"convert_hf_to_gguf.py": LLAMA_CPP_MODEL})
    result = check_ref(root, ENGINES["llama.cpp"], ARCH)
    assert result.match.path == "convert_hf_to_gguf.py"
    assert result.moved == "convert_hf_to_gguf.py"


def test_missing_registry_is_unknown(tmp_path):
    root = _tree(tmp_path, {"vllm/model_executor/models/llama.py": "class Llama: ...\n"})
    result = check_ref(root, ENGINES["vllm"], ARCH)
    assert not result.registry_found
    assert result.match is None


@pytest.mark.parametrize(
    ("in_release", "in_main", "expected"),
    [(True, True, "release"), (False, True, "main-only"), (False, False, "not-found")],
)
def test_classify(tmp_path, in_release, in_main, expected):
    with_arch = _tree(tmp_path / "a", {"vllm/model_executor/models/registry.py": VLLM_REGISTRY})
    without = _tree(
        tmp_path / "b",
        {"vllm/model_executor/models/registry.py": VLLM_REGISTRY.replace(ARCH, "Other")},
    )
    spec = ENGINES["vllm"]
    release = check_ref(with_arch if in_release else without, spec, ARCH)
    main = check_ref(with_arch if in_main else without, spec, ARCH)
    assert classify(release, main) == expected


def test_classify_unknown_without_registry_or_checkout(tmp_path):
    root = _tree(tmp_path, {"README.md": "nothing here\n"})
    empty = check_ref(root, ENGINES["vllm"], ARCH)
    assert classify(empty, empty) == "unknown"
    assert classify(None, None) == "unknown"


def test_pick_release_skips_release_candidates():
    releases = [
        {"tag_name": "v0.33.0rc3", "prerelease": False, "draft": False},
        {"tag_name": "v0.33.0.dev1", "prerelease": False, "draft": False},
        {"tag_name": "v0.32.1b1", "prerelease": True, "draft": False},
        {"tag_name": "v0.32.0", "prerelease": False, "draft": False},
    ]
    assert pick_release(releases) == "v0.32.0"
    assert pick_release([]) is None


def _fake_checkouts(tmp_path, files_by_ref):
    def checkout(spec, ref, refresh=False):
        root = _tree(tmp_path / spec.key / ref, files_by_ref[spec.key][ref])
        return Checkout(path=root, ref=ref, sha="abcdef1234567890")

    return checkout


def test_check_end_to_end_on_fake_checkouts(tmp_path):
    registry = "vllm/model_executor/models/registry.py"
    files = {
        "vllm": {"v0.32.0": {registry: VLLM_REGISTRY}, "main": {registry: VLLM_REGISTRY}},
        "sglang": {
            "v0.5.23": {"python/sglang/srt/models/x.py": "EntryClass = Other\n"},
            "main": {"python/sglang/srt/models/nemotron_h.py": SGLANG_MODEL},
        },
    }
    tags = {"vllm": "v0.32.0", "sglang": "v0.5.23"}
    statuses = check(
        CONFIG,
        engines=["vllm", "sglang"],
        checkout_fn=_fake_checkouts(tmp_path, files),
        release_fn=lambda spec: tags[spec.key],
    )
    by_engine = {s.engine: s for s in statuses}
    vllm = by_engine["vLLM"]
    assert vllm.status == "release"
    assert vllm.version == "v0.32.0"
    assert vllm.url == (
        "https://github.com/vllm-project/vllm/blob/v0.32.0/"
        "vllm/model_executor/models/registry.py#L3"
    )
    sglang = by_engine["SGLang"]
    assert sglang.status == "main-only"
    assert sglang.version == "main @ abcdef1"
    assert "/blob/abcdef1234567890/" in sglang.url


def test_check_reports_failures_as_unknown(tmp_path):
    def broken(spec, ref, refresh=False):
        raise RuntimeError("git clone failed: network down")

    statuses = check(CONFIG, engines=["vllm"], checkout_fn=broken, release_fn=lambda s: "v1")
    assert statuses[0].status == "unknown"
    assert any("network down" in n for n in statuses[0].notes)


def test_check_without_architectures_is_unknown():
    statuses = check({"model_type": "x"}, engines=["vllm"], checkout_fn=None, release_fn=None)
    assert statuses[0].status == "unknown"
    assert any("architectures" in n for n in statuses[0].notes)


def test_auto_map_flags_trust_remote_code(tmp_path):
    registry = "vllm/model_executor/models/registry.py"
    files = {"vllm": {"v1": {registry: VLLM_REGISTRY}, "main": {registry: VLLM_REGISTRY}}}
    config = {**CONFIG, "auto_map": {"AutoConfig": "configuration_x.XConfig"}}
    statuses = check(
        config,
        engines=["vllm"],
        checkout_fn=_fake_checkouts(tmp_path, files),
        release_fn=lambda spec: "v1",
    )
    assert any("trust_remote_code" in n for n in statuses[0].notes)

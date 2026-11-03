"""bench/: AIPerf parsing, cost formula, GPU samples, matrix expansion, GGUF headers, lm-eval."""

import csv
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench import gguf_check, quality, run_matrix

FIXTURES = Path(__file__).parent / "fixtures" / "bench"


# --- AIPerf export → metrics -----------------------------------------------------------------


def test_parse_aiperf_export_picks_headline_metrics():
    export = json.loads((FIXTURES / "aiperf_profile_export.json").read_text())
    m = run_matrix.parse_aiperf(export)
    assert m["aiperf_version"] == "0.13.0"
    assert m["ttft_ms_p50"] == pytest.approx(2.95625)
    assert m["ttft_ms_p99"] == pytest.approx(7.82336229)
    assert m["itl_ms_p50"] == pytest.approx(2.29539519)
    assert m["itl_ms_p99"] == pytest.approx(2.49234571)
    assert m["output_tps_total"] == pytest.approx(1669.8926, rel=1e-6)
    assert m["output_tps_per_user_p50"] == pytest.approx(435.6559, rel=1e-6)
    assert m["requests_completed"] == 20
    assert m["error_rate_pct"] == 0.0
    assert m["osl_mean"] == 33.0


def test_parse_aiperf_tolerates_missing_metrics():
    m = run_matrix.parse_aiperf({"aiperf_version": "0.13.0", "error_summary": [{"x": 1}]})
    assert m["ttft_ms_p50"] is None
    assert m["output_tps_total"] is None
    assert m["errors"] == [{"x": 1}]


# --- $ per 1M output tokens (6.9) ------------------------------------------------------------


def test_cost_per_million_output_tokens():
    # $0.70/h at 1,000 tok/s: 0.70 / 3.6M tokens × 1M = $0.1944
    assert run_matrix.cost_per_million(0.70, 1000.0) == pytest.approx(0.19444, rel=1e-4)
    assert run_matrix.cost_per_million(0.70, 0.0) is None
    assert run_matrix.cost_per_million(0.70, None) is None
    assert run_matrix.cost_per_million(0.0, 500.0) == 0.0


# --- nvidia-smi samples ----------------------------------------------------------------------


def test_summarize_gpu_samples_skips_na_rows():
    s = run_matrix.summarize_gpu_samples((FIXTURES / "nvidia_smi_samples.csv").read_text())
    assert s["samples"] == 3
    assert s["mem_used_mib_peak"] == 20992
    assert s["power_w_mean"] == pytest.approx((85.5 + 410.25 + 420.75) / 3)
    assert s["power_w_peak"] == 420.75
    assert s["util_pct_mean"] == pytest.approx((10 + 97 + 99) / 3)


def test_summarize_gpu_samples_empty():
    assert run_matrix.summarize_gpu_samples("")["samples"] == 0


# --- matrix ------------------------------------------------------------------------------------

MATRIX = """
session: test
vars: {MODEL_DIR: /models/m}
workloads:
  chat: {isl: 512, osl: 256}
  long: {isl: 32768, osl: 512}
servers:
  a:
    engine: vllm
    port: 8000
    model: m
    variants:
      - {name: first, cmd: [serve, "${MODEL_DIR}", --flag, '{"k": 1}']}
      - {name: second, cmd: [serve, "${MODEL_DIR}"]}
  b:
    engine: ollama
    port: 11434
    model: m2
    variants: [{name: only, cmd: [ollama, serve]}]
runs:
  - {server: a, workload: chat, concurrency: [1, 8], reps: 3, request_count: {1: 50, 8: 200}}
  - {server: a, workload: long, concurrency: [1], request_count: 10}
  - {server: b, workload: chat, concurrency: [1], request_count: 30}
"""


def test_load_matrix_and_expand_runs(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(MATRIX)
    matrix = run_matrix.load_matrix(path)
    runs = run_matrix.expand_runs(matrix, "a")
    assert [(r.workload, r.concurrency, r.rep, r.request_count) for r in runs] == [
        ("chat", 1, 1, 50),
        ("chat", 1, 2, 50),
        ("chat", 1, 3, 50),
        ("chat", 8, 1, 200),
        ("chat", 8, 2, 200),
        ("chat", 8, 3, 200),
        ("long", 1, 1, 10),
    ]
    assert runs[0].isl == 512 and runs[-1].osl == 512
    assert run_matrix.expand_runs(matrix, "b")[0].request_count == 30


def test_load_matrix_rejects_unknown_server_and_workload(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(MATRIX.replace("server: b,", "server: nope,"))
    with pytest.raises(run_matrix.MatrixError, match="nope"):
        run_matrix.load_matrix(path)
    path.write_text(MATRIX.replace("workload: long", "workload: huge"))
    with pytest.raises(run_matrix.MatrixError, match="huge"):
        run_matrix.load_matrix(path)


def test_render_cmd_substitutes_vars_and_env_but_keeps_json(monkeypatch):
    monkeypatch.setenv("PORT_X", "9000")
    cmd = run_matrix.render_cmd(
        ["serve", "${MODEL_DIR}", "--port", "${PORT_X}", '{"k": 1}', "$UNSET_THING"],
        {"MODEL_DIR": "/m"},
    )
    assert cmd == ["serve", "/m", "--port", "9000", '{"k": 1}', "$UNSET_THING"]


def test_run_id_is_stable():
    spec = run_matrix.RunSpec("vllm-nvfp4", "chat", 8, 2, 200, 512, 256)
    assert run_matrix.run_id(spec, "no-fp8-kv") == "vllm-nvfp4__no-fp8-kv__chat__c8__r2"


def test_aiperf_command_has_confirmed_flags(tmp_path):
    spec = run_matrix.RunSpec("a", "chat", 8, 1, 200, 512, 256)
    server = {
        "port": 8000,
        "model": "m",
        "tokenizer": "org/tok",
        "tokenizer_revision": "abc",
        "extra_inputs": {"ignore_eos": True},
        "aiperf_args": ["--use-legacy-max-tokens"],
    }
    cmd = run_matrix.aiperf_command(spec, server, tmp_path, warmup=10, aiperf="aiperf")
    joined = " ".join(cmd)
    for part in [
        "profile",
        "--model m",
        "--url http://127.0.0.1:8000",
        "--endpoint-type chat",
        "--streaming",
        "--tokenizer org/tok",
        "--tokenizer-revision abc",
        "--concurrency 8",
        "--request-count 200",
        "--warmup-request-count 10",
        "--isl 512",
        "--isl-stddev 0",
        "--osl 256",
        "--osl-stddev 0",
        "--ui-type none",
        "--no-gpu-telemetry",
        "--use-legacy-max-tokens",
    ]:
        assert part in joined, part
    assert json.loads(cmd[cmd.index("--extra-inputs") + 1]) == {"ignore_eos": True}
    assert cmd[cmd.index("--artifact-dir") + 1] == str(tmp_path)


def test_parse_vllm_capacity_lines():
    log = (
        "INFO 11-03 [kv_cache_utils.py:1087] GPU KV cache size: 1,234,567 tokens\n"
        "INFO 11-03 [kv_cache_utils.py:1091] Maximum concurrency for 40,960 tokens per request: "
        "30.14x\n"
        "INFO 11-03 [gpu_model_runner.py:2100] Model loading took 20.1 GiB memory\n"
    )
    cap = run_matrix.parse_vllm_capacity(log)
    assert cap == {
        "kv_cache_tokens": 1234567,
        "max_concurrency": 30.14,
        "max_concurrency_at_tokens": 40960,
        "model_loading_gib": 20.1,
    }
    assert run_matrix.parse_vllm_capacity("nothing here") == {}


def test_append_summary_row_writes_header_once(tmp_path):
    path = tmp_path / "summary.csv"
    run_matrix.append_summary(path, {"run_id": "x", "ttft_ms_p50": 1.5, "env_gpu": "RTX 5090"})
    run_matrix.append_summary(path, {"run_id": "y", "unknown_key": 1})
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert [r["run_id"] for r in rows] == ["x", "y"]
    assert rows[0]["env_gpu"] == "RTX 5090"
    assert rows[1]["ttft_ms_p50"] == ""
    assert path.read_text().count("run_id") == 1


def test_flatten_env_prefixes_keys():
    flat = run_matrix.flatten_env({"gpu": "RTX 5090", "versions": {"vllm": "0.27.1"}})
    assert flat == {"env_gpu": "RTX 5090", "env_versions_vllm": "0.27.1"}


def test_headline_takes_median_of_reps():
    rows = [
        {
            "server": "a",
            "variant": "v",
            "workload": "chat",
            "concurrency": 8,
            "rep": i,
            "output_tps_total": tps,
            "ttft_ms_p50": 10.0 * i,
        }
        for i, tps in [(1, 900.0), (2, 1000.0), (3, 1300.0)]
    ]
    [h] = run_matrix.headline(rows)
    assert h["reps"] == 3
    assert h["output_tps_total"] == 1000.0
    assert h["output_tps_total_min"] == 900.0
    assert h["output_tps_total_max"] == 1300.0
    assert h["ttft_ms_p50"] == 20.0


def test_parse_smi_env():
    line = "NVIDIA GeForce RTX 5090, 32607, 580.95.05, 12.0\n"
    env = run_matrix.parse_smi_env(line)
    assert env == {
        "gpu": "NVIDIA GeForce RTX 5090",
        "gpu_count": 1,
        "vram_mib": 32607,
        "driver": "580.95.05",
        "compute_cap": "12.0",
    }


# --- GGUF header -------------------------------------------------------------------------------


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_arr_i32(key: str, values: list[int]) -> bytes:
    body = struct.pack("<IQ", 5, len(values)) + b"".join(struct.pack("<i", v) for v in values)
    return _s(key) + struct.pack("<I", 9) + body


def _kv_arr_str(key: str, values: list[str]) -> bytes:
    body = struct.pack("<IQ", 8, len(values)) + b"".join(_s(v) for v in values)
    return _s(key) + struct.pack("<I", 9) + body


Q5_0, Q4_K, F32 = 6, 12, 0


def _write_gguf(path: Path, kv_heads: list[int], tensors: list[tuple[str, list[int], int]]):
    """Minimal GGUF v3: KV pairs, tensor infos, 32-byte aligned data of the right sizes."""
    kvs = [
        _kv_str("general.architecture", "nemotron_h_moe"),
        _kv_str("tokenizer.ggml.pre", "tekken"),
        _kv_arr_str("tokenizer.ggml.tokens", ["<s>", "a", "b"]),
        _kv_u32("nemotron_h_moe.block_count", len(kv_heads)),
        _kv_arr_i32("nemotron_h_moe.attention.head_count_kv", kv_heads),
    ]
    infos, offset, sizes = [], 0, []
    for name, dims, ggml_type in tensors:
        n = 1
        for d in dims:
            n *= d
        block, type_size = gguf_check.GGML_TYPES[ggml_type][1:]
        size = n // block * type_size
        infos.append(
            _s(name)
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<IQ", ggml_type, offset)
        )
        sizes.append(size)
        offset += (size + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs + infos)
    head += b"\0" * ((32 - len(head) % 32) % 32)
    data = b"".join(b"\0" * ((s + 31) // 32 * 32) for s in sizes)
    path.write_bytes(head + data)


def test_gguf_header_tensor_types_and_pretokenizer(tmp_path):
    path = tmp_path / "tiny.gguf"
    _write_gguf(
        path,
        kv_heads=[0, 2],
        tensors=[
            ("token_embd.weight", [256, 4], Q4_K),
            ("blk.0.ssm_in.weight", [256, 2], Q4_K),
            ("blk.0.ffn_up_exps.weight", [64, 8, 4], Q5_0),
            ("blk.1.attn_k.weight", [256, 1], Q4_K),
            ("blk.1.attn_norm.weight", [64], F32),
        ],
    )
    with path.open("rb") as f:
        header = gguf_check.read_header(f)
    report = gguf_check.summarize(header, path.stat().st_size, name="tiny.gguf")
    assert report["label"] == "measured"
    assert report["architecture"] == "nemotron_h_moe"
    assert report["tokenizer_ggml_pre"] == "tekken"
    types = report["tensor_types"]
    assert types["Q5_0"]["params"] == 64 * 8 * 4
    assert types["Q5_0"]["bytes"] == 64 * 8 * 4 // 32 * 22
    assert types["Q4_K"]["tensors"] == 3
    assert types["F32"]["bytes"] == 64 * 4
    assert report["params_total"] == 256 * 4 + 256 * 2 + 64 * 8 * 4 + 256 + 64
    expected_bpw = report["tensor_bytes"] * 8 / report["params_total"]
    assert report["bits_per_weight"] == pytest.approx(expected_bpw)
    assert report["tensor_bytes"] <= report["file_bytes"]
    assert report["layer_check"]["mismatches"] == []
    assert report["layer_check"]["blocks"] == 2


def test_gguf_layer_check_flags_zeroed_kv_heads(tmp_path):
    # llama.cpp#27718: an attention block whose head_count_kv says 0 loads as SSM.
    path = tmp_path / "bad.gguf"
    _write_gguf(
        path,
        kv_heads=[0, 0],
        tensors=[("blk.0.ssm_in.weight", [256, 1], Q4_K), ("blk.1.attn_k.weight", [256, 1], Q4_K)],
    )
    with path.open("rb") as f:
        report = gguf_check.summarize(gguf_check.read_header(f), path.stat().st_size, name="bad")
    [mismatch] = report["layer_check"]["mismatches"]
    assert mismatch["block"] == 1
    assert "attn" in mismatch["reason"]


def test_gguf_rejects_non_gguf(tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 64)
    with path.open("rb") as f, pytest.raises(gguf_check.GGUFError, match="magic"):
        gguf_check.read_header(f)


# --- lm-eval ---------------------------------------------------------------------------------


def test_parse_lm_eval_results():
    data = json.loads((FIXTURES / "lm_eval_results.json").read_text())
    scores = quality.parse_lm_eval(data, "gsm8k")
    assert scores["exact_match,strict-match"] == 0.884
    assert scores["exact_match,flexible-extract"] == 0.892
    assert "exact_match_stderr,strict-match" in scores
    assert "alias" not in scores and "sample_len" not in scores
    assert quality.parse_lm_eval(data, "ifeval") == {}


def test_lm_eval_command_targets_chat_completions(tmp_path):
    cmd = quality.lm_eval_command(
        base_url="http://127.0.0.1:8000",
        model="m",
        tokenizer="org/tok",
        task="ifeval",
        limit=200,
        out_dir=tmp_path,
        num_concurrent=8,
        lm_eval="lm_eval",
    )
    joined = " ".join(cmd)
    assert "--model local-chat-completions" in joined
    assert "base_url=http://127.0.0.1:8000/v1/chat/completions" in joined
    assert "model=m" in joined and "num_concurrent=8" in joined
    assert "tokenized_requests=False" in joined
    assert "--tasks ifeval" in joined and "--limit 200" in joined
    assert "--apply_chat_template" in joined
    assert cmd[cmd.index("--output_path") + 1] == str(tmp_path)


def test_find_lm_eval_results_file(tmp_path):
    nested = tmp_path / "m" / "results_2026-11-03T10-00-00.json"
    nested.parent.mkdir()
    nested.write_text("{}")
    assert quality.find_results(tmp_path) == nested
    assert quality.find_results(tmp_path / "missing") is None


def test_resolve_vars_chains_and_env_wins(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/override")
    monkeypatch.delenv("NVFP4_DIR", raising=False)
    out = run_matrix.resolve_vars(
        {"MODELS_DIR": "/workspace/models", "NVFP4_DIR": "${MODELS_DIR}/n"}
    )
    assert out == {"MODELS_DIR": "/override", "NVFP4_DIR": "/override/n"}


# --- GPU paths with a fake nvidia-smi (the Kaggle dry run crashed here; the Mac has no GPU) ----

FAKE_SMI = """#!/bin/sh
echo "$@" >> "$FAKE_SMI_ARGS"
case "$*" in
  *-lms*)
    i=0
    while true; do
      echo "2026/11/03 10:00:00.000, 0, $((20000 + i)), 400.5, 97"
      i=$((i + 1))
      sleep 0.05
    done ;;
  *) echo "1000" ;;
esac
"""

FAKE_AIPERF = """import shutil, sys, time
time.sleep(0.4)  # long enough for a few fake nvidia-smi samples
out = sys.argv[sys.argv.index("--artifact-dir") + 1]
shutil.copy(sys.argv[1], out + "/profile_export_aiperf.json")
"""


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    args = tmp_path / "smi_args.txt"
    monkeypatch.setenv("FAKE_SMI_ARGS", str(args))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return args


class _FakeServer:
    variant = {"name": "v", "thinking": "off"}
    cmd = ["serve"]

    def __init__(self, cfg):
        self.cfg = cfg

    def alive(self):
        return True


def _ctx(tmp_path, aiperf):
    return {
        "session": "t",
        "warmup": 0,
        "aiperf": aiperf,
        "hourly_price": 0.89,
        "engine_version": "x",
        "meta": {"engine": "vllm"},
        "environment": {"gpu": "fake"},
        "gpu_index": 0,
    }


def test_gpu_sampler_creates_its_directory_and_reads_samples(fake_smi, tmp_path):
    path = tmp_path / "out" / "gpu" / "run.csv"  # gpu/ doesn't exist yet
    with run_matrix.GpuSampler(path, interval_ms=500, gpu_index=0) as sampler:
        time.sleep(0.4)
    s = sampler.summary()
    assert path.exists()
    assert s["samples"] >= 2
    assert s["mem_used_mib_peak"] >= 20001
    assert s["power_w_mean"] == pytest.approx(400.5)
    assert "-lms=500" in fake_smi.read_text() and "--id=0" in fake_smi.read_text()


def test_run_one_end_to_end_with_fake_gpu_and_fake_aiperf(fake_smi, tmp_path):
    fake = tmp_path / "fake_aiperf.py"
    fake.write_text(FAKE_AIPERF)
    export = FIXTURES / "aiperf_profile_export.json"
    spec = run_matrix.RunSpec("a", "chat", 8, 1, 20, 64, 32)
    server = _FakeServer({"port": 1, "model": "m", "notes": {}})
    out = tmp_path / "session"
    row = run_matrix.run_one(
        spec, server, out, _ctx(tmp_path, [sys.executable, str(fake), str(export)])
    )
    assert row["status"] == "ok", row.get("note")
    assert row["gpu_samples"] >= 1
    assert row["gpu_mem_used_mib_peak"] >= 20000
    assert row["output_tps_total"] == pytest.approx(1669.8926, rel=1e-6)
    assert (out / "gpu" / f"{row['run_id']}.csv").exists()


def test_safe_run_one_turns_unexpected_errors_into_failed_rows(fake_smi, tmp_path):
    spec = run_matrix.RunSpec("a", "chat", 1, 1, 5, 64, 32)
    server = _FakeServer({"model": "m"})  # no "port": aiperf_command raises KeyError
    row = run_matrix.safe_run_one(spec, server, tmp_path / "s", _ctx(tmp_path, ["aiperf"]))
    assert row["status"] == "failed"
    assert "KeyError" in row["note"]
    assert row["run_id"] == "a__v__chat__c1__r1"
    assert row["traceback"]


# --- thinking off: the answer must be in `content` (Kaggle dry run 2: vLLM scored a false 0.0) --


def test_null_content_response_fails_when_thinking_is_off():
    resp = json.loads((FIXTURES / "vllm_null_content_response.json").read_text())
    ok, text = run_matrix.message_text(resp, require_content=True)
    assert not ok
    assert "content is empty" in text and "42" in text


def test_reasoning_only_response_is_fine_when_thinking_is_on():
    # 64 tokens can end mid-thought, so with thinking on reasoning counts as an answer.
    resp = json.loads((FIXTURES / "vllm_null_content_response.json").read_text())
    assert run_matrix.message_text(resp, require_content=False) == (True, "42")


def test_message_text_reads_content_and_rejects_malformed():
    good = {"choices": [{"message": {"content": "Paris."}}]}
    assert run_matrix.message_text(good, require_content=True) == (True, "Paris.")
    assert run_matrix.message_text({"choices": []}, require_content=True)[0] is False


def test_require_content_follows_the_variant_thinking_label():
    assert run_matrix.requires_content({"thinking": "off (--chat-template-kwargs ...)"}) is True
    assert run_matrix.requires_content({"thinking": "on (model default)"}) is False
    assert run_matrix.requires_content({}) is False


def test_recorded_vllm_samples_are_counted_as_empty(tmp_path):
    shutil.copy(
        FIXTURES / "lm_eval_samples_vllm_null_content.jsonl", tmp_path / "samples_gsm8k_x.jsonl"
    )
    assert quality.count_empty(tmp_path, "gsm8k") == (4, 4)
    assert quality.task_status(4, 4) == "invalid: 4/4 empty responses"


def test_recorded_llamacpp_samples_are_not_empty(tmp_path):
    shutil.copy(FIXTURES / "lm_eval_samples_llamacpp.jsonl", tmp_path / "samples_gsm8k_x.jsonl")
    assert quality.count_empty(tmp_path, "gsm8k") == (0, 4)
    assert quality.task_status(0, 4) == "ok"


def test_task_status_tolerates_a_few_empty_answers():
    assert quality.task_status(1, 250) == "ok"
    assert quality.task_status(13, 250).startswith("invalid")
    assert quality.task_status(0, 0) == "ok"  # no samples file: nothing to judge


def test_invalid_task_writes_no_score_to_the_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        quality,
        "run_task",
        lambda *a: {
            "task": "gsm8k",
            "limit": 4,
            "status": "invalid: 4/4 empty responses",
            "empty_responses": 4,
            "scores": {"exact_match,strict-match": 0.0},
        },
    )
    quality.run_quality(
        base_url="u",
        model="m",
        label="vllm-bench",
        session="s",
        results=tmp_path,
        tasks={"gsm8k": 4},
        num_concurrent=1,
        meta={},
        engine_version="0.32.0",
        environment={},
        thinking="off",
    )
    [row] = list(csv.DictReader((tmp_path / "quality_summary.csv").open()))
    assert row["status"] == "invalid: 4/4 empty responses"
    assert row["value"] == "" and row["empty_responses"] == "4"


# --- dry run 2: repeated prompts hit llama-server's cache; failed requests looked like successes


def test_every_prompt_is_unique_across_warmup_and_run(tmp_path):
    spec = run_matrix.RunSpec("a", "chat", 8, 1, 200, 512, 256)
    server = {"port": 1, "model": "m"}
    cmd = run_matrix.aiperf_command(spec, server, tmp_path, warmup=10, aiperf="aiperf")
    assert cmd[cmd.index("--num-dataset-entries") + 1] == "210"


def test_requests_completed_counts_successes_only():
    export = {
        "request_count": {"avg": 62.0},
        "completed_request_count": {"avg": 64.0},
        "request_error_rate": {"avg": 3.125},
        "error_summary": [
            {"error_details": {"type": "ServerDisconnectedError"}, "count": 2},
        ],
    }
    m = run_matrix.parse_aiperf(export)
    assert m["requests_completed"] == 62
    assert run_matrix.error_note(m) == "2 failed requests (3.1%): ServerDisconnectedError × 2"
    assert run_matrix.error_note(run_matrix.parse_aiperf({})) is None


def test_run_with_failed_requests_is_partial_and_stays_in_the_headline(fake_smi, tmp_path):
    export = json.loads((FIXTURES / "aiperf_profile_export.json").read_text())
    export["request_error_rate"] = {"avg": 3.125}
    export["error_summary"] = [{"error_details": {"type": "ServerDisconnectedError"}, "count": 2}]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export))
    fake = tmp_path / "fake_aiperf.py"
    fake.write_text(FAKE_AIPERF)
    spec = run_matrix.RunSpec("a", "chat", 32, 1, 64, 64, 32)
    server = _FakeServer({"port": 1, "model": "m", "notes": {}})
    ctx = _ctx(tmp_path, [sys.executable, str(fake), str(path)])
    row = run_matrix.run_one(spec, server, tmp_path / "s", ctx)
    assert row["status"] == "partial"
    assert "ServerDisconnectedError" in row["note"]
    assert run_matrix.headline([row])[0]["reps"] == 1


# --- recipes run as written in the 5090 session ------------------------------------------------

RECIPES = Path(__file__).parent.parent / "recipes" / "nemotron-3.5-lightning"
MATRIX_5090 = Path(__file__).parent.parent / "bench" / "matrices" / "rtx5090_lightning.yaml"
NVFP4_REV = "bee7596271d1495f6992ae224aefde4410e816b8"


def _run_vllm_recipe(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "vllm"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    full_env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env}
    out = subprocess.run(
        ["bash", str(RECIPES / "vllm.sh")],
        env={**full_env, "RUNNER": "local"},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()


def test_vllm_recipe_default_has_pinned_model_and_no_fp8_kv(tmp_path):
    args = _run_vllm_recipe(tmp_path)
    assert args[0] == "serve" and NVFP4_REV in args
    assert "--kv-cache-dtype" not in args
    assert "--no-enable-prefix-caching" in args
    assert args[args.index("--max-model-len") + 1] == "40960"


def test_vllm_recipe_card_mode_keeps_the_card_flags(tmp_path):
    args = _run_vllm_recipe(tmp_path, MODE="card")
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--enable-prefix-caching" in args
    assert "--speculative_config.model" in args


def test_5090_session_runs_each_recipe_as_written():
    matrix = run_matrix.load_matrix(MATRIX_5090)
    for name in ("recipe-vllm", "recipe-llama-server", "recipe-ollama"):
        assert matrix["servers"][name]["startup_only"] is True
    vllm = matrix["servers"]["recipe-vllm"]["variants"][0]
    assert vllm["cmd"] == ["bash", "recipes/nemotron-3.5-lightning/vllm.sh"]
    assert vllm["env"]["RUNNER"] == "local"
    # The llama-server flags in the matrix are the ones printed in ollama.md.
    recipe = (RECIPES / "ollama.md").read_text()
    llama = matrix["servers"]["recipe-llama-server"]["variants"][0]["cmd"]
    assert "--port 8001 -np 8 -c 327680 -ngl 99 -fa on --jinja --no-webui --fit off" in recipe
    assert (
        " ".join(llama[3:])
        == "--port 8001 -np 8 -c 327680 -ngl 99 -fa on --jinja --no-webui --fit off"
    )
    ollama = matrix["servers"]["recipe-ollama"]
    assert ollama["variants"][0]["env"]["OLLAMA_CONTEXT_LENGTH"] == "40960"
    run = next(s for s in ollama["setup"] if s[:2] == ["ollama", "run"])
    assert "--verbose" in run and "ollama run hf.co/bartowski" in recipe


def test_nvfp4_is_downloaded_once_for_the_matrix_and_the_recipe(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    matrix = run_matrix.load_matrix(MATRIX_5090)
    variables = run_matrix.resolve_vars(matrix["vars"])
    assert variables["NVFP4_DIR"].endswith(f"/snapshots/{NVFP4_REV}")
    step = next(s for s in matrix["prepare"] if "${NVFP4_REPO}" in s)
    assert "--local-dir" not in step and step.startswith("HF_HOME=${HF_HOME} ")
    assert matrix["servers"]["recipe-vllm"]["variants"][0]["env"]["HF_HOME"] == "${HF_HOME}"

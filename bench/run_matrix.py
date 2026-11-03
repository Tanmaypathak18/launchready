"""Run a benchmark matrix: start each server, check it, benchmark it with AIPerf, stop it.

Reads a small YAML matrix (BUILD_PLAN 6.9; see bench/matrices/). For each server, tries its
variants in order (e.g. the card's `--kv-cache-dtype fp8`, then without it), records every
startup attempt, and benchmarks the first variant that starts and answers a fixed 8-request
sequence. Each run samples nvidia-smi every 500 ms and is written as one JSON as soon as it
finishes, plus one row in <results_dir>/summary.csv with the environment flattened in.

    uv run python -m bench.run_matrix --matrix bench/matrices/rtx5090_lightning.yaml \\
        --hourly-price 0.89
    uv run python -m bench.run_matrix --matrix ... --hourly-price 0 --dry-print
"""

import csv
import json
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Annotated

import typer
import yaml

AIPERF_VERSION = "0.13.0"
AIPERF = ["uvx", "--python", "3.12", "--from", f"aiperf=={AIPERF_VERSION}", "aiperf"]
SMI_QUERY = "timestamp,index,memory.used,power.draw,utilization.gpu"
STARTUP_PROMPTS = [
    "What is 17 + 25? Answer with the number only.",
    "Name the capital of France in one word.",
    "Write one sentence about GPUs.",
    "List three primary colors, comma-separated.",
    "Translate 'good morning' into Spanish.",
    "What is 9 times 8? Answer with the number only.",
    "Give a synonym for 'fast'.",
    "Is 29 a prime number? Answer yes or no.",
]
# Columns of summary.csv, in order. Environment columns (env_*) follow; unknown keys are dropped.
SUMMARY_FIELDS = [
    "run_id",
    "label",
    "session",
    "server",
    "variant",
    "engine",
    "engine_version",
    "workload",
    "isl",
    "osl",
    "concurrency",
    "rep",
    "request_count",
    "requests_completed",
    "error_rate_pct",
    "ttft_ms_p50",
    "ttft_ms_p99",
    "itl_ms_p50",
    "itl_ms_p99",
    "output_tps_total",
    "output_tps_per_user_avg",
    "output_tps_per_user_p50",
    "osl_mean",
    "isl_mean",
    "usd_per_1m_output_tokens",
    "hourly_price_usd",
    "gpu_mem_used_mib_peak",
    "gpu_power_w_mean",
    "gpu_util_pct_mean",
    "prefix_caching",
    "model_repo",
    "model_revision",
    "gguf_file",
    "gguf_tokenizer_pre",
    "started_utc",
    "duration_s",
    "status",
    "note",
]
ENV_FIELDS = [
    "env_gpu",
    "env_gpu_count",
    "env_vram_mib",
    "env_driver",
    "env_cuda",
    "env_compute_cap",
    "env_host",
    "env_provider",
    "env_container_image",
    "env_python",
    "env_aiperf",
    "env_launchready_commit",
    "env_date_utc",
]


class MatrixError(ValueError):
    """The matrix YAML is malformed or refers to something it doesn't define."""


@dataclass(frozen=True)
class RunSpec:
    server: str
    workload: str
    concurrency: int
    rep: int
    request_count: int
    isl: int
    osl: int


# --- matrix ------------------------------------------------------------------------------------


def load_matrix(path: Path) -> dict:
    """Load and validate a matrix YAML."""
    matrix = yaml.safe_load(Path(path).read_text())
    for key in ("session", "workloads", "servers", "runs"):
        if key not in matrix:
            raise MatrixError(f"{path}: missing top-level key '{key}'")
    for name, server in matrix["servers"].items():
        for key in ("engine", "port", "model", "variants"):
            if key not in server:
                raise MatrixError(f"server '{name}': missing '{key}'")
        for v in server["variants"]:
            if "name" not in v or "cmd" not in v:
                raise MatrixError(f"server '{name}': every variant needs 'name' and 'cmd'")
    for run in matrix["runs"]:
        if run.get("server") not in matrix["servers"]:
            raise MatrixError(f"run refers to unknown server '{run.get('server')}'")
        if run.get("workload") not in matrix["workloads"]:
            raise MatrixError(f"run refers to unknown workload '{run.get('workload')}'")
    return matrix


def expand_runs(matrix: dict, server: str) -> list[RunSpec]:
    """All runs for one server, in matrix order: workload × concurrency × rep."""
    specs = []
    for run in matrix["runs"]:
        if run["server"] != server:
            continue
        wl = matrix["workloads"][run["workload"]]
        conc = run.get("concurrency", [1])
        for c in conc if isinstance(conc, list) else [conc]:
            rc = run["request_count"]
            count = int(rc[c] if isinstance(rc, dict) else rc)
            for rep in range(1, int(run.get("reps", 1)) + 1):
                specs.append(
                    RunSpec(server, run["workload"], int(c), rep, count, wl["isl"], wl["osl"])
                )
    return specs


def resolve_vars(raw: dict) -> dict:
    """Matrix vars in order; each may use earlier ones and ${ENV}. The environment wins."""
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        rendered = Template(str(value)).safe_substitute({**os.environ, **resolved})
        resolved[key] = os.environ.get(key, rendered)
    return resolved


def render_cmd(cmd: list[str], variables: dict) -> list[str]:
    """Substitute ${VAR} from the matrix's vars, then the environment; JSON braces stay intact."""
    mapping = {**os.environ, **{k: str(v) for k, v in variables.items()}}
    return [Template(str(arg)).safe_substitute(mapping) for arg in cmd]


def run_id(spec: RunSpec, variant: str) -> str:
    return f"{spec.server}__{variant}__{spec.workload}__c{spec.concurrency}__r{spec.rep}"


# --- AIPerf --------------------------------------------------------------------------------------


def aiperf_command(
    spec: RunSpec, server: dict, out_dir: Path, warmup: int, aiperf: str | list[str] = AIPERF
) -> list[str]:
    """`aiperf profile` for one run; flags checked against `aiperf profile --help` (0.13.0)."""
    base = [aiperf] if isinstance(aiperf, str) else list(aiperf)
    cmd = base + [
        "profile",
        "--model",
        server["model"],
        "--url",
        f"http://127.0.0.1:{server['port']}",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        server.get("tokenizer", server["model"]),
    ]
    if server.get("tokenizer_revision"):
        cmd += ["--tokenizer-revision", server["tokenizer_revision"]]
    cmd += [
        "--concurrency",
        str(spec.concurrency),
        "--request-count",
        str(spec.request_count),
        "--warmup-request-count",
        str(warmup),
        "--isl",
        str(spec.isl),
        "--isl-stddev",
        "0",
        "--osl",
        str(spec.osl),
        "--osl-stddev",
        "0",
        "--random-seed",
        "1234",
        "--request-timeout-seconds",
        "1800",
        "--artifact-dir",
        str(out_dir),
        "--ui-type",
        "none",
        "--no-gpu-telemetry",
        "--no-server-metrics",
    ]
    if server.get("extra_inputs"):
        cmd += ["--extra-inputs", json.dumps(server["extra_inputs"])]
    return cmd + [str(a) for a in server.get("aiperf_args", [])]


def _metric(export: dict, name: str, stat: str):
    value = export.get(name)
    return value.get(stat) if isinstance(value, dict) else None


def parse_aiperf(export: dict) -> dict:
    """Headline metrics from AIPerf's profile_export_aiperf.json."""
    completed = _metric(export, "completed_request_count", "avg")
    if completed is None:
        completed = _metric(export, "request_count", "avg")
    return {
        "aiperf_version": export.get("aiperf_version"),
        "ttft_ms_p50": _metric(export, "time_to_first_token", "p50"),
        "ttft_ms_p99": _metric(export, "time_to_first_token", "p99"),
        "itl_ms_p50": _metric(export, "inter_token_latency", "p50"),
        "itl_ms_p99": _metric(export, "inter_token_latency", "p99"),
        "output_tps_total": _metric(export, "output_token_throughput", "avg"),
        "output_tps_per_user_avg": _metric(export, "output_token_throughput_per_user", "avg"),
        "output_tps_per_user_p50": _metric(export, "output_token_throughput_per_user", "p50"),
        "osl_mean": _metric(export, "output_sequence_length", "avg"),
        "isl_mean": _metric(export, "input_sequence_length", "avg"),
        "requests_completed": int(completed) if completed is not None else None,
        "error_rate_pct": _metric(export, "request_error_rate", "avg"),
        "benchmark_duration_s": _metric(export, "benchmark_duration", "avg"),
        "errors": export.get("error_summary", []),
    }


def cost_per_million(hourly_price: float, tokens_per_s: float | None) -> float | None:
    """$ per 1M output tokens = hourly price ÷ (tokens/s × 3,600) × 1,000,000 (BUILD_PLAN 6.9)."""
    if not tokens_per_s:
        return None
    return hourly_price / (tokens_per_s * 3600) * 1_000_000


# --- GPU sampling and environment ------------------------------------------------------------


def _num(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def summarize_gpu_samples(text: str) -> dict:
    """Peak memory, mean/peak power and mean utilization from nvidia-smi CSV samples."""
    mem, power, util = [], [], []
    for row in csv.reader(io_lines(text)):
        if len(row) < 5:
            continue
        m, p, u = _num(row[2]), _num(row[3]), _num(row[4])
        if m is None or p is None or u is None:
            continue
        mem.append(m)
        power.append(p)
        util.append(u)
    if not mem:
        return {"samples": 0}
    return {
        "samples": len(mem),
        "mem_used_mib_peak": max(mem),
        "power_w_mean": statistics.fmean(power),
        "power_w_peak": max(power),
        "util_pct_mean": statistics.fmean(util),
    }


def io_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


class GpuSampler:
    """`nvidia-smi ... -lms 500` into a CSV for the duration of a run; no-op without a GPU."""

    def __init__(self, path: Path, interval_ms: int = 500, gpu_index: int | None = None):
        self.path, self.interval_ms, self.proc, self.file = path, interval_ms, None, None
        self.gpu_index = gpu_index

    def __enter__(self):
        if _which("nvidia-smi"):
            self.file = self.path.open("w")
            self.proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={SMI_QUERY}",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                    *([f"--id={self.gpu_index}"] if self.gpu_index is not None else []),
                ],
                stdout=self.file,
                stderr=subprocess.DEVNULL,
            )
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.file.close()

    def summary(self) -> dict:
        if not self.path.exists():
            return {"samples": 0, "note": "nvidia-smi not found; no GPU samples"}
        return summarize_gpu_samples(self.path.read_text())


def _which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def _run_text(cmd: list[str], timeout: int = 30) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or out.stderr).strip() or None


def parse_smi_env(text: str) -> dict:
    """nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap (one line per GPU)."""
    rows = [r for r in csv.reader(io_lines(text)) if len(r) >= 4]
    if not rows:
        return {}
    name, vram, driver, cap = (c.strip() for c in rows[0][:4])
    return {
        "gpu": name,
        "gpu_count": len(rows),
        "vram_mib": int(float(vram)),
        "driver": driver,
        "compute_cap": cap,
    }


def collect_environment(matrix: dict, hourly_price: float) -> dict:
    """Everything a measured number needs next to it (CLAUDE.md: record the environment)."""
    env = {}
    smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        env.update(parse_smi_env(smi))
    banner = _run_text(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
    env["cuda"] = m.group(1) if m else None
    commit = _run_text(["git", "rev-parse", "HEAD"])
    env.update(
        {
            "host": socket.gethostname(),
            "provider": os.environ.get("BENCH_PROVIDER"),
            "container_image": os.environ.get("BENCH_CONTAINER_IMAGE"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "aiperf": AIPERF_VERSION,
            "launchready_commit": commit if commit and len(commit) == 40 else None,
            "hourly_price_usd": hourly_price,
            "session": matrix["session"],
            "matrix_vars": matrix.get("vars", {}),
            "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    return env


def flatten_env(env: dict, prefix: str = "env") -> dict:
    flat = {}
    for key, value in env.items():
        name = f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(flatten_env(value, name))
        else:
            flat[name] = value
    return flat


def append_summary(path: Path, row: dict) -> None:
    """Append one run to summary.csv right away, writing the header for a new file."""
    fields = SUMMARY_FIELDS + ENV_FIELDS
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


HEADLINE_METRICS = [
    "ttft_ms_p50",
    "ttft_ms_p99",
    "itl_ms_p50",
    "itl_ms_p99",
    "output_tps_total",
    "output_tps_per_user_p50",
    "usd_per_1m_output_tokens",
    "gpu_mem_used_mib_peak",
]


def headline(rows: list[dict]) -> list[dict]:
    """Median of repetitions per (server, variant, workload, concurrency), with min and max."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["server"], r["variant"], r["workload"], r["concurrency"])
        groups.setdefault(key, []).append(r)
    out = []
    for (server, variant, workload, conc), rs in groups.items():
        h = {
            "server": server,
            "variant": variant,
            "workload": workload,
            "concurrency": conc,
            "reps": len(rs),
        }
        for m in HEADLINE_METRICS:
            vals = [float(r[m]) for r in rs if r.get(m) not in (None, "")]
            if vals:
                h[m] = statistics.median(vals)
                h[f"{m}_min"], h[f"{m}_max"] = min(vals), max(vals)
        out.append(h)
    return out


def parse_vllm_capacity(log: str) -> dict:
    """vLLM's startup lines for KV-cache capacity, max concurrency and weights (6.8)."""
    out = {}
    if m := re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log):
        out["kv_cache_tokens"] = int(m[-1].replace(",", ""))
    if m := re.findall(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", log):
        out["max_concurrency_at_tokens"] = int(m[-1][0].replace(",", ""))
        out["max_concurrency"] = float(m[-1][1])
    if m := re.findall(r"Model loading took ([\d.]+) GiB", log):
        out["model_loading_gib"] = float(m[-1])
    return out


# --- servers -----------------------------------------------------------------------------------


def _http(url: str, body: dict | None = None, timeout: float = 10) -> tuple[int, dict | str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500]
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, str(e)


class Server:
    """One server process (own process group) with its log file."""

    def __init__(self, name: str, cfg: dict, variant: dict, variables: dict, log_path: Path):
        self.name, self.cfg, self.variant = name, cfg, variant
        self.cmd = render_cmd(variant["cmd"], variables)
        self.env = {**os.environ, **{k: str(v) for k, v in variant.get("env", {}).items()}}
        self.env = {k: Template(v).safe_substitute(self.env) for k, v in self.env.items()}
        self.base = f"http://127.0.0.1:{cfg['port']}"
        self.log_path, self.proc, self.log = log_path, None, None
        self.start_error: str | None = None
        self.variables = variables

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.log.write(f"$ {shlex.join(self.cmd)}\n")
        self.log.flush()
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
            )
        except OSError as e:  # e.g. llama-server not built: record it, don't crash the session
            self.log.write(f"failed to start: {type(e).__name__}: {e}\n")
            self.log.flush()
            self.start_error = f"{type(e).__name__}: {e}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_healthy(self, timeout: float) -> tuple[bool, float, str]:
        t0 = time.monotonic()
        path = self.cfg.get("health_path", "/health")
        if self.start_error:
            return False, 0.0, f"could not start: {self.start_error}"
        while time.monotonic() - t0 < timeout:
            if not self.alive():
                return False, time.monotonic() - t0, f"exited with code {self.proc.returncode}"
            code, _ = _http(self.base + path, timeout=5)
            if code == 200:
                return True, time.monotonic() - t0, "healthy"
            time.sleep(5)
        return False, time.monotonic() - t0, f"not healthy after {timeout:.0f} s"

    def run_setup(self) -> list[dict]:
        """Commands to run once the server is up (e.g. `ollama pull`)."""
        results = []
        for step in self.cfg.get("setup", []):
            cmd = render_cmd(step, self.variables)
            t0 = time.monotonic()
            out = subprocess.run(cmd, capture_output=True, text=True, env=self.env, timeout=3600)
            self.log.write(f"\n$ {shlex.join(cmd)}\n{out.stdout[-4000:]}{out.stderr[-4000:]}\n")
            self.log.flush()
            results.append(
                {
                    "cmd": cmd,
                    "returncode": out.returncode,
                    "seconds": round(time.monotonic() - t0, 1),
                }
            )
            if out.returncode:
                break
        return results

    def chat(self, prompt: str, max_tokens: int = 64, timeout: float = 600) -> tuple[bool, str]:
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        code, resp = _http(self.base + "/v1/chat/completions", body, timeout=timeout)
        if code != 200 or not isinstance(resp, dict):
            return False, f"HTTP {code}: {str(resp)[:300]}"
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return False, f"no choices in response: {str(resp)[:300]}"
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        text += msg.get("reasoning") or ""
        return bool(text.strip()), text[:200]

    def version(self) -> str | None:
        if url := self.cfg.get("version_path"):
            code, resp = _http(self.base + url)
            if code == 200:
                return resp.get("version") if isinstance(resp, dict) else str(resp)[:100]
        if cmd := self.cfg.get("version_cmd"):
            text = _run_text(render_cmd(cmd, self.variables))
            return text.splitlines()[0][:200] if text else None
        return None

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def stop(self, grace: float = 60) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
            except ProcessLookupError:
                pass
        if self.log and not self.log.closed:
            self.log.close()
        wait_gpu_released()


def wait_gpu_released(limit_mib: int = 1024, timeout: float = 180) -> None:
    """Wait until GPU memory drops back near idle, so the next server starts on a clean GPU."""
    if not _which("nvidia-smi"):
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        text = _run_text(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        used = [_num(x) for x in io_lines(text or "")]
        if used and all(u is not None and u < limit_mib for u in used):
            return
        time.sleep(3)


def startup_check(server: Server, timeout: float, warmup: int) -> dict:
    """Start, wait for health, run setup, send the fixed 8-request sequence, then warm up."""
    rec = {
        "server": server.name,
        "variant": server.variant["name"],
        "cmd": server.cmd,
        "env": server.variant.get("env", {}),
        "started_utc": _now(),
    }
    server.start()
    ok, seconds, why = server.wait_healthy(timeout)
    rec.update({"healthy": ok, "seconds_to_healthy": round(seconds, 1), "health": why})
    if ok and server.cfg.get("setup"):
        rec["setup"] = server.run_setup()
        ok = all(s["returncode"] == 0 for s in rec["setup"])
    answers = []
    if ok:
        for prompt in STARTUP_PROMPTS:
            good, text = server.chat(prompt)
            answers.append({"prompt": prompt, "ok": good, "text": text})
            if not good or not server.alive():
                break
    rec["sequence"] = answers
    rec["passed"] = ok and len(answers) == len(STARTUP_PROMPTS) and all(a["ok"] for a in answers)
    if rec["passed"]:
        for _ in range(warmup):
            server.chat("Write a short paragraph about the ocean.", max_tokens=128)
        rec["engine_version"] = server.version()
    log = server.log_text()
    rec["vllm_capacity"] = parse_vllm_capacity(log) if server.cfg["engine"] == "vllm" else {}
    rec["alive"] = server.alive()
    rec["log_tail"] = log.strip().splitlines()[-40:]
    rec["log_file"] = str(server.log_path)
    return rec


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- one run -----------------------------------------------------------------------------------


def server_meta(cfg: dict, variables: dict) -> dict:
    """Model identity for every row: repo, revision, GGUF file and its pre-tokenizer."""
    meta = {
        "engine": cfg["engine"],
        "model_repo": cfg.get("model_repo"),
        "model_revision": cfg.get("model_revision"),
        "prefix_caching": cfg.get("prefix_caching"),
        "gguf_file": None,
        "gguf_tokenizer_pre": None,
    }
    if gguf := cfg.get("gguf_file"):
        path = render_cmd([gguf], variables)[0]
        meta["gguf_file"] = path
        meta["gguf_sha256"] = cfg.get("gguf_sha256")
        try:
            from bench import gguf_check

            report = gguf_check.check(path)
            meta["gguf_tokenizer_pre"] = report["tokenizer_ggml_pre"]
            meta["gguf_bits_per_weight"] = report["bits_per_weight"]
        except (OSError, ValueError) as e:
            meta["gguf_error"] = f"{type(e).__name__}: {e}"
    return meta


def run_one(spec: RunSpec, server: Server, out_dir: Path, ctx: dict) -> dict:
    """One AIPerf run with GPU sampling; returns the result row (also written to JSON)."""
    rid = run_id(spec, server.variant["name"])
    art = out_dir / "aiperf" / rid
    art.mkdir(parents=True, exist_ok=True)
    cmd = aiperf_command(spec, server.cfg, art, ctx["warmup"], ctx["aiperf"])
    row = {
        "run_id": rid,
        "label": "measured",
        "session": ctx["session"],
        **asdict(spec),
        "variant": server.variant["name"],
        "engine_version": ctx["engine_version"],
        "hourly_price_usd": ctx["hourly_price"],
        "started_utc": _now(),
        "aiperf_cmd": cmd,
        "server_cmd": server.cmd,
        **ctx["meta"],
    }
    if note := server.cfg.get("notes", {}).get(spec.concurrency):
        row["note"] = note
    t0 = time.monotonic()
    with GpuSampler(out_dir / "gpu" / f"{rid}.csv", gpu_index=ctx.get("gpu_index")) as sampler:
        with (art / "aiperf_stdout.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    row["duration_s"] = round(time.monotonic() - t0, 1)
    gpu = sampler.summary()
    row.update(
        {
            "gpu_samples": gpu.get("samples", 0),
            "gpu_mem_used_mib_peak": gpu.get("mem_used_mib_peak"),
            "gpu_power_w_mean": gpu.get("power_w_mean"),
            "gpu_power_w_peak": gpu.get("power_w_peak"),
            "gpu_util_pct_mean": gpu.get("util_pct_mean"),
        }
    )
    export = art / "profile_export_aiperf.json"
    if proc.returncode or not export.exists():
        row.update({"status": "failed", "aiperf_returncode": proc.returncode})
        row["note"] = (row.get("note", "") + f" aiperf exited {proc.returncode}").strip()
    else:
        metrics = parse_aiperf(json.loads(export.read_text()))
        row.update(metrics)
        row["usd_per_1m_output_tokens"] = cost_per_million(
            ctx["hourly_price"], metrics["output_tps_total"]
        )
        row["status"] = "ok" if server.alive() else "server died"
        if metrics["osl_mean"] and abs(metrics["osl_mean"] - spec.osl) > 0.1 * spec.osl:
            row["note"] = (
                row.get("note", "") + f" mean output {metrics['osl_mean']:.0f} "
                f"tokens vs {spec.osl} requested"
            ).strip()
    row["environment"] = ctx["environment"]
    return row


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


# --- main ------------------------------------------------------------------------------------


def _deadline_passed(deadline: float | None, margin: float = 300) -> bool:
    return deadline is not None and time.time() > deadline - margin


def main(
    matrix_path: Annotated[Path, typer.Option("--matrix", help="Matrix YAML (bench/matrices/).")],
    hourly_price: Annotated[float, typer.Option(help="GPU price in $/hour, for $ per 1M tokens.")],
    only: Annotated[list[str] | None, typer.Option(help="Run only these servers.")] = None,
    with_quality: Annotated[
        bool, typer.Option(help="Run bench/quality.py on servers marked for it while they're up.")
    ] = False,
    deadline: Annotated[
        float | None, typer.Option(help="Unix time; start nothing new after it.")
    ] = None,
    dry_print: Annotated[
        bool, typer.Option(help="Print commands and exit; start nothing.")
    ] = False,
    print_prepare: Annotated[
        bool, typer.Option(help="Print the matrix's prepare commands, one per line, and exit.")
    ] = False,
) -> None:
    matrix = load_matrix(matrix_path)
    variables = resolve_vars(matrix.get("vars", {}))
    session = os.environ.get("SESSION") or matrix["session"]
    results = Path(render_cmd([matrix.get("results_dir", "results")], variables)[0])
    out_dir = results / "bench" / session
    defaults = matrix.get("defaults", {})
    warmup = int(defaults.get("warmup_requests", 10))
    servers = [s for s in matrix["servers"] if not only or s in only]

    if print_prepare:
        for step in matrix.get("prepare", []):
            print(render_cmd([step], variables)[0])
        return
    if dry_print:
        for name in servers:
            cfg = matrix["servers"][name]
            print(f"# server {name} ({cfg['engine']})")
            for v in cfg["variants"]:
                print(f"  [{v['name']}] {shlex.join(render_cmd(v['cmd'], variables))}")
            for spec in expand_runs(matrix, name):
                cmd = aiperf_command(
                    spec, cfg, out_dir / "aiperf" / run_id(spec, "VARIANT"), warmup
                )
                print(f"  {run_id(spec, 'VARIANT')}: {shlex.join(cmd)}")
        return

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    environment = collect_environment(matrix, hourly_price)
    write_json(out_dir / "environment.json", environment)
    summary_path = results / "summary.csv"
    env_flat = flatten_env({k: v for k, v in environment.items() if k != "matrix_vars"})
    all_rows: list[dict] = []

    for name in servers:
        cfg = matrix["servers"][name]
        specs = expand_runs(matrix, name)
        chosen = None
        startup_only = cfg.get("startup_only", False)
        for variant in cfg["variants"]:
            if _deadline_passed(deadline):
                print(f"[{name}] deadline reached; skipping variant {variant['name']}")
                break
            if chosen and not cfg.get("try_all_variants", False):
                break
            log_path = out_dir / "logs" / f"{name}__{variant['name']}.log"
            server = Server(name, cfg, variant, variables, log_path)
            print(f"[{name}] starting variant {variant['name']}", flush=True)
            try:
                rec = startup_check(server, float(cfg.get("startup_timeout", 1800)), warmup)
                rec["meta"] = server_meta(cfg, variables)
                write_json(out_dir / "startup" / f"{name}__{variant['name']}.json", rec)
                print(
                    f"[{name}] {variant['name']}: passed={rec['passed']} "
                    f"({rec['seconds_to_healthy']} s to healthy; {rec['health']})",
                    flush=True,
                )
                if not rec["passed"] or chosen or startup_only:
                    continue
                chosen = variant["name"]
                ctx = {
                    "session": session,
                    "warmup": warmup,
                    "aiperf": AIPERF,
                    "hourly_price": hourly_price,
                    "engine_version": rec.get("engine_version"),
                    "meta": rec["meta"],
                    "environment": environment,
                    "gpu_index": defaults.get("gpu_index"),
                }
                for spec in specs:
                    if _deadline_passed(deadline):
                        print(f"[{name}] deadline reached; skipping remaining runs")
                        break
                    if not server.alive():
                        print(f"[{name}] server died; skipping remaining runs")
                        break
                    row = run_one(spec, server, out_dir, ctx)
                    write_json(out_dir / "runs" / f"{row['run_id']}.json", row)
                    append_summary(summary_path, {**row, **env_flat})
                    all_rows.append(row)
                    heads = headline([r for r in all_rows if r.get("status") == "ok"])
                    write_json(out_dir / "headline.json", {"label": "measured", "rows": heads})
                    print(
                        f"[{name}] {row['run_id']}: {row['status']} "
                        f"tps={row.get('output_tps_total')} ttft_p50={row.get('ttft_ms_p50')}",
                        flush=True,
                    )
                if with_quality and cfg.get("quality") and server.alive():
                    from bench import quality

                    quality.run_for_server(
                        matrix,
                        name,
                        server,
                        results,
                        rec["meta"],
                        rec.get("engine_version"),
                        environment,
                        deadline,
                    )
            finally:
                server.stop()
        if not chosen and not startup_only:
            print(f"[{name}] no variant passed; nothing benchmarked (see startup/ JSONs)")

    print(f"Results: {out_dir} (runs/, startup/, logs/, gpu/) and {summary_path}")


if __name__ == "__main__":
    typer.run(main)

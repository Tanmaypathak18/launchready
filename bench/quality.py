"""Quality check: lm-evaluation-harness GSM8K (250) and IFEval (200) on an OpenAI-compatible server.

Runs `lm_eval --model local-chat-completions` (pinned through uvx) against a server that's
already up, or starts one from a matrix. Thinking is turned off on the server (see the
matrices) so both engines answer the same way; that is stated in every result. Each task is
written to results/quality/<session>/<server>__<task>.json as soon as it finishes, plus a row
in results/quality_summary.csv.

    uv run python -m bench.quality --base-url http://127.0.0.1:8000 --model nemotron-lightning \\
        --label vllm-nvfp4
    uv run python -m bench.quality --matrix bench/matrices/rtx5090_lightning.yaml \\
        --server vllm-nvfp4
"""

import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

LM_EVAL_VERSION = "0.4.13"
LM_EVAL = [
    "uvx",
    "--python",
    "3.12",
    "--from",
    f"lm-eval[api,ifeval]=={LM_EVAL_VERSION}",
    "lm_eval",
]
DEFAULT_TASKS = {"gsm8k": 250, "ifeval": 200}
QUALITY_FIELDS = [
    "label",
    "session",
    "server",
    "engine",
    "engine_version",
    "task",
    "limit",
    "metric",
    "value",
    "stderr",
    "model_repo",
    "model_revision",
    "gguf_file",
    "gguf_tokenizer_pre",
    "thinking",
    "lm_eval_version",
    "seconds",
    "status",
    "env_gpu",
    "env_driver",
    "env_cuda",
    "date_utc",
]


def lm_eval_command(
    base_url: str,
    model: str,
    task: str,
    limit: int,
    out_dir: Path,
    num_concurrent: int = 8,
    tokenizer: str | None = None,
    lm_eval: str | list[str] = LM_EVAL,
) -> list[str]:
    """lm_eval against /v1/chat/completions, greedy, chat template applied by the server."""
    base = [lm_eval] if isinstance(lm_eval, str) else list(lm_eval)
    model_args = [
        f"base_url={base_url.rstrip('/')}/v1/chat/completions",
        f"model={model}",
        f"num_concurrent={num_concurrent}",
        "max_retries=3",
        "tokenized_requests=False",
        "timeout=1800",
    ]
    cmd = base + [
        "--model",
        "local-chat-completions",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        task,
        "--limit",
        str(limit),
        "--apply_chat_template",
        "--gen_kwargs",
        "temperature=0",
        "--seed",
        "1234",
        "--output_path",
        str(out_dir),
        "--log_samples",
    ]
    if task.startswith("gsm8k"):
        cmd.append("--fewshot_as_multiturn")
    return cmd


def find_results(out_dir: Path) -> Path | None:
    """Newest results_*.json lm-eval wrote under out_dir (it nests by model name)."""
    files = sorted(Path(out_dir).rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse_lm_eval(data: dict, task: str) -> dict:
    """Metric → value for one task (e.g. 'exact_match,strict-match')."""
    scores = data.get("results", {}).get(task, {})
    return {
        k: v
        for k, v in scores.items()
        if k not in ("alias", "sample_len") and isinstance(v, int | float)
    }


def run_task(
    base_url: str, model: str, task: str, limit: int, out_dir: Path, num_concurrent: int
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = lm_eval_command(base_url, model, task, limit, out_dir, num_concurrent)
    t0 = time.monotonic()
    with (out_dir / "lm_eval.log").open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    result = {
        "task": task,
        "limit": limit,
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - t0, 1),
    }
    path = find_results(out_dir)
    if proc.returncode or path is None:
        result["status"] = "failed"
        result["scores"] = {}
        return result
    data = json.loads(path.read_text())
    result.update(
        {
            "status": "ok",
            "scores": parse_lm_eval(data, task),
            "results_file": str(path),
            "n_samples": data.get("n-samples", {}).get(task),
        }
    )
    return result


def _append(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUALITY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def run_quality(
    *,
    base_url: str,
    model: str,
    label: str,
    session: str,
    results: Path,
    tasks: dict[str, int],
    num_concurrent: int,
    meta: dict,
    engine_version: str | None,
    environment: dict,
    thinking: str,
    deadline: float | None = None,
) -> list[dict]:
    """Run each task and write its JSON and CSV rows as soon as it finishes."""
    out = results / "quality" / session
    done = []
    for task, limit in tasks.items():
        if deadline is not None and time.time() > deadline - 300:
            print(f"[quality {label}] deadline reached; skipping {task}")
            break
        print(f"[quality {label}] {task} (limit {limit})", flush=True)
        r = run_task(base_url, model, task, limit, out / f"{label}__{task}", num_concurrent)
        record = {
            "label": "measured",
            "session": session,
            "server": label,
            "engine_version": engine_version,
            "thinking": thinking,
            "lm_eval_version": LM_EVAL_VERSION,
            **meta,
            **r,
            "environment": environment,
            "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = out / f"{label}__{task}.json"
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        rows = [
            {
                **record,
                "metric": k,
                "value": v,
                "stderr": r["scores"].get(k.replace(",", "_stderr,", 1)) if "," in k else None,
                "env_gpu": environment.get("gpu"),
                "env_driver": environment.get("driver"),
                "env_cuda": environment.get("cuda"),
            }
            for k, v in r["scores"].items()
            if "_stderr" not in k
        ] or [{**record, "metric": None, "value": None}]
        _append(results / "quality_summary.csv", rows)
        print(f"[quality {label}] {task}: {r['status']} {r['scores']}", flush=True)
        done.append(record)
    return done


def _quality_settings(matrix: dict) -> dict:
    q = matrix.get("quality", {})
    return {
        "tasks": q.get("tasks", DEFAULT_TASKS),
        "num_concurrent": int(q.get("num_concurrent", 8)),
        "thinking": q.get("thinking", "off (server-side enable_thinking=false)"),
    }


def run_for_server(matrix, name, server, results, meta, engine_version, environment, deadline):
    """Called by run_matrix while a benchmarked server is still up."""
    import os

    s = _quality_settings(matrix)
    return run_quality(
        base_url=server.base,
        model=server.cfg["model"],
        label=name,
        session=os.environ.get("SESSION") or matrix["session"],
        results=results,
        tasks=s["tasks"],
        num_concurrent=s["num_concurrent"],
        meta=meta,
        engine_version=engine_version,
        environment=environment,
        thinking=s["thinking"],
        deadline=deadline,
    )


def main(
    base_url: Annotated[
        str | None, typer.Option(help="Server already running, e.g. http://127.0.0.1:8000")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Model name the server expects.")] = None,
    label: Annotated[str, typer.Option(help="Name for the result files.")] = "server",
    matrix_path: Annotated[
        Path | None, typer.Option("--matrix", help="Start the server from this matrix instead.")
    ] = None,
    server_name: Annotated[
        str | None, typer.Option("--server", help="Server in the matrix.")
    ] = None,
    gsm8k: Annotated[int, typer.Option(help="GSM8K examples.")] = 250,
    ifeval: Annotated[int, typer.Option(help="IFEval examples.")] = 200,
    results: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
) -> None:
    tasks = {"gsm8k": gsm8k, "ifeval": ifeval}
    if matrix_path:
        from bench import run_matrix as rm

        matrix = rm.load_matrix(matrix_path)
        if server_name not in matrix["servers"]:
            raise typer.BadParameter(f"--server must be one of {list(matrix['servers'])}")
        cfg = matrix["servers"][server_name]
        variables = rm.resolve_vars(matrix.get("vars", {}))
        environment = rm.collect_environment(matrix, 0.0)
        results = Path(rm.render_cmd([matrix.get("results_dir", str(results))], variables)[0])
        for variant in cfg["variants"]:
            log = results / "quality" / matrix["session"] / f"{server_name}__server.log"
            server = rm.Server(server_name, cfg, variant, variables, log)
            try:
                rec = rm.startup_check(server, float(cfg.get("startup_timeout", 1800)), 2)
                if not rec["passed"]:
                    print(f"variant {variant['name']} failed: {rec['health']}")
                    continue
                matrix.setdefault("quality", {})["tasks"] = tasks
                run_for_server(
                    matrix,
                    server_name,
                    server,
                    results,
                    rm.server_meta(cfg, variables),
                    rec.get("engine_version"),
                    environment,
                    None,
                )
                return
            finally:
                server.stop()
        raise typer.Exit(1)
    if not base_url or not model:
        raise typer.BadParameter("give --base-url and --model, or --matrix and --server")
    run_quality(
        base_url=base_url,
        model=model,
        label=label,
        session="manual",
        results=results,
        tasks=tasks,
        num_concurrent=8,
        meta={},
        engine_version=None,
        environment={},
        thinking="as configured on the server",
    )


if __name__ == "__main__":
    typer.run(main)

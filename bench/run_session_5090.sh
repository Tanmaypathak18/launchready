#!/usr/bin/env bash
# One unattended GPU session: preflight → downloads/builds (the matrix's `prepare:` list) →
# benchmarks + quality (bench/run_matrix.py --with-quality) → pack results.
#
#   HOURLY_PRICE=0.89 bash bench/run_session_5090.sh                       # RTX 5090, Lightning
#   MATRIX=bench/matrices/kaggle_qwen3_4b.yaml HOURLY_PRICE=0 MAX_HOURS=3 \
#       bash bench/run_session_5090.sh                                     # Kaggle dry run
#
# Environment:
#   HOURLY_PRICE  required: the pod's price in $/hour (for $ per 1M tokens and the cost meter)
#   MATRIX        default bench/matrices/rtx5090_lightning.yaml
#   MAX_HOURS     default 8: hard wall-clock limit for the whole session; stages are killed at it
#   SESSION       default <matrix session>_<UTC timestamp>
#   BENCH_PROVIDER, BENCH_CONTAINER_IMAGE   optional labels recorded with every result
#   BENCH_ALLOW_NO_GPU=1   only for testing this script with bench/matrices/mock_smoke.yaml
#
# Every run's JSON is written the moment it finishes, so stopping early loses at most one run.
set -uo pipefail # not -e: a failed stage is logged and the session moves on

cd "$(dirname "$0")/.."
MATRIX=${MATRIX:-bench/matrices/rtx5090_lightning.yaml}
HOURLY_PRICE=${HOURLY_PRICE:?set HOURLY_PRICE to the pod price in \$/hour, e.g. HOURLY_PRICE=0.89}
MAX_HOURS=${MAX_HOURS:-8}
START=$(date +%s)
DEADLINE=$(awk -v s="$START" -v h="$MAX_HOURS" 'BEGIN { printf "%d", s + h * 3600 }')
export BENCH_PROVIDER=${BENCH_PROVIDER:-unknown}
export BENCH_CONTAINER_IMAGE=${BENCH_CONTAINER_IMAGE:-unknown}
BOLD=$(printf '\033[1m')
RESET=$(printf '\033[0m')

elapsed() {
  local s=$(($(date +%s) - START))
  printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
}
cost() {
  awk -v s="$(($(date +%s) - START))" -v p="$HOURLY_PRICE" 'BEGIN { printf "$%.2f", s / 3600 * p }'
}
banner() {
  echo
  echo "${BOLD}== $1 · elapsed $(elapsed) · est. cost $(cost) at \$${HOURLY_PRICE}/h" \
    "· hard stop in $(((DEADLINE - $(date +%s)) / 60)) min${RESET}"
}

# --- preflight -------------------------------------------------------------------------------
banner "preflight"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
elif [ "${BENCH_ALLOW_NO_GPU:-0}" != 1 ]; then
  echo "nvidia-smi not found: no NVIDIA GPU here." >&2
  exit 1
fi
df -h . /workspace 2>/dev/null || df -h .
if ! command -v uv >/dev/null; then
  python3 -m pip install -q "uv==0.13.6" || { echo "could not install uv" >&2; exit 1; }
fi
TIMEOUT=$(command -v timeout || command -v gtimeout) ||
  { echo "GNU timeout is required (coreutils)." >&2; exit 1; }
uv sync --locked -q || { echo "uv sync failed" >&2; exit 1; }
read -r SESSION_BASE RESULTS_DIR < <(uv run python -c '
import sys
from bench.run_matrix import load_matrix, render_cmd, resolve_vars
m = load_matrix(sys.argv[1])
print(m["session"], render_cmd([m.get("results_dir", "results")], resolve_vars(m.get("vars", {})))[0])
' "$MATRIX")
export SESSION=${SESSION:-${SESSION_BASE}_$(date -u +%Y%m%dT%H%MZ)}
OUT="$RESULTS_DIR/bench/$SESSION"
LOGS="$OUT/logs/session"
mkdir -p "$LOGS"
STAGES="$OUT/session_stages.tsv"
printf 'stage\tstatus\tstarted_utc\tseconds\n' >"$STAGES"
echo "session $SESSION · matrix $MATRIX · results in $OUT"

# run_stage <name> <cmd...>: run with whatever time is left, log it, record status and duration.
run_stage() {
  local name=$1
  shift
  local remaining=$((DEADLINE - $(date +%s) - 600)) # keep 10 min to pack results
  local t0 started
  t0=$(date +%s)
  started=$(date -u +%FT%TZ)
  banner "$name"
  if [ "$remaining" -le 60 ]; then
    echo "skipped: MAX_HOURS=$MAX_HOURS reached"
    printf '%s\tskipped (time limit)\t%s\t0\n' "$name" "$(date -u +%FT%TZ)" >>"$STAGES"
    return 0
  fi
  echo "\$ $*"
  "$TIMEOUT" --signal=TERM --kill-after=120 "$remaining" "$@" 2>&1 | tee -a "$LOGS/$name.log"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 124 ] && echo "${BOLD}stage $name hit the MAX_HOURS limit${RESET}"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$started" "$(($(date +%s) - t0))" >>"$STAGES"
  return 0
}

pack() {
  trap - EXIT INT TERM
  banner "pack results"
  # Anything still holding the GPU belongs to this session.
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "llama-server" 2>/dev/null
  pkill -f "ollama serve" 2>/dev/null
  local tarball="$RESULTS_DIR/${SESSION}.tar.gz"
  local parts=("$OUT")
  [ -d "$RESULTS_DIR/quality/$SESSION" ] && parts+=("$RESULTS_DIR/quality/$SESSION")
  [ -f "$RESULTS_DIR/summary.csv" ] && parts+=("$RESULTS_DIR/summary.csv")
  [ -f "$RESULTS_DIR/quality_summary.csv" ] && parts+=("$RESULTS_DIR/quality_summary.csv")
  # AIPerf's per-request records are large; keep summaries, logs and GPU samples.
  tar czf "$tarball" --exclude='*_raw.jsonl' --exclude='inputs.json' "${parts[@]}"
  tarball="$(cd "$(dirname "$tarball")" && pwd)/$(basename "$tarball")"
  cat "$STAGES"
  echo
  echo "${BOLD}Results saved:${RESET}"
  echo "  runs (one JSON each):  $OUT/runs/"
  echo "  startup checks:        $OUT/startup/"
  echo "  headline medians:      $OUT/headline.json"
  echo "  all runs, one per row: $RESULTS_DIR/summary.csv"
  echo "  quality:               $RESULTS_DIR/quality/$SESSION/ and $RESULTS_DIR/quality_summary.csv"
  echo "  everything, packed:    $tarball ($(du -h "$tarball" | cut -f1))"
  echo
  echo "${BOLD}################################################################################${RESET}"
  echo "${BOLD}#  COPY RESULTS OFF THE POD BEFORE YOU STOP IT. Stopping or deleting the pod     #${RESET}"
  echo "${BOLD}#  deletes them. From your laptop, for example:                                  #${RESET}"
  echo "${BOLD}#    runpodctl send $tarball${RESET}"
  echo "${BOLD}#    scp -P <port> root@<pod-ip>:$tarball .${RESET}"
  echo "${BOLD}#  or download it from the Jupyter file browser.                                #${RESET}"
  echo "${BOLD}################################################################################${RESET}"
  echo "${BOLD}Session took $(elapsed), est. cost $(cost) at \$${HOURLY_PRICE}/h.${RESET}"
}
trap pack EXIT
trap 'echo "interrupted"; exit 130' INT TERM

# --- prepare: downloads, llama.cpp build, Ollama install, GGUF header check ------------------
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i + 1))
  run_stage "prepare-$i" bash -c "$line"
done < <(uv run python -m bench.run_matrix --matrix "$MATRIX" --hourly-price "$HOURLY_PRICE" \
  --print-prepare)

# --- benchmarks + quality ----------------------------------------------------------------------
run_stage "matrix" uv run python -m bench.run_matrix --matrix "$MATRIX" \
  --hourly-price "$HOURLY_PRICE" --with-quality --deadline "$((DEADLINE - 600))"

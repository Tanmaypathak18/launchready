#!/usr/bin/env bash
# NVIDIA Nemotron 3.5 Lightning, NVFP4, on one RTX 5090 (32 GB, Blackwell SM120) with vLLM.
#
#   GPU:            1× NVIDIA GeForce RTX 5090 (32 GB). CUDA 12.8+ driver.
#   Engine:         vLLM v0.27.1, container vllm/vllm-openai:v0.27.1 (the version NVIDIA's card uses;
#                   pip-installed 0.27.1 fails on the card command, NVFP4 discussion #3)
#   Model:          nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 @ bee7596271d1495f6992ae224aefde4410e816b8
#   Drafter (opt.): nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark @ 6d127b50ecd98562cea85d0087680b7d86091fb0
#   Verified:       Not run on a 5090.
#
# Modes (MODE=...):
#   no-fp8-kv (default)  NVIDIA's flags without --kv-cache-dtype fp8: two RTX 5090 / DGX Spark reports
#                        say FP8 KV crashes at startup on SM12x (NVFP4 discussions #7, #10).
#   card                 NVIDIA's DGX Spark recipe as written (FP8 KV, prefix caching, DSpark drafter).
# Options:
#   SPEC=dspark          add NVIDIA's DSpark drafter (3 speculative tokens) to no-fp8-kv
#   PREFIX_CACHING=on    turn prefix caching on (off by default: costs 13–16% with no cache hits, vllm#60008)
#   RUNNER=local         run `vllm serve` directly (inside the vllm/vllm-openai:v0.27.1 image)
#   RUNNER=docker        (default) run it in that image with Docker
#
#   bash recipes/nemotron-3.5-lightning/vllm.sh
#   curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model": "nemotron-lightning", "messages": [{"role": "user", "content": "Hello"}]}'
set -euo pipefail

IMAGE=vllm/vllm-openai:v0.27.1
MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
MODEL_REV=bee7596271d1495f6992ae224aefde4410e816b8
DSPARK=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
# The drafter is pulled at its latest revision here; the benchmark downloads it at the pinned
# revision above (bench/matrices/rtx5090_lightning.yaml).
MODE=${MODE:-no-fp8-kv}
RUNNER=${RUNNER:-docker}
PORT=${PORT:-8000}

case "$MODE" in
  card)
    # The card's DSpark recipe, flag for flag (it serves up to the model's 1M-token context).
    ARGS=(--model "$MODEL" --revision "$MODEL_REV" --served-model-name nemotron-lightning
      --moe-backend marlin --kv-cache-dtype fp8 --enable-prefix-caching
      --gpu-memory-utilization 0.85 --speculative_config.num_speculative_tokens 3
      --mamba-backend flashinfer --mamba-cache-mode align --reasoning-parser nemotron_v3
      --speculative_config.model "$DSPARK"
      --tool-call-parser qwen3_coder --enable-auto-tool-choice)
    ;;
  no-fp8-kv)
    ARGS=(--model "$MODEL" --revision "$MODEL_REV" --served-model-name nemotron-lightning
      --moe-backend marlin --gpu-memory-utilization 0.85 --mamba-backend flashinfer
      --max-model-len "${MAX_MODEL_LEN:-40960}" --reasoning-parser nemotron_v3
      --tool-call-parser qwen3_coder --enable-auto-tool-choice)
    if [ "${PREFIX_CACHING:-off}" = on ]; then
      ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
    else
      ARGS+=(--no-enable-prefix-caching)
    fi
    if [ "${SPEC:-}" = dspark ]; then
      ARGS+=(--speculative_config.model "$DSPARK" --speculative_config.num_speculative_tokens 3)
    fi
    ;;
  *)
    echo "MODE must be no-fp8-kv or card" >&2
    exit 1
    ;;
esac
ARGS+=(--port "$PORT")

if [ "$RUNNER" = local ]; then
  exec vllm serve "${ARGS[@]}"
fi
# The image's entrypoint is `vllm serve`, so only the arguments follow the image name.
exec docker run --rm --gpus all --ipc=host -p "$PORT:$PORT" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN} "$IMAGE" "${ARGS[@]}"

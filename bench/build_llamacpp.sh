#!/usr/bin/env bash
# Build llama-server with CUDA at a pinned llama.cpp tag, for the GPU in this machine.
#   bash bench/build_llamacpp.sh v0.8.0 /workspace/llama.cpp
# Skips the build if <dir>/build/bin/llama-server already exists. If nvcc is missing (vLLM's
# container ships the CUDA runtime, not always the compiler), installs the matching CUDA
# compiler and cuBLAS headers from the image's NVIDIA apt repo.
set -euo pipefail
TAG=${1:?usage: build_llamacpp.sh <tag> <dir>}
DIR=${2:?usage: build_llamacpp.sh <tag> <dir>}
BIN="$DIR/build/bin/llama-server"

if [ -x "$BIN" ]; then
  echo "llama-server already built: $BIN"
  "$BIN" --version 2>&1 | head -3 || true
  exit 0
fi

# Compute capability without the dot: 12.0 → 120 (RTX 5090), 7.5 → 75 (T4).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
echo "Building llama.cpp $TAG for CUDA architecture $ARCH into $DIR"

export PATH=/usr/local/cuda/bin:$PATH
if ! command -v nvcc >/dev/null; then
  CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || true)
  if [ -z "$CUDA_VER" ]; then
    echo "nvcc not found and the CUDA version is unknown; install the CUDA toolkit first." >&2
    exit 1
  fi
  V=${CUDA_VER/./-}
  echo "nvcc not found; installing cuda-nvcc-$V and libcublas-dev-$V (CUDA $CUDA_VER)"
  apt-get update -qq
  apt-get install -y -qq "cuda-nvcc-$V" "libcublas-dev-$V" "cuda-cudart-dev-$V"
fi
command -v cmake >/dev/null || python3 -m pip install -q "cmake==3.31.6"
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }

if [ ! -d "$DIR/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$DIR"
fi
cmake -S "$DIR" -B "$DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$DIR/build" -j "$(nproc)" --target llama-server
"$BIN" --version 2>&1 | head -3

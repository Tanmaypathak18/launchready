#!/usr/bin/env bash
# Build llama-server with CUDA at a pinned llama.cpp tag, for the GPU in this machine.
#   bash bench/build_llamacpp.sh v0.8.0 /workspace/llama.cpp
# Skips the build if <dir>/build/bin/llama-server already exists. If nvcc is missing (vLLM's
# container ships the CUDA runtime, not always the compiler), installs the matching CUDA
# compiler and cuBLAS headers from the image's NVIDIA apt repo.
#
# llama.cpp links libcuda (CUDA::cuda_driver) for its VMM memory pool. CMake only creates that
# target if it finds a file named libcuda.so, which many images lack (Kaggle: toolkit present,
# no driver stub). So find one, install the stub if needed, and pass it explicitly; build
# without VMM only as a last resort. What was used goes to build/LAUNCHREADY_BUILD.txt.
set -euo pipefail
TAG=${1:?usage: build_llamacpp.sh <tag> <dir>}
DIR=${2:?usage: build_llamacpp.sh <tag> <dir>}
BIN="$DIR/build/bin/llama-server"

if [ -x "$BIN" ]; then
  echo "llama-server already built: $BIN"
  cat "$DIR/build/LAUNCHREADY_BUILD.txt" 2>/dev/null || true
  "$BIN" --version 2>&1 | head -3 || true
  exit 0
fi

# Compute capability without the dot: 12.0 → 120 (RTX 5090), 7.5 → 75 (T4).
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
echo "Building llama.cpp $TAG for CUDA architecture $ARCH into $DIR"

apt_install() {
  apt-get update -qq && apt-get install -y -qq "$@"
}

export PATH=/usr/local/cuda/bin:$PATH
if command -v nvcc >/dev/null; then
  CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
else
  CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || true)
  if [ -z "$CUDA_VER" ]; then
    echo "nvcc not found and the CUDA version is unknown; install the CUDA toolkit first." >&2
    exit 1
  fi
  echo "nvcc not found; installing the CUDA $CUDA_VER compiler and cuBLAS headers"
  apt_install "cuda-nvcc-${CUDA_VER/./-}" "libcublas-dev-${CUDA_VER/./-}" \
    "cuda-cudart-dev-${CUDA_VER/./-}"
fi
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
echo "CUDA $CUDA_VER at $CUDA_HOME"

find_libcuda() {
  local f
  # Driver stub from the toolkit (preferred for linking: works without a driver at build time).
  for f in "$CUDA_HOME"/lib64/stubs/libcuda.so "$CUDA_HOME"/targets/*/lib/stubs/libcuda.so \
    /usr/local/cuda*/lib64/stubs/libcuda.so /usr/local/cuda*/targets/*/lib/stubs/libcuda.so; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  # The real driver library (from the host driver, mounted into the container).
  f=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so(\.1)? /{print $NF; exit}')
  [ -n "$f" ] && [ -f "$f" ] && { echo "$f"; return 0; }
  for f in /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib64/libcuda.so /usr/lib64/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}

CMAKE_EXTRA=()
LIBCUDA=$(find_libcuda || true)
if [ -z "$LIBCUDA" ]; then
  echo "libcuda not found; installing cuda-driver-dev-${CUDA_VER/./-} (driver stub)"
  apt_install "cuda-driver-dev-${CUDA_VER/./-}" || true
  LIBCUDA=$(find_libcuda || true)
fi
if [ -n "$LIBCUDA" ]; then
  echo "linking CUDA::cuda_driver against $LIBCUDA"
  CMAKE_EXTRA+=("-DCUDA_cuda_driver_LIBRARY=$LIBCUDA")
else
  echo "WARNING: no libcuda found; building with GGML_CUDA_NO_VMM=ON (legacy memory pool)" >&2
  CMAKE_EXTRA+=("-DGGML_CUDA_NO_VMM=ON")
fi

command -v cmake >/dev/null || python3 -m pip install -q "cmake==3.31.6"
command -v git >/dev/null || apt_install git
cmake --version | head -1

if [ ! -d "$DIR/.git" ]; then
  git -c advice.detachedHead=false clone --depth 1 --branch "$TAG" \
    https://github.com/ggml-org/llama.cpp "$DIR"
fi
cmake -S "$DIR" -B "$DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF "${CMAKE_EXTRA[@]}"
cmake --build "$DIR/build" -j "$(nproc)" --target llama-server
{
  echo "llama.cpp $TAG ($(git -C "$DIR" rev-parse --short HEAD))"
  echo "CUDA $CUDA_VER, arch $ARCH, $(cmake --version | head -1)"
  echo "extra cmake flags: ${CMAKE_EXTRA[*]}"
} | tee "$DIR/build/LAUNCHREADY_BUILD.txt"
"$BIN" --version 2>&1 | head -3

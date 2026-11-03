#!/usr/bin/env bash
# Install a pinned Ollama with the official install script (skips if that version is present).
#   bash bench/install_ollama.sh 0.40.14
# The benchmark matrix starts and stops `ollama serve` itself, so no service is needed. In
# containers (Kaggle, Runpod) systemd isn't running and the installer can't start one anyway;
# where systemd does run, its service is stopped so it can't hold port 11434 instead.
set -euo pipefail
VER=${1:?usage: install_ollama.sh <version>}
if command -v ollama >/dev/null && ollama --version 2>&1 | grep -q "$VER"; then
  echo "ollama $VER already installed"
else
  # The install script unpacks a .tar.zst and looks for GPUs with lspci; images often lack both.
  missing=()
  command -v zstd >/dev/null || missing+=(zstd)
  command -v lspci >/dev/null || missing+=(pciutils)
  if [ ${#missing[@]} -gt 0 ]; then
    { apt-get update -qq && apt-get install -y -qq "${missing[@]}"; } || true
  fi
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$VER" sh
fi
if [ -d /run/systemd/system ] && systemctl is-active --quiet ollama 2>/dev/null; then
  echo "stopping the ollama systemd service: the benchmark runs its own 'ollama serve'"
  systemctl stop ollama && systemctl disable ollama || true
else
  echo "no ollama service running (systemd absent or service inactive); the matrix runs 'ollama serve'"
fi
ollama --version 2>&1 | tail -1

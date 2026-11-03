#!/usr/bin/env bash
# Install a pinned Ollama with the official install script (skips if that version is present).
#   bash bench/install_ollama.sh 0.40.14
set -euo pipefail
VER=${1:?usage: install_ollama.sh <version>}
if command -v ollama >/dev/null && ollama --version 2>&1 | grep -q "$VER"; then
  echo "ollama $VER already installed"
  exit 0
fi
# The install script unpacks a .tar.zst; container images often lack zstd.
command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; } || true
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$VER" sh
ollama --version

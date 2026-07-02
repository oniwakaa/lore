#!/usr/bin/env bash
# LORE Phase 0 setup: clone fork, build with Metal, download models
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORK_REPO="https://github.com/TheTom/llama-cpp-turboquant.git"
FORK_DIR="$ROOT/external/llama-cpp-turboquant"
MODELS_DIR="$ROOT/models"

echo "=== LORE Phase 0 Setup ==="

# 1. Clone + build llama.cpp TurboQuant fork
if [ ! -d "$FORK_DIR" ]; then
    echo "[1/3] Cloning TurboQuant fork..."
    mkdir -p "$ROOT/external"
    git clone --depth 1 "$FORK_REPO" "$FORK_DIR"
else
    echo "[1/3] Fork already cloned at $FORK_DIR"
fi

if [ ! -f "$FORK_DIR/build/bin/llama-cli" ]; then
    echo "[2/3] Building with Metal..."
    cd "$FORK_DIR"
    cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
    cmake --build build --config Release -j$(sysctl -n hw.ncpu)
    cd "$ROOT"
else
    echo "[2/3] Build already exists at $FORK_DIR/build/bin/"
fi

# 3. Download models
echo "[3/3] Downloading models..."
mkdir -p "$MODELS_DIR"

if [ ! -f "$MODELS_DIR/ornith-1.0-9b-Q4_K_M.gguf" ]; then
    echo "  Downloading Ornith-1.0-9B Q4_K_M (~5.6 GB)..."
    hf download deepreinforce-ai/Ornith-1.0-9B-GGUF \
        ornith-1.0-9b-Q4_K_M.gguf --local-dir "$MODELS_DIR/"
else
    echo "  Ornith already downloaded"
fi

if [ ! -f "$MODELS_DIR/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf" ]; then
    echo "  Downloading Falcon-H1-1.5B Q4_K_M (~1 GB)..."
    hf download tiiuae/Falcon-H1-1.5B-Instruct-GGUF \
        Falcon-H1-1.5B-Instruct-Q4_K_M.gguf --local-dir "$MODELS_DIR/"
else
    echo "  Falcon-H1 already downloaded"
fi

echo ""
echo "=== Setup complete ==="
echo "Binary: $FORK_DIR/build/bin/llama-cli"
echo "Models: $MODELS_DIR/"
ls -lh "$MODELS_DIR/"*.gguf 2>/dev/null || echo "  (no model files found)"

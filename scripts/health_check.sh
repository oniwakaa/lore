#!/usr/bin/env bash
# LORE health check: verify binary, models, memory headroom
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="$ROOT/external/llama-cpp-turboquant/build/bin/llama-cli"
MODELS_DIR="$ROOT/models"

echo "=== LORE Health Check ==="
echo ""

# 1. Binary
if [ -x "$CLI" ]; then
    echo "[OK] llama-cli: $($CLI --version 2>&1 | head -1)"
else
    echo "[FAIL] llama-cli not found at $CLI"
    exit 1
fi

# 2. Models
for f in ornith-1.0-9b-Q4_K_M.gguf Falcon-H1-1.5B-Instruct-Q4_K_M.gguf; do
    if [ -f "$MODELS_DIR/$f" ]; then
        SIZE=$(ls -lh "$MODELS_DIR/$f" | awk '{print $5}')
        echo "[OK] $f ($SIZE)"
    else
        echo "[FAIL] $f missing"
    fi
done

# 3. Memory
echo ""
echo "=== System Memory ==="
memory_pressure 2>/dev/null | head -5 || echo "  (memory_pressure not available)"
sysctl hw.memsize 2>/dev/null | awk '{printf "  Total RAM: %.0f GB\n", $2/1024/1024/1024}'

# 4. Disk
echo ""
echo "=== Disk Space ==="
df -h "$ROOT" | tail -1 | awk '{printf "  Available: %s\n", $4}'

echo ""
echo "=== Health check complete ==="

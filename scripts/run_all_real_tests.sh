#!/bin/bash
# LORE — Full Real Inference Test Suite
# Runs on M4 16GB. Takes ~10-15 minutes (model loading + inference).
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "LORE Real Inference Test Suite"
echo "=========================================="
echo ""

# Kill any leftover llama-server processes
pkill -f "llama-server.*1900" 2>/dev/null || true
sleep 2

echo "--- Phase 3.5: Wiring Test ---"
PYTHONPATH=src python3 scripts/test_wiring_real.py
echo ""

# Kill servers between tests to free memory cleanly
pkill -f "llama-server.*1900" 2>/dev/null || true
sleep 3

echo "--- Phase 4: Orchestrator Test ---"
PYTHONPATH=src python3 scripts/test_orchestrator_real.py
echo ""

# Cleanup
pkill -f "llama-server.*1900" 2>/dev/null || true
rm -rf sessions/test_wiring_* sessions/test_orch_* 2>/dev/null || true

echo "=========================================="
echo "ALL REAL INFERENCE TESTS COMPLETE"
echo "=========================================="

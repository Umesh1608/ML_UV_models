#!/bin/bash
# Phase 2 CPU jobs (log_mec RF only — 81K rows, must be sequential)
set -e

echo "=== Phase 2 CPU Track ==="
echo "Started: $(date)"

# log_mec RF fold 4 (fold 3 already running separately)
echo "[CPU] log_mec RF fold 4..."
python3 run_multi_property.py --property log_mec --model rf --fold 4

echo "=== Phase 2 CPU Track DONE ==="
echo "Finished: $(date)"

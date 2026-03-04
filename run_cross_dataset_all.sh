#!/bin/bash
# Cross-dataset benchmark runner
# Waits for bigru_solvent_v2 to finish, then runs Deep4Chem + Jung 2024
# Each experiment is a single train/val/test split matching the published protocol.

set -e
cd /home/umesh/paper1_new_cl

LOG="results/cross_dataset_runner.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "=== Cross-Dataset Benchmark Runner ==="

# 1. Wait for bigru_solvent_v2 to finish (if running)
V2_PID=219076
if kill -0 $V2_PID 2>/dev/null; then
    log "Waiting for bigru_solvent_v2 (PID $V2_PID) to finish..."
    while kill -0 $V2_PID 2>/dev/null; do
        sleep 60
    done
    log "bigru_solvent_v2 finished."
    sleep 10  # Let GPU memory clear
else
    log "bigru_solvent_v2 already finished."
fi

# 2. Deep4Chem — single random 80/10/10 (matching Joung 2020)
log "Starting Deep4Chem random 80/10/10..."
python3 run_cross_dataset.py --dataset deep4chem 2>&1 | tee -a "$LOG"
log "Deep4Chem done."

# 3. Jung 2024 — single random 72/18/10 (matching Jung 2024)
log "Starting Jung 2024 random 72/18/10..."
python3 run_cross_dataset.py --dataset jung2024 2>&1 | tee -a "$LOG"
log "Jung 2024 random done."

# 4. Jung 2024 — scaffold 80/10/10
log "Starting Jung 2024 scaffold 80/10/10..."
python3 run_cross_dataset.py --dataset jung2024 --split scaffold 2>&1 | tee -a "$LOG"
log "Jung 2024 scaffold done."

# 5. Summary
log "Generating summary..."
python3 run_cross_dataset.py --summary 2>&1 | tee -a "$LOG"
log "=== All cross-dataset benchmarks complete ==="

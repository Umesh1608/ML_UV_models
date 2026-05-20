#!/usr/bin/env bash
# RunPod environment setup for the JCIM revision work.
#
# What persists vs. what doesn't on RunPod:
#   /workspace                — persistent volume, survives pod stop/restart
#   /usr/bin/tmux, node, etc. — container filesystem, WIPED on every restart
#   system Python site-packages — also wiped on restart
#
# So this script always needs to run after a fresh pod start, but the data
# in /workspace/paper1_new_cl from a previous session is already there.
#
# Usage on a fresh pod:
#   curl -O https://raw.githubusercontent.com/Umesh1608/Paper-1_New/main/setup_runpod.sh
#   bash setup_runpod.sh
# Or after rsync:
#   bash /workspace/paper1_new_cl/setup_runpod.sh

set -e

echo "=== [1/5] System packages: tmux, rsync, curl ==="
apt update -qq
apt install -y tmux rsync curl

echo "=== [2/5] Node 20 (Claude Code requires Node 18+) ==="
apt remove -y nodejs npm libnode-dev libnode72 2>/dev/null || true
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
node --version

echo "=== [3/5] Claude Code ==="
npm install -g @anthropic-ai/claude-code
claude --version

echo "=== [4/5] Python: pip + project + core scientific stack ==="
pip install --upgrade pip setuptools wheel
WORKDIR="/workspace/paper1_new_cl"
if [ -d "$WORKDIR" ]; then
    cd "$WORKDIR"
    pip install -e . 2>/dev/null || echo "(skip: package install — data may not be present yet)"
    pip install numpy pandas scipy scikit-learn rdkit matplotlib seaborn \
                xgboost tqdm joblib statsmodels optuna
else
    echo "(skip: $WORKDIR not present — rsync data from local first, then re-run)"
fi

echo "=== [5/5] tmux session 'claude' ==="
tmux kill-session -t claude 2>/dev/null || true
tmux new-session -d -s claude -c "${WORKDIR:-/workspace}" "exec bash"
# Pre-stage the Claude Code launch command in the tmux session.
# --dangerously-skip-permissions = no per-tool approval prompts (pod is ephemeral,
# data is replaceable, no production system to break). Press Enter after auth.
tmux send-keys -t claude "clear && pwd && echo 'Ready. Press Enter to launch Claude Code in auto-accept mode:' && read -p '' && claude --dangerously-skip-permissions" Enter

echo ""
echo "=== Setup complete ==="
echo "Attach with:  tmux attach -t claude"
echo "Inside tmux:  press Enter to launch Claude in auto-accept mode (--dangerously-skip-permissions)"
echo ""
echo "Heavy deps NOT installed (install on demand):"
echo "  Chemprop:    pip install chemprop torch --extra-index-url https://download.pytorch.org/whl/cu121"
echo "  ChemBERTa:   pip install transformers tokenizers torch --extra-index-url https://download.pytorch.org/whl/cu121"
echo "  TensorFlow:  pip install 'tensorflow[and-cuda]==2.20.0'"

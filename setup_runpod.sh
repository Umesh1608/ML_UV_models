#!/usr/bin/env bash
# RunPod environment setup for the JCIM revision work.
#
# What persists vs. what doesn't on RunPod:
#   /workspace                — persistent volume, survives pod stop/restart
#   /usr/bin/tmux, node, etc. — container filesystem, WIPED on every restart
#   system Python site-packages — also wiped on restart
#   the 'work' user account   — gone on restart (rebuilt here)
#
# So this script always needs to run after a fresh pod start, but the data
# in /workspace/paper1_new_cl from a previous session is already there.
#
# Claude Code refuses --dangerously-skip-permissions when run as root, so we
# create a non-root user named 'work' with passwordless sudo and start the
# tmux session as that user.
#
# Usage on a fresh pod (MUST be run as root):
#   curl -O https://raw.githubusercontent.com/Umesh1608/Paper-1_New/main/setup_runpod.sh
#   bash setup_runpod.sh
# Or after rsync:
#   bash /workspace/paper1_new_cl/setup_runpod.sh
#
# When done, attach with:
#   su - work       # switch into work user
#   tmux attach -t claude

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Run this script as root (it needs apt + useradd)."
    exit 1
fi

echo "=== [1/5] System packages: tmux, rsync, curl, sudo ==="
apt update -qq
apt install -y tmux rsync curl sudo

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

echo "=== [5/6] non-root 'work' user (Claude Code refuses --dangerously-skip-permissions as root) ==="
if ! id -u work >/dev/null 2>&1; then
    useradd -m -s /bin/bash work
fi
echo 'work ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/work
chmod 0440 /etc/sudoers.d/work
if [ -d "$WORKDIR" ]; then
    chown -R work:work "$WORKDIR"
fi

echo "=== [6/6] tmux session 'claude' (owned by work) ==="
# Kill any existing tmux server (may be root-owned from a prior run).
tmux kill-server 2>/dev/null || true
sudo -u work bash -c "pkill -u work tmux 2>/dev/null; tmux new-session -d -s claude -c '${WORKDIR:-/workspace}' 'exec bash'"
# Pre-stage the Claude Code launch in the tmux session.
sudo -u work tmux send-keys -t claude "clear && pwd && echo 'Ready. Press Enter to launch Claude Code in auto-accept mode:' && read -p '' && claude --dangerously-skip-permissions" Enter

echo ""
echo "=== Setup complete ==="
echo "Attach with:  su - work  →  tmux attach -t claude"
echo "Inside tmux:  press Enter to launch Claude in --dangerously-skip-permissions mode"
echo ""
echo "Heavy deps NOT installed (install on demand):"
echo "  Chemprop:    pip install chemprop torch --extra-index-url https://download.pytorch.org/whl/cu121"
echo "  ChemBERTa:   pip install transformers tokenizers torch --extra-index-url https://download.pytorch.org/whl/cu121"
echo "  TensorFlow:  pip install 'tensorflow[and-cuda]==2.20.0'"

#!/bin/bash
# setup_groot_so101.sh — Configure a Brev instance for
# Isaac GR00T N1.6 post-training
set -euo pipefail

REPO_DIR="${GROOT_REPO_DIR:-$HOME/Isaac-GR00T}"
GROOT_COMMIT="ead52833afbbf4243f8cd5e7664f48a94de03b19"
DATASET_REPO_ID="izuluaga/finish_sandwich"
DATASET_ROOT="examples/SO100/finish_sandwich_lerobot"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

log() { echo "==> $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: System dependencies
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 1: System dependencies"

$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    ffmpeg libaio-dev git-lfs build-essential curl

git lfs install

# CUDA toolkit 12.8 (provides nvcc for deepspeed JIT)
if ! dpkg -s cuda-toolkit-12-8 &>/dev/null; then
    UBUNTU_VERSION=$(. /etc/os-release && echo "${VERSION_ID//.}")
    if ! apt-cache show cuda-toolkit-12-8 &>/dev/null; then
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
        log "Adding NVIDIA CUDA apt repository..."
        curl -fsSL "$KEYRING_URL" -o /tmp/cuda-keyring.deb
        $SUDO dpkg -i /tmp/cuda-keyring.deb
        rm /tmp/cuda-keyring.deb
        $SUDO apt-get update -qq
    fi
    $SUDO apt-get install -y --no-install-recommends cuda-toolkit-12-8
else
    log "cuda-toolkit-12-8 already installed"
fi

# needed to manually comment this out of the launchable's included startup script
# then run the below to undo the compatibility fixes
# sed -i '\/usr\/local\/cuda-12.8\/compat/d' ~/.bashrc
# sudo apt-get remove --purge -y cuda-compat-12-8
# unset LD_LIBRARY_PATH
# source ~/.bashrc

# CUDA forward-compat library — allows CUDA 12.8 runtime on driver 550 (H100)
# if ! dpkg -s cuda-compat-12-8 &>/dev/null; then
#     $SUDO apt-get install -y --no-install-recommends cuda-compat-12-8
# else
#     log "cuda-compat-12-8 already installed"
# fi

# Persist LD_LIBRARY_PATH for CUDA compat
# COMPAT_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
# if ! grep -qF "cuda-12.8/compat" "$HOME/.bashrc" 2>/dev/null; then
#     echo "$COMPAT_LINE" >> "$HOME/.bashrc"
# fi
# export LD_LIBRARY_PATH="/usr/local/cuda-12.8/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Also ensure CUDA_HOME is set for deepspeed
CUDA_HOME_LINE='export CUDA_HOME=/usr/local/cuda-12.8'
if ! grep -qF "CUDA_HOME" "$HOME/.bashrc" 2>/dev/null; then
    echo "$CUDA_HOME_LINE" >> "$HOME/.bashrc"
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

# Ensure uv exists and is callable in this process.
if ! command -v uv &>/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
UV_PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF "$UV_PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$UV_PATH_LINE" >> "$HOME/.bashrc"
fi

if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    log "ERROR: uv is not available in PATH and ~/.local/bin/uv was not found."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Clone Isaac-GR00T
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 2: Clone Isaac-GR00T"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$GROOT_COMMIT"
    git submodule update --init --recursive
else
    log "Repository already cloned at $REPO_DIR"
    cd "$REPO_DIR"
    CURRENT=$(git rev-parse HEAD)
    if [ "$CURRENT" != "$GROOT_COMMIT" ]; then
        log "Checking out pinned commit $GROOT_COMMIT"
        git checkout "$GROOT_COMMIT"
        git submodule update --init --recursive
    fi
fi

cd "$REPO_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Python environment (via repo's own install script)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 3: Python environment"

if [ ! -d "$REPO_DIR/.venv" ]; then
    bash scripts/deployment/dgpu/install_deps.sh
else
    log ".venv already exists, skipping install_deps.sh"
fi

# Activate for subsequent phases
source "$REPO_DIR/.venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Dataset prep (manual)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 4: Dataset prep (manual)"
log "Skipping Hugging Face dataset download/conversion by design."
log "Prepare dataset separately, then place it under:"
log "  $DATASET_ROOT/$DATASET_REPO_ID"
log "For SO100 camera schema, copy modality config with:"
log "  cp examples/SO100/modality.json <DATASET_PATH>/meta/modality.json"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Verification
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 5: Verification"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"

log ""
log "============================================================"
log " Setup complete. Environment is ready for SO-101 post-training."
log "============================================================"
log ""
log "To finetune on the sample dataset:"
log ""
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 uv run bash examples/finetune.sh \\"
echo "    --base-model-path nvidia/GR00T-N1.6-3B \\"
echo "    --dataset-path $DATASET_ROOT/$DATASET_REPO_ID \\"
echo "    --modality-config-path examples/SO100/so100_config.py \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --output-dir /tmp/so100_finetune"
log ""
log "To run open-loop evaluation after training:"
log ""
echo "  export PATH=\"$HOME/.local/bin:\$PATH\""
echo "  uv run python gr00t/eval/open_loop_eval.py \\"
echo "    --dataset-path $DATASET_ROOT/$DATASET_REPO_ID/ \\"
echo "    --embodiment-tag NEW_EMBODIMENT \\"
echo "    --model-path /tmp/so100_finetune/checkpoint-10000 \\"
echo "    --traj-ids 0 --action-horizon 16 --steps 400"
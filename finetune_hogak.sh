#!/usr/bin/env bash

# If invoked as `sh finetune_hogak.sh`, re-exec with bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# If your PyTorch build errors with:
#   undefined symbol: iJIT_NotifyEvent
# preload the shared ITT runtime in the active conda env.
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libittnotify.so" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so:${LD_PRELOAD:-}"
fi

# Finetuning Hyperparameters
MODEL_ARCH="rny008_gsm"
TEMP_ARCH="gru"
CLIP_LEN=64
BATCH_SIZE=8
NUM_WORKERS=4
NUM_EPOCHS=200 # Total epochs (including pretraining epochs if resuming in same dir)

# Directory containing the pretrained checkpoint (epoch 99)
# Ensure this directory contains checkpoint_099.pt
SAVE_DIR="exp/e2espatial_pretrain_vnl2_finetune_hogak"

echo "=== Finetuning on hogak (target: ${NUM_EPOCHS} epochs) ==="
echo "Reading checkpoints from: ${SAVE_DIR}"

python train_e2e_spatial.py hogak data/hogak/frames_224p \
  -m "${MODEL_ARCH}" -t "${TEMP_ARCH}" \
  --clip_len "${CLIP_LEN}" --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  -s "${SAVE_DIR}" \
  --resume \
  --predict_location --num_workers "${NUM_WORKERS}" \
  --wandb_project e2espatial_hogak_finetune

echo "Done."

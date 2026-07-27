#!/bin/bash

# --- Script arguments ---
CONFIG=configs/train_bmst.yaml
LOGDIR=logs/ar_diffusion_chunkwise_5b_720P_bmst
mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG=$CONFIG"

# --- Print configuration ---
echo "========================================"
echo "Starting B-MST Training..."
echo "========================================"

# --- Build and launch the torchrun command ---
torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR 2>&1 | tee $LOGDIR/log.txt
  
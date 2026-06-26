#!/bin/bash

# --- Script arguments ---
CONFIG=configs/longlive_train_long_5b_720P_chunkwise_only_3th_stage_with_HOIGen_dataset_debug.yaml
LOGDIR=logs/ar_diffusion_chunkwise_long_5b_720P_chunkwise_only_3th_stage_with_HOIGen_dataset_new_debug
mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG=$CONFIG"


pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

# --- Print configuration ---
echo "========================================"
echo "Starting single-node training..."
echo "========================================"

# --- Build and launch the torchrun command ---
torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR 2>&1 | tee $LOGDIR/log.txt

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

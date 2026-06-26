#!/bin/bash

# --- User-configurable section ---
NNODES=4                               # Total number of machines
NPROC_PER_NODE=8                       # Number of GPUs per machine
MASTER_ADDR="11.41.77.124"             # IP address of the master node (must be accessible by all nodes)
MASTER_PORT=29500                      # Port on the master node
NODE_RANK=$1                           # Rank of the current node (starting from 0), passed via command-line argument

# --- Script arguments ---
CONFIG=configs/longlive_train_long_5b_720P_chunkwise_only_3th_stage_with_HOIGen_dataset_overlap1_no_switch_infinity.yaml
LOGDIR=logs/ar_diffusion_chunkwise_long_5b_720P_chunkwise_only_3th_stage_with_HOIGen_dataset_overlap1_no_switch_infinity
mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG=$CONFIG"

# --- Argument check ---
if [ -z "$NODE_RANK" ]; then
    echo "Error: Please provide the node rank (NODE_RANK) as the first argument."
    echo "Usage: bash run_multi_node.sh 0  (on the master node)"
    echo "       bash run_multi_node.sh 1  (on the first worker node)"
    exit 1
fi

pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

# --- Print configuration ---
echo "========================================"
echo "Starting multi-node training..."
echo "Total number of nodes (NNODES): $NNODES"
echo "Processes per node (NPROC_PER_NODE): $NPROC_PER_NODE"
echo "Master node address (MASTER_ADDR): $MASTER_ADDR:$MASTER_PORT"
echo "Current node rank (NODE_RANK): $NODE_RANK"
echo "========================================"

# --- Build and launch the torchrun command ---
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --rdzv_id=job_123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --node_rank=$NODE_RANK \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR 2>&1 | tee $LOGDIR/log_node_${NODE_RANK}.txt

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

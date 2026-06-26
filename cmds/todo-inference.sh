#!/usr/bin/env bash

GPU_ID="0,1,2,3,4,5,6,7"

MODE="lora"
GENERATOR_CKPT="checkpoints/longlive_base.pt"
LORA_CKPT="checkpoints/longlive_base.pt"

DATA_PATHS=(
  "prompts/vbench/demos.txt"
  "prompts/vbench/20260209170323-gemini.txt"
  "prompts/vbench/20260209170323-vbench.txt"
  "prompts/vbench/vbench/all_dimension_extended.txt"
)

INTERACTIVE_DATA_PATHS=(
  "prompts/interactive_benchmark_memflow.jsonl"
  "prompts/interactive_polish_20260108173100/interactive_benchmark_memflow.jsonl"
)

for data_path in "${DATA_PATHS[@]}"; do
  bash cmds/inference.sh "$GPU_ID" "$data_path" "$MODE" "$GENERATOR_CKPT" "$LORA_CKPT"
done

for data_path in "${DATA_PATHS[@]}"; do
  bash cmds/inference.sh "$GPU_ID" "$data_path" "$MODE" "$GENERATOR_CKPT" "$LORA_CKPT" 21
done

for data_path in "${DATA_PATHS[@]}"; do
  bash cmds/inference_infinity.sh "$GPU_ID" "$data_path" "$MODE" "$GENERATOR_CKPT" "$LORA_CKPT"
done

for data_path in "${INTERACTIVE_DATA_PATHS[@]}"; do
  bash cmds/inference_interactive.sh "$GPU_ID" "$data_path" "$MODE" "$GENERATOR_CKPT" "$LORA_CKPT"
done

# bash /m2v_intern/zhangjiaming09/Video-Causal/input_gpu.sh "$GPU_ID"

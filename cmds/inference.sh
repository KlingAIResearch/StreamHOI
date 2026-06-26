config_path="configs/longlive_inference.yaml"

GPU_ID="$1"
data_path="$2"
mode="$3"
generator_ckpt="$4"
lora_ckpt="$5"
num_output_frames="$6"

if [[ -z "$GPU_ID" || -z "$data_path" || -z "$mode" || -z "$generator_ckpt" || ( "$mode" == "ema" && ! -e "$lora_ckpt" ) ]]; then
  echo "Usage:"
  echo "  bash cmds/inference.sh <GPU_ID> <data_path> ema <generator_ckpt> <lora_ckpt> [num_output_frames]"
  echo "  bash cmds/inference.sh <GPU_ID> <data_path> lora <generator_ckpt> <lora_ckpt> [num_output_frames]"
  if [[ "$mode" == "ema" && ! -e "$lora_ckpt" ]]; then
    echo "Error: lora_ckpt path does not exist in ema mode: '$lora_ckpt'"
  fi
  exit 1
fi

if [[ "$mode" != "ema" && "$mode" != "lora" ]]; then
  echo "Error: mode must be 'ema' or 'lora', got '$mode'"
  exit 1
fi

if [[ "$mode" == "lora" && -z "$lora_ckpt" ]]; then
  echo "Error: lora mode requires <lora_ckpt>"
  exit 1
fi

if [[ -n "$num_output_frames" && ! "$num_output_frames" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: num_output_frames must be a positive integer, got '$num_output_frames'"
  exit 1
fi

if [[ -z "$num_output_frames" ]]; then
  num_output_frames=$(awk -F':' '
    /^[[:space:]]*num_output_frames:[[:space:]]*/ {
      value = $2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]*#.*/, "", value)
      sub(/[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' "$config_path")
fi

if [[ -z "$num_output_frames" ]]; then
  echo "Error: failed to read num_output_frames from $config_path"
  exit 1
fi

# data_path 相关
data_file=$(basename "$data_path" .jsonl)
data_dir=$(basename "$(dirname "$data_path")")

# ckpt 相关
if [[ "$generator_ckpt" =~ /logs/(.+)/checkpoint_model_([0-9]+)/model\.pt$ ]]; then
  gen_name="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
else
  gen_name="$(basename "$generator_ckpt" .pt)"
fi

if [[ "$mode" == "lora" ]]; then
  if [[ "$lora_ckpt" =~ /logs/(.+)/checkpoint_model_([0-9]+)/model\.pt$ ]]; then
    lora_name="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
  else
    lora_name="$(basename "$lora_ckpt" .pt)"
  fi
fi

# git 相关
GIT_VERSION=$(git rev-parse --short HEAD)

# gpu 相关
GPU_NUM=$(awk -F',' '{print NF}' <<< "$GPU_ID")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
GPU_NAME=$(echo "$GPU_NAME" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')

# python 相关
PYTHON_ENV_DIR=$(dirname "$(dirname "$(which python)")")
PYTHON_ENV_NAME=$(basename "$PYTHON_ENV_DIR")
if [[ "$PYTHON_ENV_NAME" == ".venv" ]]; then
  PYTHON_ENV_NAME="$(basename "$(dirname "$PYTHON_ENV_DIR")")-${PYTHON_ENV_NAME}"
fi

# output 文件夹
if [[ "$mode" == "ema" ]]; then
  output_folder="videos/${gen_name}-${mode}/$(basename "$0" .sh)-${data_dir}-${data_file}-${num_output_frames}/${GPU_NAME}-$(echo "$GPU_ID" | tr -d ',')-${GIT_VERSION}-${PYTHON_ENV_NAME}"
else
  output_folder="videos/${gen_name}-${lora_name}-${mode}/$(basename "$0" .sh)-${data_dir}-${data_file}-${num_output_frames}/${GPU_NAME}-$(echo "$GPU_ID" | tr -d ',')-${GIT_VERSION}-${PYTHON_ENV_NAME}"
fi
output_log="${output_folder}.log"
mkdir -p "$output_folder"

echo GPU_ID: "$GPU_ID"
echo data_path: "$data_path"
echo mode: "$mode"
echo generator_ckpt: "$generator_ckpt"
if [[ "$mode" == "lora" ]]; then
  echo lora_ckpt: "$lora_ckpt"
fi
echo num_output_frames: "$num_output_frames"
echo output_folder: "$output_folder"
echo output_log: "$output_log"

{
  echo "===== ENV: pip list ====="
  pip list
  echo "===== START torchrun ====="
} >> "$output_log" 2>&1

torchrun_args=(
  --nproc_per_node="$GPU_NUM"
  --master_port="$((29500 + $(echo "$GPU_ID" | cut -d',' -f1)))"
  inference.py
  --config_path "$config_path"
  --cover_config
  --data_path "$data_path"
  --output_folder "$output_folder"
  --generator_ckpt "$generator_ckpt"
)

if [[ "$mode" == "ema" ]]; then
  torchrun_args+=(--use_ema)
else
  torchrun_args+=(--lora_ckpt "$lora_ckpt")
fi

if [[ -n "$num_output_frames" ]]; then
  torchrun_args+=(--num_output_frames "$num_output_frames")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" NCCL_DEBUG=WARN torchrun \
  "${torchrun_args[@]}" \
  2>&1 | tee -a "$output_log"

bash /m2v_intern/zhangjiaming09/Video-Causal/input_gpu.sh $GPU_ID

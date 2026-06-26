config_path='configs/longlive_interactive_inference_infinity_5b_sink_per_block_temporal_scale.yaml'
data_path='/m2v_intern/raozejing/StreamingCode/dataset/live_5s_final_mix_HOIGen_origin_prompt/live_5s_final_mix_HOIGen_no_camera_prompt_testset_2_prompts_list.csv'
output_folder='vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_interactive_infinity_sink5_per_block_temporal_scale_prompts2/step_000321'

echo 'config_path'=$config_path
echo 'data_path'=$data_path
echo 'output_folder'=$output_folder

mkdir -p $output_folder

generator_ckpt='/m2v_intern_v3/raozejing/logs/longlive_dmd_init_chunkwise_only_3th_stage_with_HOIGen_dataset_origin_prompts/checkpoint_model_000321/model.pt'
lora_ckpt=''

echo 'generator_ckpt'=$generator_ckpt
echo 'lora_ckpt'=$lora_ckpt

pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path $config_path \
  --data_path $data_path \
  --output_folder $output_folder\
  --generator_ckpt $generator_ckpt \
  --cover_config

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

# torchrun \
#   --nproc_per_node=8 \
#   --master_port=29500 \
#   inference.py \
#   --config_path $config_path \
#   --data_path $data_path \
#   --output_folder $output_folder\
#   --generator_ckpt $generator_ckpt \
#   --lora_ckpt $lora_ckpt \
#   --cover_config
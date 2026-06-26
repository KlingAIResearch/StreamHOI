config_path='configs/longlive_inference_infinity_5b_sink_per_block_temporal_scale.yaml'
data_path='/m2v_intern/raozejing/StreamingCode/dataset/live_5s_final_mix_HOIGen_origin_prompt/live_5s_final_mix_HOIGen_no_camera_prompt_testset.csv'
# data_path='/m2v_intern/raozejing/StreamingCode/dataset/homa/evalset_896x512.csv'
output_folder='vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink_temporal_scale_test111/step_000321'
# output_folder='vis/homa-testset-896*512-60s-Ours'
vis_sink_output_dir='/m2v_intern_v3/raozejing/vis/vis_sink_local_output_with_bar_5chunk_new'

echo 'config_path'=$config_path
echo 'data_path'=$data_path
echo 'output_folder'=$output_folder
echo 'vis_sink_output_dir'=$vis_sink_output_dir

mkdir -p $output_folder
mkdir -p $vis_sink_output_dir

generator_ckpt='/m2v_intern_v3/raozejing/logs/longlive_dmd_init_chunkwise_only_3th_stage_with_HOIGen_dataset_origin_prompts/checkpoint_model_000321/model.pt'
lora_ckpt=''

echo 'generator_ckpt'=$generator_ckpt
echo 'lora_ckpt'=$lora_ckpt

# pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
# bash /home/raozejing/input_gpu.sh 0,1,2,3,4,5,6,7
# bash /home/raozejing/input_gpu.sh 0,1,2,3,4,5,6,7

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  inference.py \
  --config_path $config_path \
  --data_path $data_path \
  --output_folder $output_folder \
  --generator_ckpt $generator_ckpt \
  --cover_config \
  --vis_sink_output_dir $vis_sink_output_dir \
  # --input_multi_frames 9

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
  # --visualize_sink_attn \
  # --visualize_local_attn \
  
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
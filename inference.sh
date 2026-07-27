config_path='configs/inference_streamhoi.yaml'
data_path='demo/cases.csv'
output_folder='demo/output'

echo 'config_path'=$config_path
echo 'data_path'=$data_path
echo 'output_folder'=$output_folder

mkdir -p $output_folder

generator_ckpt='checkpoints/streamhoi_model.pt' #download
lora_ckpt='checkpoints/streamhoi_lora.pt' #download

echo 'generator_ckpt'=$generator_ckpt
echo 'lora_ckpt'=$lora_ckpt


torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  inference.py \
  --config_path $config_path \
  --data_path $data_path \
  --output_folder $output_folder\
  --generator_ckpt $generator_ckpt \
  --lora_ckpt $lora_ckpt \
  --cover_config

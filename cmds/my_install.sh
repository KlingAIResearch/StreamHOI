# registry.corp.kuaishou.com/kml-supercomputing-project/jisihui-3.4-ft:m2v_nv_torch221_cu12_ema_0524-snapshot-18594-20250225193716-snapshot-18837-20250304014032
cd /m2v_intern/zhangjiaming09/Video-Causal/ResonInteraction
source ~/.bashrc && conda deactivate && conda activate base && which python
export http_proxy=http://10.66.29.113:11080 https_proxy=http://10.66.29.113:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
conda activate ft
which python && which pip && which conda && pip cache dir

python -c "import torch;print(torch.__version__);print(torch.cuda.is_available())" && cd ../ && bash train.sh && cd - && pwd
pip install easydict ftfy omegaconf lmdb datasets av diffusers==0.31.0 flask tensorboard

pip install /m2v_intern/zhangjiaming09/I2V_Animation/sota_animate/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl # pip install flash-attn==2.7.1.post4
python -c "import flash_attn" && pip list | grep flash

pip install -U wandb



# source /pfs/xuyifan09/envs/ResonInteraction-BDY/bin/activate
cd /m2v_intern/zhangjiaming09/Video-Causal/ResonInteraction
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
export http_proxy=http://10.66.29.113:11080 https_proxy=http://10.66.29.113:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
uv pip install -r requirements.txt
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
uv pip install git+https://github.com/openai/CLIP.git
uv pip install /m2v_intern/zhangjiaming09/I2V_Animation/sota_animate/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl # pip install flash-attn==2.7.1.post4
python -c "import flash_attn" && uv pip list | grep flash
uv pip install -U wandb
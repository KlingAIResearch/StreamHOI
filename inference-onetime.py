import os
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np

from utils.misc import set_seed
from utils.global_config import get_wan_version
from utils.wan_wrapper import VideoPath
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline import CausalInferencePipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(SCRIPT_DIR, "input_video.mp4")
CAPTION = (
    "A man in a blue shirt stands behind a wooden cutting board on a white table. "
    "He is focused on slicing a red onion with a knife. Surrounding the cutting board "
    "are various vegetables, including a carrot, a red cabbage, and a red onion, all of "
    "which are placed on the table. The man's hands are steady as he slices the onion, "
    "and his facial expression suggests concentration. The background is plain. The "
    "lighting is bright, highlighting the colors of the vegetables and the man's shirt. "
    "The scene captures a moment of culinary preparation, emphasizing the process of "
    "cutting ingredients."
)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs/longlive_inference_infinity_5b_sink_per_block_temporal_scale.yaml")
GENERATOR_CKPT = "/m2v_intern_v3/raozejing/logs/longlive_dmd_init_chunkwise_only_3th_stage_with_HOIGen_dataset_origin_prompts/checkpoint_model_000321/model.pt"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output_onetime.mp4")


def read_video_frames(video_path, num_frames=None):
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        vlen = len(vr)
        idx = list(range(vlen))
        frames = vr.get_batch(idx).asnumpy()
        return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).convert("RGB"))
    cap.release()
    return frames


def process_frames(frames_pil, height=1280, width=704):
    def cover_resize(image):
        w, h = image.size
        scale = max(width / w, height / h)
        shape = [round(h * scale), round(w * scale)]
        return torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )

    video_t = []
    for im in frames_pil:
        im = cover_resize(im)
        im = torchvision.transforms.functional.center_crop(im, (height, width))
        t = torchvision.transforms.functional.to_tensor(im)
        t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
        video_t.append(t)
    return torch.stack(video_t, dim=0)  # [T, 3, H, W]


device = torch.device("cuda:0")
torch.cuda.set_device(0)

config = OmegaConf.load(CONFIG_PATH)
config.generator_ckpt = GENERATOR_CKPT
config.distributed = False
set_seed(config.seed)

torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=device)
latent_spatial_shape = list(getattr(config, "image_or_video_shape", []))[2:]
if not latent_spatial_shape:
    latent_spatial_shape = [16, 60, 104]

state_dict = torch.load(config.generator_ckpt, map_location="cpu", mmap=True)
if "generator" in state_dict or "generator_ema" in state_dict:
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
elif "model" in state_dict:
    raw_gen_state_dict = state_dict["model"]
else:
    raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
pipeline.generator.load_state_dict(raw_gen_state_dict)

pipeline = pipeline.to(dtype=torch.bfloat16)
low_memory = True
DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

print(f"Loading video from: {VIDEO_PATH}")
frames_pil = read_video_frames(VIDEO_PATH)
print(f"Loaded {len(frames_pil)} frames")

frames_tensor = process_frames(frames_pil).to(device=device, dtype=torch.bfloat16)
frames_vae_input = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

with torch.no_grad():
    clean_latent = pipeline.vae.encode_to_latent(frames_vae_input).to(device=device, dtype=torch.bfloat16)
initial_latent = clean_latent[:, 0:1]

num_output_frames = config.num_output_frames
sampled_noise = torch.randn(
    [1, num_output_frames - 1, *latent_spatial_shape], device=device, dtype=torch.bfloat16
)

print(f"Running inference, caption: {CAPTION[:80]}...")
video, latents = pipeline.inference(
    noise=sampled_noise,
    text_prompts=[CAPTION],
    return_latents=True,
    low_memory=low_memory,
    initial_latent=initial_latent,
)

pipeline.vae.model.clear_cache()

video = rearrange(video, 'b t c h w -> b t h w c').cpu()
video = 255.0 * video
write_video(OUTPUT_PATH, video[0], fps=16)
print(f"Saved to: {OUTPUT_PATH}")

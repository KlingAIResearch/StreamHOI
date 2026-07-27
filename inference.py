# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List
import shutil

import torch
import os
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms
from einops import rearrange
from utils.misc import set_seed
from utils.global_config import get_wan_version
from utils.wan_wrapper import VideoPath
from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, I2VDataset, ImageDataset, I2VDataset_Origin_Size
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")

parser.add_argument("--cover_config", action='store_true')
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--output_folder", type=str, default=None)
parser.add_argument("--generator_ckpt", type=str, default=None)
parser.add_argument("--lora_ckpt", type=str, default=None)
parser.add_argument("--use_ema", action='store_true')
parser.add_argument("--num_output_frames", type=int, default=None)

args = parser.parse_args()

def is_rank0():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)

if args.num_output_frames is not None and args.num_output_frames <= 0:
    raise ValueError("--num_output_frames must be a positive integer")

config = OmegaConf.load(args.config_path)
if args.cover_config:
    for key in ['data_path', 'output_folder', 'generator_ckpt', 'lora_ckpt', 'num_output_frames']:
        if getattr(args, key) is not None:
            rank0_print(f"config[{key}]: {getattr(config, key)} -> {getattr(args, key)}")
            setattr(config, key, getattr(args, key))
    if args.use_ema:
        rank0_print(f"config[use_ema]: {getattr(config, 'use_ema', None)} -> True")
        rank0_print(f"config[adapter]: {getattr(config, 'adapter', None)} -> None")
        config.use_ema = True
        config.adapter = None
    if args.lora_ckpt is not None:
        rank0_print(f"config[use_ema]: {getattr(config, 'use_ema', None)} -> False")
        rank0_print(f"config[adapter]: {getattr(config, 'adapter', None)}")
        config.use_ema = False
        if getattr(config, "adapter", None) is None: rank0_print("[Warning] config.adapter is None, but lora_ckpt is provided. This is unexpected for LoRA inference."); raise AssertionError("config.adapter must not be None when lora_ckpt is provided")
rank0_print(config)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True
    rank0_print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    rank0_print(f"Single GPU mode on device {device}")

rank0_print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)
latent_spatial_shape = list(getattr(config, "image_or_video_shape", []))[2:]
if not latent_spatial_shape:
    latent_spatial_shape = [48, 80, 44] if get_wan_version("inference.py") == "2.2" else [16, 60, 104]

# Load generator checkpoint
if config.generator_ckpt:
    rank0_print(f"config.generator_ckpt: {config.generator_ckpt}")
    state_dict = torch.load(config.generator_ckpt, map_location="cpu", mmap=True)
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if len(missing) > 0:
            rank0_print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
        if len(unexpected) > 0:
            rank0_print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    rank0_print(f"LoRA enabled with config: {config.adapter}")
    rank0_print("Applying LoRA to generator (inference)...")
    # After loading base weights, apply LoRA wrapper to the generator's transformer model
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights (if lora_ckpt is provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    _mds_state = None
    if lora_ckpt_path:
        rank0_print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu", mmap=True)
        # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        rank0_print("LoRA weights loaded for generator")
        if isinstance(lora_checkpoint, dict):
            _mds_state = lora_checkpoint.get("generator_mds", None)
    else:
        rank0_print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

    # ---- Learnable Memory Distance Scaling (MDS) ----
    if bool(getattr(config, "learnable_mds", False)):
        gen_model = pipeline.generator.model
        base_model = gen_model.get_base_model() if hasattr(gen_model, "get_base_model") else gen_model
        if _mds_state:
            base_model.load_learnable_mds_from_state(_mds_state)
        else:
            rank0_print("[MDS] WARNING: learnable_mds=true but no generator_mds found in the "
                        "lora checkpoint; MDS will be inactive (all lambda=1.0).")
            for _blk in base_model.blocks:
                _blk.self_attn.mds_rho = None
                _blk.self_attn.temporal_scale = 1.0

# Move pipeline to appropriate dtype and device
rank0_print(f"dtype {pipeline.generator.model.dtype} -> torch.bfloat16")
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


dataset = ImageDataset(
    csv_path=config.data_path,
    height=1280,
    width=704,
    image_col="path",
    prompt_col="caption",
    center_crop=True,
    random_flip=False,
)

rank0_print(f"Number of prompts: {len(dataset)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def build_output_path(output_folder, save_with_index, idx, prompt, seed_idx, pipeline, config):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"
    if save_with_index:
        return os.path.join(output_folder, f'{int(idx):04d}-{seed_idx}-{model_type}.mp4')
    else:
        return os.path.join(output_folder, f'{prompt[:100].replace("/", "_")}-{seed_idx}-{model_type}.mp4')

# ----------------------------- Inference loop -----------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples
    print(f"[RANK {rank}] prompts: ", prompts)

    initial_latent = None
    if config.i2v:
        frames = batch["frames"].to(device=device, dtype=torch.bfloat16)
        frames_vae_input = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            clean_latent = pipeline.vae.encode_to_latent(
                frames_vae_input).to(device=device, dtype=torch.bfloat16)
            initial_latent = clean_latent[:, 0:1, ]

    output_paths = [
        build_output_path(config.output_folder, config.save_with_index, idx, prompt, seed_idx, pipeline, config)
        for seed_idx in range(config.num_samples)
    ]
    if all(os.path.exists(output_path) for output_path in output_paths):
        print(f"[RANK {rank}] [Skip] idx={idx}: videos already exist. Pass!")
        if config.inference_iter != -1 and i >= config.inference_iter:
            break
        continue

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames - 1, *latent_spatial_shape], device=device, dtype=torch.bfloat16
    )

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
        initial_latent=initial_latent,
    )

    if isinstance(video, VideoPath):
        pipeline.vae.model.clear_cache()
        for seed_idx in range(config.num_samples):
            output_path = output_paths[seed_idx]
            shutil.move(video.path, output_path)
        if config.inference_iter != -1 and i >= config.inference_iter:
            break
        continue

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    for seed_idx in range(config.num_samples):
        output_path = output_paths[seed_idx]
        write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()

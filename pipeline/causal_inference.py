# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os
import time

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, VideoPath
from utils.wan_wrapper_22 import Wan22_DiffusionWrapper, Wan22_VAEWrapper
from utils.global_config import set_wan_version, get_wan_version, get_seq_frame_len

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist
from tqdm import tqdm
import wan.modules.causal_model_infinity_22 as _causal_model_22_mod

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        set_wan_version(args, str(self.__class__))
        if get_wan_version(str(self.__class__)) == "2.2":
            _DiffusionWrapper = Wan22_DiffusionWrapper
            _VAEWrapper = Wan22_VAEWrapper
        else:
            _DiffusionWrapper = WanDiffusionWrapper
            _VAEWrapper = WanVAEWrapper

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = _DiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = _VAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = get_seq_frame_len(return_seq=False, return_frame=True, caller=str(self.__class__))

        self.kv_cache1 = None
        self.crossattn_cache = None
        self._wan_version = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"CausalInferencePipeline KV inference with {self.num_frame_per_block} frames per block")
            print(f'CausalInferencePipeline independent_first_frame: {self.independent_first_frame}')
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        initial_latent = False,
        visualize_sink_attn: bool = False,
        visualize_cross_attn: bool = False,
        visualize_local_attn: bool = False,
        vis_output_dir: str = "vis_sink_attn",
        sink_frames_rgb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        _causal_model_22_mod.SINK_ATTN_VIS_ENABLED = visualize_sink_attn
        _causal_model_22_mod.SINK_ATTN_SCORE_BUFFER = []
        _causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID = 0
        _causal_model_22_mod.SINK_ATTN_CURRENT_STEP_ID = 0
        _causal_model_22_mod.SINK_ATTN_CURRENT_BLOCK_ID = 0
        _causal_model_22_mod.SINK_ATTN_IS_DENOISING = False
        _causal_model_22_mod.CROSS_ATTN_VIS_ENABLED = visualize_cross_attn
        _causal_model_22_mod.CROSS_ATTN_SCORE_BUFFER = []
        _causal_model_22_mod.CROSS_ATTN_CURRENT_BLOCK_ID = 0
        _causal_model_22_mod.LOCAL_ATTN_VIS_ENABLED = False
        _causal_model_22_mod.LOCAL_ATTN_SCORE_BUFFER = []
        _causal_model_22_mod.LOCAL_ATTN_CURRENT_BLOCK_ID = 0
        if visualize_sink_attn:
            os.makedirs(vis_output_dir, exist_ok=True)

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            target_device = noise.device
            gpu_memory_preservation = get_cuda_free_memory_gb(target_device) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=target_device,
                preserved_memory_gb=gpu_memory_preservation,
            )

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        if (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)
        self._set_all_modules_sink_size(1)
        target_sink_size = getattr(self.args.model_kwargs, "sink_size", 1)
        print(f"[inference] sink_size initialized to 1, will switch to {target_sink_size} after first chunk")
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block
        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        first_chunk_done = False
        chunk_loop_id = 0
        first_chunk_start = time.time()
        for current_num_frames in tqdm(all_num_frames, desc=f"rank {dist.get_rank() if dist.is_initialized() else 0}"):
            if profile:
                block_start.record()

            if not dist.is_initialized() or dist.get_rank() == 0:
                sink_sizes = [block.self_attn.sink_size for block in self.generator.model.blocks if hasattr(block.self_attn, "sink_size")]
                print(f"[chunk {chunk_loop_id:04d}] sink_sizes: {sink_sizes}")

            if visualize_local_attn:
                _causal_model_22_mod.LOCAL_ATTN_VIS_ENABLED = (chunk_loop_id % 5 == 0 or chunk_loop_id in [1, 2, 3, 4])

            _causal_model_22_mod.SINK_ATTN_VIS_ENABLED = visualize_sink_attn and (chunk_loop_id % 5 == 0 or chunk_loop_id in [1, 2, 3, 4])
            _causal_model_22_mod.CROSS_ATTN_VIS_ENABLED = visualize_cross_attn and (chunk_loop_id == 0)

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if visualize_sink_attn or visualize_cross_attn or visualize_local_attn:
                    _causal_model_22_mod.SINK_ATTN_CURRENT_STEP_ID = index
                    _causal_model_22_mod.SINK_ATTN_CURRENT_BLOCK_ID = 0
                    _causal_model_22_mod.CROSS_ATTN_CURRENT_BLOCK_ID = 0
                    _causal_model_22_mod.LOCAL_ATTN_CURRENT_BLOCK_ID = 0
                    _causal_model_22_mod.SINK_ATTN_IS_DENOISING = True

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
            # Step 2.2: record the model's output
            output[:, current_start_frame :current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            if visualize_sink_attn or visualize_cross_attn or visualize_local_attn:
                _causal_model_22_mod.SINK_ATTN_IS_DENOISING = False
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if visualize_sink_attn and sink_frames_rgb is not None and (chunk_loop_id % 5 == 0 or chunk_loop_id in [1, 2, 3, 4]):
                print(f"[vis] chunk={_causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID} buffer_len={len(_causal_model_22_mod.SINK_ATTN_SCORE_BUFFER)} sink_frames_rgb={sink_frames_rgb.shape}")
                if len(_causal_model_22_mod.SINK_ATTN_SCORE_BUFFER) > 0:
                    buffer_save_path = os.path.join(vis_output_dir, f"sink_attn_buffer_chunk{chunk_loop_id:02d}.pt")
                    torch.save(_causal_model_22_mod.SINK_ATTN_SCORE_BUFFER, buffer_save_path)
                    print(f"[vis] sink attn buffer saved to {buffer_save_path}")
                # self._save_sink_attn_heatmaps(
                #     vis_output_dir=vis_output_dir,
                #     chunk_id=_causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID,
                #     sink_frames_rgb=sink_frames_rgb,
                #     frame_h=1280, frame_w=704,
                #     token_h=40, token_w=22,
                # )
                _causal_model_22_mod.SINK_ATTN_SCORE_BUFFER = []

            if visualize_cross_attn and chunk_loop_id == 0:
                self._save_cross_attn_heatmaps(
                    vis_output_dir=vis_output_dir,
                    chunk_id=_causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID,
                    sink_frames_rgb=sink_frames_rgb,
                    frame_h=1280, frame_w=704,
                    token_h=40, token_w=22,
                )
                _causal_model_22_mod.CROSS_ATTN_SCORE_BUFFER = []

            if visualize_local_attn and (chunk_loop_id % 5 == 0 or chunk_loop_id in [1, 2, 3, 4]):
                if len(_causal_model_22_mod.LOCAL_ATTN_SCORE_BUFFER) > 0:
                    local_buffer_save_path = os.path.join(vis_output_dir, f"local_attn_buffer_chunk{chunk_loop_id:02d}.pt")
                    torch.save(_causal_model_22_mod.LOCAL_ATTN_SCORE_BUFFER, local_buffer_save_path)
                    print(f"[vis] local attn buffer saved to {local_buffer_save_path}")
                # if chunk_loop_id >= 4:
                #     self._save_local_attn_heatmaps(
                #         vis_output_dir=vis_output_dir,
                #         chunk_id=_causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID,
                #         sink_frames_rgb=sink_frames_rgb,
                #         frame_h=1280, frame_w=704,
                #         token_h=40, token_w=22,
                #     )
                _causal_model_22_mod.LOCAL_ATTN_SCORE_BUFFER = []

            if visualize_sink_attn or visualize_cross_attn or visualize_local_attn:
                _causal_model_22_mod.SINK_ATTN_CURRENT_CHUNK_ID += 1

            chunk_loop_id += 1

            if chunk_loop_id == 1:
                first_chunk_latency = time.time() - first_chunk_start
                print(f"[Latency] First chunk denoising time: {first_chunk_latency:.3f}s")

            if not first_chunk_done and target_sink_size > 1:
                whether_per_block = getattr(self.args, "whether_set_per_block_sink_size", False)
                if whether_per_block:
                    sink_size_per_block = list(getattr(self.args, "sink_size_per_block", [target_sink_size] * self.num_transformer_blocks))
                    self._set_per_block_sink_size(sink_size_per_block)
                else:
                    self._set_all_modules_sink_size(target_sink_size)

                whether_per_block_temporal_scale = getattr(self.args, "whether_set_per_block_temporal_scale", False)
                if whether_per_block_temporal_scale:
                    default_scale = getattr(self.args, "default_temporal_scale", 1.0)
                    temporal_scale_per_block = list(getattr(self.args, "temporal_scale_per_block", [default_scale] * self.num_transformer_blocks))
                    self._set_per_block_temporal_scale(temporal_scale_per_block)

                first_chunk_done = True
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[inference] sink_size switched after first chunk (per_block={whether_per_block})")
                    if whether_per_block_temporal_scale:
                        print(f"[inference] temporal_scale set per block")

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        # if getattr(self.args.model_kwargs, "use_infinite_attention", False):
        #     video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        # else:
        #     video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        if output.shape[1] > 241:
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False, chunk_size=240)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)

        if not isinstance(video, VideoPath):
            video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: use the configured global token length.
                kv_cache_size = get_seq_frame_len(return_seq=True, return_frame=False, caller=str(self.__class__))

        if self._get_wan_version() == "2.1":
            num_heads = 12
        elif self._get_wan_version() == "2.2":
            num_heads = 24

        for block_idx in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        if self._get_wan_version() == "2.1":
            num_heads = 12
        elif self._get_wan_version() == "2.2":
            num_heads = 24

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for 1.3B-480p, 18480 for 5B-720p).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = get_seq_frame_len(return_seq=True, return_frame=False, caller=str(self.__class__))
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass

    def _set_all_modules_sink_size(self, sink_size: int):
        if hasattr(self.generator.model, "sink_size"):
            self.generator.model.sink_size = sink_size
        for module in self.generator.model.modules():
            if hasattr(module, "sink_size"):
                module.sink_size = sink_size

    def _set_per_block_sink_size(self, sink_size_per_block: list):
        for block_idx, block in enumerate(self.generator.model.blocks):
            s = sink_size_per_block[block_idx] if block_idx < len(sink_size_per_block) else sink_size_per_block[-1]
            if hasattr(block.self_attn, "sink_size"):
                block.self_attn.sink_size = s
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[inference] per-block sink_size set: {sink_size_per_block}")

    def _set_per_block_temporal_scale(self, temporal_scale_per_block: list):
        for block_idx, block in enumerate(self.generator.model.blocks):
            s = temporal_scale_per_block[block_idx] if block_idx < len(temporal_scale_per_block) else temporal_scale_per_block[-1]
            if hasattr(block.self_attn, "temporal_scale"):
                block.self_attn.temporal_scale = float(s)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[inference] per-block temporal_scale set: {temporal_scale_per_block}")

    def _save_sink_attn_heatmaps(self, vis_output_dir, chunk_id, sink_frames_rgb,
                                  frame_h=1280, frame_w=704, token_h=80, token_w=44):
        import torch.nn.functional as F
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            print("[vis] matplotlib not available, skipping heatmap save")
            return

        buffer = _causal_model_22_mod.SINK_ATTN_SCORE_BUFFER
        if len(buffer) == 0:
            return

        for sink_idx in range(1):
            if sink_frames_rgb is not None:
                sink_rgb = sink_frames_rgb[0]
                if sink_rgb.shape[0] == 3:
                    sink_rgb = sink_rgb.permute(1, 2, 0)
                sink_rgb = sink_rgb.float().cpu()
                if sink_rgb.min() < 0.0:
                    sink_rgb = (sink_rgb + 1.0) / 2.0
                elif sink_rgb.max() > 1.0:
                    sink_rgb = sink_rgb / 255.0
            else:
                sink_rgb = None

            for entry in buffer:
                step_id = entry["denoise_step_id"]
                block_id = entry["block_id"]
                score = entry["score"][0].detach().cpu()

                num_query_tokens = score.shape[0]
                num_query_frames = num_query_tokens // (token_h * token_w)
                if num_query_frames == 0:
                    continue

                # score: (num_query_tokens, sink_total_tokens)
                # For each query token at spatial position (h,w), take the score against
                # the corresponding sink token at the same (h,w) position.
                # This removes RoPE cross-position bias introduced by averaging over all sink tokens.
                first_frame_sink_tokens = token_h * token_w
                score_query = score[:num_query_frames * token_h * token_w, :first_frame_sink_tokens]
                # score_query: (num_query_frames * token_h * token_w, token_h * token_w)
                # diagonal: score of each query token against its spatially corresponding sink token
                sink_pos = torch.arange(first_frame_sink_tokens)
                query_pos = sink_pos.unsqueeze(0).expand(num_query_frames, -1).reshape(-1)  # repeat for each query frame
                score_first_sink = score_query[torch.arange(num_query_frames * token_h * token_w), query_pos]  # (num_query_tokens,)
                score_spatial = score_first_sink.reshape(num_query_frames, token_h, token_w)
                score_avg = score_spatial.mean(dim=0)

                heatmap = score_avg.unsqueeze(0).unsqueeze(0).float()
                heatmap = F.interpolate(heatmap, size=(frame_h, frame_w), mode="bilinear", align_corners=False)
                heatmap = heatmap.squeeze().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

                fig, ax = plt.subplots(1, 1, figsize=(frame_w / 100, frame_h / 100), dpi=100)
                if sink_rgb is not None:
                    ax.imshow(sink_rgb.numpy())
                ax.imshow(heatmap, cmap="jet", alpha=0.5)
                ax.axis("off")

                rank_id = dist.get_rank() if dist.is_initialized() else 0
                save_path = os.path.join(
                    vis_output_dir,
                    f"rank{rank_id:02d}_chunk{chunk_id:04d}_step{step_id:02d}_block{block_id:02d}_sink{sink_idx:02d}.png"
                )
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

    def _save_cross_attn_heatmaps(self, vis_output_dir, chunk_id, sink_frames_rgb,
                                   frame_h=1280, frame_w=704, token_h=40, token_w=22):
        import torch.nn.functional as F
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[vis] matplotlib not available, skipping cross attn heatmap save")
            return

        buffer = _causal_model_22_mod.CROSS_ATTN_SCORE_BUFFER
        if len(buffer) == 0:
            return

        if sink_frames_rgb is not None:
            sink_rgb = sink_frames_rgb[0]
            if sink_rgb.shape[0] == 3:
                sink_rgb = sink_rgb.permute(1, 2, 0)
            sink_rgb = sink_rgb.float().cpu()
            if sink_rgb.min() < 0.0:
                sink_rgb = (sink_rgb + 1.0) / 2.0
            elif sink_rgb.max() > 1.0:
                sink_rgb = sink_rgb / 255.0
        else:
            sink_rgb = None

        rank_id = dist.get_rank() if dist.is_initialized() else 0

        for entry in buffer:
            step_id = entry["denoise_step_id"]
            block_id = entry["block_id"]
            score = entry["score"][0].detach().cpu()  # (num_query_tokens, text_len)

            num_query_tokens = score.shape[0]
            num_query_frames = num_query_tokens // (token_h * token_w)
            if num_query_frames == 0:
                continue

            score_spatial = score[:num_query_frames * token_h * token_w]  # (num_query_tokens, text_len)

            # max pooling over text tokens: each video token takes its most attended text token
            score_max, _ = score_spatial.max(dim=-1)  # (num_query_tokens,)
            score_map = score_max.reshape(num_query_frames, token_h, token_w).mean(dim=0)  # (token_h, token_w)

            heatmap = score_map.unsqueeze(0).unsqueeze(0).float()
            heatmap = F.interpolate(heatmap, size=(frame_h, frame_w), mode="bilinear", align_corners=False)
            heatmap = heatmap.squeeze().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            fig, ax = plt.subplots(1, 1, figsize=(frame_w / 100, frame_h / 100), dpi=100)
            if sink_rgb is not None:
                ax.imshow(sink_rgb.numpy())
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            ax.axis("off")

            save_path = os.path.join(
                vis_output_dir,
                f"rank{rank_id:02d}_chunk{chunk_id:04d}_step{step_id:02d}_block{block_id:02d}_cross.png"
            )
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def _save_local_attn_heatmaps(self, vis_output_dir, chunk_id, sink_frames_rgb,
                                   frame_h=1280, frame_w=704, token_h=40, token_w=22):
        import torch.nn.functional as F
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[vis] matplotlib not available, skipping local attn heatmap save")
            return

        buffer = _causal_model_22_mod.LOCAL_ATTN_SCORE_BUFFER
        if len(buffer) == 0:
            return

        if sink_frames_rgb is not None:
            sink_rgb = sink_frames_rgb[0]
            if sink_rgb.shape[0] == 3:
                sink_rgb = sink_rgb.permute(1, 2, 0)
            sink_rgb = sink_rgb.float().cpu()
            if sink_rgb.min() < 0.0:
                sink_rgb = (sink_rgb + 1.0) / 2.0
            elif sink_rgb.max() > 1.0:
                sink_rgb = sink_rgb / 255.0
        else:
            sink_rgb = None

        rank_id = dist.get_rank() if dist.is_initialized() else 0

        for entry in buffer:
            step_id = entry["denoise_step_id"]
            block_id = entry["block_id"]
            score = entry["score"][0].detach().cpu()  # (num_query_tokens, last4_tokens)

            num_query_tokens = score.shape[0]
            num_query_frames = num_query_tokens // (token_h * token_w)
            if num_query_frames == 0:
                continue

            # last 1 frame has token_h*token_w tokens, use diagonal (same spatial position) to remove RoPE bias
            spatial_tokens = token_h * token_w
            score_q = score[:num_query_frames * spatial_tokens, :spatial_tokens]
            diag_pos = torch.arange(spatial_tokens).unsqueeze(0).expand(num_query_frames, -1).reshape(-1)
            score_diag = score_q[torch.arange(num_query_frames * spatial_tokens), diag_pos]
            score_map = score_diag.reshape(num_query_frames, token_h, token_w).mean(dim=0)

            heatmap = score_map.unsqueeze(0).unsqueeze(0).float()
            heatmap = F.interpolate(heatmap, size=(frame_h, frame_w), mode="bilinear", align_corners=False)
            heatmap = heatmap.squeeze().numpy()
            import numpy as np
            heatmap = np.log1p(heatmap - heatmap.min())
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            fig, ax = plt.subplots(1, 1, figsize=(frame_w / 100, frame_h / 100), dpi=100)
            if sink_rgb is not None:
                ax.imshow(sink_rgb.numpy())
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            ax.axis("off")

            save_path = os.path.join(
                vis_output_dir,
                f"rank{rank_id:02d}_chunk{chunk_id:04d}_step{step_id:02d}_block{block_id:02d}_local.png"
            )
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def clear_kv_cache(self):
        """
        Fully release KV cache and cross-attention cache tensors so the old cache can be reclaimed.
        """
        for attr_name in ("kv_cache1", "crossattn_cache"):
            cache = getattr(self, attr_name, None)
            if cache is not None:
                del cache
            setattr(self, attr_name, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_wan_version(self):
        if self._wan_version is None:
            self._wan_version = get_wan_version(str(self.__class__))
        return self._wan_version
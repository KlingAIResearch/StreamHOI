# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG
from tqdm import tqdm


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[Recache] num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        self.generator.model.block_mask = block_mask
        
        context_timestep = torch.ones([batch_size, num_recache_frames], device=device, dtype=torch.int64) * self.args.context_noise
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        to_cpu: bool = False,
        initial_latent=None,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
            initial_latent: Optional initial latent for I2V conditioning.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )

        conditional_dict_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            target_device = noise.device
            gpu_memory_preservation = get_cuda_free_memory_gb(target_device) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=target_device,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        if (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
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
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)
        self._set_all_modules_sink_size(1)
        target_sink_size = getattr(self.args.model_kwargs, "sink_size", 1)
        first_chunk_done = False

        # initial_latent prefill (same as CausalInferencePipeline)
        conditional_dict_first = conditional_dict_list[0]
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict_first,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict_first,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        segment_idx = 0
        next_switch_frame_idx = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in tqdm(all_num_frames, desc=f"rank {dist.get_rank() if dist.is_initialized() else 0}"):
            noisy_input = noise[
                :, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames
            ]

            if next_switch_frame_idx is not None and current_start_frame >= next_switch_frame_idx:
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, conditional_dict_list[segment_idx])
                next_switch_frame_idx = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}")
                    print(f"text_prompts_list[segment_idx({segment_idx})]: {text_prompts_list[segment_idx]}")

            conditional_dict = conditional_dict_list[segment_idx]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

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

            current_start_frame += current_num_frames

        if to_cpu:
            if hasattr(self, 'kv_cache1'):
                del self.kv_cache1
            if hasattr(self, 'crossattn_cache'):
                del self.crossattn_cache
            torch.cuda.empty_cache()

        # if getattr(self.args.model_kwargs, "MDS_with_relativeRope", False):
        #     video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        # else:
        #     video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        
        if to_cpu:
            self.vae.model.clear_cache()
            video = video.cpu()

        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video

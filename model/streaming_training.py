# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any, Optional
from einops import rearrange

from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory
from utils.global_config import get_seq_frame_len
from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline


class StreamingTrainingModel:
    """
    A model wrapper specifically for streaming / serialized training.

    Current design:
    1. Reuse KV cache and cross-attention cache across chunks
    2. Support DMDSwitch mid-video prompt switching
    3. For I2V:
       - first training chunk is [initial_latent] + [20 generated frames] => 21 frames
       - later chunks are compressed into [bridge_frame] + [20 generated frames] => 21 frames
    4. Only newly generated frames are supervised
    """

    def __init__(self, base_model, config):
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, "image_or_video_shape", None)

        # Streaming configuration
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.max_length = getattr(config, "streaming_max_length", 57)
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 20)

        # Components from base model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func

        # Model configuration
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(
            base_model.inference_pipeline,
            "frame_seq_length",
            get_seq_frame_len(return_seq=False, return_frame=True, caller=str(self.__class__)),
        )

        # Inference pipeline
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline

        self.reset_state()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] StreamingTrainingModel initialized:")
            print(f"[StreamingTrain-Model] chunk_size={self.chunk_size}, max_length={self.max_length}")
            print(f"[StreamingTrain-Model] min_new_frame={self.min_new_frame}")
            print(f"[StreamingTrain-Model] base_model type: {type(self.base_model).__name__}")

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compress a long chunk into a fixed training chunk:
        use the last frame of the prefix as image-latent bridge frame,
        then keep the last (chunk_size - 1) frames.

        Example in current setup:
        input  : [21 previous frames] + [20 new frames] => 41 frames
        output : [bridge frame] + [20 new frames]       => 21 frames
        """
        total_frames = frames.shape[1]

        if total_frames <= 1:
            return frames

        process_frames = min(self.chunk_size, total_frames)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Processing first frame encoding for loss: "
                f"total_frames={total_frames}, processing last {process_frames} frames"
            )

        with torch.no_grad():
            # Prefix part whose last frame will become the bridge frame
            frames_to_decode = frames[:, :-(process_frames - 1), ...]
            pixels = self.base_model.vae.decode_to_pixel(frames_to_decode)
        
            last_frame_pixel = pixels[:, -1:, ...].to(self.dtype)
            last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")
        
            image_latent = self.base_model.vae.encode_to_latent(last_frame_pixel).to(self.dtype)
        
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print(f"[DEBUG-bridge] image_latent(bridge): mean={image_latent.mean().item():.4f}, std={image_latent.std().item():.4f}, min={image_latent.min().item():.4f}, max={image_latent.max().item():.4f}")
        #     print(f"[DEBUG-bridge] remaining_frames:      mean={frames[:, -(process_frames-1):].mean().item():.4f}, std={frames[:, -(process_frames-1):].std().item():.4f}, min={frames[:, -(process_frames-1):].min().item():.4f}, max={frames[:, -(process_frames-1):].max().item():.4f}")

        # # Directly use the last frame of the prefix latent as bridge frame.
        # # Round-trip (latent->pixel->latent) compresses std by ~2x (0.58 vs 1.08),
        # # causing VAE CausalConv3d to produce color saturation in the first 4 frames.
        # image_latent = frames[:, -(process_frames):-(process_frames - 1), ...].detach()

        remaining_frames = frames[:, -(process_frames - 1):, ...]
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Processed first frame encoding: "
                f"{frames.shape} -> {processed_frames.shape}"
            )

        return processed_frames

    def reset_state(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] Resetting streaming training state")

        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,
            "previous_frames": None,
            "temp_max_length": None,
            "initial_latent_for_loss": None,
        }

        self.inference_pipeline.clear_kv_cache()

    def _should_switch_prompt(self, chunk_start_frame: int, chunk_size: int) -> bool:
        if not isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] Not a switch pipeline, no switching")
            return False

        if self.state.get("has_switched", False):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] Already switched, not switching again")
            return False

        switch_info = self.state["conditional_info"].get("switch_info", {})
        switch_frame_index = switch_info.get("switch_frame_index")

        if switch_frame_index is None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] No switch_frame_index, not switching")
            return False

        chunk_end_frame = chunk_start_frame + chunk_size
        should_switch = chunk_start_frame <= switch_frame_index < chunk_end_frame

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Switch check: "
                f"switch_frame={switch_frame_index}, "
                f"chunk=[{chunk_start_frame}, {chunk_end_frame}), "
                f"should_switch={should_switch}"
            )

        return should_switch

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        cond_info = self.state["conditional_info"]

        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(
                            f"[StreamingTrain-Model] Using switch conditional_dict "
                            f"for chunk starting at frame {chunk_start_frame}"
                        )
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Using original conditional_dict "
                f"for chunk starting at frame {chunk_start_frame}"
            )
        return cond_info["conditional_dict"]

    def _generate_chunk(
        self,
        noise_chunk: torch.Tensor,
        chunk_start_frame: int,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                f"StreamingTrain-Model: Before generate chunk {chunk_start_frame}",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] _generate_chunk: "
                f"chunk_start_frame={chunk_start_frame}, "
                f"chunk_size={noise_chunk.shape[1]}"
            )
            print(f"[StreamingTrain-Model] requires_grad={requires_grad}")

        current_conditional_dict = self._get_current_conditional_dict(chunk_start_frame)

        kwargs = {
            "noise": noise_chunk,
            "conditional_dict": current_conditional_dict,
            "current_start_frame": chunk_start_frame,
            "requires_grad": requires_grad,
            "return_sim_step": False,
        }

        if isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            switch_info = self.state["conditional_info"].get("switch_info", {})
            if switch_info and self._should_switch_prompt(chunk_start_frame, noise_chunk.shape[1]):
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print(
                        f"[StreamingTrain-Model] Switching prompt at frame "
                        f"{switch_info['switch_frame_index']}"
                    )

                relative_switch_index = max(0, switch_info["switch_frame_index"] - chunk_start_frame)
                kwargs["switch_frame_index"] = relative_switch_index
                kwargs["switch_conditional_dict"] = switch_info["switch_conditional_dict"]

                if self.state["previous_frames"] is not None:
                    kwargs["switch_recache_frames"] = self.state["previous_frames"]
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(
                            f"[StreamingTrain-Model] Passed previous_frames for switch recache: "
                            f"{self.state['previous_frames'].shape}"
                        )

                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(
                        f"[StreamingTrain-Model] Adding switch parameters: "
                        f"relative_switch_index={relative_switch_index}"
                    )

                self.state["has_switched"] = True

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] Calling pipeline.generate_chunk_with_cache")

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: Before pipeline.generate_chunk_with_cache",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        output, denoised_timestep_from, denoised_timestep_to = \
            self.inference_pipeline.generate_chunk_with_cache(**kwargs)

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: After pipeline.generate_chunk_with_cache",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        return output, denoised_timestep_from, denoised_timestep_to

    def setup_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        switch_conditional_dict: Optional[Dict] = None,
        switch_frame_index: Optional[int] = None,
        temp_max_length: Optional[int] = None,
    ):
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: Before setup_sequence",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] Setting up new sequence:")
            print(f"[StreamingTrain-Model] image_or_video_shape={self.image_or_video_shape}")
            print(
                f"[StreamingTrain-Model] initial_latent shape: "
                f"{initial_latent.shape if initial_latent is not None else None}"
            )
            print(f"[StreamingTrain-Model] switch_frame_index={switch_frame_index}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] init kv_cache1: "
                    f"{self.inference_pipeline.kv_cache1[0]['k'].shape}"
                )

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] init crossattn_cache: "
                    f"{self.inference_pipeline.crossattn_cache[0]['k'].shape}"
                )

        self.reset_state()
        self.state["temp_max_length"] = temp_max_length
        self.state["initial_latent_for_loss"] = (
            initial_latent.detach().clone() if initial_latent is not None else None
        )

        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] Starting with initial_latent, "
                    f"length={self.state['current_length']}"
                )
        else:
            self.state["current_length"] = 0
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] Starting with empty sequence")

        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }

        if switch_conditional_dict is not None and switch_frame_index is not None:
            self.state["conditional_info"]["switch_info"] = {
                "switch_conditional_dict": switch_conditional_dict,
                "switch_frame_index": switch_frame_index,
            }
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] DMDSwitch info saved: "
                    f"switch_frame_index={switch_frame_index}"
                )

        if initial_latent is not None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] Initializing cache with initial_latent")

            timestep = torch.zeros(
                [batch_size, initial_latent.shape[1]],
                device=self.device,
                dtype=torch.int64,
            )
            with torch.no_grad():
                self.inference_pipeline.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0,
                )

            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(
                    "StreamingTrain-Model: After initial latent processing",
                    device=self.device,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                )
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] No initial latent")

    def can_generate_more(self) -> bool:
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = (
            current_length < temp_max_length
            and (current_length + self.min_new_frame) <= temp_max_length
        )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] can_generate_more: "
                f"current_length={current_length}, "
                f"temp_max_length={temp_max_length}, "
                f"global_max_length={self.max_length}, "
                f"can_generate={can_generate}"
            )

        return can_generate

    def generate_next_chunk(self, requires_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Current behavior:
        - first chunk:
            [initial_latent] + [20 generated frames] => 21 frames
        - later chunks:
            [21 previous frames] + [20 generated frames] => 41 frames
            then _process_first_frame_encoding() compresses it into:
            [bridge frame] + [20 generated frames] => 21 frames
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] generate_next_chunk called: requires_grad={requires_grad}")

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            gen_training_mode = self.generator.training
            gen_params_requiring_grad = sum(1 for p in self.generator.parameters() if p.requires_grad)
            gen_params_total = sum(1 for p in self.generator.parameters())
            print(f"[DEBUG-SeqModel] Generator training mode: {gen_training_mode}")
            print(f"[DEBUG-SeqModel] Generator params requiring grad: {gen_params_requiring_grad}/{gen_params_total}")

        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")

        current_length = self.state["current_length"]
        batch_size = self.image_or_video_shape[0]

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Generating chunk: current_length={current_length}")

        previous_frames = self.state.get("previous_frames")

        if previous_frames is not None:
            max_new_frames = min(
                self.state["temp_max_length"] - current_length + 1,
                self.chunk_size,
            )
            possible_new_frames = list(range(self.min_new_frame, max_new_frames, 5))

            if len(possible_new_frames) == 0:
                raise ValueError(
                    f"No valid possible_new_frames. "
                    f"min_new_frame={self.min_new_frame}, "
                    f"max_new_frames={max_new_frames}, "
                    f"chunk_size={self.chunk_size}, "
                    f"current_length={current_length}, "
                    f"temp_max_length={self.state['temp_max_length']}"
                )

            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(
                    selected_idx,
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(possible_new_frames) - 1)

            new_frames_to_generate = possible_new_frames[selected_idx]

            overlap_frames = self.chunk_size - new_frames_to_generate
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                overlap_frames_to_use = 0
                new_frames_to_generate = self.chunk_size

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] With auto overlap: "
                    f"generating {new_frames_to_generate} new frames, "
                    f"reusing {overlap_frames_to_use} overlap frames"
                )
        else:
            # First chunk: reserve one slot for initial_latent_for_loss
            overlap_frames_to_use = 1
            new_frames_to_generate = self.chunk_size - 1

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] First chunk: generating "
                    f"{new_frames_to_generate} frames (prepended with initial latent)"
                )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Random frame selection: selected={new_frames_to_generate}")
            print(f"[StreamingTrain-Model] Auto overlap calculation: overlap_frames={overlap_frames_to_use}")

        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *self.image_or_video_shape[2:]],
            device=self.device,
            dtype=self.dtype,
        )

        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self._generate_chunk(
            noise_chunk=noise_chunk,
            chunk_start_frame=current_length,
            requires_grad=requires_grad,
        )

        if previous_frames is not None:
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
        elif self.state.get("initial_latent_for_loss") is not None:
            first_frame = self.state["initial_latent_for_loss"]
            full_chunk = torch.cat([first_frame, generated_new_frames], dim=1)
        else:
            full_chunk = generated_new_frames

        frames_to_save = full_chunk.detach().clone()[:, -self.chunk_size:, ...]
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Saved last {frames_to_save.shape[1]} frames as previous_frames")

        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)

        if previous_frames is not None:
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            # gradient_mask[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate, ...] = True
            grad_start = overlap_frames_to_use + self.num_frame_per_block
            gradient_mask[:, grad_start:overlap_frames_to_use + new_frames_to_generate, ...] = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Built chunk with overlap: shape={full_chunk.shape}")
                print(
                    f"[StreamingTrain-Model] Gradient mask: "
                    f"{new_frames_to_generate} frames will have gradients "
                    f"out of {full_chunk.shape[1]}"
                )
        else:
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            if full_chunk.shape[1] > 1:
                gradient_mask[:, 1:, ...] = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] First chunk built: shape={full_chunk.shape}")
                print(
                    f"[StreamingTrain-Model] Gradient mask: "
                    f"{full_chunk.shape[1] - 1} generated frames supervised, first frame excluded"
                )

        self.state["current_length"] += new_frames_to_generate
        self.state["previous_frames"] = frames_to_save

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Updated state: current_length={self.state['current_length']}")
            if self.state["previous_frames"] is not None:
                print(
                    f"[StreamingTrain-Model] Saved "
                    f"{self.state['previous_frames'].shape[1]} frames as previous_frames for next chunk"
                )

        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length,
            "chunk_frames": full_chunk.shape[1],
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,
            "overlap_frames_used": overlap_frames_to_use,
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[StreamingTrain-Model] current_training_chunk: "
                f"({self.state['current_length'] - new_frames_to_generate} "
                f"-> {self.state['current_length']})/"
                f"{self.state['temp_max_length']}"
            )

        return full_chunk, info

    def compute_generator_loss(
        self,
        chunk: torch.Tensor,
        chunk_info: Dict[str, Any],
        debug_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: Before compute generator loss",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        unconditional_dict = self.state["conditional_info"]["unconditional_dict"]
        gradient_mask = chunk_info.get("gradient_mask", None)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Using conditional_dict and unconditional_dict "
                f"for loss calculation at frame {chunk_start_frame}"
            )

        dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss(
            image_or_video=chunk,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=chunk_info["denoised_timestep_from"],
            denoised_timestep_to=chunk_info["denoised_timestep_to"],
            debug_step=debug_step,
            debug_current_length=chunk_info.get("current_length"),
        )

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: After DMD loss computation",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        dmd_log_dict.update({
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", chunk.shape[1]),
        })

        return dmd_loss, dmd_log_dict

    def _clear_cache_gradients(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] Clearing cache gradients")

        if hasattr(self.inference_pipeline, "kv_cache1") and self.inference_pipeline.kv_cache1 is not None:
            for cache_block in self.inference_pipeline.kv_cache1:
                if "k" in cache_block and cache_block["k"].requires_grad:
                    cache_block["k"] = cache_block["k"].detach()
                if "v" in cache_block and cache_block["v"].requires_grad:
                    cache_block["v"] = cache_block["v"].detach()

        if hasattr(self.inference_pipeline, "crossattn_cache") and self.inference_pipeline.crossattn_cache is not None:
            for cache_block in self.inference_pipeline.crossattn_cache:
                if "k" in cache_block and cache_block["k"].requires_grad:
                    cache_block["k"] = cache_block["k"].detach()
                if "v" in cache_block and cache_block["v"].requires_grad:
                    cache_block["v"] = cache_block["v"].detach()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[StreamingTrain-Model] Cache gradients cleared")

    def compute_critic_loss(
        self,
        chunk: torch.Tensor,
        chunk_info: Dict[str, Any],
        debug_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: Before compute critic loss",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] compute_critic_loss: chunk_shape={chunk.shape}")
            for k, v in chunk_info.items():
                if k == "gradient_mask":
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v[0, :, 0, 0, 0]}")
                else:
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v}")
            print(f"[StreamingTrain-Model] chunk requires_grad: {chunk.requires_grad}")

        if chunk.requires_grad:
            chunk = chunk.detach()

        self._clear_cache_gradients()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: After chunk detachment and cache cleanup",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        gradient_mask = chunk_info.get("gradient_mask", None)

        batch_size, num_frame = chunk.shape[:2]

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Preparing critic loss: "
                f"batch_size={batch_size}, num_frame={num_frame}"
            )

        denoised_timestep_from = chunk_info.get("denoised_timestep_from", None)
        denoised_timestep_to = chunk_info.get("denoised_timestep_to", None)

        min_timestep = (
            denoised_timestep_to
            if (getattr(self.base_model, "ts_schedule", False) and denoised_timestep_to is not None)
            else getattr(self.base_model, "min_score_timestep")
        )
        max_timestep = (
            denoised_timestep_from
            if (getattr(self.base_model, "ts_schedule_max", False) and denoised_timestep_from is not None)
            else getattr(self.base_model, "num_train_timestep")
        )

        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=getattr(self.base_model, "num_frame_per_block", 4),
            uniform_timestep=True,
        ).to(self.device)

        if getattr(self.base_model, "timestep_shift") > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = (
                timestep_shift * (critic_timestep / 1000)
                / (1 + (timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )

        critic_timestep = critic_timestep.clamp(self.base_model.min_step, self.base_model.max_step)

        if self.config.i2v:
            critic_timestep[:, 0] = 0

        critic_noise = torch.randn_like(chunk)

        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        if self.config.i2v:
            noisy_chunk[:, 0] = chunk[:, 0]

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Added noise, timestep range: "
                f"[{critic_timestep.min().item()}, {critic_timestep.max().item()}]"
            )

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: Before fake score computation",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: After fake score computation",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        denoising_loss_type = getattr(self.base_model.args, "denoising_loss_type", "mse")
        if denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))

        gradient_mask_flat = gradient_mask.flatten(0, 1) if gradient_mask is not None else None
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat,
        )

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(
                "StreamingTrain-Model: After denoising loss computation",
                device=self.device,
                rank=dist.get_rank() if dist.is_initialized() else 0,
            )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Critic loss computed: {denoising_loss.item()}")

        del conditional_dict, critic_noise, noisy_chunk
        if "flow_pred" in locals():
            del flow_pred
        if "pred_fake_noise" in locals():
            del pred_fake_noise

        if debug_step is not None:
            self.base_model._debug_save_latents(
                original_latent=chunk,
                pred_fake=pred_fake_image,
                pred_real=None,
                step=debug_step,
                current_length=chunk_info.get("current_length", 0),
            )

        del pred_fake_image

        critic_log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", num_frame),
        }

        return denoising_loss, critic_log_dict

    def get_sequence_length(self) -> int:
        return self.state.get("current_length", 0)
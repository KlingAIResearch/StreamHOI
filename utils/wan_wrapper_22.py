# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from utils.global_config import get_seq_frame_len
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl

# from wan.modules.model import WanModel
# from wan.modules.vae import _video_vae
# from wan.modules.causal_model import CausalWanModel

from wan.modules.model_22 import WanModel22
from wan.modules.vae_21 import _video_vae as _video_vae21
from wan.modules.vae_22 import _video_vae as _video_vae22
from wan.modules.causal_model_22 import CausalWanModel22
from wan.modules.causal_model_infinity_22 import CausalWanModel22 as CausalWanModelInfinity22

from .wan_wrapper import WanVAEWrapper, WanDiffusionWrapper


class Wan22_VAEWrapper(WanVAEWrapper):
    def __init__(
            self,
            z_dim=48,
            c_dim=160,
            vae_pth="wan_models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
        ):
        super().__init__()
        self.mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=torch.float32,
        )
        self.std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=torch.float32,
        )
        self.dtype = torch.bfloat16
        # init model
        self.model = (
            _video_vae22(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            )
            .eval()
            .requires_grad_(False)
        )

    def encode(self, pixel):
        device, dtype = pixel[0].device, self.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.model.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        return output


class Wan22_DiffusionWrapper(WanDiffusionWrapper):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            MDS_with_relativeRope=False
    ):
        super().__init__()

        if is_causal:
            if MDS_with_relativeRope:
                self.model = CausalWanModelInfinity22.from_pretrained(
                    f"wan_models/Wan-AI/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                self.model = CausalWanModel22.from_pretrained(
                    f"wan_models/Wan-AI/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel22.from_pretrained(f"wan_models/Wan-AI/{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = get_seq_frame_len(return_seq=False, return_frame=True, caller=str(self.__class__)) * local_attn_size if local_attn_size > 21 else get_seq_frame_len(return_seq=True, return_frame=False, caller=str(self.__class__))  # [1, 21, 48, 30, 52]  # To 2.2
        self.post_init()

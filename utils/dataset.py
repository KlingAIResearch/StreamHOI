# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import datasets

import pandas as pd
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from torchvision import transforms


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }

class I2VDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        num_frames=None,                 # 返回的视频帧数
        sample_mode="all",         # "uniform" or "head" or "all"
        return_video=True,
        return_image=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()
        
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.return_video = return_video
        self.return_image = return_image

        self.center_crop = center_crop
        self.random_flip = random_flip

        # 仅做 “crop + flip + toTensor + normalize”
        # resize 由我们自己做（为了完全复用你现有的 cover-resize 逻辑）
        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # 用 randint + index 做一个“固定 seed 下可复现”的遍历扰动
        #data_id = torch.randint(0, len(self.video_paths), (1,))[0]
        data_id = (index) % len(self.video_paths)

        video_path = self.video_paths[int(data_id)]
        # if video_path.startswith("/m2v_intern/xuziyi06/"):
        #     video_path = video_path.replace(
        #         "/m2v_intern/xuziyi06/",
        #         "/m2v_intern/raozejing/xuziyi06/"
        #     )
        if video_path.startswith("/share/xuziyi/"):
            video_path = video_path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )

        text = self.texts[int(data_id)]
        

        out = {"prompts": text, "idx": index, "data_id": int(data_id), "video_path": video_path}

        # 先读视频帧（因为 image 可以直接取首帧，避免重复解码）
        frames_pil = None
        if self.return_video:
            frames_pil = self._read_video_frames(video_path, num_frames=self.num_frames, mode=self.sample_mode)
            # frames_pil: List[PIL.Image]，每帧 RGB

        # image：如果需要就取首帧；如果已经读了 video，直接用 video 的第 0 帧
        if self.return_image:
            if frames_pil is not None and len(frames_pil) > 0:
                image = frames_pil[0]
            else:
                image = self._read_first_frame(video_path)
            out["images"] = self._process_single_image(image)

        # video：对每一帧做相同 resize + 同一组 crop/flip 逻辑（下面用“参数锁定”保证一致）
        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(frames_pil)

        return out

    # ---------- processing helpers ----------

    def _cover_resize_pil(self, image: Image.Image) -> Image.Image:
        """完全复用你现在的 cover-resize 逻辑：先 resize 到能覆盖目标 H/W，再 crop。"""
        target_height, target_width = self.height, self.width
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        shape = [round(h * scale), round(w * scale)]  # [H, W]
        image = torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )
        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        image = self._cover_resize_pil(image)
        image = self.post_processor(image)  # [3, H, W], float, [-1,1]
        return image

    def _process_video_frames_consistent(self, frames_pil):
        """
        关键点：保证 video 内所有帧用同一组 crop 参数、同一次 flip。
        - resize：每帧独立做 cover-resize（因为原始帧尺寸相同，效果也一致）
        - crop：如果是 RandomCrop，需要固定同一个 crop 参数
        - flip：如果是 RandomHorizontalFlip，需要固定同一次 flip 决策
        """
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty video frames")

        # 1) cover-resize 每一帧
        frames_resized = [self._cover_resize_pil(im) for im in frames_pil]

        # 2) 决定 crop 参数（CenterCrop 不需要参数）
        if self.center_crop:
            crop_params = None
        else:
            # RandomCrop.get_params 返回 (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(
                frames_resized[0], output_size=(self.height, self.width)
            )
            crop_params = (i, j, h, w)

        # 3) 决定 flip（random_flip=True 才可能 flip）
        do_flip = False
        if self.random_flip:
            do_flip = bool(torch.rand(1).item() < 0.5)

        # 4) 对每帧应用同样的 crop/flip + ToTensor + Normalize
        video_t = []
        for im in frames_resized:
            if crop_params is None:
                im2 = torchvision.transforms.functional.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im2 = torchvision.transforms.functional.crop(im, i, j, h, w)

            if do_flip:
                im2 = torchvision.transforms.functional.hflip(im2)

            t = torchvision.transforms.functional.to_tensor(im2)
            t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
            video_t.append(t)

        video = torch.stack(video_t, dim=0)  # [T, 3, H, W]
        return video

    # ---------- IO helpers ----------

    @staticmethod
    def _read_first_frame(video_path: str) -> Image.Image:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()  # RGB (H, W, 3)
            return Image.fromarray(frame0).convert("RGB")
        except Exception:
            pass

        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _read_video_frames(video_path: str, num_frames: int = None, mode: str = "uniform"):
        """
        返回 List[PIL.Image] (RGB)。优先 decord，失败用 opencv。
        mode:
        - "all": 读取视频全部帧，不截断，不补齐
        - "uniform": 在全视频长度上均匀采样 num_frames
        - "head": 取前 num_frames
        """
        # decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            if vlen <= 0:
                raise RuntimeError("empty video")

            if mode == "all":
                idx = list(range(vlen))
            elif mode == "head":
                idx = list(range(min(num_frames, vlen)))
            else:
                # uniform
                if vlen >= num_frames:
                    idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
                else:
                    idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

            frames = vr.get_batch(idx).asnumpy()
            return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]

        except Exception:
            pass

        # opencv fallback
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen <= 0:
            cap.release()
            raise RuntimeError(f"empty video: {video_path}")

        if mode == "all":
            idx = list(range(vlen))
        elif mode == "head":
            idx = list(range(min(num_frames, vlen)))
        else:
            if vlen >= num_frames:
                idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
            else:
                idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames
    
class I2V_TwoText_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        switch_prompt_col="caption_extend",
        num_frames=None,                 # 返回的视频帧数
        sample_mode="all",         # "uniform" or "head" or "all"
        return_video=True,
        return_image=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()
        if switch_prompt_col in self.df.columns:
            self.switch_texts = self.df[switch_prompt_col].fillna("").tolist()
        else:
            raise ValueError(f"Column '{switch_prompt_col}' not found in csv: {csv_path}")
        
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.return_video = return_video
        self.return_image = return_image

        self.center_crop = center_crop
        self.random_flip = random_flip

        # 仅做 “crop + flip + toTensor + normalize”
        # resize 由我们自己做（为了完全复用你现有的 cover-resize 逻辑）
        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # 用 randint + index 做一个“固定 seed 下可复现”的遍历扰动
        #data_id = torch.randint(0, len(self.video_paths), (1,))[0]
        data_id = (index) % len(self.video_paths)

        video_path = self.video_paths[int(data_id)]
        # if video_path.startswith("/m2v_intern/xuziyi06/"):
        #     video_path = video_path.replace(
        #         "/m2v_intern/xuziyi06/",
        #         "/m2v_intern/raozejing/xuziyi06/"
        #     )
        if video_path.startswith("/share/xuziyi/"):
            video_path = video_path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )
        text = self.texts[int(data_id)]
        switch_text = self.switch_texts[int(data_id)]

        out = {"prompts": text, "switch_prompts": switch_text, "idx": index, "data_id": int(data_id), "video_path": video_path}

        # 先读视频帧（因为 image 可以直接取首帧，避免重复解码）
        frames_pil = None
        if self.return_video:
            frames_pil = self._read_video_frames(video_path, num_frames=self.num_frames, mode=self.sample_mode)
            # frames_pil: List[PIL.Image]，每帧 RGB

        # image：如果需要就取首帧；如果已经读了 video，直接用 video 的第 0 帧
        if self.return_image:
            if frames_pil is not None and len(frames_pil) > 0:
                image = frames_pil[0]
            else:
                image = self._read_first_frame(video_path)
            out["images"] = self._process_single_image(image)

        # video：对每一帧做相同 resize + 同一组 crop/flip 逻辑（下面用“参数锁定”保证一致）
        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(frames_pil)

        return out

    # ---------- processing helpers ----------

    def _cover_resize_pil(self, image: Image.Image) -> Image.Image:
        """完全复用你现在的 cover-resize 逻辑：先 resize 到能覆盖目标 H/W，再 crop。"""
        target_height, target_width = self.height, self.width
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        shape = [round(h * scale), round(w * scale)]  # [H, W]
        image = torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )
        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        image = self._cover_resize_pil(image)
        image = self.post_processor(image)  # [3, H, W], float, [-1,1]
        return image

    def _process_video_frames_consistent(self, frames_pil):
        """
        关键点：保证 video 内所有帧用同一组 crop 参数、同一次 flip。
        - resize：每帧独立做 cover-resize（因为原始帧尺寸相同，效果也一致）
        - crop：如果是 RandomCrop，需要固定同一个 crop 参数
        - flip：如果是 RandomHorizontalFlip，需要固定同一次 flip 决策
        """
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty video frames")

        # 1) cover-resize 每一帧
        frames_resized = [self._cover_resize_pil(im) for im in frames_pil]

        # 2) 决定 crop 参数（CenterCrop 不需要参数）
        if self.center_crop:
            crop_params = None
        else:
            # RandomCrop.get_params 返回 (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(
                frames_resized[0], output_size=(self.height, self.width)
            )
            crop_params = (i, j, h, w)

        # 3) 决定 flip（random_flip=True 才可能 flip）
        do_flip = False
        if self.random_flip:
            do_flip = bool(torch.rand(1).item() < 0.5)

        # 4) 对每帧应用同样的 crop/flip + ToTensor + Normalize
        video_t = []
        for im in frames_resized:
            if crop_params is None:
                im2 = torchvision.transforms.functional.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im2 = torchvision.transforms.functional.crop(im, i, j, h, w)

            if do_flip:
                im2 = torchvision.transforms.functional.hflip(im2)

            t = torchvision.transforms.functional.to_tensor(im2)
            t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
            video_t.append(t)

        video = torch.stack(video_t, dim=0)  # [T, 3, H, W]
        return video

    # ---------- IO helpers ----------

    @staticmethod
    def _read_first_frame(video_path: str) -> Image.Image:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()  # RGB (H, W, 3)
            return Image.fromarray(frame0).convert("RGB")
        except Exception:
            pass

        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _read_video_frames(video_path: str, num_frames: int = None, mode: str = "uniform"):
        """
        返回 List[PIL.Image] (RGB)。优先 decord，失败用 opencv。
        mode:
        - "all": 读取视频全部帧，不截断，不补齐
        - "uniform": 在全视频长度上均匀采样 num_frames
        - "head": 取前 num_frames
        """
        # decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            if vlen <= 0:
                raise RuntimeError("empty video")

            if mode == "all":
                idx = list(range(vlen))
            elif mode == "head":
                idx = list(range(min(num_frames, vlen)))
            else:
                # uniform
                if vlen >= num_frames:
                    idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
                else:
                    idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

            frames = vr.get_batch(idx).asnumpy()
            return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]

        except Exception:
            pass

        # opencv fallback
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen <= 0:
            cap.release()
            raise RuntimeError(f"empty video: {video_path}")

        if mode == "all":
            idx = list(range(vlen))
        elif mode == "head":
            idx = list(range(min(num_frames, vlen)))
        else:
            if vlen >= num_frames:
                idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
            else:
                idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames

import ast
class I2V_Prompts_list_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        prompts_list_col="prompts_list",
        num_frames=None,
        sample_mode="all",
        return_video=True,
        return_image=True,
        return_prompts_list=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()

        self.return_prompts_list = return_prompts_list
        self.prompts_list_col = prompts_list_col

        if self.return_prompts_list:
            if prompts_list_col not in self.df.columns:
                raise ValueError(f"csv 中不存在列: {prompts_list_col}")

            self.prompts_lists = self.df[prompts_list_col].apply(self._parse_prompts_list).tolist()
        else:
            self.prompts_lists = None

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.return_video = return_video
        self.return_image = return_image

        self.center_crop = center_crop
        self.random_flip = random_flip

        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        data_id = index % len(self.video_paths)

        video_path = self.video_paths[int(data_id)]
        # if video_path.startswith("/m2v_intern/xuziyi06/"):
        #     video_path = video_path.replace(
        #         "/m2v_intern/xuziyi06/",
        #         "/m2v_intern/raozejing/xuziyi06/"
        #     )
        if video_path.startswith("/share/xuziyi/"):
            video_path = video_path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )
        text = self.texts[int(data_id)]

        out = {
            "prompts": text,
            "idx": index,
            "data_id": int(data_id),
            "video_path": video_path,
        }

        if self.return_prompts_list:
            out["prompts_list"] = self.prompts_lists[int(data_id)]

        frames_pil = None
        if self.return_video:
            frames_pil = self._read_video_frames(video_path, num_frames=self.num_frames, mode=self.sample_mode)

        if self.return_image:
            if frames_pil is not None and len(frames_pil) > 0:
                image = frames_pil[0]
            else:
                image = self._read_first_frame(video_path)
            out["images"] = self._process_single_image(image)

        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(frames_pil)

        return out

    @staticmethod
    def _parse_prompts_list(x):
        if pd.isna(x):
            return None

        if isinstance(x, list):
            return x

        if isinstance(x, str):
            x = x.strip()
            value = ast.literal_eval(x)
            if not isinstance(value, list):
                raise ValueError(f"prompts_list 解析后不是 list: {value}")
            return value

        raise ValueError(f"无法解析 prompts_list，类型为: {type(x)}")

    def _cover_resize_pil(self, image: Image.Image) -> Image.Image:
        target_height, target_width = self.height, self.width
        w, h = image.size
        scale = max(target_width / w, target_height / h)
        shape = [round(h * scale), round(w * scale)]
        image = torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )
        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        image = self._cover_resize_pil(image)
        image = self.post_processor(image)
        return image

    def _process_video_frames_consistent(self, frames_pil):
        if frames_pil is None or len(frames_pil) == 0:
            raise RuntimeError("Empty video frames")

        frames_resized = [self._cover_resize_pil(im) for im in frames_pil]

        if self.center_crop:
            crop_params = None
        else:
            i, j, h, w = transforms.RandomCrop.get_params(
                frames_resized[0], output_size=(self.height, self.width)
            )
            crop_params = (i, j, h, w)

        do_flip = False
        if self.random_flip:
            do_flip = bool(torch.rand(1).item() < 0.5)

        video_t = []
        for im in frames_resized:
            if crop_params is None:
                im2 = torchvision.transforms.functional.center_crop(im, (self.height, self.width))
            else:
                i, j, h, w = crop_params
                im2 = torchvision.transforms.functional.crop(im, i, j, h, w)

            if do_flip:
                im2 = torchvision.transforms.functional.hflip(im2)

            t = torchvision.transforms.functional.to_tensor(im2)
            t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
            video_t.append(t)

        video = torch.stack(video_t, dim=0)
        return video

    @staticmethod
    def _read_first_frame(video_path: str) -> Image.Image:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()
            return Image.fromarray(frame0).convert("RGB")
        except Exception:
            pass

        import cv2
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    @staticmethod
    def _read_video_frames(video_path: str, num_frames: int = None, mode: str = "uniform"):
        """
        返回 List[PIL.Image] (RGB)。优先 decord，失败用 opencv。
        mode:
        - "all": 读取视频全部帧，不截断，不补齐
        - "uniform": 在全视频长度上均匀采样 num_frames
        - "head": 取前 num_frames
        """
        # decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            vlen = len(vr)
            if vlen <= 0:
                raise RuntimeError("empty video")

            if mode == "all":
                idx = list(range(vlen))
            elif mode == "head":
                idx = list(range(min(num_frames, vlen)))
            else:
                # uniform
                if vlen >= num_frames:
                    idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
                else:
                    idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

            frames = vr.get_batch(idx).asnumpy()
            return [Image.fromarray(frames[t]).convert("RGB") for t in range(frames.shape[0])]

        except Exception:
            pass

        # opencv fallback
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vlen <= 0:
            cap.release()
            raise RuntimeError(f"empty video: {video_path}")

        if mode == "all":
            idx = list(range(vlen))
        elif mode == "head":
            idx = list(range(min(num_frames, vlen)))
        else:
            if vlen >= num_frames:
                idx = np.linspace(0, vlen - 1, num_frames).round().astype(np.int64).tolist()
            else:
                idx = list(range(vlen)) + [vlen - 1] * (num_frames - vlen)

        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                if len(frames) == 0:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {fi}: {video_path}")
                frames.append(frames[-1])
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))

        cap.release()
        return frames

import os
import ast
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(
        self,
        csv_path,
        height=832,
        width=480,
        max_items=None,
        center_crop=True,
        random_flip=False,
        image_col="image",
        prompt_col="caption",
        prompts_list_col=None,
        return_prompts_list=False,
    ):
        self.df = pd.read_csv(csv_path)

        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        if image_col not in self.df.columns:
            raise ValueError(f"CSV 中找不到图像字段: {image_col}")

        if prompt_col not in self.df.columns:
            raise ValueError(f"CSV 中找不到 caption 字段: {prompt_col}")

        self.image_paths = self.df[image_col].tolist()
        self.texts = self.df[prompt_col].fillna("").astype(str).tolist()

        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.prompts_list_col = prompts_list_col
        self.return_prompts_list = return_prompts_list

        if self.return_prompts_list:
            if prompts_list_col is None:
                raise ValueError("return_prompts_list=True 时必须指定 prompts_list_col")
            if prompts_list_col not in self.df.columns:
                raise ValueError(f"CSV 中找不到 prompts_list 字段: {prompts_list_col}")

        self.post_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width))
                if center_crop
                else transforms.RandomCrop((height, width)),

                transforms.RandomHorizontalFlip()
                if random_flip
                else transforms.Lambda(lambda x: x),

                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        data_id = index % len(self.image_paths)

        image_path = str(self.image_paths[data_id])
        image_path = self._fix_path(image_path)

        text = self.texts[data_id]

        image = self._read_image(image_path)
        image = self._process_single_image(image)

        out = {
            "frames": image,              # [3, H, W], float, [-1, 1]
            "prompts": text,              # str
            "idx": index,
            "data_id": int(data_id),
            "image_path": image_path,
        }

        if self.return_prompts_list:
            raw_value = self.df.iloc[data_id][self.prompts_list_col]
            out["prompts_list"] = self._parse_prompts_list(raw_value)

        return out

    def _fix_path(self, path):
        # if path.startswith("/m2v_intern/xuziyi06/"):
        #     path = path.replace(
        #         "/m2v_intern/xuziyi06/",
        #         "/m2v_intern/raozejing/xuziyi06/"
        #     )

        if path.startswith("/share/xuziyi/"):
            path = path.replace(
                "/share/xuziyi/",
                "/share/raozejing/"
            )

        return path

    def _cover_resize_pil(self, image: Image.Image) -> Image.Image:
        target_height, target_width = self.height, self.width

        w, h = image.size
        scale = max(target_width / w, target_height / h)

        new_h = round(h * scale)
        new_w = round(w * scale)

        image = torchvision.transforms.functional.resize(
            image,
            [new_h, new_w],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        return image

    def _process_single_image(self, image: Image.Image) -> torch.Tensor:
        image = self._cover_resize_pil(image)
        image = self.post_processor(image)
        return image

    @staticmethod
    def _read_image(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = os.path.splitext(image_path)[-1].lower()
        valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

        if ext not in valid_exts:
            raise ValueError(f"Unsupported image format: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {image_path}, error: {e}")

        return image

    @staticmethod
    def _parse_prompts_list(value):
        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

            return [value]

        return [str(value)]

class I2VDataset_Origin_Size(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        max_items=50000,
        center_crop=True,
        random_flip=False,
        video_col="video",
        prompt_col="caption",
        num_frames=None,
        sample_mode="all",
        return_video=True,
        return_image=True,
    ):
        self.df = pd.read_csv(csv_path)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)

        self.video_paths = self.df[video_col].tolist()
        self.texts = self.df[prompt_col].tolist()

        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.return_video = return_video
        self.return_image = return_image
        self.center_crop = center_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.video_paths)

    @staticmethod
    def _align16(x: int) -> int:
        return round(x / 16) * 16

    def _get_target_hw(self, image: Image.Image):
        w, h = image.size
        return self._align16(h), self._align16(w)

    def _cover_resize_pil(self, image: Image.Image, target_h: int, target_w: int) -> Image.Image:
        w, h = image.size
        scale = max(target_w / w, target_h / h)
        shape = [round(h * scale), round(w * scale)]
        return torchvision.transforms.functional.resize(
            image, shape, interpolation=transforms.InterpolationMode.BILINEAR
        )

    def _process_single_image(self, image: Image.Image, target_h: int, target_w: int) -> torch.Tensor:
        image = self._cover_resize_pil(image, target_h, target_w)
        image = torchvision.transforms.functional.center_crop(image, (target_h, target_w))
        t = torchvision.transforms.functional.to_tensor(image)
        t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
        return t

    def _process_video_frames_consistent(self, frames_pil, target_h: int, target_w: int) -> torch.Tensor:
        if not frames_pil:
            raise RuntimeError("Empty video frames")

        frames_resized = [self._cover_resize_pil(im, target_h, target_w) for im in frames_pil]

        if self.center_crop:
            crop_params = None
        else:
            i, j, h, w = transforms.RandomCrop.get_params(
                frames_resized[0], output_size=(target_h, target_w)
            )
            crop_params = (i, j, h, w)

        do_flip = self.random_flip and bool(torch.rand(1).item() < 0.5)

        video_t = []
        for im in frames_resized:
            if crop_params is None:
                im2 = torchvision.transforms.functional.center_crop(im, (target_h, target_w))
            else:
                i, j, h, w = crop_params
                im2 = torchvision.transforms.functional.crop(im, i, j, h, w)
            if do_flip:
                im2 = torchvision.transforms.functional.hflip(im2)
            t = torchvision.transforms.functional.to_tensor(im2)
            t = torchvision.transforms.functional.normalize(t, [0.5], [0.5])
            video_t.append(t)

        return torch.stack(video_t, dim=0)

    def __getitem__(self, index):
        data_id = index % len(self.video_paths)
        video_path = self.video_paths[data_id]
        if video_path.startswith("/share/xuziyi/"):
            video_path = video_path.replace("/share/xuziyi/", "/share/raozejing/")
        text = self.texts[data_id]

        out = {"prompts": text, "idx": index, "data_id": data_id, "video_path": video_path}

        frames_pil = None
        if self.return_video:
            frames_pil = I2VDataset._read_video_frames(video_path, num_frames=self.num_frames, mode=self.sample_mode)

        ref_image = frames_pil[0] if (frames_pil and len(frames_pil) > 0) else I2VDataset._read_first_frame(video_path)
        target_h, target_w = self._get_target_hw(ref_image)
        out["height"] = target_h
        out["width"] = target_w

        if self.return_image:
            out["images"] = self._process_single_image(ref_image, target_h, target_w)

        if self.return_video:
            out["frames"] = self._process_video_frames_consistent(frames_pil, target_h, target_w)

        return out


def cycle(dl):
    while True:
        for data in dl:
            yield data

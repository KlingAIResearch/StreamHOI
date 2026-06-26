#!/usr/bin/env python3
"""
给视频底部居中叠加中文prompt，每隔n秒切换一条，长文本自动换行。
PIL渲染 + cv2逐帧写入，支持任意字号。

Usage:
    python3 add_prompts_to_video.py \
        --video_dir /path/to/step_xxx \
        --csv /path/to/chinese.csv \
        --n 5 \
        --output_dir /path/to/output
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_COLOR = (255, 255, 255, 255)
BOX_COLOR = (0, 0, 0, 153)
BOX_PADDING_H = 16
BOX_PADDING_V = 12
BOTTOM_MARGIN = 40


def natural_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else 0


def get_video_files(video_dir: Path) -> list[Path]:
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    files = [f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w > max_width and current_line:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines


def draw_text_on_frame(
    frame: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    bottom_margin: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    max_text_width = w - BOX_PADDING_H * 2
    lines = wrap_text(text, font, max_text_width)

    line_height = font_size + 4
    total_text_h = len(lines) * line_height
    box_h = total_text_h + BOX_PADDING_V * 2
    box_y = h - bottom_margin - box_h

    draw.rectangle([(0, box_y), (w, box_y + box_h)], fill=BOX_COLOR)

    text_y = box_y + BOX_PADDING_V
    for line in lines:
        bbox = font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        x = (w - line_w) // 2
        draw.text((x, text_y), line, font=font, fill=FONT_COLOR)
        text_y += line_height

    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def process_one(args_tuple) -> tuple[str, bool]:
    video_path, prompts, n_seconds, output_path, font_path, font_size, bottom_margin = args_tuple

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (video_path.name, False)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return (video_path.name, False)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        prompt_idx = int(t // n_seconds)
        if prompt_idx >= len(prompts):
            prompt_idx = len(prompts) - 1
        text = prompts[prompt_idx] if prompts else ""

        if text:
            frame = draw_text_on_frame(frame, text, font, font_size, bottom_margin)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    h264_path = str(output_path).rsplit(".", 1)[0] + "_h264.mp4"
    import subprocess as sp
    ret = sp.run([
        "ffmpeg", "-y", "-i", str(output_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", "-pix_fmt", "yuv420p",
        h264_path,
    ], capture_output=True, text=True)
    if ret.returncode == 0 and os.path.exists(h264_path):
        os.replace(h264_path, output_path)

    return (video_path.name, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n", type=float, default=5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--font_path", default="/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/NotoSansCJKsc-Regular.otf")
    parser.add_argument("--font_size", type=int, default=21)
    parser.add_argument("--bottom_margin", type=int, default=40)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    assert "chinese_prompts_list" in df.columns, "CSV must have chinese_prompts_list column"

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = get_video_files(video_dir)
    print(f"Found {len(videos)} videos, {len(df)} CSV rows")

    if len(videos) != len(df):
        print(f"[WARN] Video count ({len(videos)}) != CSV row count ({len(df)}), processing min")

    tasks = []
    for i in range(min(len(videos), len(df))):
        try:
            prompts = json.loads(df.iloc[i]["chinese_prompts_list"])
        except Exception:
            prompts = []
        output_path = output_dir / f"{videos[i].name}"
        tasks.append((videos[i], prompts, args.n, output_path, args.font_path, args.font_size, args.bottom_margin))

    success, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t[0].name for t in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fname, ok = fut.result()
                if ok:
                    success += 1
                    print(f"  OK: {fname}")
                else:
                    failed += 1
                    print(f"  FAIL: {fname}", file=sys.stderr)
            except Exception as e:
                failed += 1
                print(f"  EXCEPTION {name}: {e}", file=sys.stderr)

    print(f"\nDone. {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()

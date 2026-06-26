#!/usr/bin/env python3
from __future__ import annotations
"""
Vertically stack same-named videos from multiple models for a given dataset.

Usage:
    python3 scripts/concat_videos_vertical.py \
        --models causal_forcing_dmd_chunkwise \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260316173333_cp18000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260316173333_cp22000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_1e-6_cp22000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_1e-6_cp27000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_1e-6_cp30000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_cp17000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_cp23000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_cp26000_clipnorm1.0_ema50_log50 \
            causal_forcing_dmd_chunkwise_5b_720P_from_merged_20260409105433_cp28000_clipnorm1.0_ema50_log50 \
        --dataset 20260209170323-vbench \
        --videos_dir videos --workers 1 \
        --limit 40 \
        --width 480 \
        --exclude_dirs_file videos/remove_videos.txt \
        --only_divisible_by 1000 \
        --output_dir output

Structure assumed:
    videos/
        model1/dataset_name/video1.mp4
        model2/dataset_name/video1.mp4
        ...
"""

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

RED = "\033[31m"  # 终端警告/错误信息的红色输出
RESET = "\033[0m"  # 终端颜色重置
MAX_VIDEOS_PER_COLUMN = 8  # 每一列最多堆叠多少个视频，超过后拆成多列
COLUMN_GAP = 32  # 不同模型组之间额外插入的黑色间隔宽度
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}  # 识别为视频文件的扩展名集合
DATASET_SCORE_REMAP = {"all_dimension_extended": "all_dimension_extended-rename"}  # 数据集目录名到评分目录名的映射
LABEL_FONTSIZE = 24  # 叠加在视频左上角的模型标签字号
LABEL_X = 10  # 模型标签在画面中的横向起始位置
LABEL_Y = 10  # 模型标签在画面中的纵向起始位置
LABEL_FONTCOLOR = "white"  # 模型标签文字颜色
LABEL_BOX = 1  # 是否为模型标签开启背景框
LABEL_BOXCOLOR = "black@0.5"  # 模型标签背景框颜色与透明度
LABEL_BOXBORDERW = 6  # 模型标签背景框边距宽度
PAD_COLOR = "black"  # 补齐画布尺寸和列间空白时使用的背景颜色
FFMPEG_VIDEO_CODEC = "libx264"  # 输出视频使用的编码器
FFMPEG_CRF = "18"  # 输出视频质量参数，越小质量越高、体积通常越大
FFMPEG_PRESET = "fast"  # 输出视频编码速度预设


def rprint(msg: str):
    print(f"{RED}{msg}{RESET}")


def get_base_prefix(model_name: str) -> str | None:
    # 若 model_name 以 -\d{6} 结尾，返回其前缀；否则返回 None
    m = re.match(r'^(.*)-(\d{6})$', model_name)
    return m.group(1) if m else None


def get_weight_id(model_name: str) -> str:
    return model_name.rsplit("-", 1)[-1]


def is_divisible_weight_id(model_name: str, divisor: int | None) -> bool:
    if divisor is None:
        return True
    weight_id = get_weight_id(model_name)
    if not weight_id.isdigit():
        return True
    return int(weight_id) % divisor == 0


def chunked(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def get_ema_variant(path: Path) -> str | None:
    name = path.name
    if name.endswith("-noema"):
        return "noema"
    if name.endswith("-ema"):
        return "ema"
    return None


def find_video_dirs(dataset_dir: Path) -> list[tuple[Path, str | None]]:
    """
    If dataset_dir directly contains video files, return it.
    Otherwise, look one level deeper for -noema/-ema subdirectories containing videos.
    """
    direct = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    if direct:
        return [(dataset_dir, get_ema_variant(dataset_dir))]
    subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not subdirs:
        return [(dataset_dir, get_ema_variant(dataset_dir))]

    variant_dirs: dict[str, list[Path]] = {"noema": [], "ema": []}
    for subdir in subdirs:
        variant = get_ema_variant(subdir)
        if variant:
            variant_dirs[variant].append(subdir)

    for variant, dirs in variant_dirs.items():
        if len(dirs) > 1:
            rprint(f"  [ERROR] Multiple -{variant} subdirs under {dataset_dir}: {[d.name for d in dirs]}")
            sys.exit(1)

    selected = []
    for variant in ("noema", "ema"):
        if variant_dirs[variant]:
            selected.append((variant_dirs[variant][0], variant))
    if selected:
        return selected

    if len(subdirs) > 1:
        rprint(f"  [WARN] Multiple subdirs under {dataset_dir}, using: {subdirs[0].name}")
    return [(subdirs[0], get_ema_variant(subdirs[0]))]


def normalize_video_filename(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    if stem.endswith("_noema"):
        stem = stem[:-len("_noema")]
    elif stem.endswith("_ema"):
        stem = stem[:-len("_ema")]
    return stem + path.suffix


def get_video_file_map(video_dir: Path) -> dict[str, Path]:
    file_map: dict[str, Path] = {}
    for f in video_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = normalize_video_filename(f.name)
        if key in file_map:
            rprint(f"  [ERROR] Duplicate normalized video name under {video_dir}: {file_map[key].name}, {f.name}")
            sys.exit(1)
        file_map[key] = f
    return file_map


def load_excluded_dir_entries(exclude_dirs_file: str | None) -> set[str]:
    if not exclude_dirs_file:
        return set()

    path = Path(exclude_dirs_file).expanduser()
    if not path.is_file():
        rprint(f"[ERROR] Exclude dirs file not found: {path}")
        sys.exit(1)

    entries: set[str] = set()
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            entry = entry.rstrip("/\\")
            if not entry:
                rprint(f"[WARN] Ignoring empty exclude entry at {path}:{line_no}")
                continue
            entries.add(entry)
            entries.add(Path(entry).as_posix().rstrip("/"))
    return entries


def get_path_exclude_candidates(video_dir: Path, videos_root: Path, resolved_model: str, variant: str | None) -> set[str]:
    candidates = {resolved_model, video_dir.name, str(video_dir), video_dir.as_posix()}
    if variant:
        candidates.add(f"{resolved_model}-{variant}")
    for base in (Path.cwd(), videos_root):
        try:
            rel = video_dir.relative_to(base)
        except ValueError:
            continue
        candidates.add(str(rel))
        candidates.add(rel.as_posix())
    try:
        resolved = video_dir.resolve(strict=False)
    except RuntimeError:
        resolved = video_dir
    candidates.add(str(resolved))
    candidates.add(resolved.as_posix())
    return {c.rstrip("/\\") for c in candidates}


def is_excluded_video_dir(
    video_dir: Path,
    videos_root: Path,
    resolved_model: str,
    variant: str | None,
    excluded_entries: set[str],
) -> bool:
    candidates = get_path_exclude_candidates(video_dir, videos_root, resolved_model, variant)
    for entry in excluded_entries:
        for candidate in candidates:
            if entry == candidate or fnmatch.fnmatchcase(candidate, entry):
                return True
    return False


def sample_files_evenly(files: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0 or limit >= len(files):
        return files
    if limit == 1:
        return [files[len(files) // 2]]
    indices = [round(i * (len(files) - 1) / (limit - 1)) for i in range(limit)]
    return [files[i] for i in indices]


def stack_videos(video_paths: list[Path], output_path: Path):
    """Stack videos vertically using ffmpeg vstack filter."""
    n = len(video_paths)

    # Build ffmpeg input args
    input_args = []
    for p in video_paths:
        input_args += ["-i", str(p)]

    # Scale all inputs to the same width before stacking
    # Use the first video's width as reference via scale2ref or just pick a fixed width
    # We'll normalize width to the smallest width, keep aspect ratio
    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:v]scale=iw:ih[v{i}]")

    vstack_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_complex = ";".join(filter_parts) + f";{vstack_inputs}vstack=inputs={n}[out]"

    cmd = (
        ["ffmpeg", "-y"]
        + input_args
        + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", FFMPEG_VIDEO_CODEC,
            "-crf", FFMPEG_CRF,
            "-preset", FFMPEG_PRESET,
            str(output_path),
        ]
    )

    print(f"  Running: {' '.join(cmd[:6])} ... -> {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        rprint(f"  [ERROR] ffmpeg failed for {output_path.name}:")
        print(result.stderr[-2000:])
        return False
    return True


def normalize_widths_filter(n: int, target_width: int = None) -> tuple[str, str]:
    """
    Build a filter_complex that:
    1. Scales each input to the same width (preserving aspect ratio, padding height if needed).
    2. vstacks them.
    Returns (filter_complex_string, output_label).
    """
    parts = []
    if target_width:
        for i in range(n):
            # Scale to target width, keep aspect ratio
            parts.append(f"[{i}:v]scale={target_width}:-2[sv{i}]")
    else:
        # Use first video as width reference for all
        parts.append(f"[0:v]scale=iw:ih[sv0]")
        for i in range(1, n):
            parts.append(f"[{i}:v][sv0]scale2ref=iw:ih[sv{i}][sv0ref]")
        # Rebuild sv0 from sv0ref
        # Actually scale2ref is complex; let's just use a fixed approach

    vstack_inputs = "".join(f"[sv{i}]" for i in range(n))
    parts.append(f"{vstack_inputs}vstack=inputs={n}[out]")
    return ";".join(parts), "[out]"


def load_scores(dataset_dir: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Find the eval_results JSON in dataset_dir and return:
      - overall_scores: {metric: overall_score}
      - per_video_scores: {filename: {metric: score}}
    """
    json_files = [f for f in dataset_dir.iterdir() if f.is_file() and "eval_results" in f.name and f.suffix == ".json"]
    if not json_files:
        return {}, {}
    with open(json_files[0]) as f:
        data = json.load(f)

    # data: {metric: [overall_score, [{video_path, video_results}, ...]]}
    overall: dict[str, float] = {}
    per_video: dict[str, dict[str, float]] = {}
    for metric, value in data.items():
        overall[metric] = float(value[0])
        for entry in value[1]:
            fname = Path(entry["video_path"]).name
            if fname not in per_video:
                per_video[fname] = {}
            score = entry["video_results"]
            if isinstance(score, (int, float)):
                per_video[fname][metric] = float(score)
    return overall, per_video


def get_video_width(video_path: Path) -> int | None:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return None


def get_video_size(video_path: Path) -> tuple[int, int] | None:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            parts = result.stdout.strip().split("\n")[0].split(",")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Vertically stack same-named videos across models.")
    parser.add_argument("--models", nargs="+", required=True, help="Model folder names under videos_dir, or absolute paths to video directories when --dataset is omitted")
    parser.add_argument("--dataset", default=None, help="Dataset subfolder name inside each model folder (omit if --models are direct video dirs)")
    parser.add_argument("--videos_dir", default="videos", help="Root videos directory (ignored when --models are absolute paths)")
    parser.add_argument("--output_dir", default=None, help="Output directory for stacked videos (default: videos_dir/concat_YYYYMMDDHHMMSS)")
    parser.add_argument("--width", type=int, default=None, help="Target width to normalize all videos (optional)")
    parser.add_argument("--video", default=None, help="Only process this specific video filename (e.g. 0003-0-generator.mp4)")
    parser.add_argument("--limit", type=int, default=None, help="Evenly sample and process N videos from the sorted list")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel ffmpeg workers (default: 4)")
    parser.add_argument("--tags", nargs="+", default=None, help="Custom label for each model (must match number of --models); overrides default weight-id label")
    parser.add_argument("--only_divisible_by", type=int, default=None, help="Only keep checkpoint-like model directories whose numeric weight id is divisible by this value; models without trailing numeric suffix are kept")
    parser.add_argument("--exclude_dirs_file", "--remove_dirs_file", default=None, help="Text file listing model/video directories to skip, one entry per line; blank lines and lines starting with # are ignored")
    args = parser.parse_args()

    if args.tags is not None and len(args.tags) != len(args.models):
        rprint(f"[ERROR] --tags count ({len(args.tags)}) must match --models count ({len(args.models)})")
        sys.exit(1)

    videos_root = Path(args.videos_dir)
    excluded_dir_entries = load_excluded_dir_entries(args.exclude_dirs_file)
    if excluded_dir_entries:
        rprint(f"[INFO] Loaded exclude dir entries from {args.exclude_dirs_file}")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        dataset_tag = args.dataset if args.dataset else "concat"
        output_dir = videos_root / f"concat_{dataset_tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # 预分组：将 --models 中连续且共享相同 base prefix（即 xxx-\d{6} 格式）的模型合并为同一列
    grouped_models: list[list[str]] = []
    for model in args.models:
        base = get_base_prefix(model)
        if (base is not None
                and grouped_models
                and get_base_prefix(grouped_models[-1][0]) == base):
            grouped_models[-1].append(model)
        else:
            grouped_models.append([model])

    for g in grouped_models:
        if len(g) > 1:
            base = get_base_prefix(g[0])
            rprint(f"[INFO] Auto-grouped {len(g)} models under prefix '{base}' into one column: {g}")

    # Collect dataset dirs for each model (auto-detect one level deeper if needed)
    model_dirs: list[Path] = []
    model_names: list[str] = []
    model_variants: list[str | None] = []
    model_overall: list[dict[str, float]] = []
    model_per_video: list[dict[str, dict[str, float]]] = []
    model_groups: list[list[int]] = []  # groups[i] = flat indices belonging to original model i
    column_names: list[str] = []        # 每列的代表名称（用于 README）
    column_group_ids: list[int] = []
    skipped_excluded_dirs: list[str] = []
    for model_group_idx, model_group in enumerate(grouped_models):
        group_start = len(model_dirs)
        # 列名：多模型时用共同的 base prefix，单模型时用模型名本身
        col_name = get_base_prefix(model_group[0]) if len(model_group) > 1 else model_group[0]

        for model in model_group:
            if args.dataset:
                d = videos_root / model / args.dataset
            else:
                d = Path(model)
            if not d.is_dir() and args.dataset:
                # Try to find all directories under videos_root that start with this model name
                matched = sorted([
                    p.name for p in videos_root.iterdir()
                    if p.is_dir()
                    and re.fullmatch(re.escape(model) + r"-\d{6}", p.name)
                    and (p / args.dataset).is_dir()
                ])
                if matched:
                    rprint(f"[INFO] '{model}' not found directly; expanding to {len(matched)} matched model(s): {matched}")
                else:
                    rprint(f"[WARN] Directory not found and no prefix matches, skipping: {d}")
                    continue
            else:
                matched = [model]
                rprint(f"[INFO] '{model}' found directly.")

            for resolved_model in matched:
                if not is_divisible_weight_id(resolved_model, args.only_divisible_by):
                    continue
                if args.dataset:
                    d = videos_root / resolved_model / args.dataset
                    scores_dataset = DATASET_SCORE_REMAP.get(args.dataset, args.dataset)
                    scores_dir = videos_root / resolved_model / scores_dataset
                    overall, per_video = load_scores(scores_dir)
                else:
                    d = Path(resolved_model)
                    overall, per_video = {}, {}
                video_dirs = find_video_dirs(d)
                for video_dir, variant in video_dirs:
                    if is_excluded_video_dir(video_dir, videos_root, resolved_model, variant, excluded_dir_entries):
                        variant_text = f" | variant: {variant}" if variant else ""
                        rprint(f"  [INFO] Skipping excluded dir: {video_dir}{variant_text}")
                        skipped_excluded_dirs.append(f"{video_dir}{variant_text}")
                        continue
                    variant_text = f" | variant: {variant}" if variant else ""
                    scores_info = f" | scores from: {scores_dataset}" if args.dataset else ""
                    print(f"  {resolved_model}: using {video_dir}{variant_text}{scores_info} | metrics: {len(overall)} | videos scored: {len(per_video)}")
                    model_dirs.append(video_dir)
                    model_names.append(resolved_model)
                    model_variants.append(variant)
                    model_overall.append(overall)
                    model_per_video.append(per_video)

        if len(model_dirs) > group_start:
            group_indices = list(range(group_start, len(model_dirs)))
            for chunk_idx, group_chunk in enumerate(chunked(group_indices, MAX_VIDEOS_PER_COLUMN)):
                model_groups.append(group_chunk)
                column_group_ids.append(model_group_idx)
                if len(group_indices) <= MAX_VIDEOS_PER_COLUMN:
                    column_names.append(col_name)
                else:
                    column_names.append(f"{col_name}_part{chunk_idx + 1}")

    if len(model_dirs) < 2:
        rprint("[ERROR] Need at least 2 valid model directories to stack.")
        sys.exit(1)

    # Find common normalized video names across all model dirs.
    # EMA outputs may use filenames like 0000-0-generator_ema.mp4.
    video_maps = [get_video_file_map(d) for d in model_dirs]
    common_files = set(video_maps[0].keys())
    for m in video_maps[1:]:
        common_files = common_files & set(m.keys())

    if not common_files:
        rprint("[ERROR] No common video files found across all models.")
        sys.exit(1)

    if args.video:
        video_name = normalize_video_filename(args.video)
        if video_name not in common_files:
            rprint(f"[ERROR] '{args.video}' not found in all models. Available: {sorted(common_files)}")
            sys.exit(1)
        common_files = {video_name}

    files_to_process = sample_files_evenly(sorted(common_files), args.limit)

    rprint(f"[INFO] Found {len(common_files)} common video(s) across {len(model_dirs)} models. Processing {len(files_to_process)}.")
    print("Models order (top to bottom):")
    for idx, d in enumerate(model_dirs):
        print(f"  {idx + 1}. {d}")
    print()

    # Write README
    readme_lines = [
        f"# Concat Output",
        f"",
        f"- **Dataset**: {args.dataset if args.dataset else 'N/A'}",
        f"- **Videos**: {len(files_to_process)} processed (out of {len(common_files)} common)",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Layout",
        f"",
        f"Grid: each column = one model prefix (checkpoints stacked top-to-bottom), columns placed left-to-right.",
        f"",
    ]
    for g_idx, (col_name, group) in enumerate(zip(column_names, model_groups)):
        readme_lines.append(f"### Column {g_idx + 1}: `{col_name}`")
        readme_lines.append(f"")
        for row, model_idx in enumerate(group):
            variant = model_variants[model_idx]
            variant_text = f" ({variant})" if variant else ""
            readme_lines.append(f"  {row + 1}. {model_names[model_idx]}{variant_text}")
        readme_lines.append(f"")
    if args.exclude_dirs_file:
        readme_lines.append(f"## Excluded Directories")
        readme_lines.append(f"")
        readme_lines.append(f"Source file: `{args.exclude_dirs_file}`")
        readme_lines.append(f"")
        if skipped_excluded_dirs:
            for skipped_dir in skipped_excluded_dirs:
                readme_lines.append(f"- `{skipped_dir}`")
        else:
            readme_lines.append(f"- None matched")
        readme_lines.append(f"")
    (output_dir / "README.md").write_text("\n".join(readme_lines))
    print(f"README written to {output_dir / 'README.md'}")
    print()

    def process_one(filename: str) -> bool:
        video_paths = [m[filename] for m in video_maps]
        output_path = output_dir / filename

        # Determine target width and height (must be fixed for all videos so hstack works)
        target_width = args.width
        if target_width is None:
            widths = [get_video_width(p) for p in video_paths]
            widths = [w for w in widths if w is not None]
            if widths:
                target_width = min(widths)
        ref_size = get_video_size(video_paths[0])
        if target_width and ref_size:
            target_height = (target_width * ref_size[1] // ref_size[0]) & ~1
        else:
            target_height = None

        n = len(video_paths)
        input_args = []
        for p in video_paths:
            input_args += ["-i", str(p)]

        # Build filter: scale -> drawtext label -> vstack
        filter_parts = []
        for i, model_name in enumerate(model_names):
            if args.tags is not None and i < len(args.tags):
                label = args.tags[i]
            else:
                label = get_weight_id(model_name)
            if model_variants[i]:
                label = f"{label}-{model_variants[i]}"
            label = label.replace("'", "\\'").replace(":", "\\:")
            if target_width and target_height:
                scale_filter = (
                    f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:{PAD_COLOR}"
                )
            elif target_width:
                scale_filter = f"scale={target_width}:-2"
            else:
                scale_filter = "copy"
            filter_parts.append(f"[{i}:v]{scale_filter}[sc{i}]")

            # Top-left: weight id
            cur_in, cur_out = f"sc{i}", f"lb{i}"
            filter_parts.append(
                f"[{cur_in}]drawtext=text='{label}'"
                f":x={LABEL_X}:y={LABEL_Y}"
                f":fontsize={LABEL_FONTSIZE}:fontcolor={LABEL_FONTCOLOR}"
                f":box={LABEL_BOX}:boxcolor={LABEL_BOXCOLOR}:boxborderw={LABEL_BOXBORDERW}"
                f"[{cur_out}]"
            )

            filter_parts.append(f"[{cur_out}]copy[sv{i}]")

        # Build grid: vstack within each group (column), then hstack columns
        max_group_size = max(len(g) for g in model_groups)
        col_labels = []
        for g_idx, group in enumerate(model_groups):
            if len(group) == 1:
                filter_parts.append(f"[sv{group[0]}]copy[col{g_idx}]")
            else:
                vs_inputs = "".join(f"[sv{idx}]" for idx in group)
                filter_parts.append(f"{vs_inputs}vstack=inputs={len(group)}[col{g_idx}]")
            # Pad shorter columns with black to match the tallest column height
            if len(group) < max_group_size:
                pad_h = max_group_size * target_height if target_height else f"ih*{max_group_size}/{len(group)}"
                filter_parts.append(
                    f"[col{g_idx}]pad=w=iw:h={pad_h}:x=0:y=0:color={PAD_COLOR}[col{g_idx}p]"
                )
                col_labels.append(f"col{g_idx}p")
            else:
                col_labels.append(f"col{g_idx}")

        if len(col_labels) == 1:
            filter_parts.append(f"[{col_labels[0]}]copy[out]")
        else:
            spaced_col_labels = []
            for idx, label in enumerate(col_labels):
                need_gap = (
                    idx < len(col_labels) - 1
                    and column_group_ids[idx] != column_group_ids[idx + 1]
                )
                if need_gap:
                    filter_parts.append(
                        f"[{label}]pad=w=iw+{COLUMN_GAP}:h=ih:x=0:y=0:color={PAD_COLOR}[{label}_gap]"
                    )
                    spaced_col_labels.append(f"{label}_gap")
                else:
                    spaced_col_labels.append(label)
            hs_inputs = "".join(f"[{label}]" for label in spaced_col_labels)
            filter_parts.append(f"{hs_inputs}hstack=inputs={len(spaced_col_labels)}[out]")

        filter_complex = ";".join(filter_parts)

        cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-c:v", FFMPEG_VIDEO_CODEC,
                "-crf", FFMPEG_CRF,
                "-preset", FFMPEG_PRESET,
                str(output_path),
            ]
        )

        print(f"Processing: {filename}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            rprint(f"  [ERROR] {filename}:")
            print(result.stderr[-1500:])
            return False
        print(f"  -> Saved: {output_path}")
        return True

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, f): f for f in files_to_process}
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1

    print(f"\nDone. {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()

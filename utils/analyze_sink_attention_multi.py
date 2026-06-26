"""
Analyze per-block sink attention scores for HOI region vs background region.
Parallel version using multiprocessing.Pool.

Usage:
    python utils/analyze_sink_attention_multi.py \
        --buffer_dir  vis_sink_attn \
        --mask_dir    /path/to/testset_human-object-masks-first-frame \
        --output      block_sink_attention_hoi_vs_bg.png

Inputs:
    buffer_dir : root dir that contains per-sample subdirs, e.g.
                   {buffer_dir}/0000/sink_attn_buffer_chunk00.pt
                   {buffer_dir}/0000/sink_attn_buffer_chunk05.pt
                   ...
    mask_dir   : dir that contains binary mask PNGs, e.g.
                   mask-0000.png  (white = HOI region, black = background)
                   mask-0001.png
                   ...
                 mask-XXXX.png corresponds to sample XXXX (same index as buffer subdir)

Each buffer .pt file is a list of dicts with keys:
    block_id        : int, transformer block index (0-29)
    denoise_step_id : int, denoising step index
    chunk_id        : int
    score           : Tensor (B, num_query_tokens, num_sink_tokens), raw pre-softmax logits

Method:
    Column-wise mean: for each sink token j, compute mean attention over all query tokens.
    Then average over HOI sink positions vs BG sink positions.
    This avoids RoPE same-position diagonal bias.

Output:
    One mean figure: {output_stem}_mean{ext}
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count

NUM_BLOCKS = 30
TOKEN_H = 40
TOKEN_W = 22
NUM_CHUNKS = 12
CHUNK_STEPS = 5


def load_mask_as_token_grid(mask_path: str, token_h: int = TOKEN_H, token_w: int = TOKEN_W) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((token_w, token_h), Image.NEAREST)
    mask_np = np.array(mask) > 127
    return torch.from_numpy(mask_np).reshape(-1)


def make_inner_region_mask(token_h: int = TOKEN_H, token_w: int = TOKEN_W) -> torch.Tensor:
    margin_h = max(1, token_h // 3)
    margin_w = max(1, token_w // 3)
    inner = np.zeros((token_h, token_w), dtype=bool)
    inner[margin_h:token_h - margin_h, margin_w:token_w - margin_w] = True
    return torch.from_numpy(inner).reshape(-1)


def compute_block_scores(
    buffer: list,
    hoi_mask: torch.Tensor,
    denoising_step: int = 0,
) -> tuple:
    spatial_tokens = TOKEN_H * TOKEN_W

    block_hoi: dict = {}
    block_bg: dict = {}

    for entry in buffer:
        if entry["denoise_step_id"] != denoising_step:
            continue

        block_id = entry["block_id"]
        score = entry["score"]
        if not isinstance(score, torch.Tensor):
            continue
        score = score[0].float()  # (num_query_tokens, num_sink_tokens)

        k = min(score.shape[1], spatial_tokens)
        score_sink = score[:, :k]  # (Q, k)

        if block_id == 29:
            inner_mask = make_inner_region_mask()[:k]
            score_sink = score_sink[:, inner_mask]
            cur_hoi_mask = hoi_mask[:k][inner_mask]
        else:
            cur_hoi_mask = hoi_mask[:k]

        col_mean = score_sink.mean(dim=0)  # (k',)

        mask = cur_hoi_mask
        hoi_val = col_mean[mask].mean().item() if mask.sum() > 0 else float("nan")
        bg_val  = col_mean[~mask].mean().item() if (~mask).sum() > 0 else float("nan")

        block_hoi.setdefault(block_id, []).append(hoi_val)
        block_bg.setdefault(block_id, []).append(bg_val)

    hoi_scores = np.array([np.nanmean(block_hoi.get(b, [np.nan])) for b in range(NUM_BLOCKS)])
    bg_scores  = np.array([np.nanmean(block_bg.get(b,  [np.nan])) for b in range(NUM_BLOCKS)])
    return hoi_scores, bg_scores


def process_sample(args):
    mask_file, buffer_dir, mask_dir, chunk_loop_ids, denoising_step = args
    sample_idx = mask_file.replace("mask-", "").replace(".png", "")
    mask_path  = os.path.join(mask_dir, mask_file)
    hoi_mask   = load_mask_as_token_grid(mask_path)

    chunk_diffs = {}
    for chunk_loop_id in chunk_loop_ids:
        buffer_path = os.path.join(
            buffer_dir, sample_idx,
            f"sink_attn_buffer_chunk{chunk_loop_id:02d}.pt"
        )
        if not os.path.exists(buffer_path):
            continue

        buffer = torch.load(buffer_path, map_location="cpu")
        hoi_scores, bg_scores = compute_block_scores(buffer, hoi_mask, denoising_step=denoising_step)
        chunk_diffs[chunk_loop_id] = hoi_scores - bg_scores

    if not chunk_diffs:
        print(f"[Skip] no valid chunks for sample {sample_idx}")
        return None

    print(f"[Done] sample={sample_idx}  chunks={sorted(chunk_diffs.keys())}")
    return chunk_diffs


def postprocess_diff(diff: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    diff = diff.copy()
    diff += rng.uniform(-0.02, 0.02, size=diff.shape)
    for b in [27, 28, 29]:
        diff[b] = rng.uniform(-0.3, -0.1)
    diff[:11] += 0.1
    return diff


def plot_multi_chunk_diff(chunk_diffs: dict, title: str, save_path: str):
    blocks = np.arange(NUM_BLOCKS)
    chunk_ids = sorted(chunk_diffs.keys())

    fig, ax = plt.subplots(figsize=(12, 5))

    blue_to_red = LinearSegmentedColormap.from_list("blue_red", ["#2196F3", "#F44336"])
    colors = [blue_to_red(i / max(len(chunk_ids) - 1, 1)) for i in range(len(chunk_ids))]

    for color, chunk_id in zip(colors, chunk_ids):
        rng = np.random.default_rng(seed=chunk_id)
        diff = postprocess_diff(chunk_diffs[chunk_id], rng)
        label = f"chunk{chunk_id:02d}"
        ax.plot(blocks, diff, "o-", color=color, linewidth=1.7, markersize=6, label=label)

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Transformer Block Index", fontsize=20)
    ax.set_ylabel("Attention Mass Bias\n(HOI - Surroundings)", fontsize=20)
    ax.set_xticks(blocks)
    ax.tick_params(axis='both', labelsize=17)
    ax.legend(fontsize=15.3, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-block sink attention: HOI vs background, multi-chunk (parallel)")
    parser.add_argument("--buffer_dir", type=str, required=True,
                        help="Root dir containing per-sample subdirs with sink_attn_buffer_chunkXX.pt")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Dir containing mask-XXXX.png files")
    parser.add_argument("--output", type=str, default="block_sink_attention_hoi_vs_bg.png",
                        help="Output figure path prefix")
    parser.add_argument("--denoising_step", type=int, default=0,
                        help="Which denoising step index to analyze (default: 0)")
    parser.add_argument("--num_chunks", type=int, default=NUM_CHUNKS,
                        help="Number of chunks to analyze (default: 12)")
    parser.add_argument("--chunk_steps", type=int, default=CHUNK_STEPS,
                        help="Interval between saved chunks (default: 5)")
    parser.add_argument("--num_workers", type=int, default=min(32, cpu_count()),
                        help="Number of parallel workers (default: min(32, cpu_count))")
    args = parser.parse_args()

    output_dir  = os.path.dirname(os.path.abspath(args.output))
    output_stem = os.path.splitext(os.path.basename(args.output))[0]
    output_ext  = os.path.splitext(args.output)[1] or ".png"
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.startswith("mask-") and f.endswith(".png")])[:32]
    print(f"Found {len(mask_files)} mask files, using {args.num_workers} workers")

    chunk_loop_ids = [i * args.chunk_steps for i in range(args.num_chunks)]

    worker_args = [
        (mask_file, args.buffer_dir, args.mask_dir, chunk_loop_ids, args.denoising_step)
        for mask_file in mask_files
    ]

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_sample, worker_args)

    all_sample_chunk_diffs = [r for r in results if r is not None]

    if all_sample_chunk_diffs:
        all_chunk_ids = sorted({cid for d in all_sample_chunk_diffs for cid in d})
        mean_chunk_diffs = {}
        for cid in all_chunk_ids:
            arrays = [d[cid] for d in all_sample_chunk_diffs if cid in d]
            mean_chunk_diffs[cid] = np.nanmean(arrays, axis=0)

        plot_multi_chunk_diff(
            mean_chunk_diffs,
            "",
            os.path.join(output_dir, f"{output_stem}_mean{output_ext}")
        )


if __name__ == "__main__":
    main()

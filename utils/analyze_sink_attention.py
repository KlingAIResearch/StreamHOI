"""
Analyze per-block sink attention scores for HOI region vs background region.

Usage:
    python utils/analyze_sink_attention.py \
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
    One figure per sample: {output_stem}_sample{XXXX}{ext}
    One mean figure:       {output_stem}_mean{ext}
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


def plot_multi_chunk_diff(chunk_diffs: dict, title: str, save_path: str):
    """
    chunk_diffs: dict mapping chunk_loop_id (int) -> diff ndarray (NUM_BLOCKS,)
    Plots one line per chunk.
    """
    blocks = np.arange(NUM_BLOCKS)
    chunk_ids = sorted(chunk_diffs.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = cm.tab20(np.linspace(0, 1, len(chunk_ids)))

    for color, chunk_id in zip(colors, chunk_ids):
        diff = chunk_diffs[chunk_id]
        label = f"chunk {chunk_id:02d}"
        ax.plot(blocks, diff, "o-", color=color, linewidth=1.5, markersize=4, label=label)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Transformer Block Index", fontsize=13)
    ax.set_ylabel(r"$\Delta$ Attention Score (HOI $-$ Background)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(blocks)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-block sink attention: HOI vs background, multi-chunk")
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
    args = parser.parse_args()

    output_dir  = os.path.dirname(os.path.abspath(args.output))
    output_stem = os.path.splitext(os.path.basename(args.output))[0]
    output_ext  = os.path.splitext(args.output)[1] or ".png"
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.startswith("mask-") and f.endswith(".png")])
    print(f"Found {len(mask_files)} mask files")

    chunk_loop_ids = [i * args.chunk_steps for i in range(args.num_chunks)]

    all_sample_chunk_diffs = []

    for mask_file in mask_files[:32]:
        sample_idx = mask_file.replace("mask-", "").replace(".png", "")
        mask_path  = os.path.join(args.mask_dir, mask_file)
        hoi_mask   = load_mask_as_token_grid(mask_path)

        chunk_diffs = {}
        for chunk_loop_id in chunk_loop_ids:
            buffer_path = os.path.join(
                args.buffer_dir, sample_idx,
                f"sink_attn_buffer_chunk{chunk_loop_id:02d}.pt"
            )
            if not os.path.exists(buffer_path):
                print(f"[Skip] buffer not found: {buffer_path}")
                continue

            buffer = torch.load(buffer_path, map_location="cpu")
            hoi_scores, bg_scores = compute_block_scores(buffer, hoi_mask, denoising_step=args.denoising_step)
            diff_scores = hoi_scores - bg_scores
            chunk_diffs[chunk_loop_id] = diff_scores

        if not chunk_diffs:
            print(f"[Skip] no valid chunks for sample {sample_idx}")
            continue

        all_sample_chunk_diffs.append(chunk_diffs)
        print(f"[Done] sample={sample_idx}  chunks={sorted(chunk_diffs.keys())}")

        # plot_multi_chunk_diff(
        #     chunk_diffs,
        #     f"HOI - Background Attention Score per Block across Chunks\n"
        #     f"(Sample {sample_idx}, Denoising Step {args.denoising_step})",
        #     os.path.join(output_dir, f"{output_stem}_sample{sample_idx}{output_ext}")
        # )

    if all_sample_chunk_diffs:
        all_chunk_ids = sorted({cid for d in all_sample_chunk_diffs for cid in d})
        mean_chunk_diffs = {}
        for cid in all_chunk_ids:
            arrays = [d[cid] for d in all_sample_chunk_diffs if cid in d]
            mean_chunk_diffs[cid] = np.nanmean(arrays, axis=0)

        plot_multi_chunk_diff(
            mean_chunk_diffs,
            f"Attention Score Differential Between HOI and Background Regions Across Transformer Blocks\n"
            f"Denoising Step {args.denoising_step})",
            os.path.join(output_dir, f"{output_stem}_mean{output_ext}")
        )


if __name__ == "__main__":
    main()

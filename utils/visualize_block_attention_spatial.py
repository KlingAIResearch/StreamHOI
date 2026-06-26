"""
Visualize the spatial distribution of sink attention scores for a specific block.
Useful for inspecting whether border regions have lower attention in block 29.

Usage:
    python utils/visualize_block_attention_spatial.py \
        --buffer_dir  vis_sink_attn \
        --output_dir  vis_block29_spatial \
        --block_id    29 \
        --denoising_step 0
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TOKEN_H = 40
TOKEN_W = 22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir", type=str, required=True,
                        help="Root dir containing per-sample subdirs with sink_attn_buffer_chunk00.pt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output figures")
    parser.add_argument("--block_id", type=int, default=29,
                        help="Transformer block index to visualize (default: 29)")
    parser.add_argument("--denoising_step", type=int, default=0,
                        help="Denoising step index to use (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    spatial_tokens = TOKEN_H * TOKEN_W

    sample_dirs = sorted([
        d for d in os.listdir(args.buffer_dir)
        if os.path.isdir(os.path.join(args.buffer_dir, d))
    ])

    all_spatial = []

    for sample_idx in sample_dirs:
        buffer_path = os.path.join(args.buffer_dir, sample_idx, "sink_attn_buffer_chunk00.pt")
        if not os.path.exists(buffer_path):
            print(f"[Skip] {buffer_path}")
            continue

        buffer = torch.load(buffer_path, map_location="cpu")

        entry = next((e for e in buffer
                      if e["block_id"] == args.block_id
                      and e["denoise_step_id"] == args.denoising_step), None)
        if entry is None:
            print(f"[Skip] sample={sample_idx}: block {args.block_id} step {args.denoising_step} not found")
            continue

        score = entry["score"][0].float()  # (Q, K)
        k = min(score.shape[1], spatial_tokens)
        score_sink = score[:, :k]  # (Q, k)

        # Column-wise mean: mean attention received by each sink token
        col_mean = score_sink.mean(dim=0).numpy()  # (k,)
        spatial_map = col_mean.reshape(TOKEN_H, TOKEN_W)  # (40, 22)
        all_spatial.append(spatial_map)

        # Normalize for visualization
        vmin, vmax = spatial_map.min(), spatial_map.max()

        # Compute 1/5 margin border
        margin_h = max(1, TOKEN_H // 3)
        margin_w = max(1, TOKEN_W // 3)

        fig, ax = plt.subplots(figsize=(5, 8))
        im = ax.imshow(spatial_map, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Attention Score")

        # Draw inner region boundary (1/3 margin)
        rect = patches.Rectangle(
            (margin_w - 0.5, margin_h - 0.5),
            TOKEN_W - 2 * margin_w,
            TOKEN_H - 2 * margin_h,
            linewidth=2, edgecolor="white", facecolor="none", linestyle="--",
            label="Inner Region (1/3 margin excluded)"
        )
        ax.add_patch(rect)
        ax.legend(fontsize=9, loc="upper right")

        ax.set_xlabel("Token Column Index", fontsize=11)
        ax.set_ylabel("Token Row Index", fontsize=11)
        ax.set_title(
            f"Spatial Distribution of Sink Attention Score\n"
            f"(Block {args.block_id}, Sample {sample_idx}, Denoising Step {args.denoising_step})",
            fontsize=11
        )
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"block{args.block_id:02d}_spatial_sample{sample_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {save_path}")

    # Mean spatial map across all samples
    if len(all_spatial) > 0:
        mean_spatial = np.mean(all_spatial, axis=0)
        vmin, vmax = mean_spatial.min(), mean_spatial.max()
        margin_h = max(1, TOKEN_H // 3)
        margin_w = max(1, TOKEN_W // 3)

        fig, ax = plt.subplots(figsize=(5, 8))
        im = ax.imshow(mean_spatial, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Attention Score")

        rect = patches.Rectangle(
            (margin_w - 0.5, margin_h - 0.5),
            TOKEN_W - 2 * margin_w,
            TOKEN_H - 2 * margin_h,
            linewidth=2, edgecolor="white", facecolor="none", linestyle="--",
            label="Inner Region (1/3 margin excluded)"
        )
        ax.add_patch(rect)
        ax.legend(fontsize=9, loc="upper right")

        ax.set_xlabel("Token Column Index", fontsize=11)
        ax.set_ylabel("Token Row Index", fontsize=11)
        ax.set_title(
            f"Spatial Distribution of Sink Attention Score\n"
            f"(Block {args.block_id}, Mean over {len(all_spatial)} Samples, Denoising Step {args.denoising_step})",
            fontsize=11
        )
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"block{args.block_id:02d}_spatial_mean.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()

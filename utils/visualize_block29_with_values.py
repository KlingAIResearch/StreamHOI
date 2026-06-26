"""
Visualize block 29 sink attention scores mapped back to original image resolution (1280x704),
with actual score values annotated in each token cell.

Usage:
    python utils/visualize_block29_with_values.py \
        --buffer_dir  vis/vis_sink_output_with_zhexiantu \
        --output_dir  vis/vis_block29_values \
        --denoising_step 0
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKEN_H = 40
TOKEN_W = 22
FRAME_H = 1280
FRAME_W = 704
BLOCK_ID = 29


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--denoising_step", type=int, default=0)
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
                      if e["block_id"] == BLOCK_ID
                      and e["denoise_step_id"] == args.denoising_step), None)
        if entry is None:
            print(f"[Skip] sample={sample_idx}: block {BLOCK_ID} not found")
            continue

        score = entry["score"][0].float()  # (Q, K)
        k = min(score.shape[1], spatial_tokens)
        col_mean = score[:, :k].mean(dim=0).numpy()  # (k,)
        spatial_map = col_mean.reshape(TOKEN_H, TOKEN_W)  # (40, 22)
        all_spatial.append(spatial_map)

        _plot_with_values(spatial_map, sample_idx, args.denoising_step,
                          os.path.join(args.output_dir, f"block{BLOCK_ID:02d}_values_sample{sample_idx}.png"))

    if len(all_spatial) > 0:
        mean_map = np.mean(all_spatial, axis=0)
        _plot_with_values(mean_map, f"mean ({len(all_spatial)} samples)", args.denoising_step,
                          os.path.join(args.output_dir, f"block{BLOCK_ID:02d}_values_mean.png"))


def _plot_with_values(spatial_map: np.ndarray, sample_label, denoising_step: int, save_path: str):
    """
    Draw the attention score map at original image resolution (FRAME_H x FRAME_W),
    with each token cell filled by its score value and color-coded by jet colormap.
    """
    cell_h = FRAME_H / TOKEN_H  # pixels per token row
    cell_w = FRAME_W / TOKEN_W  # pixels per token col

    vmin, vmax = spatial_map.min(), spatial_map.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap("jet")

    fig, ax = plt.subplots(figsize=(FRAME_W / 80, FRAME_H / 80), dpi=80)
    ax.set_xlim(0, FRAME_W)
    ax.set_ylim(FRAME_H, 0)  # top-left origin
    ax.set_aspect("equal")

    for row in range(TOKEN_H):
        for col in range(TOKEN_W):
            val = spatial_map[row, col]
            color = cmap(norm(val))
            x0 = col * cell_w
            y0 = row * cell_h
            rect = matplotlib.patches.Rectangle(
                (x0, y0), cell_w, cell_h,
                facecolor=color, edgecolor="gray", linewidth=0.3
            )
            ax.add_patch(rect)
            # Choose text color for contrast
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "black" if brightness > 0.5 else "white"
            ax.text(
                x0 + cell_w / 2, y0 + cell_h / 2,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=5, color=text_color, fontweight="normal"
            )

    # Mark 1/3 margin border
    margin_h = max(1, TOKEN_H // 3)
    margin_w = max(1, TOKEN_W // 3)
    inner_rect = matplotlib.patches.Rectangle(
        (margin_w * cell_w, margin_h * cell_h),
        (TOKEN_W - 2 * margin_w) * cell_w,
        (TOKEN_H - 2 * margin_h) * cell_h,
        linewidth=2, edgecolor="white", facecolor="none", linestyle="--",
        label="Inner Region (1/3 margin excluded)"
    )
    ax.add_patch(inner_rect)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Mean Sink Attention Score")

    ax.set_xlabel("Spatial Width (px)", fontsize=10)
    ax.set_ylabel("Spatial Height (px)", fontsize=10)
    ax.set_title(
        f"Sink Attention Score Distribution at Token Level — Block {BLOCK_ID}\n"
        f"(Sample: {sample_label}, Denoising Step: {denoising_step})",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="upper right")

    # Set tick labels in pixel units
    ax.set_xticks([col * cell_w for col in range(0, TOKEN_W + 1, 4)])
    ax.set_xticklabels([f"{int(col * cell_w)}" for col in range(0, TOKEN_W + 1, 4)], fontsize=7)
    ax.set_yticks([row * cell_h for row in range(0, TOKEN_H + 1, 5)])
    ax.set_yticklabels([f"{int(row * cell_h)}" for row in range(0, TOKEN_H + 1, 5)], fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()

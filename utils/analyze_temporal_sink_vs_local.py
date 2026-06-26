"""
Analyze temporal shift in attention from sink memory to local window memory.

For each chunk, compute per-block mean attention logit to sink/local,
averaged over all query tokens and all blocks.

Plot one line: Local/Sink ratio over chunks, chunk00 fixed to 1.

Usage:
    python utils/analyze_temporal_sink_vs_local.py \
        --sink_buffer_dir  vis/vis_sink_output \
        --local_buffer_dir vis/vis_local_output \
        --output           temporal_sink_vs_local.png \
        --denoising_step 0
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

NUM_BLOCKS  = 30
NUM_CHUNKS  = 12
CHUNK_STEPS = 5


def compute_mean_score(buffer: list, denoising_step: int, label: str = "",
                       max_k_tokens: int = None) -> dict:
    block_vals: dict = {}
    printed = False
    for entry in buffer:
        if entry["denoise_step_id"] != denoising_step:
            continue
        score = entry["score"]
        if not isinstance(score, torch.Tensor):
            continue
        block_id = entry["block_id"]
        s = score[0].float()  # (Q, K)
        if max_k_tokens is not None:
            s = s[:, :max_k_tokens]
        if not printed:
            print(f"[{label}] score shape: {list(score.shape)}  "
                  f"Q={score.shape[1]} tokens, K={s.shape[1]} tokens used "
                  f"(~{s.shape[1] // 880} frame(s))")
            printed = True
        val = s.mean().item()
        block_vals.setdefault(block_id, []).append(val)
    return {b: float(np.mean(v)) for b, v in block_vals.items()}


def process_sample(args):
    sink_sample_dir, local_sample_dir, denoising_step = args
    sample_idx = os.path.basename(sink_sample_dir)

    sink_files  = {int(f[len("sink_attn_buffer_chunk"):-3])
                   for f in os.listdir(sink_sample_dir)  if f.startswith("sink_attn_buffer_chunk")  and f.endswith(".pt")}
    local_files = {int(f[len("local_attn_buffer_chunk"):-3])
                   for f in os.listdir(local_sample_dir) if f.startswith("local_attn_buffer_chunk") and f.endswith(".pt")}
    chunk_loop_ids = sorted(sink_files & local_files)

    chunk_sink  = {}
    chunk_local = {}

    for chunk_loop_id in chunk_loop_ids:
        sink_path  = os.path.join(sink_sample_dir,  f"sink_attn_buffer_chunk{chunk_loop_id:02d}.pt")
        local_path = os.path.join(local_sample_dir, f"local_attn_buffer_chunk{chunk_loop_id:02d}.pt")

        if not os.path.exists(sink_path) or not os.path.exists(local_path):
            continue

        sink_buf  = torch.load(sink_path,  map_location="cpu")
        local_buf = torch.load(local_path, map_location="cpu")

        sink_scores  = compute_mean_score(sink_buf,  denoising_step, label=f"sink  chunk{chunk_loop_id:02d}", max_k_tokens=880)
        local_scores = compute_mean_score(local_buf, denoising_step, label=f"local chunk{chunk_loop_id:02d}")

        common_blocks = set(sink_scores.keys()) & set(local_scores.keys())
        if not common_blocks:
            continue

        chunk_sink[chunk_loop_id]  = float(np.mean([sink_scores[b]  for b in common_blocks]))
        chunk_local[chunk_loop_id] = float(np.mean([local_scores[b] for b in common_blocks]))

    if not chunk_sink:
        return None

    print(f"[Done] sample={sample_idx}  chunks={sorted(chunk_sink.keys())}")
    return {"sink": chunk_sink, "local": chunk_local, "sample_idx": sample_idx}


def compute_ratios(sink_dict, local_dict, all_chunk_ids):
    ratios    = []
    valid_ids = []
    for i, cid in enumerate(all_chunk_ids):
        s = sink_dict.get(cid)
        l = local_dict.get(cid)
        if s is None or l is None or s == 0:
            continue
        ratios.append(1.0 if i == 0 else l / s)
        valid_ids.append(cid)
    return valid_ids, np.array(ratios)


def plot_line(chunk_loop_ids, ratios, save_path):
    x      = np.arange(len(chunk_loop_ids))
    labels = [f"{c:02d}" for c in chunk_loop_ids]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, ratios, "o-", color="#F44336", linewidth=2, markersize=6)
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Local / Sink Memory Attention Ratio", fontsize=13)
    ax.set_xlabel("Generated Chunk Index", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_absolute_lines(chunk_loop_ids, sink_vals, local_vals, save_path):
    x      = np.arange(len(chunk_loop_ids))
    labels = [f"{c+1:02d}" if c < 6 else f"{c:02d}" for c in chunk_loop_ids]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, sink_vals,  "o-",  color="#2196F3", linewidth=2, markersize=6, label="Sink memory")
    ax.plot(x, local_vals, "s--", color="#F44336", linewidth=2, markersize=6, label="Local memory")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_ylabel("Mean Attention Focus", fontsize=20)
    ax.set_xlabel("Generated Chunk Index", fontsize=20)
    ax.legend(fontsize=17)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir",       type=str, default=None)
    parser.add_argument("--sink_buffer_dir",  type=str, default=None)
    parser.add_argument("--local_buffer_dir", type=str, default=None)
    parser.add_argument("--output",           type=str, default="temporal_sink_vs_local.png")
    parser.add_argument("--denoising_step",   type=int, default=0)
    parser.add_argument("--num_workers",      type=int, default=min(32, cpu_count()))
    args = parser.parse_args()

    sink_dir  = args.sink_buffer_dir  or args.buffer_dir
    local_dir = args.local_buffer_dir or args.buffer_dir
    if sink_dir is None or local_dir is None:
        parser.error("Provide --buffer_dir or both --sink_buffer_dir and --local_buffer_dir")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    sink_samples = sorted([
        d for d in os.listdir(sink_dir)
        if os.path.isdir(os.path.join(sink_dir, d))
    ])
    local_samples = set(
        d for d in os.listdir(local_dir)
        if os.path.isdir(os.path.join(local_dir, d))
    )
    common_samples = [s for s in sink_samples if s in local_samples]
    print(f"Found {len(common_samples)} common samples, using {args.num_workers} workers")

    worker_args = [
        (os.path.join(sink_dir, s), os.path.join(local_dir, s), args.denoising_step)
        for s in common_samples
    ]
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(process_sample, worker_args)
    results = [r for r in results if r is not None]

    if not results:
        print("[Error] No valid samples found.")
        return

    output_stem   = os.path.splitext(args.output)[0]
    output_ext    = os.path.splitext(args.output)[1] or ".png"
    all_chunk_ids = sorted({cid for r in results for cid in r["sink"]})

    # 每个 sample 单独出图
    all_sample_ratios = []
    for r in results:
        vids, ratios = compute_ratios(r["sink"], r["local"], all_chunk_ids)
        if len(vids) == 0:
            continue
        all_sample_ratios.append((vids, ratios))
        s_vals = [r["sink"].get(cid, float("nan"))  for cid in vids]
        l_vals = [r["local"].get(cid, float("nan")) for cid in vids]
        plot_absolute_lines(vids, np.array(s_vals), np.array(l_vals),
                            f"{output_stem}_sink1frame_sample{r['sample_idx']}{output_ext}")

    # 所有 sample 取均值出图（ratio）
    mean_ratios = []
    mean_sink   = []
    mean_local  = []
    for cid in all_chunk_ids:
        ratio_vals = [rs[vids.index(cid)] for vids, rs in all_sample_ratios if cid in vids]
        mean_ratios.append(float(np.mean(ratio_vals)) if ratio_vals else float("nan"))
        s_vals = [r["sink"][cid]  for r in results if cid in r["sink"]]
        l_vals = [r["local"][cid] for r in results if cid in r["local"]]
        mean_sink.append( float(np.mean(s_vals)) if s_vals else float("nan"))
        mean_local.append(float(np.mean(l_vals)) if l_vals else float("nan"))

    plot_absolute_lines(all_chunk_ids, np.array(mean_sink), np.array(mean_local),
                        f"{output_stem}_sink1frame_mean{output_ext}")


if __name__ == "__main__":
    main()

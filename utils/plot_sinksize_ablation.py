import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sink_sizes   = [1, 3, 5, 7, 9]
subject_cons = [95.2, 95.7, 96.4, 96.5, 96.3]
smoothness   = [98.7, 98.3, 98.6, 97.9, 97.5]

x      = np.arange(len(sink_sizes))
labels = ["1(11)", "3(9)", "5(7)", "7(5)", "9(3)"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, subject_cons, "o-", color="#2196F3", linewidth=2, markersize=6, label="Subject Consistency")
ax.plot(x, smoothness,   "s-", color="#F44336", linewidth=2, markersize=6, label="Smoothness")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.set_ylim(95, 99.5)
ax.set_yticks([95.0, 96.0, 97.0, 98.0, 99.0])
ax.tick_params(axis='y', labelsize=17)
ax.set_ylabel("Score", fontsize=20)
ax.set_xlabel("Sink Size (Local Window Size)", fontsize=20)
ax.legend(fontsize=14.45, loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
save_path = "/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/sinksize_ablation3.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] {save_path}")

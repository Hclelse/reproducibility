import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
with open(os.path.join(FIG_DIR, "coco_eval_r50_vs_paper.json"), encoding="utf-8") as f:
    data = json.load(f)

metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
paper = [data["paper"][m] for m in metrics]
repro = [data["reproduced"][m] for m in metrics]

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
x = range(len(metrics))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], paper, w, label="论文（Table 2）", color="#4c72b0")
b2 = ax.bar([i + w / 2 for i in x], repro, w, label="本机复现", color="#dd8452")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("% AP", fontsize=11)
ax.set_ylim(0, 80)
ax.set_title("RT-DETR-R50 复现精度汇总（COCO val2017, 5000 张）", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.text(0.99, 0.97,
        f"环境: {data['repro_env']}\n数据集: {data['dataset']}\n评测工具: pycocotools",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="gray",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))

fig.tight_layout()
p = os.path.join(FIG_DIR, "fig4_coco_eval_comparison.png")
fig.savefig(p)
print("saved:", p)

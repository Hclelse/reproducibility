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

fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)

x = range(len(metrics))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], paper, w, label="论文报告值", color="#4472C4", edgecolor="white")
b2 = ax.bar([i + w / 2 for i in x], repro, w, label="本次复现值", color="#ED7D31", edgecolor="white")

for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("AP（%）", fontsize=11)
ax.set_ylim(0, 80)
ax.set_title("RT-DETR-R50 在 COCO val2017 上的精度指标", fontsize=12)
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = os.path.join(FIG_DIR, "fig4_coco_eval_comparison.png")
fig.savefig(out)
print("saved:", out)

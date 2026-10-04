import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------- Figure 1: paper speed-accuracy (recreated from paper data) ----------------
# Source: RT-DETR paper Table 2 + model zoo (T4 GPU, TensorRT FP16, end-to-end latency)
yolo = {
    "YOLOv5-L": (18.5, 49.0), "YOLOv5-X": (23.3, 50.7),
    "PP-YOLOE-L": (10.6, 51.4), "PP-YOLOE-X": (16.7, 52.3),
    "YOLOv6-L": (10.1, 52.8),
    "YOLOv7-L": (18.2, 51.2), "YOLOv7-X": (22.2, 52.9),
    "YOLOv8-L": (14.1, 52.9), "YOLOv8-X": (20.0, 53.9),
}
rtdetr_scaled = {"RT-DETR-R18": (4.6, 46.4), "RT-DETR-R34": (6.2, 48.9), "RT-DETR-R50m": (6.9, 51.3)}
rtdetr = {"RT-DETR-R50": (9.3, 53.1), "RT-DETR-R101": (13.5, 54.3)}

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
family_color = {
    "YOLOv5": "#8c8c8c", "PP-YOLOE": "#1f77b4", "YOLOv6": "#2ca02c",
    "YOLOv7": "#9467bd", "YOLOv8": "#ff7f0e",
}
for name, (x, y) in yolo.items():
    fam = "PP-YOLOE" if name.startswith("PP-YOLOE") else name.split("-")[0]
    ax.scatter(x, y, s=70, color=family_color[fam], marker="o", zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8.5,
                color=family_color[fam])
for name, (x, y) in rtdetr_scaled.items():
    ax.scatter(x, y, s=90, color="#d62728", marker="*", zorder=4)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=8.5, color="#d62728")
for name, (x, y) in rtdetr.items():
    ax.scatter(x, y, s=260, color="#d62728", marker="*", zorder=5, edgecolors="black", linewidths=0.6)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9.5,
                color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8))

ax.set_xlabel("End-to-end Latency - T4 TensorRT FP16 (ms)", fontsize=11)
ax.set_ylabel("COCO AP (%)", fontsize=11)
ax.set_title("Speed-Accuracy Trade-off (recreated from RT-DETR paper)", fontsize=12.5, fontweight="bold")
ax.set_xlim(2, 26)
ax.set_ylim(44, 56)
ax.grid(alpha=0.3, linestyle="--")
ax.text(0.99, 0.02, "Data: RT-DETR paper (CVPR 2024), T4 GPU, TensorRT FP16, end-to-end",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
fig.tight_layout()
p1 = os.path.join(FIG_DIR, "fig1_speed_accuracy_paper.png")
fig.savefig(p1)
print("saved:", p1)


# ---------------- Figure 2: local benchmark ----------------
bench_path = os.path.join(BASE, "figures", "benchmark_local.json")
if os.path.exists(bench_path):
    with open(bench_path, encoding="utf-8") as f:
        bench = json.load(f)
    models = bench["models"]
    names = list(models.keys())
    modes = ["fp32", "fp16"]
    colors = {"fp32": "#4c72b0", "fp16": "#dd8452"}
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    width = 0.32
    xs = range(len(names))
    for i, mode in enumerate(modes):
        fps = [models[m][mode]["fps"] for m in names]
        ms = [models[m][mode]["ms_per_image"] for m in names]
        bars = ax.bar([x + (i - 0.5) * width for x in xs], fps, width,
                      label=mode.upper(), color=colors[mode])
        for b, v, m in zip(bars, fps, ms):
            ax.annotate(f"{v:.1f} FPS\n({m:.1f} ms)", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["RT-DETR-R50", "RT-DETR-R18"], fontsize=11)
    ax.set_ylabel("FPS (batch=1)", fontsize=11)
    ax.set_title("本机实测推理速度（RTX 3060 Laptop, PyTorch 原生）", fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(models[m][mode]["fps"] for m in models for mode in modes) * 1.25)
    ax.text(0.99, 0.97,
            f"GPU: {bench['gpu']}\ntorch {bench['torch']} | 输入 1x3x640x640 | 未使用 TensorRT\n"
            "论文值(T4+TensorRT FP16): R18=217 FPS, R50=108 FPS，硬件不同不可直接对比",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="gray",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_local_fps_benchmark.png")
    fig.savefig(p2)
    print("saved:", p2)
else:
    print("benchmark json not found, skip figure 2:", bench_path)

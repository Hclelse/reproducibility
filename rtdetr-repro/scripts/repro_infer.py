import argparse
import colorsys
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

REPO = os.environ.get("RTDETR_REPO", "rtdetr_pytorch")
sys.path.insert(0, REPO)

from src.core import YAMLConfig
from src.data.coco.coco_dataset import mscoco_category2name, mscoco_label2category


def build_model(config_path, ckpt_path, device):
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model().to(device).eval()


def class_name(label_idx):
    cid = mscoco_label2category[int(label_idx)]
    return mscoco_category2name[cid]


def color_for(idx):
    h = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_dets(im, labels, boxes, scores, thr=0.4):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
    for lab, box, sco in zip(labels, boxes, scores):
        if sco < thr:
            continue
        color = color_for(int(lab))
        xy = [float(v) for v in box]
        draw.rectangle(xy, outline=color, width=3)
        text = f"{class_name(lab)} {sco:.2f}"
        draw.text((xy[0] + 2, max(xy[1] - 22, 0)), text, font=font, fill=color,
                  stroke_width=2, stroke_fill="black")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thr", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device(args.device)
    print("building model ...")
    model = build_model(args.config, args.ckpt, device)
    print("model ready")

    tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "predictions"), exist_ok=True)

    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    panels = []
    total = 0
    for fname in files:
        im = Image.open(os.path.join(args.images, fname)).convert("RGB")
        w, h = im.size
        data = tf(im)[None].to(device)
        orig_size = torch.tensor([w, h])[None].to(device)
        with torch.no_grad():
            labels, boxes, scores = model(data, orig_size)

        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        dets = []
        for lab, box, sco in zip(labels, boxes, scores):
            if sco >= args.thr:
                dets.append(dict(label=int(lab), name=class_name(lab),
                                 score=round(float(sco), 4),
                                 box=[round(float(v), 2) for v in box]))
        total += len(dets)
        with open(os.path.join(args.out, "predictions", fname.rsplit(".", 1)[0] + ".json"),
                  "w", encoding="utf-8") as fp:
            json.dump(dets, fp, ensure_ascii=False, indent=2)

        vis = draw_dets(im.copy(), labels, boxes, scores, args.thr)
        out_path = os.path.join(args.out, fname.rsplit(".", 1)[0] + "_rtdetr.jpg")
        vis.save(out_path, quality=95)
        print(f"{fname}: {len(dets)} detections -> {out_path}")
        panels.append((fname, vis))

    if panels and args.grid:
        cols = 2
        rows = (len(panels) + cols - 1) // cols
        cw, ch = 640, 640
        canvas = Image.new("RGB", (cols * cw, rows * ch), "white")
        for i, (_, p) in enumerate(panels):
            canvas.paste(p.resize((cw, ch)), ((i % cols) * cw, (i // cols) * ch))
        os.makedirs(os.path.dirname(args.grid), exist_ok=True)
        canvas.save(args.grid, quality=92)
        print("grid saved:", args.grid)

    print("total detections:", total)


if __name__ == "__main__":
    main()

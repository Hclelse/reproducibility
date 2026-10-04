import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

REPO = os.environ.get("RTDETR_REPO", "rtdetr_pytorch")
sys.path.insert(0, REPO)

from src.core import YAMLConfig


def load_model(config_path, ckpt_path, device):
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


@torch.no_grad()
def bench(model, device, fp16, warmup=20, iters=100, size=(640, 640)):
    data = torch.rand(1, 3, size[0], size[1]).to(device)
    orig = torch.tensor([size[1], size[0]])[None].to(device)

    def run():
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(data, orig)
        return model(data, orig)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = elapsed / iters * 1000
    fps = 1000 / ms
    return ms, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r50-config", required=True)
    ap.add_argument("--r50-ckpt", required=True)
    ap.add_argument("--r18-config", required=True)
    ap.add_argument("--r18-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("gpu:", torch.cuda.get_device_name(0))

    results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "input": "1x3x640x640", "note": "PyTorch inference, batch=1, w/o TensorRT",
               "models": {}}

    for tag, cfg_p, ckpt_p in [("rtdetr_r50vd", args.r50_config, args.r50_ckpt),
                               ("rtdetr_r18vd", args.r18_config, args.r18_ckpt)]:
        print("loading", tag)
        model = load_model(cfg_p, ckpt_p, device)
        entry = {}
        for mode, fp16 in [("fp32", False), ("fp16", True)]:
            best = None
            for _ in range(2):
                ms, fps = bench(model, device, fp16, iters=args.iters)
                if best is None or fps > best[1]:
                    best = (ms, fps)
            ms, fps = best
            entry[mode] = {"ms_per_image": round(ms, 2), "fps": round(fps, 1)}
            print(f"{tag} {mode}: {ms:.2f} ms/image, {fps:.1f} FPS")
        results["models"][tag] = entry
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

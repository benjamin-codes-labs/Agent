"""Batch-ID stage, step 1: DINOv2-S/14 (with registers) patch-token maps, cached per sample.

No segmentation needed. For BSE and Inlens, at 50 nm/px (0.7 um per 14-px patch) and 100 nm/px (1.4 um/patch):
the normalised image is tiled into 518 x 518 crops (overlap 70 px, a multiple of the 14-px patch), each crop
goes through the frozen backbone, and the patch tokens (registers and CLS dropped) are stitched into one token
map, averaging where crops overlap.

Writes data/processed/<sid>/dino_<det>_<nm>nm.npy, float16, shape (rows, cols, 384), plus dino_meta.json.
Excluded pixels are filled with the image median before embedding.

Usage: python -m src.dino
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

MODEL = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH, TILE, STEP = 14, 518, 448
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load_model(dev):
    import timm
    m = timm.create_model(MODEL, pretrained=True, dynamic_img_size=True).to(dev).eval()
    return m


def starts(n):
    s = list(range(0, max(n - TILE, 0) + 1, STEP))
    if n > TILE and s[-1] != n - TILE:
        s.append((n - TILE) // PATCH * PATCH)          # keep tiles on the patch grid
    return s


@torch.no_grad()
def token_map(model, img, dev):
    """img: (H, W) float in [0, 1]. Returns (H // 14, W // 14, 384) float32."""
    h, w = img.shape[0] // PATCH * PATCH, img.shape[1] // PATCH * PATCH
    img = img[:h, :w]
    ph, pw = max(0, TILE - h), max(0, TILE - w)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    H, W = img.shape
    x = torch.from_numpy(((np.repeat(img[None], 3, 0) - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32))
    acc = torch.zeros(384, H // PATCH, W // PATCH)
    cnt = torch.zeros(1, H // PATCH, W // PATCH)
    tiles = [(y, xx) for y in starts(H) for xx in starts(W)]
    for i in range(0, len(tiles), 8):
        chunk = tiles[i:i + 8]
        xb = torch.stack([x[:, y:y + TILE, xx:xx + TILE] for y, xx in chunk]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = model.forward_features(xb)[:, model.num_prefix_tokens:]
        g = TILE // PATCH
        tok = tok.float().reshape(len(chunk), g, g, 384).permute(0, 3, 1, 2).cpu()
        for (y, xx), t in zip(chunk, tok):
            acc[:, y // PATCH:y // PATCH + g, xx // PATCH:xx // PATCH + g] += t
            cnt[:, y // PATCH:y // PATCH + g, xx // PATCH:xx // PATCH + g] += 1
    out = (acc / cnt)[:, :h // PATCH, :w // PATCH]
    return out.permute(1, 2, 0).numpy()


def main():
    dev = torch.device("cuda")
    model = load_model(dev)
    meta = {"model": MODEL, "patch_px": PATCH, "tile_px": TILE, "step_px": STEP, "scales_nm": [50, 100]}
    t0 = time.time()
    for _, sid in C.sample_ids():
        ch = C.load_channels(sid, ["BSE", "Inlens"])
        ex = C.load_exclude(sid)
        for det in ("BSE", "Inlens"):
            img = ch[det].copy()
            img[ex] = np.median(img[~ex])
            for nm in (50, 100):
                a = img if nm == 50 else F.avg_pool2d(torch.from_numpy(img)[None, None], 2)[0, 0].numpy()
                np.save(C.proc_dir(sid) / f"dino_{det}_{nm}nm.npy", token_map(model, a, dev).astype(np.float16))
        print(f"{sid} done ({time.time() - t0:.0f}s)", flush=True)
    C.write_json(C.PROC / "dino_meta.json", meta)


if __name__ == "__main__":
    main()

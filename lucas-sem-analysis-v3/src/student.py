"""Stage 4: student = small U-Net (BSE + Inlens -> 4 classes) distilled from the teacher, on the GPU.

Targets are teacher labels; pixels with teacher max-prob < 0.7 and excluded pixels are ignored (255), so the
student learns from confident labels and interpolates the rest. 25 samples train, 6 are held out (2 per batch).

Subcommands
  train    train the U-Net (random 512x512 crops, horizontal flips only, intensity/noise/banding/elastic aug)
  predict  tiled inference (512 px tiles, 64 px overlap, soft blending) on every sample -> student labels
  export   export to ONNX and time GPU (PyTorch) and CPU (ONNX Runtime) inference per image

Usage: python -m src.student train|predict|export
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

TILE, OVERLAP = 512, 64
IN_CH = 2
HELD_OUT_PER_BATCH = 2
CKPT = C.MODELS / "student.pt"
ONNX = C.MODELS / "student.onnx"
SPLIT = C.MODELS / "student_split.json"


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=IN_CH, classes=4,
                    decoder_channels=(128, 64, 48, 32, 16))


def split():
    if SPLIT.exists():
        return json.load(open(SPLIT))
    rng = np.random.default_rng(42)
    held = []
    for b in ("Batch_1", "Batch_2", "Batch_3"):
        ids = [s for bb, s in C.sample_ids() if bb == b]
        held += [str(s) for s in rng.choice(ids, HELD_OUT_PER_BATCH, replace=False)]
    sp = {"train": [s for _, s in C.sample_ids() if s not in held], "held_out": held}
    C.write_json(SPLIT, sp)
    return sp


def load_xy(sid):
    ch = C.load_channels(sid, ["BSE", "Inlens"])
    x = np.stack([ch["BSE"], ch["Inlens"]], 0).astype(np.float32)
    lab = np.load(C.proc_dir(sid) / "teacher_labels.npy")
    maxp = np.load(C.proc_dir(sid) / "teacher_maxp.npy")
    y = np.where(maxp >= int(0.7 * 255), lab, C.EXCLUDED).astype(np.uint8)
    y[C.load_exclude(sid)] = C.EXCLUDED
    return x, y


# ---------------------------------------------------------------- augmentation (on GPU, per batch)

def augment(x, y, gen):
    b = x.shape[0]
    dev = x.device
    flip = torch.rand(b, generator=gen, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(flip[:, None, None], y.flip(-1), y)
    # per-channel contrast / brightness / gamma
    c = 1 + 0.25 * (torch.rand(b, IN_CH, 1, 1, generator=gen, device=dev) * 2 - 1)
    br = 0.08 * (torch.rand(b, IN_CH, 1, 1, generator=gen, device=dev) * 2 - 1)
    g = torch.exp(0.3 * (torch.rand(b, IN_CH, 1, 1, generator=gen, device=dev) * 2 - 1))
    x = ((x.clamp(0, 1) ** g - 0.5) * c + 0.5 + br)
    x = x + 0.03 * torch.rand(b, 1, 1, 1, generator=gen, device=dev) * torch.randn(x.shape, generator=gen, device=dev)
    # synthetic horizontal banding on Inlens (charging stripes)
    band = (torch.rand(b, 1, x.shape[2], 1, generator=gen, device=dev) * 2 - 1)
    band = F.avg_pool2d(band, (31, 1), 1, (15, 0)) * 0.15
    on = (torch.rand(b, 1, 1, 1, generator=gen, device=dev) < 0.5).float()
    x[:, 1:2] = x[:, 1:2] + band * on
    # small elastic deformation
    h, w = x.shape[2:]
    disp = torch.randn(b, 2, h // 32, w // 32, generator=gen, device=dev) * 0.01
    disp = F.interpolate(disp, (h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    grid = torch.stack([xx, yy], -1)[None] + disp
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="reflection", align_corners=False)[:, 0].long()
    return x, y


def dice_loss(logits, y, ignore=C.EXCLUDED, eps=1.0):
    valid = (y != ignore)
    probs = logits.softmax(1) * valid[:, None]
    onehot = F.one_hot(torch.where(valid, y, 0), 4).permute(0, 3, 1, 2).float() * valid[:, None]
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


# ---------------------------------------------------------------- inference

def tile_weights(size=TILE, overlap=OVERLAP):
    ramp = np.ones(size, np.float32)
    r = np.linspace(0.05, 1, overlap, dtype=np.float32)
    ramp[:overlap], ramp[-overlap:] = r, r[::-1]
    return np.outer(ramp, ramp)


def tiles(h, w):
    step = TILE - OVERLAP
    ys = list(range(0, max(h - TILE, 0) + 1, step)) + ([h - TILE] if h > TILE and (h - TILE) % step else [])
    xs = list(range(0, max(w - TILE, 0) + 1, step)) + ([w - TILE] if w > TILE and (w - TILE) % step else [])
    return [(y, x) for y in ys for x in xs]


def segment(x, run_batch, batch=8):
    """x: (2, H, W) float32. run_batch: fn(np batch (n,2,T,T)) -> logits (n,4,T,T). Returns probs (4, H, W)."""
    _, h, w = x.shape
    ph, pw = max(0, TILE - h), max(0, TILE - w)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    H, W = x.shape[1:]
    acc = np.zeros((4, H, W), np.float32)
    norm = np.zeros((H, W), np.float32)
    wt = tile_weights()
    pos = tiles(H, W)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = np.stack([x[:, y:y + TILE, xx:xx + TILE] for y, xx in chunk])
        logits = run_batch(xb)
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        for (y, xx), pi in zip(chunk, p):
            acc[:, y:y + TILE, xx:xx + TILE] += pi * wt
            norm[y:y + TILE, xx:xx + TILE] += wt
    return (acc / norm)[:, :h, :w]


def torch_runner(model, dev):
    @torch.no_grad()
    def run(xb):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            return model(torch.from_numpy(xb).to(dev)).float().cpu().numpy()
    return run


# ---------------------------------------------------------------- commands

def cmd_train(args):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    sp = split()
    data = {s: load_xy(s) for s in sp["train"]}
    held = {s: load_xy(s) for s in sp["held_out"]}
    print(f"train {len(data)} samples, held-out {sp['held_out']}", flush=True)
    model = build_model().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = args.crops // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    keys = list(data)
    best = -1
    for ep in range(args.epochs):
        model.train()
        t, tot = time.time(), 0.0
        for _ in range(steps_per_epoch):
            xb, yb = [], []
            for _ in range(args.batch):
                x, y = data[keys[rng.integers(len(keys))]]
                h, w = y.shape
                y0, x0 = rng.integers(0, h - TILE + 1), rng.integers(0, w - TILE + 1)
                xb.append(x[:, y0:y0 + TILE, x0:x0 + TILE]); yb.append(y[y0:y0 + TILE, x0:x0 + TILE])
            xt = torch.from_numpy(np.stack(xb)).to(dev, non_blocking=True)
            yt = torch.from_numpy(np.stack(yb)).to(dev, non_blocking=True).long()
            xt, yt = augment(xt, yt, gen)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xt)
                loss = F.cross_entropy(logits.float(), yt, ignore_index=C.EXCLUDED) + dice_loss(logits.float(), yt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        msg = f"epoch {ep + 1}/{args.epochs} loss {tot / steps_per_epoch:.4f} ({time.time() - t:.0f}s)"
        if (ep + 1) % 3 == 0 or ep + 1 == args.epochs:
            model.eval()
            agree = evaluate(model, held, dev)
            msg += f" | held-out agreement with confident teacher px {100 * agree:.2f}%"
            if agree > best:
                best = agree
                torch.save(model.state_dict(), CKPT)
        print(msg, flush=True)
    print(f"best held-out agreement {100 * best:.2f}% -> {CKPT}")


def evaluate(model, held, dev):
    run = torch_runner(model, dev)
    hit = n = 0
    for x, y in held.values():
        pred = segment(x, run).argmax(0)
        m = y != C.EXCLUDED
        hit += int((pred[m] == y[m]).sum()); n += int(m.sum())
    return hit / n


def cmd_predict(args):
    dev = torch.device("cuda")
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    run = torch_runner(model, dev)
    for _, sid in C.sample_ids():
        ch = C.load_channels(sid, ["BSE", "Inlens"])
        x = np.stack([ch["BSE"], ch["Inlens"]], 0).astype(np.float32)
        torch.cuda.synchronize(); t = time.time()
        probs = segment(x, run)
        torch.cuda.synchronize(); dt = time.time() - t
        from .physics import constrain_siox
        from .teacher import clean
        exclude = C.load_exclude(sid)
        probs = constrain_siox(probs, ch["BSE"], exclude, class_axis=0)
        pred = clean(probs.argmax(0).astype(np.uint8), exclude)
        np.save(C.proc_dir(sid) / "student_labels.npy", pred)
        np.save(C.proc_dir(sid) / "student_maxp.npy", (probs.max(0) * 255).astype(np.uint8))
        print(f"{sid}: {dt:.2f}s", flush=True)


def cmd_export(args):
    import onnxruntime as ort
    dev = torch.device("cuda")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, IN_CH, TILE, TILE)
    torch.onnx.export(model, dummy, str(ONNX), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    sid = "kbdh4tri"
    ch = C.load_channels(sid, ["BSE", "Inlens"])
    x = np.stack([ch["BSE"], ch["Inlens"]], 0).astype(np.float32)
    timings = {}
    gpu = model.to(dev)
    run = torch_runner(gpu, dev)
    segment(x, run)                                     # warm-up
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3):
        p_gpu = segment(x, run)
    torch.cuda.synchronize(); timings["gpu_torch_s"] = round((time.time() - t) / 3, 3)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    run_cpu = lambda xb: sess.run(None, {"x": xb})[0]
    segment(x, run_cpu, batch=4)
    t = time.time()
    p_cpu = segment(x, run_cpu, batch=4)
    timings["cpu_onnx_s"] = round(time.time() - t, 3)
    timings["image_shape_halfres"] = list(x.shape[1:])
    timings["gpu_vs_cpu_label_agreement"] = round(float((p_gpu.argmax(0) == p_cpu.argmax(0)).mean()), 5)
    timings["device"] = torch.cuda.get_device_name(0)
    C.write_json(C.OUT_MET / "inference_timing.json", timings)
    print(timings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "export"])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--crops", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()

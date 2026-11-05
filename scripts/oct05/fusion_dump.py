#!/usr/bin/env python3
"""D3 (declared 2026-10-05 in docs/TODO.md): tile-fusion / decode variants of ONE checkpoint, dumped as eval-frame
probability maps (float16, key 'prob', the layout eval_v5.py --pred-dir reads). Inference only, VAL only.

Variants (fusion x decode):
  A_decl   uniform PROBABILITY mean, stride 387, probabilities resampled x1.5   (the declared path; must reproduce the cache)
  A_c      same fusion, LOGITS resampled then sigmoid                            (decode variant c)
  UL387_c  uniform LOGIT mean, stride 387, logit resample
  G256_a   Gaussian-weighted logit mean, stride 256 (50 %), probability resample
  G256_c   same, logit resample
  G128_c   Gaussian-weighted logit mean, stride 128 (75 %), logit resample
  WHOLE_c  one forward pass on the reflect-padded frame, logit resample

    python fusion_dump.py CKPT DATA SPLIT OUT_ROOT TAG [REF_PRED_DIR]
"""
import glob, os, sys
import numpy as np
import torch
from scipy.ndimage import zoom
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from train_v5 import TILE_FOR, build_model, norm01, IMA_M, IMA_S, DEV  # noqa: E402
from eval_v5 import eval_frame_shape, resample_to_eval  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402

ckpt, data, split, out_root, tag = sys.argv[1:6]
ref = sys.argv[6] if len(sys.argv) > 6 else None
TILE = TILE_FOR["nnunet"]; S0 = int(round(TILE * 392 / 518))
state = torch.load(ckpt, map_location="cuda")
out_ch = int([v for k, v in state.items() if "decoder.seg_layers." in k and k.endswith(".weight")][0].shape[0])
assert out_ch == 1, out_ch
model = build_model("nnunet", out_ch=1, in_ch=3).to("cuda").eval()
print("load", model.load_state_dict(state, strict=False), flush=True)
LCLIP = 30.0


@torch.no_grad()
def fwd(t):
    tt = torch.from_numpy(t.astype(np.float32))[None].repeat(3, 1, 1)
    tt = ((tt - IMA_M) / IMA_S)[None].to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(tt)
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o.float()[0, 0].cpu().numpy()


def gauss(th, tw):
    y = np.arange(th) - (th - 1) / 2; x = np.arange(tw) - (tw - 1) / 2
    g = np.exp(-(y[:, None] ** 2) / (2 * (TILE / 8) ** 2) - (x[None] ** 2) / (2 * (TILE / 8) ** 2))
    return np.maximum(g, g.max() * 1e-3)


def tiles(H, W, stride):
    ys = list(range(0, max(1, H - TILE + 1), stride)) or [0]
    xs = list(range(0, max(1, W - TILE + 1), stride)) or [0]
    if ys[-1] != max(0, H - TILE): ys.append(max(0, H - TILE))
    if xs[-1] != max(0, W - TILE): xs.append(max(0, W - TILE))
    return ys, xs


def fuse(img, stride):
    H, W = img.shape; ys, xs = tiles(H, W, stride)
    pu = np.zeros((H, W)); lu = np.zeros((H, W)); lg = np.zeros((H, W)); cn = np.zeros((H, W)); wg = np.zeros((H, W))
    for y in ys:
        for x in xs:
            t = img[y:y + TILE, x:x + TILE]; th, tw = t.shape
            lo = np.clip(fwd(t)[:th, :tw], -LCLIP, LCLIP); g = gauss(th, tw)
            pu[y:y + th, x:x + tw] += 1 / (1 + np.exp(-lo)); lu[y:y + th, x:x + tw] += lo
            lg[y:y + th, x:x + tw] += lo * g; cn[y:y + th, x:x + tw] += 1; wg[y:y + th, x:x + tw] += g
    cn = np.maximum(cn, 1)
    return pu / cn, lu / cn, lg / wg


def whole(img):
    H, W = img.shape; ph, pw = (-H) % 128, (-W) % 128
    return np.clip(fwd(np.pad(img, ((0, ph), (0, pw)), mode="reflect"))[:H, :W], -LCLIP, LCLIP)


sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def to_eval(m, hw, as_logit):
    if as_logit:
        return sig(resample_to_eval(m[None].astype(np.float32), hw))
    return resample_to_eval(m[None].astype(np.float32), hw)


def save(v, nm, ch):
    d = os.path.join(out_root, f"FUS_{v}_{tag}"); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, nm.replace(".h5", ".npz")), prob=ch.astype(np.float16))


worst = 0.0; n = 0
for p in sorted(glob.glob(os.path.join(data, "*.h5"))):
    fr = read_frame_h5(p)
    if str(fr["attrs"].get("split")) != split:
        continue
    nm = os.path.basename(p); im = norm01(np.asarray(fr["image"], float)); hw = eval_frame_shape(im.shape)
    pu, lu, _ = fuse(im, S0); _, _, g256 = fuse(im, 256); _, _, g128 = fuse(im, 128)
    a = to_eval(pu, hw, False)
    save("A_decl", nm, a); save("A_c", nm, to_eval(logit(pu), hw, True)); save("UL387_c", nm, to_eval(lu, hw, True))
    save("G256_a", nm, to_eval(sig(g256), hw, False)); save("G256_c", nm, to_eval(g256, hw, True))
    save("G128_c", nm, to_eval(g128, hw, True))
    try:
        save("WHOLE_c", nm, to_eval(whole(im), hw, True))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); print("OOM whole", nm, im.shape, flush=True)
    if ref and os.path.exists(os.path.join(ref, nm.replace(".h5", ".npz"))):
        r = np.load(os.path.join(ref, nm.replace(".h5", ".npz")))["prob"].astype(np.float32)
        worst = max(worst, float(np.abs(r - a.astype(np.float16).astype(np.float32)).max()))
    n += 1
    if n % 10 == 0: print(n, nm, "max|A_decl - cache| so far", worst, flush=True)
print("DONE", n, "frames; max |A_decl - cached declared dump| =", worst, flush=True)

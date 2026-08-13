#!/usr/bin/env python3
"""Train the semantic foreground with TEMPORAL context. RUNS ON KAJMAN / PANDA (free L40S).

The measurement that motivates this: on synthetic sequences the tracker's identity switch rate
is 0.6-2.3 % and barely moves between a perfect foreground and the real one -- association is
solved -- while **tracks per object doubles from ~1.7 to ~3.4** and completeness falls from
0.65 to 0.48. Each microtubule is followed by three disconnected tracks instead of one. The
failure is the foreground fragmenting, and a fragment caused by noise in frame t is not a
fragment in t+-1. Temporal context is the information that can repair exactly that.

No architecture change is needed, and that is the point
-------------------------------------------------------
``DinoSeg`` already takes **three** input channels: the frozen DINOv2 backbone expects RGB and
the high-resolution branch starts with ``Conv2d(3, 48, ...)``. Today the same grayscale frame
is replicated into all three. Putting ``(t-1, t, t+1)`` there instead makes the temporal model
the same network, and makes the **single frame the exact degenerate case** ``(t, t, t)`` rather
than an approximation of one. A separate video model that drifts from the still model is the
failure this project's deployed wrapper already documents.

Two consequences worth stating rather than discovering later:

* ImageNet normalisation uses a different mean/std per channel, so t-1 and t+1 are scaled
  slightly differently -- the temporal axis is not symmetric. Keeping it is deliberate: the
  backbone is frozen and was pretrained with those statistics, and single-frame mode has to
  stay bit-identical to the deployed model. The decoder is free to absorb the offset.
* Every augmentation -- crop, scale, flip, polarity inversion -- must be applied IDENTICALLY to
  all three frames. Augmenting them independently would destroy the temporal relationship the
  model is supposed to learn, while still looking like it trains.

Training mixes temporal and single-frame batches (``--p-single``) so one network serves both.

    ssh prusek@kajman.utia.cas.cz
    cd /home/prusek/mt_enc_exp/mt34_work
    SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base \
    CALIB=/home/prusek/mt_enc_exp/calib_reg418_morph.json \
    MASK_HW=1.0 POS_W=8 CLDICE_W=0.1 \
    PYTHONPATH=src:synth ~/dinov3_env/bin/python scripts/train_temporal.py --epochs 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "synth"))

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

from cbdice_loss import soft_cbdice  # noqa: E402
from dino_seg import IMA_M, IMA_S, DinoSeg, dice, soft_cldice  # noqa: E402
from mt_bench.fg_quality import (foreground_quality, mean_properties,  # noqa: E402
                                 passes_overfiring_gate, quality_score, select_checkpoint)
from mt_sequence import MotionConfig, generate_sequence  # noqa: E402
from train_gated import load_val, prob_channels  # noqa: E402

DEV = "cuda"
UP = 1.5
THR = 0.35
#: Background pool. The canonical copy lives under ``BIOCEV/datasets``, which is a symlink to
#: ``/disk2`` -- **tulen-local storage that kajman and panda cannot see**, even though the home
#: directory is NFS-shared between them. The first entry is a copy on the shared home so any of
#: the three machines can train; the tulen path is kept as a fallback.
BG_DIRS = [
    "/home/prusek/mt_enc_exp/irm_backgrounds_v2",
    "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
]


def background_paths() -> list:
    """First background directory that actually has files on THIS machine."""
    for d in BG_DIRS:
        found = sorted(glob.glob(os.path.join(d, "*.tif")))
        if found:
            return found
    raise SystemExit(
        "no backgrounds found in any of: " + ", ".join(BG_DIRS) + "\n"
        "On kajman/panda the BIOCEV path is a dead symlink into tulen-local /disk2.")


class TemporalOnlineDS(torch.utils.data.Dataset):
    """Fresh 3-frame synthetic sequences, with the SAME augmentation applied to every frame.

    Mirrors ``dino_seg.OnlineDS`` step for step -- same background sampling, same percentile
    normalisation, same polarity inversion and flips -- and differs only in generating a
    sequence instead of a frame, and in supervising on the MIDDLE frame.
    """

    def __init__(self, cfg, bg_paths, ori_fn, crop=518, gen=640, epoch_len=5000,
                 p_single=0.35, mcfg=None):
        self.cfg, self.bg, self.ori_fn = cfg, bg_paths, ori_fn
        self.crop, self.gen, self.epoch_len = crop, gen, epoch_len
        self.p_single = p_single
        self.mcfg = mcfg or MotionConfig()

    def __len__(self):
        return self.epoch_len

    def _background(self, rng):
        g = self.gen
        b = None
        for _ in range(8):
            b = np.asarray(Image.open(self.bg[int(rng.integers(len(self.bg)))]).convert("F"),
                           np.float32)
            while b.ndim > 2:
                b = b[..., 0]
            if b.shape[0] >= g and b.shape[1] >= g:
                break
        if b is None or b.shape[0] < g or b.shape[1] < g:
            b = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))),
                           np.float32)
        H, W = b.shape
        y0, x0 = int(rng.integers(0, H - g + 1)), int(rng.integers(0, W - g + 1))
        return b[y0:y0 + g, x0:x0 + g]

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng((((wid + 1) * 1000003) * (i + 1)) ^ random.getrandbits(48))
        g = self.gen
        bg = self._background(rng)

        imgs, per_frame, _ = generate_sequence(bg, rng, self.cfg, n_frames=3, mcfg=self.mcfg)
        mid = len(imgs) // 2
        stack = []
        for im in imgs:
            lo, hi = np.percentile(im, [1, 99])
            stack.append(np.clip((im - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32))
        stack = np.stack(stack)                                    # (3, g, g)
        gt = self.ori_fn(per_frame[mid], (g, g)).astype(np.float32)

        # --- augmentation, applied to the WHOLE stack at once ---------------------------
        if random.random() < 0.5:                                  # polarity inversion
            stack = 1.0 - stack
        c = self.crop
        y, x = random.randint(0, g - c), random.randint(0, g - c)
        stack = stack[:, y:y + c, x:x + c]
        gt = gt[:, y:y + c, x:x + c]
        if random.random() < 0.5:
            stack = stack[:, :, ::-1].copy()
            gt = gt[::-1, :, ::-1].copy()
        if random.random() < 0.5:
            stack = stack[:, ::-1].copy()
            gt = gt[::-1, ::-1, :].copy()

        # --- temporal dropout: the single frame is the degenerate case ------------------
        if random.random() < self.p_single:
            stack = np.repeat(stack[mid:mid + 1], 3, axis=0)

        t = (torch.from_numpy(stack) - IMA_M) / IMA_S
        return t, torch.from_numpy(gt)


@torch.no_grad()
def validate_single_frame(model, frames, empty) -> dict:
    """The hard gate: quality on ONE frame, which is what the deployed model does today.

    ``prob_channels`` replicates the frame into all three channels, i.e. exactly the degenerate
    case, so this measures the temporal model in single-frame mode without a separate path.
    """
    from scipy.ndimage import zoom
    from train_gated import norm01
    model.eval()
    per = []
    for f in frames:
        img = zoom(norm01(np.asarray(f["image"], dtype=float)), UP, order=1)
        mask = prob_channels(model, img).max(axis=0) > THR
        per.append(foreground_quality(mask, f["polylines"], up=UP))
    props = mean_properties(per)
    if empty:
        fg_e = []
        for f in empty:
            img = zoom(norm01(np.asarray(f["image"], dtype=float)), UP, order=1)
            fg_e.append(float((prob_channels(model, img).max(axis=0) > THR).mean()))
        props["fg_empty"] = float(np.mean(fg_e))
        props["fg"] = max(props.get("fg", 0.0), props["fg_empty"])
    model.train()
    model.dino.eval()
    return props


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--val-data", default="data/real/mt34_eval")
    ap.add_argument("--ckpt-dir", default="/disk2/prusek/temporal_ckpt")
    ap.add_argument("--out", default="/home/prusek/mt_enc_exp/dino_seg_ori_temporal.pth")
    ap.add_argument("--report", default="data/enc_sensitivity_testset/train_temporal.json")
    ap.add_argument("--epoch-len", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--p-single", type=float, default=0.35,
                    help="fraction of batches shown as (t,t,t); keeps one net serving both")
    ap.add_argument("--drift-px", type=float, default=1.0)
    ap.add_argument("--synth-fg", type=float, default=0.016)
    ap.add_argument("--cbdice-w", type=float, default=0.0,
                    help="weight on centerline-BOUNDARY Dice. It targets the measured failure "
                         "directly -- the mask is shattered into 7x too many components while "
                         "coverage, width and branch topology all match the oracle -- by "
                         "weighting the topological terms with the local radius, so a break in "
                         "a two-pixel filament costs more than the same break in a thick one.")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    frames, empty = load_val(args.val_data)
    if not frames:
        raise SystemExit(f"no VAL frames under {args.val_data}")
    print(f"VAL: {len(frames)} real frames + {len(empty)} empty probe(s); "
          f"selection only -- training data is synthetic", flush=True)

    from gen_train import build_cfg, ori_channels
    params = json.load(open(os.environ["CALIB"]))["best_params"] if os.environ.get("CALIB") else {}
    cfg = build_cfg(params, mask_hw=float(os.environ.get("MASK_HW", "1.0")))
    bgs = background_paths()
    print(f"backgrounds: {len(bgs)} from {os.path.dirname(bgs[0])}", flush=True)
    ds = TemporalOnlineDS(cfg, bgs, ori_channels, epoch_len=args.epoch_len,
                          p_single=args.p_single,
                          mcfg=MotionConfig(drift_px_std=args.drift_px))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     drop_last=True, persistent_workers=True, prefetch_factor=3)

    model = DinoSeg().to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 1e-3)
    pos_w = torch.tensor(float(os.environ.get("POS_W", "1")), device=DEV)
    cldice_w = float(os.environ.get("CLDICE_W", "0.5"))
    aux_w = float(os.environ.get("AUX_W", "0.4"))

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        model.dino.eval()
        tot = n = 0
        for im, mk in dl:
            im, mk = im.to(DEV), mk.to(DEV)
            out = model(im)
            aux = None
            if isinstance(out, (tuple, list)):
                out, aux = out

            def sloss(o):
                base = (F.binary_cross_entropy_with_logits(o, mk, pos_weight=pos_w)
                        + dice(o, mk) + cldice_w * soft_cldice(o, mk))
                # cbDice saturates once the thresholded mask is topologically right, so it is
                # ADDED to the pixel losses rather than replacing any of them.
                return base + (args.cbdice_w * soft_cbdice(o, mk) if args.cbdice_w > 0 else 0.0)

            loss = sloss(out) + (aux_w * sloss(aux) if aux is not None else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"epoch {ep}/{args.epochs} loss={tot / max(n, 1):.4f}", flush=True)

        if ep % args.val_every == 0 or ep == args.epochs:
            props = validate_single_frame(model, frames, empty)
            ckpt = os.path.join(args.ckpt_dir, f"ep{ep:03d}.pth")
            # Trainable params only: requires_grad=False keeps the frozen ViT-L out of the
            # gradients but NOT out of state_dict(), so a full save is 1.2 GB per epoch.
            torch.save({k: v for k, v in model.state_dict().items()
                        if not k.startswith("dino.")}, ckpt)
            props.update({"epoch": ep, "ckpt": ckpt, "loss": tot / max(n, 1)})
            history.append(props)
            print(f"  VAL(single-frame) score={quality_score(props):.3f} "
                  f"cc/gt={props.get('cc_per_gt', float('nan')):.2f} "
                  f"gaps/mt={props.get('gaps_per_mt', float('nan')):.2f} | "
                  f"fg={100 * props.get('fg', float('nan')):.2f}% "
                  f"rec2={props.get('rec2', float('nan')):.3f} | "
                  f"gate={'PASS' if passes_overfiring_gate(props, args.synth_fg) else 'FAIL'}",
                  flush=True)
            with open(args.report, "w") as fh:
                json.dump({"history": history, "synth_fg": args.synth_fg,
                           "p_single": args.p_single}, fh, indent=1)

    best = select_checkpoint(history, synth_fg=args.synth_fg, min_frames=len(frames))
    print("\n=== selection (single-frame quality is the gate) ===", flush=True)
    if best is None:
        print("NO checkpoint passed -- nothing selected.", flush=True)
        raise SystemExit(2)
    chosen = history[best]
    print(f"  epoch {chosen['epoch']}  continuity {quality_score(chosen):.3f}  "
          f"(v4b reference on this split: 0.841)", flush=True)
    model.load_state_dict(torch.load(chosen["ckpt"], map_location=DEV), strict=False)
    torch.save(model.state_dict(), args.out)
    with open(args.report, "w") as fh:
        json.dump({"history": history, "selected": chosen, "out": args.out,
                   "p_single": args.p_single, "synth_fg": args.synth_fg}, fh, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

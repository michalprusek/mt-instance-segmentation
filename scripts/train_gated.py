#!/usr/bin/env python3
"""Train the semantic foreground with fg_quality-GATED checkpoint selection. RUNS ON TULEN.

Why a new script rather than a flag on ``dino_seg.py``: that trainer has **no validation at
all**. It runs N epochs and overwrites one fixed path with the final weights, so "which
checkpoint" has never been a decision the project could make -- and §17k showed the metric it
would have used (tolerant coverage F1) ranks foregrounds at chance anyway. Everything about
the model, the data and the loss is imported from ``dino_seg`` unchanged; the only additions
are per-epoch checkpoints, a validation pass, and a selection rule.

The selection rule is ``mt_bench.fg_quality.select_checkpoint``: minimise the continuity score
(``cc_per_gt`` / ``gaps_per_mt`` / ``endp_per_kpx``, 0.79-0.82 ranking accuracy) SUBJECT TO an
over-firing ceiling and a collapse floor. Unconstrained, all three metrics are maximised by a
mask that floods the frame -- see that module for the measurement.

Validation runs on the **real MT-34 VAL split**. Training data stays 100% synthetic; VAL is
used only to pick a checkpoint, which is the same licence the instancer's hyperparameters
already have, and TEST is untouched. Every epoch's metrics are written to the run json, so the
counterfactual -- what "last epoch" or "best coverage F1" would have selected -- is
recoverable without retraining. That comparison is the experiment.

    cd /home/prusek/mt_enc_exp/mt34_work
    SEG_MODE=ori SEG_BACKBONE=dinov2 SEG_INPUT=raw SEG_ARCH=base \
    ONLINE=1 CALIB=/home/prusek/mt_enc_exp/calib_reg418_morph.json \
    MASK_HW=1.0 POS_W=8 CLDICE_W=0.1 EPOCH_LEN=5000 NWORKERS=10 \
    PYTHONPATH=src ~/dinov3_env/bin/python scripts/train_gated.py --epochs 30 --val-every 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")

from dino_seg import DinoSeg, IMA_M, IMA_S, dice, soft_cldice  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fg_quality import (foreground_quality, mean_properties,  # noqa: E402
                                 passes_overfiring_gate, quality_score, select_checkpoint)

DEV = "cuda"
TILE, STRIDE = 518, 392          # identical to learn_amodal.prob_channels
UP = 1.5                         # the eval frame everything else lives in
THR = 0.35


def norm01(a, p=(1, 99)):
    """Whole-frame percentile stretch -- ``learn_amodal.norm01`` verbatim.

    Deliberately NOT the FOV-based variant: it was tested on VAL and lost (0.412 vs 0.438).
    Prediction-time preprocessing must match what the eval pipeline does, or the gate would be
    selecting checkpoints for a distribution nothing else sees.
    """
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


@torch.no_grad()
def prob_channels(model, img01: np.ndarray) -> np.ndarray:
    """Tiled K-channel prediction from a LIVE model (no checkpoint round-trip)."""
    H, W = img01.shape
    out_ch = None
    ys = list(range(0, max(1, H - TILE + 1), STRIDE))
    xs = list(range(0, max(1, W - TILE + 1), STRIDE))
    if ys[-1] != H - TILE:
        ys.append(max(0, H - TILE))
    if xs[-1] != W - TILE:
        xs.append(max(0, W - TILE))
    acc = cnt = None
    for y in ys:
        for x in xs:
            t = img01[y:y + TILE, x:x + TILE]
            th, tw = t.shape
            tt = torch.from_numpy(t.astype(np.float32))[None].repeat(3, 1, 1)
            tt = ((tt - IMA_M) / IMA_S)[None].to(DEV)
            o = model(tt)
            if isinstance(o, (tuple, list)):
                o = o[0]
            o = torch.sigmoid(o)[0].cpu().numpy()
            if acc is None:
                out_ch = o.shape[0]
                acc = np.zeros((out_ch, H, W))
                cnt = np.zeros((H, W))
            acc[:, y:y + th, x:x + tw] += o[:, :th, :tw]
            cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)[None]


def load_val(data_dir: str, split: str = "val"):
    """VAL frames, split into those with GT and the EMPTY ones.

    ``training_img_102`` has zero annotated microtubules and lands in VAL. It cannot enter the
    continuity battery (there is no microtubule to be continuous along), but it is the purest
    over-firing probe the benchmark has: every predicted pixel there is a false positive.
    """
    frames, empty = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(path)
        if str(fr["attrs"].get("split")) != split:
            continue
        row = {"name": os.path.basename(path), "image": fr["image"],
               "polylines": fr["polylines"]}
        (frames if fr["polylines"] else empty).append(row)
    return frames, empty


@torch.no_grad()
def _fg_mask(model, image) -> np.ndarray:
    img = zoom(norm01(np.asarray(image, dtype=float)), UP, order=1)
    return prob_channels(model, img).max(axis=0) > THR


@torch.no_grad()
def validate(model, frames, empty) -> dict:
    """Foreground-quality battery averaged over the VAL frames. Cheap: no instancer."""
    model.eval()
    per_frame = [foreground_quality(_fg_mask(model, f["image"]), f["polylines"], up=UP)
                 for f in frames]
    props = mean_properties(per_frame)
    if empty:
        fg_empty = float(np.mean([_fg_mask(model, f["image"]).mean() for f in empty]))
        props["fg_empty"] = fg_empty
        # The ceiling applies to the WORST observed foreground fraction. An annotated frame
        # hides over-firing behind real microtubules; an empty field cannot.
        props["fg"] = max(props.get("fg", 0.0), fg_empty)
    model.train()
    model.dino.eval()
    return props


def build_loader(batch_size: int = 6):
    """Exactly ``dino_seg.main``'s data path -- online generation or the on-disk set."""
    if os.environ.get("ONLINE"):
        sys.path.insert(0, "/home/prusek/mt_enc_exp/synth")
        from gen_train import build_cfg, ori_channels
        from mt_generator import generate_frame

        from dino_seg import OnlineDS
        params = json.load(open(os.environ["CALIB"]))["best_params"]
        mhw = float(os.environ.get("MASK_HW", "1.0"))
        if os.environ.get("DR") == "1":
            from dr_cfg import build_cfg_dr
            cfg = build_cfg_dr(params, mask_hw=mhw)
        else:
            cfg = build_cfg(params, mask_hw=mhw)
        bg = sorted(glob.glob(
            "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2/*.tif"))
        ds = OnlineDS(cfg, bg, generate_frame, ori_channels,
                      epoch_len=int(os.environ.get("EPOCH_LEN", "5000")))
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=int(os.environ.get("NWORKERS", "10")),
            drop_last=True, persistent_workers=True, prefetch_factor=3)
    from dino_seg import DS
    return torch.utils.data.DataLoader(DS(), batch_size=batch_size, shuffle=True,
                                       num_workers=4, drop_last=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--val-data", default="data/real/mt34_eval")
    ap.add_argument("--ckpt-dir", default="/home/prusek/mt_enc_exp/gated_ckpt")
    ap.add_argument("--out", default="/home/prusek/mt_enc_exp/dino_seg_ori_gated.pth")
    ap.add_argument("--report", default="data/enc_sensitivity_testset/train_gated.json")
    ap.add_argument("--synth-fg", type=float, default=0.016,
                    help="mean in-domain synthetic foreground fraction; the ceiling is 3x it")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    frames, empty = load_val(args.val_data)
    if not frames:
        raise SystemExit(f"no VAL frames with GT under {args.val_data}")
    print(f"VAL: {len(frames)} real frames + {len(empty)} empty-field over-firing probe(s) "
          f"(selection only -- training data is synthetic)", flush=True)

    dl = build_loader()
    model = DinoSeg().to(DEV)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 1e-3)
    pos_w = torch.tensor(float(os.environ.get("POS_W", "1")), device=DEV)
    cldice_w = float(os.environ.get("CLDICE_W", "0.5"))
    aux_w = float(os.environ.get("AUX_W", "0.4"))

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        model.dino.eval()
        tot = 0.0
        for im, mk in dl:
            im, mk = im.to(DEV), mk.to(DEV)
            out = model(im)
            aux = None
            if isinstance(out, (tuple, list)):
                out, aux = out

            def sloss(o):
                return (F.binary_cross_entropy_with_logits(o, mk, pos_weight=pos_w)
                        + dice(o, mk) + cldice_w * soft_cldice(o, mk))

            loss = sloss(out) + (aux_w * sloss(aux) if aux is not None else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep}/{args.epochs} loss={tot / len(dl):.4f}", flush=True)

        if ep % args.val_every == 0 or ep == args.epochs:
            props = validate(model, frames, empty)
            ckpt = os.path.join(args.ckpt_dir, f"ep{ep:03d}.pth")
            # Trainable params only (~7 MB vs 1.23 GB): `requires_grad=False` keeps the frozen
            # DINOv2 ViT-L out of the gradients but NOT out of `state_dict()`, so a full save
            # per epoch would write ~20 GB of byte-identical backbone onto the NFS home. The
            # winner is re-expanded to a full state_dict below, because learn_amodal loads
            # strictly.
            torch.save({k: v for k, v in model.state_dict().items()
                        if not k.startswith("dino.")}, ckpt)
            props.update({"epoch": ep, "ckpt": ckpt, "loss": tot / len(dl)})
            history.append(props)
            gate = passes_overfiring_gate(props, args.synth_fg)
            print(f"  VAL score={quality_score(props):.3f} "
                  f"cc/gt={props.get('cc_per_gt', float('nan')):.2f} "
                  f"gaps/mt={props.get('gaps_per_mt', float('nan')):.2f} "
                  f"endp/kpx={props.get('endp_per_kpx', float('nan')):.1f} | "
                  f"fg={100 * props.get('fg', float('nan')):.2f}% "
                  f"rec2={props.get('rec2', float('nan')):.3f} "
                  f"prec2={props.get('prec2', float('nan')):.3f} | "
                  f"gate={'PASS' if gate else 'FAIL'}", flush=True)
            with open(args.report, "w") as fh:
                json.dump({"history": history, "synth_fg": args.synth_fg}, fh, indent=1)

    # ``min_frames`` is mandatory here: without it a checkpoint that predicts nothing on most
    # VAL frames is scored only on the few it fired on, and can win. See fg_quality.
    best = select_checkpoint(history, synth_fg=args.synth_fg, min_frames=len(frames))
    print("\n=== selection ===", flush=True)
    if best is None:
        # A real outcome, not a crash: every checkpoint over-fired or collapsed. Shipping the
        # least-bad one would reintroduce exactly the failure the gate exists to catch.
        print("NO checkpoint passed the over-firing/collapse gate -- nothing selected.",
              flush=True)
        raise SystemExit(2)

    chosen = history[best]
    last = history[-1]

    def cov_f1(h) -> float:
        """The selection metric the project has always used: tolerant coverage F1."""
        p, r = h.get("prec2", 0.0), h.get("rec2", 0.0)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # The counterfactual, on the same run: dino_seg.py would have shipped the final epoch, and
    # the project's usual metric would have picked the best coverage F1. If all three coincide
    # the gate changed nothing -- a result worth reporting, not one worth hiding.
    by_cov = max(history, key=cov_f1)
    for label, h in (("fg_quality gate", chosen), ("'last epoch'", last),
                     ("'best coverage F1'", by_cov)):
        print(f"  {label:20s} -> epoch {h['epoch']:3d}  "
              f"continuity {quality_score(h):.3f}  covF1 {cov_f1(h):.3f}", flush=True)

    # Re-expand to a FULL state_dict: the frozen backbone is identical at every epoch, so
    # loading the winner's trainable params into the live model reconstructs it exactly.
    model.load_state_dict(torch.load(chosen["ckpt"], map_location=DEV), strict=False)
    torch.save(model.state_dict(), args.out)
    with open(args.report, "w") as fh:
        json.dump({"history": history, "synth_fg": args.synth_fg,
                   "selected": chosen, "last": last, "by_coverage": by_cov,
                   "out": args.out}, fh, indent=1)
    print(f"wrote {args.out} and {args.report}", flush=True)


if __name__ == "__main__":
    main()

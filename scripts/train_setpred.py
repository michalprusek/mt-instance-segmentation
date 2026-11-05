"""Train the curve set predictor on synthetic frames. Screen on synth before touching real data.

The decision rule this run exists to answer: does a set predictor beat instancer A's 0.695 on
SYNTHETIC data, where ground truth is exact and n is unlimited? If it does not win there, the
domain gap is not worth arguing about and the idea is dropped.

    PYTHONPATH=src:scripts python scripts/train_setpred.py --calib <calib.json> \
        --ckpt-dir <dir> --epochs 30
"""
from __future__ import annotations

import argparse, glob, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "synth"))

from build_synth_eval import clip_polyline  # noqa: E402
from setpred_model import CurveSetNet, curve_l1, resample_curve  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)


class CurveDS(torch.utils.data.Dataset):
    """Fresh synthetic frame per item; yields the image and its exact per-instance centerlines."""

    def __init__(self, cfg, bg_paths, gen_fn, crop=512, epoch_len=2000, n_points=16,
                 max_inst=400, min_len=20.0):
        self.cfg, self.bg, self.gen_fn = cfg, bg_paths, gen_fn
        self.crop, self.epoch_len = crop, epoch_len
        self.n_points, self.max_inst, self.min_len = n_points, max_inst, min_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng((((wid + 1) * 1000003) * (i + 1)) ^ random.getrandbits(48))
        c = self.crop
        for _ in range(8):
            b = np.asarray(Image.open(self.bg[int(rng.integers(len(self.bg)))]).convert("F"), np.float32)
            if b.shape[0] >= c and b.shape[1] >= c:
                break
        y0 = int(rng.integers(0, b.shape[0] - c + 1))
        x0 = int(rng.integers(0, b.shape[1] - c + 1))
        bg = b[y0:y0 + c, x0:x0 + c]
        img, instances, _ = self.gen_fn(bg, rng, self.cfg)

        curves = []
        for ins in instances:
            cl = np.asarray(ins["centerline"], dtype=np.float64)   # (N,2) = (x=col, y=row)
            if len(cl) < 2:
                continue
            # The generator returns the FULL geometric curve, including the part outside the
            # field of view -- centerlines reach col=-382 on a 512 px crop. Use the project's
            # own convention (longest contiguous run inside the frame, stubs dropped) so the
            # training target is exactly what build_synth_eval.py scores against; a boolean
            # mask would splice two separate visible runs into one curve that jumps the frame.
            seg = clip_polyline(cl, (c, c), self.min_len)
            if seg is None:
                continue
            k = resample_curve(torch.from_numpy(seg.astype(np.float32)), self.n_points)
            curves.append(k / float(c))                            # normalise to [0,1]
        if len(curves) > self.max_inst:
            curves = curves[:self.max_inst]
        tgt = torch.stack(curves) if curves else torch.zeros(0, self.n_points, 2)
        x = torch.from_numpy(norm01(img))[None].repeat(3, 1, 1)
        return x, tgt


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    return xs, [b[1] for b in batch]


def match_and_loss(obj, pts, targets, w_pt=5.0, w_obj=1.0, eos=0.1):
    """Hungarian assignment of predicted curves to GT, then L1 on matched pairs + BCE objectness.

    The matching lives ONLY in the loss -- inference emits the set directly -- so the whole model
    stays differentiable end to end and there is no combinatorial step to transfer to real data.
    """
    B, Q = obj.shape
    total_pt = obj.new_zeros(())
    tgt_obj = torch.zeros_like(obj)
    n_match = 0
    for b in range(B):
        t = targets[b].to(obj.device)
        if t.numel() == 0:
            continue
        cost = curve_l1(pts[b][:, None], t[None])                   # (Q, T), direction-invariant
        cost = cost - 0.05 * torch.sigmoid(obj[b])[:, None]         # prefer confident queries
        qi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
        qi = torch.as_tensor(qi, device=obj.device)
        ti = torch.as_tensor(ti, device=obj.device)
        total_pt = total_pt + curve_l1(pts[b][qi], t[ti]).sum()
        tgt_obj[b, qi] = 1.0
        n_match += len(qi)
    w = torch.where(tgt_obj > 0.5, torch.ones_like(tgt_obj), torch.full_like(tgt_obj, eos))
    loss_obj = (F.binary_cross_entropy_with_logits(obj, tgt_obj, reduction="none") * w).mean()
    loss_pt = total_pt / max(1, n_match)
    return w_obj * loss_obj + w_pt * loss_pt, float(loss_obj), float(loss_pt), n_match


def build_backbone(feat_ch: int):
    """nnU-Net's ResEnc-M trunk, reused as a feature extractor.

    Same topology v5H was trained with, so its weights warm-start everything except the head:
    the trunk already knows what a microtubule looks like, and only the curve decoder is new.
    """
    from train_v5 import build_nnunet
    return build_nnunet(feat_ch, deep_supervision=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--init", default=None, help="v5H checkpoint to warm-start the trunk from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-len", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--feat-ch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    from mt_generator import generate_frame
    from gen_train import build_cfg
    cal = json.load(open(args.calib))
    cfg = build_cfg(cal["best_params"], mask_hw=1.0)
    for k, v in (cal.get("gen_overrides") or {}).items():
        setattr(cfg, k, v)

    bgs = []
    for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
              "/disk2/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2"):
        bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
        if bgs:
            break
    if not bgs:
        raise SystemExit("no IRM backgrounds found")
    print(f"backgrounds: {len(bgs)}", flush=True)

    backbone = build_backbone(args.feat_ch)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        sd = sd.get("model", sd)
        own = backbone.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        backbone.load_state_dict(keep, strict=False)
        print(f"warm-started {len(keep)}/{len(own)} backbone tensors from {os.path.basename(args.init)}",
              flush=True)

    model = CurveSetNet(backbone, backbone_ch=args.feat_ch, n_queries=args.queries,
                        n_points=args.points, n_layers=args.layers).to(DEV)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model {n_par:.1f}M params, {args.queries} queries x {args.points} points", flush=True)

    ds = CurveDS(cfg, bgs, generate_frame, crop=args.crop, epoch_len=args.epoch_len,
                 n_points=args.points, max_inst=args.queries)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     collate_fn=collate, pin_memory=True, drop_last=True,
                                     persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (args.epoch_len // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")

    hist, step = [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = np.zeros(4)
        t0 = time.time()
        for x, tg in dl:
            x = x.to(DEV, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
                obj, pts = model(x)
                loss, lo, lp, nm = match_and_loss(obj.float(), pts.float(), tg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            agg += [float(loss), lo, lp, nm]; step += 1
        n = max(1, len(dl))
        rec = {"epoch": ep, "loss": agg[0] / n, "obj": agg[1] / n, "pt": agg[2] / n,
               "gt_per_frame": agg[3] / n / args.batch, "secs": time.time() - t0}
        hist.append(rec)
        print(f"epoch {ep}/{args.epochs} loss={rec['loss']:.4f} obj={rec['obj']:.4f} "
              f"pt={rec['pt']:.5f} gt/frame={rec['gt_per_frame']:.1f} {rec['secs']:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args)},
                   os.path.join(args.ckpt_dir, f"ep{ep:03d}.pth"))
        json.dump(hist, open(os.path.join(args.ckpt_dir, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Score the curve set predictor, and score instancer A on the SAME frames.

The comparison has to be head to head on identical images or it says nothing: instancer A's
historical 0.695 on synthetic data was measured on a set that no longer exists, and re-quoting it
against a freshly generated one would be exactly the kind of favourable mismatch that gets a paper
rejected. So both methods are run here, now, on the same frames, through the same metric.

    PYTHONPATH=src:scripts:synth python scripts/eval_setpred.py --ckpt <ep030.pth> --synth 40
    PYTHONPATH=src:scripts:synth python scripts/eval_setpred.py --ckpt <ep030.pth> \
        --data data/real/mt34_htw_eval --split val
"""
from __future__ import annotations

import argparse, glob, json, os, random, sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from instance.instancer_a import instance_a                      # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from instance.oracle import oracle_instance_masks                # noqa: E402
from mt_bench.cvat_import import read_frame_h5                   # noqa: E402
from build_synth_eval import clip_polyline                        # noqa: E402
from setpred_model import CurveSetNet                            # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL, LCOV, PCOV = 5.0, 0.95, 0.95
KAPPA_MAX = 0.25


def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    a = ck["args"]
    from train_setpred import build_backbone
    bb = build_backbone(a["feat_ch"])
    m = CurveSetNet(bb, backbone_ch=a["feat_ch"], n_queries=a["queries"],
                    n_points=a["points"], n_layers=a["layers"])
    m.load_state_dict(ck["model"])
    return m.to(DEV).eval(), a


@torch.no_grad()
def predict_curves(model, img01, obj_thr, min_len):
    """Return predicted polylines in PIXEL coordinates of the (square-padded) input."""
    h, w = img01.shape
    side = max(h, w)
    pad = np.zeros((side, side), np.float32)
    pad[:h, :w] = img01
    x = torch.from_numpy(pad)[None, None].repeat(1, 3, 1, 1).to(DEV)
    obj, pts = model(x)
    keep = torch.sigmoid(obj[0]) > obj_thr
    out = []
    for p in pts[0][keep].cpu().numpy():
        q = p * side                                   # normalised -> pixels
        d = np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
        if d >= min_len:
            out.append(q.astype(np.float64))
    return out


def score_frame(polys, gt, shape):
    masks = oracle_instance_masks(polys, shape, half_width=1.0, up=1.0)
    return centerline_f1(masks, gt, tol=TOL, length_coverage=LCOV, precision_coverage=PCOV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None, help="h5 benchmark dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--synth", type=int, default=0, help="score N freshly generated synth frames")
    ap.add_argument("--calib", default="/disk2/prusek/mt_work/calib_reg418_morph_snr3.json")
    ap.add_argument("--seed", type=int, default=9999, help="held out from training seeds")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--obj-thrs", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--min-len", type=float, default=20.0)
    ap.add_argument("--up", type=float, default=1.5)
    ap.add_argument("--vs-instancer", action="store_true",
                    help="also run v5H foreground + instancer A on the same frames")
    ap.add_argument("--fg-ckpt", default="/disk2/prusek/mt_work/weights/dino_seg_v5H.pth")
    ap.add_argument("--fg-thr", type=float, default=0.97)
    ap.add_argument("--params", default="src/instance/params_a_model_v5H.json")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    model, margs = load_model(args.ckpt)
    print(f"setpred: {margs['queries']} queries x {margs['points']} pts", flush=True)

    frames = []          # (name, img01_at_eval_scale, gt_polylines_at_eval_scale)
    if args.synth:
        sys.path.insert(0, os.path.join(_ROOT, "synth"))
        from mt_generator import generate_frame
        from gen_train import build_cfg
        cal = json.load(open(args.calib))
        cfg = build_cfg(cal["best_params"], mask_hw=1.0)
        for k, v in (cal.get("gen_overrides") or {}).items():
            setattr(cfg, k, v)
        bgs = []
        for d in ("/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2",
                  "/disk2/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2"):
            bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
            if bgs:
                break
        c = args.crop
        for i in range(args.synth):
            rng = np.random.default_rng(args.seed * 100003 + i)
            for _ in range(8):
                b = np.asarray(Image.open(bgs[int(rng.integers(len(bgs)))]).convert("F"), np.float32)
                if b.shape[0] >= c and b.shape[1] >= c:
                    break
            y0 = int(rng.integers(0, b.shape[0] - c + 1)); x0 = int(rng.integers(0, b.shape[1] - c + 1))
            img, inst, _ = generate_frame(b[y0:y0 + c, x0:x0 + c], rng, cfg)
            # Clip to the field of view exactly as build_synth_eval.py does. Without this the
            # reference contains curves that are mostly OUTSIDE the image (centerlines reach
            # col=-382 on a 512 px crop), no prediction can cover 95 % of their length, and the
            # score is 0 for a reason that has nothing to do with the model.
            gt = []
            for s_ in inst:
                cl = np.asarray(s_["centerline"], float)
                if len(cl) < 2:
                    continue
                seg = clip_polyline(cl, (c, c), args.min_len)
                if seg is not None:
                    gt.append(seg)
            frames.append((f"synth_{i:03d}", norm01(img), gt, 1.0))
    else:
        for p in sorted(glob.glob(os.path.join(args.data, "*.h5"))):
            fr = read_frame_h5(p)
            if str(fr["attrs"].get("split")) != args.split:
                continue
            im = zoom(norm01(np.asarray(fr["image"], float)), args.up, order=1)
            gt = [np.asarray(q, float) * args.up for q in fr["polylines"]]
            frames.append((os.path.basename(p), im, gt, args.up))
    print(f"{len(frames)} frames", flush=True)

    results = {}
    for ot in [float(t) for t in args.obj_thrs.split(",")]:
        rows = []
        for nm, im, gt, up in frames:
            polys = predict_curves(model, im, ot, args.min_len * up)
            r = score_frame(polys, gt, im.shape)
            r["name"] = nm
            r["task"] = "htw" if nm.startswith("htw_") else ("585" if nm.startswith("alice") else
                                                             ("synth" if nm.startswith("synth") else "586"))
            r["n_pred"] = len(polys)
            rows.append(r)
        agg = aggregate_benchmark(rows)
        agg["_frames"] = [{k: v for k, v in r.items() if k != "pairs"} for r in rows]
        results[f"{ot}"] = agg
        print(f"  obj_thr={ot:<4} F1={agg['mean_f1']:.4f} tp={agg['total_tp']} "
              f"fp={agg['total_fp']} fn={agg['total_fn']}", flush=True)

    out = {"ckpt": args.ckpt, "n_frames": len(frames), "setpred": results}

    if args.vs_instancer:
        print("--- instancer A on the SAME frames ---", flush=True)
        from train_v5 import build_nnunet, predict as fg_predict          # noqa
        sd = torch.load(args.fg_ckpt, map_location=DEV)
        sd = sd.get("model", sd)
        fg = build_nnunet(1, deep_supervision=False).to(DEV).eval()
        fg.load_state_dict(sd, strict=False)
        params = json.load(open(args.params)); params.pop("kappa_max", None)
        rows = []
        for nm, im, gt, up in frames:
            with torch.no_grad():
                x = torch.from_numpy(im)[None, None].repeat(1, 3, 1, 1).to(DEV)
                prob = torch.sigmoid(fg(x))[0, 0].cpu().numpy()
            mask = prob > args.fg_thr
            polys, _ = instance_a(mask, KAPPA_MAX, params, channels=prob[None], prob=prob)
            r = score_frame(polys, gt, im.shape)
            r["name"] = nm; r["task"] = "synth"; r["n_pred"] = len(polys)
            rows.append(r)
        aggA = aggregate_benchmark(rows)
        aggA["_frames"] = [{k: v for k, v in r.items() if k != "pairs"} for r in rows]
        out["instancer_a"] = aggA
        print(f"  instancer A  F1={aggA['mean_f1']:.4f} tp={aggA['total_tp']} "
              f"fp={aggA['total_fp']} fn={aggA['total_fn']}", flush=True)

    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=1, default=float)
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()

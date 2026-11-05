#!/usr/bin/env python3
"""Turn the manual ImageJ ROI annotation into a benchmark block, rejoining the fragments.

The annotation is genuine human work -- 16 051 traced curves over 303 crossing-dense frames -- but
it is BROKEN: the annotator traced filaments in interference microscopy, and wherever a microtubule
lifts off the coverslip its contrast passes through zero at h = lambda/(8*n) = 56.4 nm and reverses
sign, so the detached stretch does not read as the same filament and the trace stops. Debris and
crossings interrupt it the same way. One microtubule is therefore stored as two or more polylines.

Verified by eye before writing any code (`data/enc_sensitivity_testset/roi_joins_zoom.png`): the
gaps sit on dark stretches inside bright filaments, on bright debris, and at crossings, and the
filament visibly continues straight through every one of them.

Leaving that unrepaired would make the block score a method for finding whole microtubules as if
finding them were an error -- the metric's precision criterion punishes a prediction that spans a
gap the ground truth does not.

THE JOIN RULE. Two polyline ends are joined only when all of these hold, which is deliberately
stricter than proximity:
  1. they are within ``--max-gap`` pixels;
  2. their outgoing tangents are anti-parallel (cos <= ``--cos-tangent``), so each continues into
     the other rather than running alongside;
  3. the straight line between them agrees with both tangents (cos >= ``--cos-join``), so the
     curve carries through instead of doubling back;
  4. the curvature the join implies stays under the project's derived kappa_max -- a join sharper
     than a microtubule can physically bend is refused.
Each end may be used at most once, taken in order of increasing cost, so two filaments that merely
end near each other cannot both claim the same partner.

Everything is geometric and image-independent apart from the reported diagnostics, so the rule
cannot be tuned to flatter any method.

    python scripts/build_roi_dataset.py --xml <cvat_export/annotations.xml> \
        --img-dir <images> --out data/real/roi303_eval
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

KAPPA_MAX = 0.25          # rad/px at an 8 px baseline; derived, not tuned (see CLAUDE.md)


def densify(p, step=1.0):
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return p
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def read_xml(path):
    root = ET.parse(path).getroot()
    out = []
    for im in root.findall("image"):
        pl = []
        for p in im.findall("polyline"):
            q = np.array([[float(v) for v in t.split(",")] for t in p.get("points").split(";")])
            if len(q) >= 2:
                pl.append(q)
        out.append(dict(name=im.get("name"), W=int(im.get("width")),
                        H=int(im.get("height")), polys=pl))
    return out


def ends_of(polys, back=12.0):
    """(polyline index, which end, point, outgoing tangent) for every polyline end."""
    out = []
    for i, q in enumerate(polys):
        d = densify(q)
        if len(d) < 4:
            continue
        k = min(int(back), len(d) - 2)
        for side, (pt, ref) in enumerate(((d[0], d[k]), (d[-1], d[-1 - k]))):
            v = pt - ref
            n = np.linalg.norm(v)
            if n > 1e-6:
                out.append((i, side, pt, v / n))
    return out


def join_curvature(a_tan, b_tan, gap):
    """Turn angle the join implies, per pixel of gap -- compared against kappa_max."""
    cos = float(np.clip(-np.dot(a_tan, b_tan), -1, 1))
    return float(np.arccos(cos) / max(gap, 1.0))


def join_polylines(polys, max_gap, cos_tan, cos_join):
    """Return merged polylines and the list of accepted joins."""
    E = ends_of(polys)
    if len(E) < 2:
        return list(polys), []
    P = np.array([e[2] for e in E])
    tree = cKDTree(P)
    pairs = tree.query_pairs(max_gap, output_type="ndarray")
    cand = []
    for a, b in pairs:
        ia, sa, pa, ta = E[a]
        ib, sb, pb, tb = E[b]
        if ia == ib:
            continue                                  # never close a polyline onto itself
        u = pb - pa
        g = float(np.linalg.norm(u))
        if g < 1e-6:
            continue
        if float(np.dot(ta, tb)) > cos_tan:
            continue                                  # not anti-parallel: they run alongside
        uh = u / g
        if float(np.dot(ta, uh)) < cos_join or float(np.dot(tb, -uh)) < cos_join:
            continue                                  # the gap does not continue either curve
        if join_curvature(ta, tb, g) > KAPPA_MAX:
            continue                                  # sharper than a microtubule can bend
        cost = g * (2.0 - float(np.dot(ta, uh)) - float(np.dot(tb, -uh)))
        cand.append((cost, a, b, g))
    cand.sort()

    used, adj, joins = set(), {}, []
    for cost, a, b, g in cand:
        if a in used or b in used:
            continue                                  # each end may be claimed once
        ia, ib = E[a][0], E[b][0]
        # refuse a join that would close a loop of already-joined pieces
        seen, stack = {ia}, [ia]
        while stack:
            n = stack.pop()
            for m in adj.get(n, []):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        if ib in seen:
            continue
        used.add(a)
        used.add(b)
        adj.setdefault(ia, []).append(ib)
        adj.setdefault(ib, []).append(ia)
        joins.append((E[a], E[b], g))

    # walk each connected chain, orienting the pieces so the merged curve runs end to end
    merged, done = [], set()
    end_side = {}
    for (ia, sa, _pa, _ta), (ib, sb, _pb, _tb), _g in joins:
        end_side.setdefault(ia, {})[ib] = sa
        end_side.setdefault(ib, {})[ia] = sb
    for i in range(len(polys)):
        if i in done:
            continue
        comp, stack = [], [i]
        while stack:
            n = stack.pop()
            if n in done:
                continue
            done.add(n)
            comp.append(n)
            stack.extend(m for m in adj.get(n, []) if m not in done)
        if len(comp) == 1:
            merged.append(polys[i])
            continue
        # order the chain from one free end
        start = next((n for n in comp if len(adj.get(n, [])) == 1), comp[0])
        order, prev, cur = [start], None, start
        while True:
            nxt = next((m for m in adj.get(cur, []) if m != prev), None)
            if nxt is None or nxt in order:
                break
            order.append(nxt)
            prev, cur = cur, nxt
        chain = []
        for k, n in enumerate(order):
            q = np.asarray(polys[n], float)
            if k == 0:
                nxt = order[1]
                if end_side[n][nxt] == 0:             # joins from its FIRST point -> reverse
                    q = q[::-1]
            else:
                pr = order[k - 1]
                if end_side[n][pr] == 1:              # joins from its LAST point -> reverse
                    q = q[::-1]
            chain.append(q)
        merged.append(np.concatenate(chain, axis=0))
    return merged, joins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--raw-dir", default=None,
                    help="directory of the ORIGINAL tifs; the exported PNGs are 8-bit and "
                         "contrast-stretched, so the raw image is preferred when available")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap", type=float, default=45.0)
    ap.add_argument("--cos-tangent", type=float, default=-0.90)
    ap.add_argument("--cos-join", type=float, default=0.90)
    ap.add_argument("--exclude-names", default=None,
                    help="file of frame stems to leave out (frames already in another split)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excl = set()
    if args.exclude_names and os.path.isfile(args.exclude_names):
        excl = {l.strip() for l in open(args.exclude_names) if l.strip()}
        print(f"excluding {len(excl)} frames that already live in another split")

    frames = read_xml(args.xml)
    os.makedirs(args.out, exist_ok=True)
    import h5py

    tot_in = tot_out = tot_join = n_frames = skipped = 0
    len_before, len_after = [], []
    for fr in frames:
        stem = os.path.splitext(fr["name"])[0]
        if stem in excl or f"training_{stem}" in excl:
            skipped += 1
            continue
        polys = fr["polys"]
        if not polys:
            continue
        merged, joins = join_polylines(polys, args.max_gap, args.cos_tangent, args.cos_join)
        tot_in += len(polys)
        tot_out += len(merged)
        tot_join += len(joins)
        for q in polys:
            len_before.append(float(np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1)).sum()))
        for q in merged:
            len_after.append(float(np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1)).sum()))
        n_frames += 1
        if args.dry_run:
            continue

        img = None
        if args.raw_dir:
            for cand in (f"base__training_{stem}.tif", f"training_{stem}.tif", f"{stem}.tif"):
                p = os.path.join(args.raw_dir, cand)
                if os.path.isfile(p):
                    img = np.asarray(Image.open(p))
                    break
        src = "raw_tif"
        if img is None:
            img = np.asarray(Image.open(os.path.join(args.img_dir, fr["name"])))
            src = "exported_png_8bit"
        while img.ndim > 2:
            img = img[..., 0]
        if img.shape != (fr["H"], fr["W"]):
            raise SystemExit(f"{stem}: image {img.shape} != annotation frame "
                             f"{(fr['H'], fr['W'])} -- refusing to write a mismatched frame")

        with h5py.File(os.path.join(args.out, f"roi_{stem}.h5"), "w") as f:
            f.create_dataset("image", data=img.astype(np.float32), compression="gzip")
            g = f.create_group("polylines")
            for k, q in enumerate(merged):
                g.create_dataset(f"pl_{k:04d}", data=np.asarray(q, np.float64))
            f.attrs.update(height=fr["H"], width=fr["W"], n_polylines=len(merged),
                           n_manual=len(merged), reviewed=True, source_task="roi303",
                           split=args.split, image_source=src,
                           sources=",".join(["manual"] * len(merged)),
                           join_max_gap=args.max_gap, join_cos_tangent=args.cos_tangent,
                           join_cos_join=args.cos_join, n_joins=len(joins),
                           n_polylines_before_join=len(polys))

    lb, la = np.array(len_before), np.array(len_after)
    print(f"\n{n_frames} frames written ({skipped} skipped as already in another split)")
    print(f"polylines {tot_in} -> {tot_out}   ({tot_join} joins, "
          f"{100 * (tot_in - tot_out) / max(tot_in, 1):.1f} % fewer curves)")
    print(f"length p25/50/75 before: {np.percentile(lb,25):6.0f}/{np.percentile(lb,50):6.0f}/"
          f"{np.percentile(lb,75):6.0f} px")
    print(f"length p25/50/75 after : {np.percentile(la,25):6.0f}/{np.percentile(la,50):6.0f}/"
          f"{np.percentile(la,75):6.0f} px")
    if not args.dry_run:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()

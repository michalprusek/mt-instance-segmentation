#!/usr/bin/env python3
"""How badly is the manual ROI annotation broken into fragments, and can the pieces be rejoined?

The annotator traced dark filaments in interference microscopy. Where a microtubule lifts off the
coverslip its contrast passes through zero at h = lambda/(8*n) = 56.4 nm and reverses sign, so the
detached stretch does not look like the same filament and the trace stops. One microtubule is then
stored as two or more polylines with a gap between them.

Rejoining them cannot be done by proximity alone -- two different filaments ending near each other
would be merged. Three conditions together identify a genuine break:

  1. the two ends are close;
  2. the outgoing tangents are ANTI-parallel, i.e. each continues into the other (cos ~ -1), and
     the joining direction agrees with both, so the curve carries straight through rather than
     doubling back;
  3. the image ALONG the gap carries filament-like contrast -- ideally of the OPPOSITE polarity to
     the annotated parts, which is the physical signature of a detached segment.

Condition 3 is what makes the join a physical statement rather than a geometric guess, and it is
measurable without any model.

    python scripts/roi_fragmentation.py --xml <cvat_export/annotations.xml> --img-dir <images>
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


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
        out.append((im.get("name"), int(im.get("width")), int(im.get("height")), pl))
    return out


def end_frames(polys, back=12.0):
    """Both ends of every polyline, with the OUTGOING tangent (pointing away from the curve)."""
    ends = []
    for i, q in enumerate(polys):
        d = densify(q)
        if len(d) < 4:
            continue
        k = min(int(back), len(d) - 2)
        for which, (pt, ref) in enumerate(((d[0], d[k]), (d[-1], d[-1 - k]))):
            v = pt - ref
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            ends.append(dict(poly=i, side=which, p=pt, t=v / n))
    return ends


def signed_contrast(img, pts, half=7.0):
    if len(pts) < 3:
        return np.nan
    t = np.gradient(pts, axis=0)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    H, W = img.shape

    def take(p):
        c = np.round(p[:, 0]).astype(int)
        r = np.round(p[:, 1]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        v = np.full(len(p), np.nan)
        v[ok] = img[r[ok], c[ok]]
        return v

    return np.nanmedian(take(pts) - np.nanmean(np.stack([take(pts + n * half),
                                                         take(pts - n * half)]), axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--max-gap", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    frames = read_xml(args.xml)[:args.limit]
    nn, cos_t, cos_join, gap_pol, ref_pol = [], [], [], [], []
    n_poly = n_end = 0

    for name, W, H, polys in frames:
        ip = os.path.join(args.img_dir, name)
        img = np.asarray(Image.open(ip), float) if os.path.exists(ip) else None
        ends = end_frames(polys)
        n_poly += len(polys)
        n_end += len(ends)
        if len(ends) < 4:
            continue
        P = np.array([e["p"] for e in ends])
        owner = np.array([e["poly"] for e in ends])
        tree = cKDTree(P)
        d, j = tree.query(P, k=6)
        if img is not None:
            rp = [signed_contrast(img, densify(q)) for q in polys]
            rp = np.array([v for v in rp if np.isfinite(v)])
            ref = float(np.median(rp)) if len(rp) else np.nan
        else:
            ref = np.nan
        for i in range(len(ends)):
            # nearest endpoint that belongs to a DIFFERENT polyline
            k = next((jj for dd, jj in zip(d[i], j[i]) if owner[jj] != owner[i]), None)
            if k is None:
                continue
            dist = float(np.linalg.norm(P[i] - P[k]))
            nn.append(dist)
            if dist > args.max_gap:
                continue
            ti, tk = ends[i]["t"], ends[k]["t"]
            cos_t.append(float(np.dot(ti, tk)))
            u = P[k] - P[i]
            nu = np.linalg.norm(u)
            # append unconditionally, NaN when undefined, so this stays index-aligned with
            # cos_t -- a conditional append silently misaligns the two and any joint
            # condition below would then compare unrelated pairs
            cos_join.append(float(np.dot(ti, u / nu)) if nu > 1e-6 else np.nan)
            if img is not None and nu > 3:
                seg = np.linspace(P[i], P[k], max(int(nu), 4))
                gp = signed_contrast(img, seg)
                if np.isfinite(gp) and np.isfinite(ref):
                    gap_pol.append(gp)
                    ref_pol.append(ref)

    nn = np.array(nn); cos_t = np.array(cos_t); cos_join = np.array(cos_join)
    gap_pol = np.array(gap_pol); ref_pol = np.array(ref_pol)
    print(f"FRAGMENTATION OF THE MANUAL ROI ANNOTATION  ({len(frames)} frames, "
          f"{n_poly} polylines, {n_end} endpoints)\n")
    print(f"distance to the nearest endpoint of ANOTHER polyline:")
    for q in (10, 25, 50, 75):
        print(f"    p{q:<3d} {np.percentile(nn, q):7.1f} px")
    for t in (5, 10, 20, 40, 60):
        print(f"    within {t:3d} px: {100 * (nn <= t).mean():5.1f} % of endpoints")

    print(f"\nfor the {len(cos_t)} pairs closer than {args.max_gap:.0f} px:")
    print(f"    cos(tangent_i, tangent_k)  p25/50/75 = "
          f"{np.percentile(cos_t,25):+.2f}/{np.percentile(cos_t,50):+.2f}/{np.percentile(cos_t,75):+.2f}")
    print("      (-1 = the two ends point away from each other = ONE filament cut in two)")
    print(f"    fraction with cos < -0.8 : {100 * (cos_t < -0.8).mean():5.1f} %")
    if len(cos_join):
        print(f"    cos(tangent_i, joining direction) p50 = {np.nanmedian(cos_join):+.2f}"
              "   (+1 = the gap continues the curve)")
        ok = np.isfinite(cos_join)
        both = (cos_t < -0.8) & ok & (cos_join > 0.8)
        print(f"    BOTH conditions (continues straight through): {100 * both.mean():5.1f} % "
              f"= {int(both.sum())} joins")

    if len(gap_pol):
        same = np.sign(gap_pol) == np.sign(ref_pol)
        print(f"\ncontrast ALONG the gap, for {len(gap_pol)} close pairs:")
        print(f"    annotated filaments: median {np.median(ref_pol):+.2f}")
        print(f"    the gap itself     : median {np.median(gap_pol):+.2f}")
        print(f"    same polarity as the filaments: {100 * same.mean():5.1f} %   "
              f"opposite: {100 * (~same).mean():5.1f} %")
        print("      opposite polarity in the gap = a DETACHED stretch the annotator skipped")


if __name__ == "__main__":
    main()

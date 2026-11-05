"""SpheroSeg projects flagged ``verified`` -> benchmark frames with per-polyline provenance.

What a verified microtubule project is, measured on the first nine (2026-10-13):

- **One image, not a movie.** The segmentation-source channel is a single static IRM frame
  (one md5 over every frame of the container), so the 108-300 "frames" of a project are one
  image with copies of one annotation. The benchmark unit is the PROJECT.
- **The annotation started as a model proposal.** The hub writes the model's N-th centerline
  with the id ``polyline_N``; the editor gives everything a human creates a different id
  (``polygon_<ms>_*`` for a drawn curve, ``polyline_<ms>_a|b_*`` for the two halves of a split).
  Copies made by track propagation carry a uuid. So the id says who made a curve -- but it
  cannot say that a human MOVED a model curve, and a DELETED proposal leaves no trace at all.
- **Both gaps close by regenerating the proposal.** The seed model is deterministic; re-running
  it on the same frame gives back the list the annotator was shown, and diffing the stored
  annotation against it yields the fate of every proposal (kept / trimmed / extended /
  reshaped / split / absorbed / deleted) and of every stored curve.

That diff is the point of this block. A score against these polylines is agreement-biased
towards the seed model by construction (``docs/protocol.md``, agreement bias: +0.17 to +0.30 F1
on the model-drawn half of MT-34), so the frames are written with ``split="dev"`` and must not
be pooled with roi303 / htw.

Coordinates are ``(x=col, y=row)`` at native 1x, as everywhere else in ``mt_bench``.

Run:  python -m mt_bench.spheroseg_verified --root data/real/spheroseg_verified
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

RE_MODEL = re.compile(r"^polyline_(\d{1,4})$")
RE_SPLIT = re.compile(r"^polyline_\d{13}_[ab]_")
RE_DRAWN = re.compile(r"^polygon_\d{13}_")
RE_COPY = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: Two curves are "the same curve" when this fraction of each lies within SAME_TOL of the other.
#: 1 px is twice the sub-pixel jitter a second inference run puts on an untouched centerline
#: (GPU / torch numerics) and well under the smallest edit a person makes with a mouse.
SAME_TOL = 1.0
SAME_FRAC = 0.98
#: A proposal with no surviving descendant whose filament is still annotated (merged into a
#: neighbour, or re-drawn by hand) was replaced, not rejected. 3 px is the half-width of a
#: microtubule image in these frames; half its length must be covered.
REPLACE_TOL = 3.0
REPLACE_FRAC = 0.5
#: Vertex identity, px. Stored coordinates are float32 values printed as JSON doubles.
EXACT_TOL = 1e-3

#: Fates of a stored curve that mean a person made or changed its geometry.
HUMAN_FATES = ("trimmed", "extended", "reshaped", "split", "drawn")


def id_class(poly_id) -> str:
    """Who created a polyline, read off its id: ``model | split | drawn | copy | other``."""
    s = str(poly_id)
    if RE_MODEL.match(s):
        return "model"
    if RE_SPLIT.match(s):
        return "split"
    if RE_DRAWN.match(s):
        return "drawn"
    if RE_COPY.match(s):
        return "copy"
    return "other"


def to_xy(poly: dict) -> np.ndarray:
    """Hub polyline dict -> ``(N, 2)`` float64 ``(x=col, y=row)``."""
    return np.array([[q["x"], q["y"]] for q in poly["points"]], dtype=np.float64)


def length(a: np.ndarray) -> float:
    return float(np.hypot(*np.diff(a, axis=0).T).sum()) if len(a) > 1 else 0.0


def densify(a: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Resample a polyline to points ``step`` px apart along its arc."""
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return a.copy()
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(a, axis=0).T))])
    if s[-1] == 0.0:
        return a[:1].copy()
    t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(t, s, a[:, 0]), np.interp(t, s, a[:, 1])], axis=1)


def _coverage_matrix(curves_a, curves_b, tol: float) -> np.ndarray:
    """``M[i, j]`` = fraction of the length of ``curves_a[i]`` within ``tol`` of ``curves_b[j]``."""
    dense_a = [densify(a) for a in curves_a]
    trees_b = [cKDTree(densify(b)) for b in curves_b]
    m = np.zeros((len(curves_a), len(curves_b)), dtype=np.float64)
    for j, tree in enumerate(trees_b):
        for i, pts in enumerate(dense_a):
            m[i, j] = float((tree.query(pts, distance_upper_bound=tol)[0] <= tol).mean())
    return m


def _union_coverage(curve: np.ndarray, others, tol: float) -> float:
    """Fraction of ``curve`` within ``tol`` of ANY curve in ``others``."""
    if not len(others):
        return 0.0
    tree = cKDTree(np.concatenate([densify(o) for o in others]))
    return float((tree.query(densify(curve), distance_upper_bound=tol)[0] <= tol).mean())


def classify(stored: list[dict], proposals: list[np.ndarray]) -> tuple[list[dict], list[dict]]:
    """Fate of every stored polyline and of every proposal.

    ``stored`` are hub polyline dicts of the frame the annotator saved; ``proposals`` are the
    regenerated model centerlines in emission order (index ``k`` <-> id ``polyline_{k+1}``).

    Matching is geometric, with the id only as a shortcut for the exact case: a re-inference
    on different hardware can emit one marginal fragment more or fewer, which shifts every
    later index while leaving the curves where they were.

    Returns ``(stored_fates, proposal_fates)``; each entry is a dict with ``fate`` and the
    numbers the decision was made from.
    """
    s_xy = [to_xy(p) for p in stored]
    s_cls = [id_class(p.get("id")) for p in stored]
    n_s, n_p = len(s_xy), len(proposals)
    sp = _coverage_matrix(s_xy, proposals, SAME_TOL) if n_s and n_p else np.zeros((n_s, n_p))
    ps = _coverage_matrix(proposals, s_xy, SAME_TOL).T if n_s and n_p else np.zeros((n_s, n_p))

    out_s: list[dict] = []
    children: dict[int, list[int]] = defaultdict(list)
    for i, (a, cls) in enumerate(zip(s_xy, s_cls)):
        rec = {"id": str(stored[i].get("id")), "id_class": cls, "length_px": length(a),
               "proposal": None, "stored_on_proposal": 0.0, "proposal_on_stored": 0.0,
               "exact": False}
        if cls == "drawn" or n_p == 0:
            rec["fate"] = "drawn" if cls == "drawn" else "unmatched"
            # how much of a hand-drawn curve re-traces something the model had proposed
            rec["stored_on_proposal"] = _union_coverage(a, proposals, REPLACE_TOL)
            out_s.append(rec)
            continue
        if cls == "split":
            j = int(np.argmax(sp[i]))
            rec.update(fate="split", proposal=j, stored_on_proposal=float(sp[i, j]),
                       proposal_on_stored=float(ps[i, j]))
            children[j].append(i)
            out_s.append(rec)
            continue
        # model id (or a propagated copy of one): same index first, geometry otherwise
        m = RE_MODEL.match(rec["id"])
        k = int(m.group(1)) - 1 if m else -1
        if 0 <= k < n_p and a.shape == proposals[k].shape \
                and float(np.abs(a - proposals[k]).max()) < EXACT_TOL:
            rec.update(fate="kept", proposal=k, stored_on_proposal=1.0,
                       proposal_on_stored=1.0, exact=True)
            children[k].append(i)
            out_s.append(rec)
            continue
        j = int(np.argmax(sp[i] + ps[i]))
        c_sp, c_ps = float(sp[i, j]), float(ps[i, j])
        if c_sp >= SAME_FRAC and c_ps >= SAME_FRAC:
            fate = "kept"
        elif c_sp >= SAME_FRAC:
            fate = "trimmed"        # lies on the proposal, covers only part of it
        elif c_ps >= SAME_FRAC:
            fate = "extended"       # contains the proposal and runs on past it
        elif max(c_sp, c_ps) >= 0.5:
            fate = "reshaped"
        else:
            fate = "unmatched"      # the id says model, no regenerated curve agrees
        rec.update(fate=fate, stored_on_proposal=c_sp, proposal_on_stored=c_ps)
        if fate != "unmatched":
            rec["proposal"] = j
            children[j].append(i)
        out_s.append(rec)

    out_p: list[dict] = []
    for j, b in enumerate(proposals):
        rec = {"id": f"polyline_{j + 1}", "length_px": length(b), "children": children.get(j, [])}
        kids = [out_s[i] for i in rec["children"]]
        survives = _union_coverage(b, s_xy, REPLACE_TOL)
        rec["covered_by_annotation"] = survives
        if kids:
            fates = {k["fate"] for k in kids}
            if "split" in fates:
                rec["fate"] = "split"
            else:
                # one model-id child is the rule; if a later index matched too, keep the closest
                rec["fate"] = max(kids, key=lambda k: k["stored_on_proposal"]
                                  + k["proposal_on_stored"])["fate"]
        else:
            rec["fate"] = "absorbed" if survives >= REPLACE_FRAC else "deleted"
        out_p.append(rec)
    return out_s, out_p


def pick_proposals(stored: list[dict], candidates: dict[str, list[np.ndarray]]) -> tuple[str, int]:
    """Which regenerated proposal list is the one this annotation started from.

    Decided by evidence, not by a table: the right list reproduces stored ``polyline_N``
    vertices exactly at the same index, a wrong model reproduces none.
    """
    best, best_n = "", -1
    for tag, props in candidates.items():
        n = 0
        for p in stored:
            m = RE_MODEL.match(str(p.get("id")))
            if not m:
                continue
            k = int(m.group(1)) - 1
            a = to_xy(p)
            if 0 <= k < len(props) and a.shape == props[k].shape \
                    and float(np.abs(a - props[k]).max()) < EXACT_TOL:
                n += 1
        if n > best_n:
            best, best_n = tag, n
    return best, best_n


def canonical_frame(frames: list[dict]) -> dict:
    """The frame the annotator edited: the one that still carries the editor's own ids.

    Sibling frames hold uuid COPIES made by track propagation, so "the richest frame" picks a
    copy: in one project frame 1 is the untouched proposal plus five propagated copies of the
    five curves the annotator had edited on frame 0 -- 66 polylines against the edited
    frame's 59. The edited frame is the one WITHOUT copies; among those, the earliest (the
    editor opens a video on frame 0, and in all nine projects of 2026-10-13 that is the one).
    """
    def copies(fr):
        return sum(id_class(p.get("id")) == "copy" for p in fr["polygons"])
    return min(frames, key=lambda fr: (copies(fr), fr["frame_index"]))


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")


def _read_png16(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("I;16", "I;16B", "I;16L"):
        raise ValueError(f"{path}: expected a 16-bit PNG, got mode {im.mode}")
    return np.array(im, dtype=np.uint16)


def _source_png(root: str, project_id: str, video_id: str, channel: str) -> str:
    """The static segmentation-source frame, as fetched by ``spheroseg_verified_fetch.sh``."""
    return os.path.join(root, "raw", "images", project_id, "images", video_id, "frames", "0000",
                        f"{channel}.png")


def build(root: str) -> list[dict]:
    """Diff every project against its regenerated proposal and write one h5 per project."""
    from mt_bench.cvat_import import write_frame_h5

    by_proj: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(os.path.join(root, "raw", "segmentations.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            by_proj[r["project_id"]].append(r)
    meta = {m["project_id"]: m for m in json.load(open(os.path.join(root, "raw", "projects.json")))}

    os.makedirs(os.path.join(root, "h5"), exist_ok=True)
    report = []
    for pid, frames in by_proj.items():
        frames.sort(key=lambda r: r["frame_index"])
        fr = canonical_frame(frames)
        stored = [p for p in fr["polygons"] if len(p.get("points", [])) >= 2]
        cands = {}
        for path in sorted(glob.glob(os.path.join(root, "proposals", f"{pid[:8]}__*.json"))):
            tag = os.path.basename(path)[len(pid[:8]) + 2:-5]
            cands[tag] = [to_xy(p) for p in json.load(open(path))["polylines"]]
        tag, n_exact = pick_proposals(stored, cands)
        proposals = cands[tag]
        s_f, p_f = classify(stored, proposals)

        png = _source_png(root, pid, fr["video_id"], meta[pid]["source_channel"])
        image = _read_png16(png)
        slug = _slug(fr["title"])
        fates = [s["fate"] for s in s_f]
        sources = ["manual" if f in HUMAN_FATES else "file" for f in fates]
        review_min = (np.datetime64(fr["verified_at"]) - np.datetime64(
            min(r["seg_created"] for r in frames))) / np.timedelta64(1, "m")
        attrs = {
            "source_task": "spheroseg_verified", "split": "dev", "reviewed": True,
            "project_id": pid, "title": fr["title"], "annotator": fr["owner_md5"][:8],
            "verified_at": fr["verified_at"], "frame_index": int(fr["frame_index"]),
            "seed_threshold": float(fr["threshold"]), "proposal_source": tag,
            "proposal_exact_matches": int(n_exact),
            "image_source": "hub_png_16bit", "source_channel": meta[pid]["source_channel"],
            "pixel_size_um": float(meta[pid].get("pixel_size_um") or float("nan")),
            # CVAT semantics, so score_by_provenance.py reads this block like MT-34:
            # manual = a person made or changed the geometry, file = model output as emitted.
            "sources": ",".join(sources), "n_manual": int(sum(s == "manual" for s in sources)),
            "fates": ",".join(fates), "ids": ",".join(s["id"] for s in s_f),
            "proposal_fates": ",".join(p["fate"] for p in p_f),
        }
        out = os.path.join(root, "h5", f"{slug}.h5")
        write_frame_h5(out, image, [to_xy(p) for p in stored], attrs)
        import h5py
        with h5py.File(out, "a") as h:
            grp = h.create_group("proposals")
            for j, b in enumerate(proposals):
                grp.create_dataset(f"pl_{j:04d}", data=b)

        s_len, p_len = Counter(), Counter()
        for s in s_f:
            s_len[s["fate"]] += s["length_px"]
        for p in p_f:
            p_len[p["fate"]] += p["length_px"]
        # Length accounting, blind to how the length is cut into instances: what the annotator
        # ADDED is annotation with no proposal under it, what they REMOVED is proposal with no
        # annotation left on it. Everything else they did was re-cutting the same length.
        s_xy = [to_xy(p) for p in stored]
        added = sum(length(a) * (1.0 - _union_coverage(a, proposals, REPLACE_TOL)) for a in s_xy)
        removed = sum(p["length_px"] * (1.0 - p["covered_by_annotation"]) for p in p_f)
        swallowed = _coverage_matrix(proposals, s_xy, REPLACE_TOL) >= REPLACE_FRAC if s_xy else None
        n_joined = int((swallowed.sum(axis=0) >= 2).sum()) if swallowed is not None else 0
        for i, rec in enumerate(s_f):   # which proposals each stored curve covers: >= 2 is a join
            rec["swallowed"] = [int(j) for j in np.nonzero(swallowed[:, i])[0]] if swallowed is not None else []
        report.append({
            "project_id": pid, "slug": slug, "title": fr["title"], "h5": out,
            "shape": list(image.shape), "frames_in_project": len(frames),
            "annotator": fr["owner_md5"][:8], "seed_threshold": fr["threshold"],
            "proposal_source": tag, "proposal_exact_matches": n_exact,
            "review_minutes": float(review_min),
            "n_stored": len(s_f), "n_proposals": len(p_f),
            "stored_fates": dict(Counter(fates)), "stored_length_by_fate": dict(s_len),
            "proposal_fates": dict(Counter(p["fate"] for p in p_f)),
            "proposal_length_by_fate": dict(p_len),
            "stored_length_px": sum(s_len.values()), "proposal_length_px": sum(p_len.values()),
            "added_length_px": added, "removed_length_px": removed, "n_joined": n_joined,
            "stored": s_f, "proposals": p_f,
        })
    with open(os.path.join(root, "provenance.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def print_report(report: list[dict]) -> None:
    order_p = ("kept", "trimmed", "extended", "reshaped", "split", "absorbed", "deleted")
    order_s = ("kept", "trimmed", "extended", "reshaped", "split", "drawn", "unmatched")
    print("proposals  " + " ".join(f"{k:>8s}" for k in order_p)
          + " |  stored  " + " ".join(f"{k:>8s}" for k in order_s) + " | min  seed    exact  project")
    tot_p, tot_s = Counter(), Counter()
    for r in report:
        tot_p.update(r["proposal_fates"])
        tot_s.update(r["stored_fates"])
        print(f"{r['n_proposals']:9d}  " + " ".join(f"{r['proposal_fates'].get(k, 0):8d}" for k in order_p)
              + f" | {r['n_stored']:7d}  " + " ".join(f"{r['stored_fates'].get(k, 0):8d}" for k in order_s)
              + f" | {r['review_minutes']:4.0f} {r['proposal_source']:7s} {r['proposal_exact_matches']:5d}  {r['title'][:46]}")
    print(f"{sum(tot_p.values()):9d}  " + " ".join(f"{tot_p.get(k, 0):8d}" for k in order_p)
          + f" | {sum(tot_s.values()):7d}  " + " ".join(f"{tot_s.get(k, 0):8d}" for k in order_s) + " | TOTAL")
    print("\nlength, px    annotation   proposal   added by hand   removed by hand   joined  split")
    tot = Counter()
    for r in report:
        row = {"gt": r["stored_length_px"], "pr": r["proposal_length_px"], "add": r["added_length_px"],
               "rem": r["removed_length_px"], "join": r["n_joined"], "split": r["proposal_fates"].get("split", 0)}
        tot.update(row)
        print(f"            {row['gt']:10.0f} {row['pr']:10.0f} {row['add']:8.0f} ({100 * row['add'] / row['gt']:4.1f} %)"
              f" {row['rem']:8.0f} ({100 * row['rem'] / row['pr']:4.1f} %) {row['join']:7d} {row['split']:6d}  {r['title'][:46]}")
    print(f"            {tot['gt']:10.0f} {tot['pr']:10.0f} {tot['add']:8.0f} ({100 * tot['add'] / tot['gt']:4.1f} %)"
          f" {tot['rem']:8.0f} ({100 * tot['rem'] / tot['pr']:4.1f} %) {tot['join']:7d} {tot['split']:6d}  TOTAL")


def render_overlays(report: list[dict], out_dir: str) -> int:
    """One picture per project: what the model proposed, and what the annotator did with it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from mt_bench.cvat_import import read_frame_h5
    from mt_bench.overlay import _norm01

    colour = {"kept": "#3ddc3d", "trimmed": "#ffd400", "extended": "#ffd400", "reshaped": "#ffd400",
              "split": "#ff8c00", "drawn": "#ff3df2", "unmatched": "#9aa0ff",
              "deleted": "#ff2a2a", "absorbed": "#35c9ff"}
    os.makedirs(out_dir, exist_ok=True)
    for r in report:
        fr = read_frame_h5(r["h5"])
        import h5py
        with h5py.File(r["h5"], "r") as h:
            props = [h["proposals"][k][:] for k in sorted(h["proposals"].keys())]
        hh, ww = fr["image"].shape
        fig, ax = plt.subplots(figsize=(ww / 100.0, hh / 100.0), dpi=200)
        ax.imshow(_norm01(fr["image"]), cmap="gray", interpolation="nearest")
        for p, rec in zip(props, r["proposals"]):
            if rec["fate"] in ("deleted", "absorbed"):
                ax.plot(p[:, 0], p[:, 1], "--", lw=0.7, color=colour[rec["fate"]], alpha=0.95)
        for p, rec in zip(fr["polylines"], r["stored"]):
            ax.plot(p[:, 0], p[:, 1], "-", lw=0.5 if rec["fate"] == "kept" else 0.8,
                    color=colour[rec["fate"]], alpha=0.9)
        keys = ("kept", "trimmed", "split", "drawn", "absorbed", "deleted")
        labels = {"kept": "model, kept", "trimmed": "model, moved by hand", "split": "split by hand",
                  "drawn": "drawn by hand", "absorbed": "proposal replaced", "deleted": "proposal deleted"}
        ax.legend([Line2D([0], [0], color=colour[k], lw=1.5, ls="--" if k in ("deleted", "absorbed") else "-")
                   for k in keys], [labels[k] for k in keys], loc="lower right", fontsize=5, framealpha=0.8)
        ax.set_xlim(0, ww)
        ax.set_ylim(hh, 0)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(os.path.join(out_dir, f"{r['slug']}.jpg"), dpi=200, pil_kwargs={"quality": 90})
        plt.close(fig)
    return len(report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/real/spheroseg_verified")
    ap.add_argument("--overlays", default="data/enc_sensitivity_testset/spheroseg_verified/diff")
    args = ap.parse_args()
    report = build(args.root)
    print_report(report)
    if args.overlays:
        print(f"{render_overlays(report, args.overlays)} overlays -> {args.overlays}")


if __name__ == "__main__":
    main()

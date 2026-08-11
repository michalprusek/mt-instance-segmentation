"""Build the MT-34 real instance-segmentation benchmark from CVAT tasks 585 + 586.

MT-34 = 12 Alice frames (CVAT task 585 / job 557 -- the same 12 scenes as
``data/real/alice_eval``, but re-exported so the 2026-06-04 human corrections are included)
+ the first 22 frames of CVAT task 586 (``v7_pysoax_general_mt``), which are exactly the
frames a human reviewed: ``source="manual"`` edits stop at frame id 21.

Everything is stored at NATIVE 1x resolution with ``(x=col, y=row)`` vertices, matching the
``alice_eval`` layout. The x1.5 upscale is an evaluation-pipeline convention applied at scoring
time (image ``zoom(..., 1.5)``, GT vertices x1.5) -- not a property of this data.

Run:  python -m mt_bench.build_mt34 --cvat-dir <dir with cvat_585/ and cvat_586/>
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil

import tifffile

from mt_bench.cvat_import import assign_split, parse_cvat_xml, write_frame_h5

# Task 586 was pre-annotated with v7+PySOAX and then human-corrected frame by frame.
# Manual edits stop after frame id 21, so "the first 22 frames" is exactly the reviewed block.
LAST_REVIEWED_FRAME_ID_586 = 21

SOURCES = [
    ("585", "cvat_585/annotations.xml", None),
    ("586", "cvat_586/annotations.xml", LAST_REVIEWED_FRAME_ID_586),
]


def collect_frames(cvat_dir: str) -> list[dict]:
    """Parse both CVAT exports and keep the frames that belong in MT-34."""
    frames = []
    for task, rel, max_id in SOURCES:
        path = os.path.join(cvat_dir, rel)
        for fr in parse_cvat_xml(path):
            if max_id is not None and fr["frame_id"] > max_id:
                continue
            fr["source_task"] = task
            frames.append(fr)
    return frames


def build(cvat_dir: str, out_dir: str, tif_dir: str | None = None) -> dict:
    tif_dir = tif_dir or os.path.join(out_dir, "tif")
    os.makedirs(out_dir, exist_ok=True)

    frames = collect_frames(cvat_dir)

    # Split is assigned SEPARATELY per source so each source is balanced val/test
    # (Alice 6/6, new-22 11/11) rather than letting the larger source dominate one half.
    splits: dict[str, str] = {}
    for task, _, _ in SOURCES:
        names = [f["name"] for f in frames if f["source_task"] == task]
        splits.update(assign_split(names))

    rows, total_pl = [], 0
    for fr in frames:
        tif_path = os.path.join(tif_dir, fr["name"])
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"missing image for annotated frame: {tif_path}")
        image = tifffile.imread(tif_path)
        if image.shape != (fr["height"], fr["width"]):
            # A mismatch means the image files do not correspond to the annotations;
            # silently continuing would produce a benchmark whose GT sits on the wrong pixels.
            raise ValueError(
                f"{fr['name']}: image shape {image.shape} != CVAT "
                f"{(fr['height'], fr['width'])}"
            )

        n_manual = sum(1 for s in fr["sources"] if s == "manual")
        stem = os.path.splitext(fr["name"])[0]
        out_h5 = os.path.join(out_dir, f"{stem}.h5")
        write_frame_h5(out_h5, image, fr["polylines"], {
            "source_task": fr["source_task"],
            "frame_id": fr["frame_id"],
            "split": splits[fr["name"]],
            "n_manual": n_manual,
            "reviewed": bool(n_manual > 0),
            "sources": ",".join(fr["sources"]),
        })
        total_pl += len(fr["polylines"])
        rows.append({
            "name": fr["name"], "source_task": fr["source_task"],
            "frame_id": fr["frame_id"], "width": fr["width"], "height": fr["height"],
            "n_polylines": len(fr["polylines"]), "n_manual": n_manual,
            "split": splits[fr["name"]], "reviewed": int(n_manual > 0),
        })

    rows.sort(key=lambda r: (r["source_task"], r["name"]))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Keep the raw CVAT exports next to the data so the benchmark is reproducible
    # without re-hitting the CVAT server.
    prov = os.path.join(out_dir, "cvat")
    os.makedirs(prov, exist_ok=True)
    for task, rel, _ in SOURCES:
        shutil.copy(os.path.join(cvat_dir, rel), os.path.join(prov, f"task_{task}.xml"))

    return {"n_frames": len(rows), "n_polylines": total_pl, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvat-dir", required=True,
                    help="directory containing cvat_585/annotations.xml and cvat_586/...")
    ap.add_argument("--out-dir", default="data/real/mt34_eval")
    args = ap.parse_args()

    res = build(args.cvat_dir, args.out_dir)
    rows = res["rows"]
    print(f"MT-34 built: {res['n_frames']} frames, {res['n_polylines']} polylines")
    for task in ("585", "586"):
        sub = [r for r in rows if r["source_task"] == task]
        n_rev = sum(r["reviewed"] for r in sub)
        n_val = sum(1 for r in sub if r["split"] == "val")
        print(f"  task {task}: {len(sub)} frames, {sum(r['n_polylines'] for r in sub)} polylines, "
              f"{sum(r['n_manual'] for r in sub)} manual, {n_rev} reviewed, "
              f"{n_val} val / {len(sub) - n_val} test")


if __name__ == "__main__":
    main()

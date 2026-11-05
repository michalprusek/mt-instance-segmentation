#!/usr/bin/env python3
"""Read the 2026-09-15 STEP-1 batch (``step1_score.sh``) into the consolidation table.

Consumes the per-threshold sweep JSONs ``eval_v5.py`` writes (each threshold carries ``_frames``
rows with tp/fp/fn per frame) and the ``run_oracle_eval.py`` JSONs. Nothing is re-run.

The rule this script enforces, declared in ``docs/research/2026-09-15-consolidation.md`` §2/§8
BEFORE any TEST number was read:

  * the decode threshold is chosen PER SEED on VAL (pooled micro-F1 over the block's frames) and
    APPLIED to TEST -- the TEST json holds the whole grid, but only the VAL-chosen row is quoted;
  * the primary statistic is pooled MICRO-F1; the per-frame macro over frames with >= 10 GT is
    reported alongside; the all-frames macro is printed for continuity with older numbers only;
  * the 100-frame block is split by source (htw / alice / 586) from the frame names; htw is the
    only source whose threshold is chosen on its own VAL frames, because it is the block the
    consolidation names as the mandatory secondary test set. A second htw column applies the
    roi303-VAL threshold unchanged (pure transfer, nothing chosen on any htw frame).

Under one-to-one matching n_gt = tp + fn, so the >= 10-GT filter needs no extra field.

    python scripts/step1_read.py --dir runs/step1 [--min-gt 10]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SEEDS = ("SPARSE35", "SEED2", "SEED3")
PARAMS = ("der", "fit")
PARAM_LABEL = {"der": "derived (min_length 15.0)", "fit": "fitted (min_length 44.74)"}


def source_of(name: str) -> str:
    """Frame-name -> source. In the 100-frame block the task-586 frames are named
    ``training_img_*`` (CVAT export naming); roi303 frames are ``roi_img_*`` and never share a
    sweep file with the 100-frame block, so the two never collide."""
    if name.startswith("htw_"):
        return "htw"
    if name.startswith("alice_"):
        return "alice"
    if name.startswith("roi_img_"):
        return "roi303"
    return "586"


def load(path: str) -> dict | None:
    """A sweep json, or the merge of its single-node shards ``<stem>_t<thr>.json`` (the roi303
    sweeps were re-run one threshold node per process; the shards carry the same layout)."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    shards = sorted(glob.glob(path[:-5] + "_t*.json"))
    if not shards:
        return None
    merged: dict | None = None
    for sh in shards:
        with open(sh) as fh:
            d = json.load(fh)
        if merged is None:
            merged = d
        else:
            merged["results"].update(d["results"])
    return merged


def stats(frames: list[dict], min_gt: int) -> dict:
    """Pooled micro + the two macros from per-frame tp/fp/fn rows."""
    tp = sum(f["tp"] for f in frames)
    fp = sum(f["fp"] for f in frames)
    fn = sum(f["fn"] for f in frames)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    micro = 2 * p * r / (p + r) if p + r else 0.0
    scored = [f for f in frames if f["tp"] + f["fn"] > 0]
    big = [f for f in frames if f["tp"] + f["fn"] >= min_gt]
    return {
        "micro": micro, "micro_p": p, "micro_r": r,
        "macro_all": float(np.mean([f["f1"] for f in scored])) if scored else float("nan"),
        "macro_big": float(np.mean([f["f1"] for f in big])) if big else float("nan"),
        "n": len(frames), "n_big": len(big), "tp": tp, "fp": fp, "fn": fn,
    }


def rows_by_thr(sweep: dict, source: str | None) -> dict[float, list[dict]]:
    out = {}
    for thr, res in sweep["results"].items():
        fr = res["_frames"]
        if source:
            fr = [f for f in fr if source_of(f["name"]) == source]
        out[float(thr)] = fr
    return out


def choose_thr(sweep: dict, source: str | None, min_gt: int) -> tuple[float, dict]:
    """The VAL selection rule: pooled micro-F1, ties to the LOWER threshold (more foreground,
    the conservative direction for a model whose failure is recall)."""
    best = None
    for thr in sorted(rows_by_thr(sweep, source)):
        s = stats(rows_by_thr(sweep, source)[thr], min_gt)
        if best is None or s["micro"] > best[1]["micro"] + 1e-12:
            best = (thr, s)
    return best


def oracle_line(o: dict | None) -> str:
    """run_oracle_eval.py writes ``a/pooled`` (mean_f1, micro_f1, ...) and ``a/<task>`` blocks
    holding ``{"agg": {...}}``; quote the pooled micro first, then each task's micro."""
    if o is None:
        return "  (oracle json missing)"
    pooled = o.get("a/pooled")
    txt = f"  micro={pooled['micro_f1']:.4f}  macro-all={pooled['mean_f1']:.4f}" if pooled else ""
    per = {k.split("/", 1)[1]: v["agg"] for k, v in o.items()
           if k.startswith("a/") and isinstance(v, dict) and "agg" in v}
    if per:
        txt += "  per-task micro: " + ", ".join(f"{k}={v['micro_f1']:.4f}" for k, v in per.items())
    return txt or f"  keys: {list(o)[:8]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--min-gt", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D = args.dir
    report: dict = {"min_gt": args.min_gt, "seeds": {}}

    print("=" * 100)
    print("STEP 1 -- SPARSE35 x 3 seeds, ep040, 0.80/0.80 tol 5 end_tol 0; threshold chosen on VAL (pooled micro)")
    print("=" * 100)
    for prm in PARAMS:
        print(f"\n##### instancer constant: {PARAM_LABEL[prm]}")
        hdr = f"{'seed':<9}{'block':<10}{'thr(VAL)':>9}{'VAL micro':>10}{'TEST micro':>11}{'TEST P':>8}{'TEST R':>8}{'macro>=10':>10}{'n>=10':>6}{'macro all':>10}{'n':>5}"
        print(hdr)
        acc: dict[str, list] = {}
        for seed in SEEDS:
            rv = load(f"{D}/sweep_roiVAL_{seed}_{prm}.json")
            rt = load(f"{D}/sweep_roiTEST_{seed}_{prm}.json")
            hv = load(f"{D}/sweep_hbVAL_{seed}_{prm}.json")
            ht = load(f"{D}/sweep_hbTEST_{seed}_{prm}.json")
            entry = report["seeds"].setdefault(seed, {}).setdefault(prm, {})
            # roi303: choose on roi303 VAL, apply to roi303 TEST
            if rv and rt:
                thr, sv = choose_thr(rv, None, args.min_gt)
                st = stats(rows_by_thr(rt, None)[thr], args.min_gt)
                entry["roi303"] = {"thr": thr, "val": sv, "test": st}
                acc.setdefault("roi303", []).append(st)
                print(f"{seed:<9}{'roi303':<10}{thr:>9.3f}{sv['micro']:>10.4f}{st['micro']:>11.4f}{st['micro_p']:>8.3f}{st['micro_r']:>8.3f}{st['macro_big']:>10.4f}{st['n_big']:>6d}{st['macro_all']:>10.4f}{st['n']:>5d}")
                roi_thr = thr
            else:
                roi_thr = None
                print(f"{seed:<9}{'roi303':<10}   (missing: {'VAL ' if not rv else ''}{'TEST' if not rt else ''})")
            # htw: choose on htw VAL frames (own-block rule) ...
            if hv and ht:
                thr, sv = choose_thr(hv, "htw", args.min_gt)
                st = stats(rows_by_thr(ht, "htw")[thr], args.min_gt)
                entry["htw"] = {"thr": thr, "val": sv, "test": st}
                acc.setdefault("htw", []).append(st)
                print(f"{seed:<9}{'htw':<10}{thr:>9.3f}{sv['micro']:>10.4f}{st['micro']:>11.4f}{st['micro_p']:>8.3f}{st['micro_r']:>8.3f}{st['macro_big']:>10.4f}{st['n_big']:>6d}{st['macro_all']:>10.4f}{st['n']:>5d}")
                # ... and the pure-transfer column: the roi303-VAL threshold, nothing chosen on htw
                if roi_thr is not None and roi_thr in rows_by_thr(ht, "htw"):
                    st2 = stats(rows_by_thr(ht, "htw")[roi_thr], args.min_gt)
                    entry["htw_transfer"] = {"thr": roi_thr, "test": st2}
                    acc.setdefault("htw@roi-thr", []).append(st2)
                    print(f"{'':<9}{'htw@roi':<10}{roi_thr:>9.3f}{'-':>10}{st2['micro']:>11.4f}{st2['micro_p']:>8.3f}{st2['micro_r']:>8.3f}{st2['macro_big']:>10.4f}{st2['n_big']:>6d}{st2['macro_all']:>10.4f}{st2['n']:>5d}")
                # the other two sources of the 100-frame block, at the htw-VAL threshold, for the record
                for src in ("alice", "586"):
                    thr_s, _ = choose_thr(hv, src, args.min_gt)
                    st3 = stats(rows_by_thr(ht, src)[thr_s], args.min_gt)
                    entry[src] = {"thr": thr_s, "test": st3}
                    acc.setdefault(src, []).append(st3)
                    print(f"{'':<9}{src:<10}{thr_s:>9.3f}{'-':>10}{st3['micro']:>11.4f}{st3['micro_p']:>8.3f}{st3['micro_r']:>8.3f}{st3['macro_big']:>10.4f}{st3['n_big']:>6d}{st3['macro_all']:>10.4f}{st3['n']:>5d}")
            else:
                print(f"{seed:<9}{'htw':<10}   (missing: {'VAL ' if not hv else ''}{'TEST' if not ht else ''})")
        print("--- 3-seed mean +- sd (TEST)")
        for blk, lst in acc.items():
            if len(lst) < 2:
                continue
            m = np.array([s["micro"] for s in lst]); b = np.array([s["macro_big"] for s in lst]); a = np.array([s["macro_all"] for s in lst])
            print(f"{blk:<12} n_seeds={len(lst)}  micro {m.mean():.4f} +- {m.std(ddof=1):.4f}   macro>=10 {b.mean():.4f} +- {b.std(ddof=1):.4f}   macro-all {a.mean():.4f} +- {a.std(ddof=1):.4f}")
            report.setdefault("summary", {}).setdefault(prm, {})[blk] = {
                "micro_mean": float(m.mean()), "micro_sd": float(m.std(ddof=1)),
                "macro_big_mean": float(b.mean()), "macro_big_sd": float(b.std(ddof=1)),
                "macro_all_mean": float(a.mean()), "macro_all_sd": float(a.std(ddof=1)), "n_seeds": len(lst)}

    print("\n##### ORACLE ceiling (GT-rasterised mask through instancer A), same convention")
    for tag in ("roiTEST_der", "roiTEST_fit", "roiVAL_der", "hbTEST_der", "hbTEST_fit", "hbVAL_der"):
        print(f"{tag:<12}" + oracle_line(load(f"{D}/oracle_{tag}.json")))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

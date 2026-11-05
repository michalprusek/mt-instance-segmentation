#!/usr/bin/env python3
"""Gate before any b80 number enters a declared read: l_bridge b60 must equal l_hyst2 bridge0.98_22.5, frame by frame, at the same cut."""
import json, os
U = "/data2/prusek/mt_work/runs/ult/"; KH = "bridge0.98_22.5"; ok = True
for m in ("DEB_S3", "DEB_S4", "DEB_S5", "DEB_S6"):
    for f in ("roiv2_", "hball_", "syn96b_"):
        a = json.load(open(f"{U}fold/{f}{m}.json")); b = json.load(open(f"{U}bridge/{f}{m}.json"))
        ra = {n: r[KH] for n, _, r in a["rows"]}; rb = {n: r["b60"] for n, _, r in b["rows"]}
        same = set(ra) == set(rb) and all(all(ra[n][q] == rb[n][q] for q in ("tp", "fp", "fn")) for n in ra)
        ok &= same and a["cut"] == b["cut"]
        print(m, f, "cut", a["cut"], b["cut"], "frames", len(ra), "PASS" if same else "FAIL")
print("GATE", "PASS" if ok else "FAIL")
for m in ("AA_S3", "AA_S4", "AA_S5", "AA_S6"):
    print(m, {f: json.load(open(f"{U}hyst2/{f}{m}.json"))["cut"] for f in ("roiv2_", "hb_", "syn96b_") if os.path.exists(f"{U}hyst2/{f}{m}.json")}, {f: os.path.exists(f"{U}hyst2/{f}{m}.json") for f in ("mt34_",)})

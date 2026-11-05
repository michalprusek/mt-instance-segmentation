#!/usr/bin/env python3
"""Production check on the e2e account (docs/TODO.md 2026-11-03 08:05): upload the check frames once, segment through the real queue, save what the API returns. usage: prod_e2e.py TAG [PROJECT_ID]. Never prints credentials."""
import json, os, re, sys, time
import requests
TAG = sys.argv[1]; PID = sys.argv[2] if len(sys.argv) > 2 else None; D = os.path.dirname(os.path.abspath(__file__)) + "/../prodcheck/"
B = "https://spherosegapp.utia.cas.cz/api"; acc = open(os.path.expanduser("~/.claude/spheroseg-e2e-account.txt")).read()
em = re.search(r"email:\s*(\S+)", acc).group(1); pw = re.search(r"password:\s*(\S+)", acc).group(1)
s = requests.Session(); r = s.post(B + "/auth/login", json={"email": em, "password": pw}, timeout=30); r.raise_for_status(); assert "access_token" in r.cookies; del acc, pw   # auth travels in the session cookies
FR = ["roi_median.png", "roi_densest.png", "htw_median.tif", "htw_densest.tif", "mt34_median.tif", "mt34_densest.png"]
if PID is None:
    r = s.post(B + "/projects", json={"title": "claude e2e V16 check 2026-11-03", "description": "Deployment check, VAL frames of the research benchmark", "type": "microtubules"}, timeout=30); print(r.status_code, r.text[:300]); r.raise_for_status(); PID = r.json().get("data", r.json())["id"]
    files = [("images", (f, open(D + f, "rb"), "image/png" if f.endswith("png") else "image/tiff")) for f in FR]
    r = s.post(f"{B}/projects/{PID}/images", files=files, timeout=300); print("upload", r.status_code, r.text[:400]); r.raise_for_status()
print("project", PID)
r = s.get(f"{B}/projects/{PID}/images", params={"limit": 50}, timeout=60); r.raise_for_status(); body = r.json(); data = body.get("data", body); imgs = data.get("images", data) if isinstance(data, dict) else data
ids = {i.get("name") or i.get("originalName") or i.get("filename"): i["id"] for i in imgs}; print("images", {k: (v[:8]) for k, v in ids.items()})
r = s.post(B + "/queue/batch", json={"imageIds": list(ids.values()), "projectId": PID, "model": "microtubule", "threshold": 0.98, "forceResegment": True, "detectHoles": True}, timeout=60); print("queue", r.status_code, r.text[:300]); r.raise_for_status(); t0 = time.time()
out = {}
while time.time() - t0 < 600:
    time.sleep(5); left = 0
    for n, i in ids.items():
        if n in out: continue
        st = s.get(f"{B}/images/{i}", timeout=30)
        q = st.json().get("data", st.json()) if st.ok else {}
        stat = q.get("segmentationStatus") or (q.get("image") or {}).get("segmentationStatus")
        if stat in ("segmented", "no_segmentation", "failed"):
            rr = s.get(f"{B}/segmentation/images/{i}/results", timeout=120); res = rr.json().get("data", rr.json()) if rr.ok else {}
            meta = {k: v for k, v in res.items() if k != "polygons" and not isinstance(v, (list, dict))}; out[n] = dict(status=stat, id=i, t=round(time.time() - t0, 1), meta=meta, model=meta.get("model_used"), threshold=meta.get("threshold_used"), processingTime=meta.get("processing_time"), n=len(res.get("polygons") or []), polys=[[(p["x"], p["y"]) for p in g.get("points", [])] for g in (res.get("polygons") or [])], w=res.get("imageWidth"), h=res.get("imageHeight"))
            print(n, stat, "n", out[n]["n"], "model", out[n]["model"], "thr", out[n]["threshold"], "proc", out[n]["processingTime"], "wall", out[n]["t"], flush=True)
        else: left += 1
    if not left: break
json.dump(dict(project=PID, frames=out), open(D + f"prod_{TAG}.json", "w")); print("saved", D + f"prod_{TAG}.json", "done", len(out), "of", len(ids))

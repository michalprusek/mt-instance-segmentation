#!/usr/bin/env python3
"""Essays production check on the e2e account (docs/TODO.md 2026-11-03 10:40).
usage: prod_essays.py upload OUTDIR | prod_essays.py rerun JOB_ID OUTDIR | prod_essays.py fetch JOB_ID OUTDIR
Polls the job, samples peak GPU memory is left to the caller, downloads the result ZIP and unpacks it."""
import glob, io, json, os, re, sys, time, zipfile
import requests
B = "https://spherosegapp.utia.cas.cz/api"
ND2 = os.path.expanduser("~/PycharmProjects/BIOCEV/datasets/Alice_MT_intensity")
WELLS = ["060", "064", "066", "069"]  # MT-34 VAL, one per condition; 059 061 063 065 067 070 are TEST and stay out
acc = open(os.path.expanduser("~/.claude/spheroseg-e2e-account.txt")).read()
em = re.search(r"email:\s*(\S+)", acc).group(1); pw = re.search(r"password:\s*(\S+)", acc).group(1)
s = requests.Session(); r = s.post(B + "/auth/login", json={"email": em, "password": pw}, timeout=60); r.raise_for_status()
cmd = sys.argv[1]
if cmd == "upload":
    out = sys.argv[2]; handles = []; files = []
    for w in WELLS:
        p = glob.glob(f"{ND2}/*{w}.nd2"); assert len(p) == 1, (w, p)
        h = open(p[0], "rb"); handles.append(h); files.append(("files", (os.path.basename(p[0]), h, "application/octet-stream")))
    t0 = time.time()
    r = s.post(B + "/essays/upload", files=files, data={"options": json.dumps({"solutionName": "TIRF"}), "folderName": "claude e2e essays V16 check 2026-11-03"}, timeout=3600)
    print("upload", r.status_code, r.text[:300], f"{time.time()-t0:.0f}s", flush=True); r.raise_for_status()
    body = r.json(); job = (body.get("data") or body); job = job.get("jobId") or job.get("id") or (job.get("job") or {}).get("id")
elif cmd == "rerun":
    job = sys.argv[2]; out = sys.argv[3]
    r = s.post(f"{B}/essays/jobs/{job}/rerun", timeout=120); print("rerun", r.status_code, r.text[:300], flush=True); r.raise_for_status()
    body = r.json(); d = body.get("data") or body; job = d.get("jobId") or d.get("id") or (d.get("job") or {}).get("id") or job
else:
    job = sys.argv[2]; out = sys.argv[3]
print("job", job, flush=True); t0 = time.time(); last = None
while time.time() - t0 < 3 * 3600:
    r = s.get(f"{B}/essays/jobs/{job}", timeout=60); r.raise_for_status(); d = r.json(); d = d.get("data") or d; d = d.get("job") or d
    line = {k: d.get(k) for k in ("status", "progress", "message", "device", "deviceReason", "error")}
    if line != last: print(f"{time.time()-t0:7.0f}s", line, flush=True); last = line
    if d.get("status") in ("completed", "failed", "cancelled"): break
    time.sleep(20)
os.makedirs(out, exist_ok=True); json.dump(d, open(f"{out}/job.json", "w"), indent=1, default=str)
if d.get("status") == "completed":
    r = s.post(f"{B}/essays/jobs/{job}/download-token", timeout=60); r.raise_for_status(); t = r.json(); t = (t.get("data") or t).get("token")
    r = s.get(f"{B}/essays/jobs/{job}/download", params={"token": t}, timeout=3600); r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(out); print("unpacked", len(r.content), "bytes ->", out, flush=True)
print("PROD_ESSAYS_DONE", job, d.get("status"), flush=True)

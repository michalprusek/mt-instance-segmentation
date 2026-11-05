#!/usr/bin/env python3
"""Did neurite_soma_merged survive the merged image? Re-segment the e2e account's neurite project through the queue. Never prints credentials."""
import collections, os, re, time, requests
PID = "e46f2460-8800-41c0-967b-28d636329f6f"; B = "https://spherosegapp.utia.cas.cz/api"; acc = open(os.path.expanduser("~/.claude/spheroseg-e2e-account.txt")).read()
s = requests.Session(); r = s.post(B + "/auth/login", json={"email": re.search(r"email:\s*(\S+)", acc).group(1), "password": re.search(r"password:\s*(\S+)", acc).group(1)}, timeout=30); r.raise_for_status(); del acc
r = s.get(f"{B}/projects/{PID}/images", params={"limit": 10}, timeout=60); r.raise_for_status(); d = r.json().get("data", r.json()); imgs = d.get("images", d) if isinstance(d, dict) else d; print("images", len(imgs))
def res(i):
    x = s.get(f"{B}/segmentation/images/{i}/results", timeout=60).json(); x = x.get("data", x); P = x.get("polygons") or []
    return len(P), x.get("model_used"), x.get("updatedAt"), dict(collections.Counter((g.get("class"), g.get("type")) for g in P))
i = imgs[0]["id"]; before = res(i); print("before", before)
r = s.post(B + "/queue/batch", json={"imageIds": [i], "projectId": PID, "forceResegment": True, "threshold": 0.5}, timeout=60); print("queue", r.status_code, r.text[:160])
for _ in range(36):
    time.sleep(5); a = res(i)
    if a[2] != before[2]: break
print("after ", a)

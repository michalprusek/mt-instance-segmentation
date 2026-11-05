#!/usr/bin/env python3
"""ARM 64 / 63b recipes (docs/TODO.md 2026-11-02 14:30): V18 = V16 + debris_rate_range x 0.5; V19 = V16 + width_thin_prob 0.35 x 0.5. Refuses to overwrite."""
import json, os
V = json.load(open("calib_V16_debris.json")); assert V["gen_overrides"]["debris_rate_range"] == [0.0, 40.0] or V["gen_overrides"]["debris_rate_range"] == [0, 40]
for name, key, val, note in (("calib_V18_debris05.json", "debris_rate_range", [0.0, 20.0], "V16 with debris_rate_range x 0.5"), ("calib_V19_thin05.json", "width_thin_prob", 0.175, "V16 + thin regime at half the ARM 63 dose")):
    assert not os.path.exists(name), name
    c = json.loads(json.dumps(V)); c["gen_overrides"][key] = val; c["_arm64"] = "2026-11-02: " + note + ", docs/TODO.md 14:30"
    json.dump(c, open(name, "w"), indent=1); print("wrote", name)

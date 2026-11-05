import json, sys
s = open(sys.argv[1]).read()
old = '"THIN_S6": 0.999}'
assert s.count(old) == 1
add = ', '.join(f'"{m}": {c}' for m, c in json.load(open(sys.argv[3])).items())
s = s.replace(old, '"THIN_S6": 0.999, ' + add + '}')
a = 'if m.startswith(("DEB", "THIN"))'; assert s.count(a) == 1
s = s.replace(a, 'if m.startswith(("DEB", "THIN", "DOSE"))')
b = 'G["THIN + L"] = [(f"THIN_S{k}", "L") for k in (3, 4, 5, 6)]; G["THIN + L+H"] = [(f"THIN_S{k}", "H") for k in (3, 4, 5, 6)]'; assert s.count(b) == 1
s = s.replace(b, 'G = {"DEB + L+H": G["DEB + L+H"]}; G["DOSE + L+H"] = [(f"DOSE_S{k}", "H") for k in (3, 4, 5, 6)]; G["THIN05 + L+H"] = [(f"THIN05_S{k}", "H") for k in (3, 4, 5, 6)]; G["THIN + L+H"] = [(f"THIN_S{k}", "H") for k in (3, 4, 5, 6)]')
c = 'offgt_survive_thin.json'; assert s.count(c) == 1
s = s.replace(c, 'offgt_survive_64.json')
open(sys.argv[2], "w").write(s)

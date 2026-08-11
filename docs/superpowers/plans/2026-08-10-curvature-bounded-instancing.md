# Curvature-Bounded MT Instancing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 34-frame MT-34 real benchmark, then replace PySOAX's greedy minimum-angle
junction handling with a curvature-bounded instancer (hard `|dθ/ds| ≤ κ_max`), tuned oracle-first
and then run on semantic-model masks; plus a synth-trained nnU-Net as a candidate semantic model.

**Architecture:** Two instancers share one geometry core. **A** = skeleton → junction-cluster
contraction → arcs → window-fitted tangents → per-junction min-cost perfect matching with a hard
curvature forbid → chaining → gap linking. **B** = lift the K=6 amodal orientation channels into a
joint (pixel, θ-bin) graph where a bin transition is a priced edge, and extract instances as
curvature-penalized shortest paths, removing only `(p,θ)` nodes so a crossing MT keeps its pixel.

**Tech Stack:** Python 3, NumPy, SciPy, scikit-image, networkx, h5py, tifffile, Optuna, pytest.
GPU work (v4b inference, nnU-Net) on tulen via `~/dinov3_env` / `~/nnunet_env`.

## Global Constraints

- **Synth-only holds.** Instancer and semantic model train on synthetic data only. The MT-34
  human annotations are **test data**. Hyperparameters may be tuned on the VAL split only.
- **Primary metric unchanged:** `centerline_f1(..., tol=5.0, length_coverage=0.95,
  precision_coverage=0.95)` at **UP = 1.5**. Never change it — comparability with 0.697 / 0.519 /
  0.326 depends on it.
- **Eval scale convention (verified on tulen):** image is `scipy.ndimage.zoom(img, 1.5, order=1)`;
  GT polylines are **pre-scaled by 1.5** (`alice_gt/*.npy` max x = 1677.1 = 1118.1 × 1.5).
  Repo-side `.h5` stores **native 1×** coords (alice_eval convention); the ×1.5 happens at eval.
- **Polyline vertex order is `(x=col, y=row)`** — the transpose of NumPy `[row, col]`. Every
  conversion must be covered by a test.
- **HTW is sealed.** Do not touch `data/real/htw_eval/`.
- **Compute on tulen** (`ssh prusek@tulen.utia.cas.cz`), never the local MacBook. Visual results
  go to `data/enc_sensitivity_testset/`.
- **The repo is not a git repository** — replace every "commit" step with the stated verification
  command. Do not run `git init`.
- Best semantic model = `dino_seg_ori_v4b.pth` (`SEG_ARCH=base`, `SEG_MODE=ori`, K=6).

---

## File Structure

| file | responsibility |
|---|---|
| `requirements.txt` | pinned deps for the local CPU-side work |
| `pytest.ini` | test discovery config |
| `src/mt_bench/__init__.py` | package marker |
| `src/mt_bench/cvat_import.py` | CVAT XML → polylines; h5 writer; split assignment |
| `src/mt_bench/build_mt34.py` | driver: build `data/real/mt34_eval/` from CVAT + images |
| `src/mt_bench/overlay.py` | polyline-on-image overlay renders (verification gate) |
| `src/mt_bench/gt_stats.py` | κ_max estimation, crossing/parallel characterization |
| `src/instance/__init__.py` | package marker |
| `src/instance/geometry.py` | resample, window tangent/curvature, turn angle, κ bound |
| `src/instance/skeleton_graph.py` | skeleton → junction contraction → arcs |
| `src/instance/matching.py` | per-junction min-cost matching with hard κ forbid |
| `src/instance/instancer_a.py` | assembles 1–7 into the A instancer |
| `src/instance/lifted.py` | (pixel, θ) graph construction |
| `src/instance/instancer_b.py` | curvature-penalized path extraction in the lifted graph |
| `src/instance/oracle.py` | GT → oracle binary mask + oracle K=6 orientation channels |
| `src/instance/metrics.py` | centerline_f1 (ported) + junction-identity / fragmentation / bundle |
| `tests/…` | one test module per source module |
| `scripts/run_oracle_eval.py` | oracle benchmark driver (PySOAX baseline, A, B) |
| `scripts/tune_instancer.py` | Optuna on VAL |

---

## Task 1: Repo tooling + package skeleton

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `src/mt_bench/__init__.py`,
  `src/instance/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: importable packages `mt_bench` and `instance` when run from the repo root.

- [ ] **Step 1: Write `requirements.txt`**

```
numpy>=1.24
scipy>=1.11
scikit-image>=0.22
networkx>=3.2
h5py>=3.10
tifffile>=2024.1.30
matplotlib>=3.8
optuna>=3.5
pytest>=8.0
```

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = src
addopts = -q
```

- [ ] **Step 3: Create empty `__init__.py` for `src/mt_bench`, `src/instance`, `tests`**

- [ ] **Step 4: Verify**

Run: `python -m pytest --collect-only`
Expected: exits 0, "no tests ran" is fine.

---

## Task 2: CVAT import — polylines with a tested transpose

**Files:**
- Create: `src/mt_bench/cvat_import.py`, `tests/test_cvat_import.py`

**Interfaces:**
- Produces:
  - `parse_cvat_xml(path: str) -> list[FrameAnn]` where
    `FrameAnn = {"frame_id": int, "name": str, "width": int, "height": int,
    "polylines": list[np.ndarray], "sources": list[str]}` and each polyline is
    `(N, 2) float64` in `(x=col, y=row)`.
  - `write_frame_h5(out_path, image: np.ndarray, polylines, attrs: dict) -> None`
    writing `image`, `polylines/pl_0000…`, attrs `height,width,n_polylines`.
  - `assign_split(names: list[str]) -> dict[str, str]` mapping name → `"val"`/`"test"`,
    alternating over the sorted list (even index → val).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, textwrap, h5py
from mt_bench.cvat_import import parse_cvat_xml, write_frame_h5, assign_split

XML = textwrap.dedent("""\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><id>1</id></task></meta>
  <image id="0" name="a.tif" width="100" height="50">
    <polyline label="microtubule" source="manual" points="1.00,2.00;3.00,4.00"/>
    <polyline label="microtubule" source="file" points="5.50,6.50;7.00,8.00;9.00,10.00"/>
  </image>
  <image id="1" name="b.tif" width="100" height="50"/>
</annotations>
""")

def test_parse_keeps_x_col_y_row_order(tmp_path):
    p = tmp_path / "ann.xml"; p.write_text(XML)
    frames = parse_cvat_xml(str(p))
    assert len(frames) == 2
    f0 = frames[0]
    assert f0["name"] == "a.tif" and f0["width"] == 100 and f0["height"] == 50
    assert len(f0["polylines"]) == 2
    # first vertex is (x=1, y=2): x is the COLUMN, y is the ROW
    np.testing.assert_allclose(f0["polylines"][0], [[1.0, 2.0], [3.0, 4.0]])
    assert f0["sources"] == ["manual", "file"]
    assert frames[1]["polylines"] == []

def test_write_frame_h5_roundtrip(tmp_path):
    img = np.arange(50 * 100, dtype=np.float32).reshape(50, 100)
    pls = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    out = tmp_path / "f.h5"
    write_frame_h5(str(out), img, pls, {"split": "val"})
    with h5py.File(out, "r") as h:
        assert h.attrs["height"] == 50 and h.attrs["width"] == 100
        assert h.attrs["n_polylines"] == 1 and h.attrs["split"] == "val"
        np.testing.assert_allclose(h["polylines/pl_0000"][:], pls[0])
        assert h["image"].shape == (50, 100)

def test_assign_split_alternates_over_sorted_names():
    got = assign_split(["c", "a", "b", "d"])
    assert got == {"a": "val", "b": "test", "c": "val", "d": "test"}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_cvat_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mt_bench.cvat_import'`

- [ ] **Step 3: Implement `src/mt_bench/cvat_import.py`**

```python
"""CVAT 'CVAT for images 1.1' XML → microtubule polylines, and the on-disk h5 format.

Coordinate convention: CVAT `points="x,y;x,y"` are (x=col, y=row) — the transpose of
NumPy [row, col] indexing — and we keep them in that order end-to-end, because
`alice_eval/*.h5` and `centerline_f1` both expect (x, y).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import h5py
import numpy as np


def _parse_points(text: str) -> np.ndarray:
    return np.array([[float(v) for v in pair.split(",")]
                     for pair in text.strip().split(";")], dtype=np.float64)


def parse_cvat_xml(path: str) -> list[dict]:
    root = ET.parse(path).getroot()
    frames = []
    for im in root.findall("image"):
        polylines, sources = [], []
        for pl in im.findall("polyline"):
            pts = _parse_points(pl.get("points"))
            if len(pts) >= 2:
                polylines.append(pts)
                sources.append(pl.get("source", "unknown"))
        frames.append({
            "frame_id": int(im.get("id")),
            "name": im.get("name"),
            "width": int(im.get("width")),
            "height": int(im.get("height")),
            "polylines": polylines,
            "sources": sources,
        })
    return frames


def write_frame_h5(out_path: str, image: np.ndarray,
                   polylines, attrs: dict | None = None) -> None:
    image = np.asarray(image, dtype=np.float32)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("image", data=image, compression="gzip")
        grp = h.create_group("polylines")
        for i, p in enumerate(polylines):
            grp.create_dataset(f"pl_{i:04d}", data=np.asarray(p, dtype=np.float64))
        h.attrs["height"] = int(image.shape[0])
        h.attrs["width"] = int(image.shape[1])
        h.attrs["n_polylines"] = int(len(polylines))
        for k, v in (attrs or {}).items():
            h.attrs[k] = v


def assign_split(names) -> dict:
    """Alternating val/test over the sorted names (even index → val).

    Matches the existing project convention of a deterministic alternating
    6-val/6-test split of the sorted Alice frames.
    """
    return {n: ("val" if i % 2 == 0 else "test")
            for i, n in enumerate(sorted(names))}
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cvat_import.py -v`
Expected: 3 passed.

---

## Task 3: Build the MT-34 dataset on disk

**Files:**
- Create: `src/mt_bench/build_mt34.py`, `data/real/mt34_eval/README.md`
- Uses: exported CVAT XML for tasks 585 and 586 (already in the session scratchpad); source
  images from tulen.

**Interfaces:**
- Consumes: `parse_cvat_xml`, `write_frame_h5`, `assign_split` from Task 2.
- Produces: `data/real/mt34_eval/{*.h5, tif/*.tif, manifest.csv}` and a printed summary.
  `manifest.csv` columns: `name,source_task,frame_id,width,height,n_polylines,n_manual,split,reviewed`.

- [ ] **Step 1: Locate the source images on tulen**

Run: `ssh prusek@tulen.utia.cas.cz 'ls ~/BIOCEV/datasets/microtubules/cvat_export | head'`
The 22 frames are `training_img_{1,10,100..109,11,110..118}.tif`. Alice tifs already exist locally
at `data/real/alice_eval/tif/`.

- [ ] **Step 2: Copy the 22 tifs locally**

```bash
mkdir -p data/real/mt34_eval/tif
scp prusek@tulen.utia.cas.cz:'~/BIOCEV/datasets/microtubules/cvat_export/training_img_{1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,118}.tif' data/real/mt34_eval/tif/
```

If that path does not hold the tifs, fall back to `~/mt_enc_exp/newdata/s7_general_mt.zip`.

- [ ] **Step 3: Write `src/mt_bench/build_mt34.py`**

It must: parse both XMLs; take **all** frames of 585 and frames with `frame_id <= 21` of 586;
verify each image's on-disk shape equals the XML `(width, height)` (**abort on mismatch** — a
mismatch means the image files do not correspond to the annotations); write one `.h5` per frame
with native 1× coords; compute the split **separately per source** (Alice 6/6, new-22 11/11) so
each source is balanced; record `reviewed = n_manual > 0`; write `manifest.csv`.

- [ ] **Step 4: Run the builder**

Run: `python -m mt_bench.build_mt34`
Expected: 34 h5 files, 957 polylines total (229 + 728), no shape mismatches.

- [ ] **Step 5: Write `data/real/mt34_eval/README.md`**

Document: composition table, provenance (CVAT tasks 585/586 + job 557), **native 1× coords with
the ×1.5 eval convention spelled out**, `(x=col, y=row)` order, mixed frame sizes, the split, the
three edge-case frames, and both caveats (calibration-corpus leakage; PySOAX-seeded GT agreement
bias).

- [ ] **Step 6: Verify**

Run: `python -c "import glob,h5py;fs=glob.glob('data/real/mt34_eval/*.h5');print(len(fs),sum(h5py.File(f)['image'].attrs.get('n_polylines',0) if False else h5py.File(f).attrs['n_polylines'] for f in fs))"`
Expected: `34 957`

---

## Task 4: Overlay verification gate

**Files:**
- Create: `src/mt_bench/overlay.py`, `tests/test_overlay.py`

**Interfaces:**
- Produces: `render_overlay(image, polylines, out_png, title=None) -> None` and
  `render_all(h5_dir, out_dir) -> int` (returns frames rendered).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from mt_bench.overlay import render_overlay

def test_overlay_draws_polyline_at_row_y_col_x(tmp_path):
    # A 1-px-wide bright image; polyline (x=col) 10..10, (y=row) 2..8 is VERTICAL.
    img = np.zeros((20, 30), dtype=np.float32)
    pl = np.array([[10.0, 2.0], [10.0, 8.0]])
    out = tmp_path / "o.png"
    render_overlay(img, [pl], str(out))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_overlay.py -v` → FAIL.

- [ ] **Step 3: Implement** — matplotlib, `imshow(img, cmap="gray")` then
  `plot(pl[:,0], pl[:,1])` per polyline (x → horizontal axis, y → vertical), one distinct color
  per instance, `savefig(dpi=110)`, `close()`.

- [ ] **Step 4: Run tests** — expected PASS.

- [ ] **Step 5: Render all 34 and INSPECT**

Run: `python -c "from mt_bench.overlay import render_all; print(render_all('data/real/mt34_eval','data/enc_sensitivity_testset/mt34_overlays'))"`
Expected: `34`. **Then open several PNGs and confirm polylines lie on microtubules.** If they are
transposed or offset, stop and fix before any metric is computed.

---

## Task 5: Geometry core — resampling, window tangent, curvature

**Files:**
- Create: `src/instance/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `resample(points: np.ndarray, ds: float) -> np.ndarray` — arc-length resampling, `(x, y)`.
  - `polyline_curvature(points, ds: float) -> np.ndarray` — `|Δθ|/Δs` per interior vertex, rad/px.
  - `window_tangent(points, end: str, window: float) -> tuple[float, float]` — returns
    `(theta, kappa)`; `theta` is the **outgoing** direction at `end ∈ {"start","end"}` in radians,
    estimated by PCA over the vertices within `window` px of that end; `kappa` is the signed
    curvature from a circle fit over the same window (0.0 if the fit is degenerate).
  - `turn_penalty(theta_in: float, theta_out: float) -> float` — absolute turn in `[0, π]`
    between an incoming heading and an outgoing heading.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.geometry import resample, polyline_curvature, window_tangent, turn_penalty

def test_resample_gives_constant_spacing():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    out = resample(pts, ds=1.0)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(d, 1.0, atol=1e-6)
    assert np.allclose(out[0], [0.0, 0.0]) and np.allclose(out[-1], [10.0, 10.0], atol=1.0)

def test_straight_line_has_zero_curvature():
    pts = resample(np.array([[0.0, 0.0], [50.0, 0.0]]), ds=1.0)
    assert np.max(polyline_curvature(pts, ds=1.0)) < 1e-6

def test_right_angle_corner_has_large_curvature():
    pts = resample(np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]), ds=1.0)
    assert np.max(polyline_curvature(pts, ds=1.0)) > 1.0  # ~pi/2 rad over 1 px

def test_circle_curvature_matches_inverse_radius():
    R, t = 40.0, np.linspace(0, np.pi / 2, 400)
    pts = np.stack([R * np.cos(t), R * np.sin(t)], axis=1)
    k = polyline_curvature(resample(pts, ds=1.0), ds=1.0)
    assert abs(np.median(k) - 1.0 / R) < 0.15 / R

def test_window_tangent_points_outward_along_x():
    # Horizontal line; the "end" tangent must point in +x (outgoing), the "start" in -x.
    pts = resample(np.array([[0.0, 5.0], [60.0, 5.0]]), ds=1.0)
    th_end, _ = window_tangent(pts, "end", window=12.0)
    th_start, _ = window_tangent(pts, "start", window=12.0)
    assert abs(np.cos(th_end) - 1.0) < 1e-3
    assert abs(np.cos(th_start) + 1.0) < 1e-3

def test_window_tangent_beats_one_pixel_estimate_on_shallow_angle():
    # 20-degree line: a single 8-connected step can only report 0 or 45 degrees.
    ang = np.deg2rad(20.0)
    pts = resample(np.array([[0.0, 0.0], [60 * np.cos(ang), 60 * np.sin(ang)]]), ds=1.0)
    pts = np.round(pts)  # pixel quantization, as a real skeleton would be
    th, _ = window_tangent(pts, "end", window=12.0)
    assert abs(np.rad2deg(th) - 20.0) < 4.0

def test_turn_penalty_is_zero_for_collinear_through_path():
    # Arm A arrives heading +x; arm B leaves heading +x -> a straight through-path.
    assert turn_penalty(0.0, 0.0) < 1e-9
    assert abs(turn_penalty(0.0, np.pi) - np.pi) < 1e-9
```

- [ ] **Step 2: Run to verify failure** — FAIL, module missing.

- [ ] **Step 3: Implement `src/instance/geometry.py`.**
  `resample`: cumulative chord length + `np.interp` per coordinate.
  `polyline_curvature`: `theta = arctan2(dy, dx)` on segments; wrap the difference to `(-π, π]`
  with `np.arctan2(np.sin(d), np.cos(d))`; divide by `ds`.
  `window_tangent`: select vertices within `window` px of the chosen end; PCA (first eigenvector
  of the centred covariance); orient it **outward** (away from the polyline body) by checking the
  sign of its dot product with `end_vertex − window_centroid`; `kappa` from a least-squares circle
  fit (`A = [2x, 2y, 1]`, `b = x²+y²`), returning `0.0` when the normal equations are singular.
  `turn_penalty`: `|wrap(theta_out − theta_in)|` — but note the calling convention in
  `matching.py` passes the **incoming heading** and the **outgoing heading**, so collinear = 0.

- [ ] **Step 4: Run tests** — expected 7 passed.

---

## Task 6: κ_max from GT + benchmark characterization

**Files:**
- Create: `src/mt_bench/gt_stats.py`, `tests/test_gt_stats.py`

**Interfaces:**
- Consumes: `polyline_curvature`, `resample` (Task 5).
- Produces:
  - `curvature_quantile(polylines, ds=2.0, q=99.5) -> float` — κ_max in rad/px.
  - `count_crossings(polylines, tol=2.0) -> list[dict]` with keys `i, j, x, y, angle_deg`.
  - `count_parallel_pairs(polylines, gap_lo=2.0, gap_hi=6.0, min_len=20.0) -> int`.
  - `characterize(h5_dir) -> dict` — per-source aggregates + a κ histogram PNG.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from mt_bench.gt_stats import curvature_quantile, count_crossings, count_parallel_pairs

def test_curvature_quantile_of_straight_lines_is_near_zero():
    pls = [np.array([[0.0, float(i)], [100.0, float(i)]]) for i in range(10)]
    assert curvature_quantile(pls, ds=2.0, q=99.5) < 1e-3

def test_count_crossings_finds_one_perpendicular_x():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])   # horizontal
    b = np.array([[50.0, 0.0], [50.0, 100.0]])   # vertical
    cr = count_crossings([a, b], tol=2.0)
    assert len(cr) == 1
    assert abs(cr[0]["angle_deg"] - 90.0) < 5.0

def test_count_crossings_ignores_disjoint_lines():
    a = np.array([[0.0, 10.0], [100.0, 10.0]])
    b = np.array([[0.0, 80.0], [100.0, 80.0]])
    assert count_crossings([a, b], tol=2.0) == []

def test_count_parallel_pairs_detects_close_bundle():
    a = np.array([[0.0, 50.0], [100.0, 50.0]])
    b = np.array([[0.0, 54.0], [100.0, 54.0]])   # 4 px apart, in [2, 6]
    far = np.array([[0.0, 200.0], [100.0, 200.0]])
    assert count_parallel_pairs([a, b, far]) == 1
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** `count_crossings`: resample both polylines at ds=1, `cKDTree`
  query_ball_tree at `tol`, cluster contiguous hits, and for each cluster compute the angle
  between the two local tangents (fold to `[0, 90]`). Skip clusters at a shared endpoint.
  `count_parallel_pairs`: for each pair, the fraction of arc length where the nearest-neighbour
  distance falls in `[gap_lo, gap_hi]`; count the pair if that length ≥ `min_len`.

- [ ] **Step 4: Run tests** — expected 4 passed.

- [ ] **Step 5: Characterize MT-34 and record κ_max**

Run: `python -m mt_bench.gt_stats data/real/mt34_eval`
Expected: prints per-source MT/frame, crossings/frame + angle histogram, parallel pairs/frame, and
κ_max. **Write the resulting κ_max into `docs/protocol.md`** and cross-check it against the
literature breaking curvature 0.43 µm⁻¹ (Alice pixel size must be stated; if unknown, report
κ_max in rad/px only and say so).

---

## Task 7: Oracle masks + oracle orientation channels

**Files:**
- Create: `src/instance/oracle.py`, `tests/test_oracle.py`

**Interfaces:**
- Produces:
  - `oracle_mask(polylines, shape, half_width=1.0, up=1.5) -> np.ndarray[bool]` — union mask in
    the **upscaled** frame; `shape` is the native `(H, W)`.
  - `oracle_ori_channels(polylines, shape, K=6, half_width=1.0, up=1.5) -> np.ndarray` —
    `(K, H*up, W*up)` float32; each MT is painted into the channel of its **local** tangent bin,
    so a crossing writes into two channels (amodal).
  - `oracle_instance_masks(polylines, shape, half_width=1.0, up=1.5) -> list[np.ndarray]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.oracle import oracle_mask, oracle_ori_channels

def test_oracle_mask_is_in_upscaled_frame():
    m = oracle_mask([np.array([[0.0, 5.0], [20.0, 5.0]])], (30, 40), up=1.5)
    assert m.shape == (45, 60) and m.any()

def test_oracle_mask_marks_row_y_times_up():
    m = oracle_mask([np.array([[0.0, 5.0], [39.0, 5.0]])], (30, 40), half_width=1.0, up=1.5)
    rows = np.where(m.any(axis=1))[0]
    assert abs(rows.mean() - 7.5) < 1.6   # y=5 -> row 7.5 after 1.5x

def test_crossing_writes_into_two_orientation_channels():
    horiz = np.array([[0.0, 25.0], [49.0, 25.0]])
    vert = np.array([[25.0, 0.0], [25.0, 49.0]])
    ch = oracle_ori_channels([horiz, vert], (50, 50), K=6, up=1.5)
    # at the crossing pixel, at least two DIFFERENT channels must be lit
    r = int(round(25 * 1.5)); c = int(round(25 * 1.5))
    lit = (ch[:, r - 2:r + 3, c - 2:c + 3].max(axis=(1, 2)) > 0.5).sum()
    assert lit >= 2
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** Scale polylines by `up`, resample at ds=0.5, stamp a disk of radius
  `half_width` at each sample. For channels, compute the local tangent angle mod 180°, map to bin
  `int(theta_deg // (180/K)) % K`, stamp into that channel.

- [ ] **Step 4: Run tests** — expected 3 passed.

---

## Task 8: Metrics module — port + new diagnostics

**Files:**
- Create: `src/instance/metrics.py`, `tests/test_metrics.py`
- Source: copy `centerline_f1`, `aggregate_f1`, `rasterise_polyline` **verbatim** from
  `tulen:~/mt_enc_exp/scripts/centerline_f1.py` (a local copy is in the session scratchpad).

**Interfaces:**
- Produces: `centerline_f1`, `aggregate_f1`, `rasterise_polyline` (unchanged semantics), plus
  - `junction_identity(pred_instances, gt_polylines, crossings, tol=5.0) -> dict` with
    `{"n_crossings", "n_preserved", "rate", "by_angle": {...}}`; a crossing is *preserved* when
    each of the two GT MTs is covered near the crossing by a **single** predicted instance that
    also covers both sides of the crossing.
  - `fragmentation(pred_instances, gt_polylines, tol=5.0) -> float` — mean number of predicted
    instances covering ≥20% of a GT MT.
  - `bundle_recovery(pred_instances, gt_polylines, pairs) -> float`.
  - `max_curvature(pred_polylines, ds=2.0) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.metrics import centerline_f1, junction_identity, fragmentation, max_curvature
from instance.oracle import oracle_instance_masks

def test_centerline_f1_perfect_prediction_scores_one():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    masks = oracle_instance_masks(gt, (60, 60), half_width=1.0, up=1.0)
    r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0

def test_fragmentation_counts_a_split_microtubule_as_two():
    gt = [np.array([[5.0, 20.0], [55.0, 20.0]])]
    left = oracle_instance_masks([np.array([[5.0, 20.0], [28.0, 20.0]])], (60, 60), up=1.0)[0]
    right = oracle_instance_masks([np.array([[32.0, 20.0], [55.0, 20.0]])], (60, 60), up=1.0)[0]
    assert fragmentation([left, right], gt, tol=5.0) == 2.0

def test_junction_identity_zero_when_both_mts_are_cut_at_the_crossing():
    h = np.array([[5.0, 30.0], [55.0, 30.0]])
    v = np.array([[30.0, 5.0], [30.0, 55.0]])
    crossings = [{"i": 0, "j": 1, "x": 30.0, "y": 30.0, "angle_deg": 90.0}]
    frags = [np.array([[5.0, 30.0], [27.0, 30.0]]), np.array([[33.0, 30.0], [55.0, 30.0]]),
             np.array([[30.0, 5.0], [30.0, 27.0]]), np.array([[30.0, 33.0], [30.0, 55.0]])]
    masks = [oracle_instance_masks([f], (60, 60), up=1.0)[0] for f in frags]
    r = junction_identity(masks, [h, v], crossings, tol=5.0)
    assert r["n_crossings"] == 1 and r["n_preserved"] == 0

def test_junction_identity_one_when_both_pass_through_intact():
    h = np.array([[5.0, 30.0], [55.0, 30.0]])
    v = np.array([[30.0, 5.0], [30.0, 55.0]])
    crossings = [{"i": 0, "j": 1, "x": 30.0, "y": 30.0, "angle_deg": 90.0}]
    masks = [oracle_instance_masks([p], (60, 60), up=1.0)[0] for p in (h, v)]
    r = junction_identity(masks, [h, v], crossings, tol=5.0)
    assert r["n_preserved"] == 1 and r["rate"] == 1.0

def test_max_curvature_flags_a_kinked_prediction():
    kinked = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]])
    assert max_curvature([kinked], ds=1.0) > 1.0
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** Port the three functions verbatim; add the diagnostics. For
  `junction_identity`, sample each GT MT at ±`W_j = 15` px along its arc from the crossing point,
  find which predicted instances cover the "before" and "after" samples within `tol`, and count
  the crossing as preserved for that MT when the **same** instance covers both. A crossing counts
  as preserved only when it is preserved for **both** MTs.

- [ ] **Step 4: Run tests** — expected 5 passed, plus the ported smoke test.

---

## Task 9: Skeleton graph — junction contraction and arcs

**Files:**
- Create: `src/instance/skeleton_graph.py`, `tests/test_skeleton_graph.py`

**Interfaces:**
- Produces: `build_arc_graph(mask, merge_radius=3.0, min_arc_len=3) -> ArcGraph` where
  `ArcGraph` is a dataclass with `arcs: list[np.ndarray]` (each `(N,2)` in `(x=col, y=row)`),
  `junctions: list[np.ndarray]` (junction centroids, `(x, y)`), and
  `arc_ends: list[tuple[int|None, int|None]]` — for each arc, the junction index at its start and
  end (`None` = free endpoint).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.oracle import oracle_mask
from instance.skeleton_graph import build_arc_graph

def _mask(polylines, shape=(80, 80)):
    return oracle_mask(polylines, shape, half_width=1.0, up=1.0)

def test_isolated_line_yields_one_arc_and_no_junctions():
    g = build_arc_graph(_mask([np.array([[10.0, 40.0], [70.0, 40.0]])]))
    assert len(g.arcs) == 1 and len(g.junctions) == 0
    assert g.arc_ends[0] == (None, None)

def test_x_crossing_yields_one_junction_and_four_arms():
    h = np.array([[10.0, 40.0], [70.0, 40.0]])
    v = np.array([[40.0, 10.0], [40.0, 70.0]])
    g = build_arc_graph(_mask([h, v]))
    assert len(g.junctions) == 1, "the Y-Y bridge must be contracted into ONE junction"
    arms = sum(1 for a, b in g.arc_ends for e in (a, b) if e == 0)
    assert arms == 4

def test_arc_vertices_are_x_col_y_row():
    g = build_arc_graph(_mask([np.array([[10.0, 40.0], [70.0, 40.0]])]))
    arc = g.arcs[0]
    assert abs(arc[:, 1].mean() - 40.0) < 2.0   # y (row) is constant at 40
    assert arc[:, 0].ptp() > 50.0               # x (col) spans the line
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** `skeletonize` → neighbour count via 3×3 convolution → endpoint mask
  (`==1`) and junction mask (`>=3`). **Contract** junction pixels: label the junction mask with a
  dilation of `merge_radius` so nearby degree-3 nodes (the Y–Y bridge) merge into one component;
  the junction position is that component's centroid. Remove junction pixels from the skeleton;
  each remaining connected component is an arc — order its pixels by walking from one endpoint.
  Attach each arc end to a junction when it is within `merge_radius + 1.5` px of that junction's
  pixels. Convert to `(x, y)` on output.

- [ ] **Step 4: Run tests** — expected 3 passed.

---

## Task 10: Curvature-bounded junction matching

**Files:**
- Create: `src/instance/matching.py`, `tests/test_matching.py`

**Interfaces:**
- Consumes: `geometry.window_tangent`, `geometry.turn_penalty`.
- Produces:
  - `ArmEnd` dataclass: `arc_idx: int`, `which: str` (`"start"`/`"end"`), `theta: float`
    (**outgoing**, i.e. pointing away from the junction), `kappa: float`, `pos: np.ndarray`.
  - `match_junction(arms: list[ArmEnd], kappa_max: float, w_theta=1.0, w_kappa=10.0,
    w_gap=0.02, c_open=1.2, gap_len=1.0) -> list[tuple[int, int]]` — index pairs into `arms`;
    unmatched arms are simply absent. Pairs whose implied `|Δθ| / gap_len > kappa_max` are
    **forbidden** (never returned).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.matching import ArmEnd, match_junction

def _arm(i, which, deg, kappa=0.0, pos=(0.0, 0.0)):
    return ArmEnd(arc_idx=i, which=which, theta=np.deg2rad(deg), kappa=kappa,
                  pos=np.array(pos, dtype=float))

def test_perpendicular_x_pairs_opposite_arms():
    # Outgoing headings of the four arms of a + shaped crossing.
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 90), _arm(3, "start", 270)]
    pairs = {tuple(sorted(p)) for p in match_junction(arms, kappa_max=0.5, gap_len=4.0)}
    assert pairs == {(0, 1), (2, 3)}

def test_shallow_crossing_still_pairs_the_collinear_arms():
    # 20-degree crossing: arms at 0/180 and 20/200.
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 20), _arm(3, "start", 200)]
    pairs = {tuple(sorted(p)) for p in match_junction(arms, kappa_max=0.5, gap_len=4.0)}
    assert pairs == {(0, 1), (2, 3)}, "must not pair 0 with 3 (a 160-degree kink)"

def test_kappa_max_forbids_a_sharp_join():
    # Only two arms, meeting at 90 degrees over a 4 px gap -> 0.39 rad/px.
    arms = [_arm(0, "end", 0), _arm(1, "start", 270)]
    assert match_junction(arms, kappa_max=0.05, gap_len=4.0) == []
    assert len(match_junction(arms, kappa_max=0.5, gap_len=4.0)) == 1

def test_t_junction_leaves_the_stem_open():
    # Through-line (0/180) plus one stem at 90: the stem must stay unmatched.
    arms = [_arm(0, "end", 0), _arm(1, "start", 180), _arm(2, "end", 90)]
    pairs = match_junction(arms, kappa_max=0.5, gap_len=4.0)
    assert len(pairs) == 1 and tuple(sorted(pairs[0])) == (0, 1)

def test_curvature_continuity_breaks_a_tie():
    # Two candidate partners at the same angle; prefer the one with matching curvature.
    arms = [_arm(0, "end", 0, kappa=0.05),
            _arm(1, "start", 180, kappa=0.05),
            _arm(2, "start", 180, kappa=-0.30)]
    pairs = match_junction(arms, kappa_max=0.5, w_kappa=10.0, gap_len=4.0)
    assert tuple(sorted(pairs[0])) == (0, 1)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.**
  Two arms `i, j` form a through-path when arm `i`'s **incoming** heading (`theta_i + π`)
  continues into arm `j`'s **outgoing** heading (`theta_j`). So
  `dtheta = turn_penalty(arms[i].theta + np.pi, arms[j].theta)`.
  Forbid the pair when `dtheta / gap_len > kappa_max`.
  Otherwise `cost = w_theta * dtheta + w_kappa * |kappa_i + kappa_j| + w_gap * gap_len`
  (the curvature signs are opposite across a junction because the outgoing tangents point in
  opposite directions, hence the `+`).
  Solve with `networkx.max_weight_matching(G, maxcardinality=False)` on a graph whose edge weight
  is `2 * c_open - cost` (so an edge is only taken when it beats leaving **both** arms open),
  keeping only edges with positive weight.

- [ ] **Step 4: Run tests** — expected 5 passed.

---

## Task 11: Instancer A end-to-end

**Files:**
- Create: `src/instance/instancer_a.py`, `tests/test_instancer_a.py`

**Interfaces:**
- Produces: `instance_a(mask, kappa_max, params: dict | None = None) ->
  tuple[list[np.ndarray], list[np.ndarray]]` returning `(polylines, masks)`; `polylines` are
  `(N,2)` `(x, y)` and `masks` are bool arrays of `mask.shape`. `params` keys: `merge_radius`,
  `window`, `w_theta`, `w_kappa`, `w_gap`, `c_open`, `min_length`, `link_max_gap`, `ds`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.oracle import oracle_mask
from instance.instancer_a import instance_a
from instance.geometry import polyline_curvature

KMAX = 0.30

def test_two_isolated_lines_give_two_instances():
    a = np.array([[10.0, 20.0], [90.0, 20.0]])
    b = np.array([[10.0, 70.0], [90.0, 70.0]])
    m = oracle_mask([a, b], (100, 100), half_width=1.0, up=1.0)
    pls, _ = instance_a(m, KMAX)
    assert len(pls) == 2

def test_perpendicular_crossing_gives_two_instances_not_four():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    m = oracle_mask([h, v], (100, 100), half_width=1.0, up=1.0)
    pls, _ = instance_a(m, KMAX)
    assert len(pls) == 2, f"crossing must not fragment; got {len(pls)}"
    assert all(len(p) > 60 for p in pls), "each instance must span the whole line"

def test_shallow_crossing_gives_two_instances():
    ang = np.deg2rad(15.0)
    c = np.array([50.0, 50.0])
    h = np.array([c - [40, 0], c + [40, 0]])
    d = np.array([c - [40 * np.cos(ang), 40 * np.sin(ang)],
                  c + [40 * np.cos(ang), 40 * np.sin(ang)]])
    m = oracle_mask([h, d], (100, 100), half_width=1.0, up=1.0)
    pls, _ = instance_a(m, KMAX)
    assert len(pls) == 2

def test_output_polylines_respect_the_curvature_bound():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    m = oracle_mask([h, v], (100, 100), half_width=1.0, up=1.0)
    pls, _ = instance_a(m, KMAX)
    for p in pls:
        assert np.max(polyline_curvature(p, ds=2.0)) <= KMAX + 1e-6

def test_close_parallels_stay_two_instances():
    a = np.array([[10.0, 50.0], [90.0, 50.0]])
    b = np.array([[10.0, 54.0], [90.0, 54.0]])
    m = oracle_mask([a, b], (100, 100), half_width=1.0, up=1.0)
    pls, _ = instance_a(m, KMAX)
    assert len(pls) == 2
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** `build_arc_graph` → resample each arc at `ds` → for every arc end
  attached to a junction build an `ArmEnd` with `window_tangent` → `match_junction` per junction
  (with `gap_len` = twice `merge_radius`) → union-find over the matched `(arc, which)` pairs →
  concatenate each chain in traversal order, inserting a straight bridge through the junction
  centroid → drop chains shorter than `min_length` → gap-link the remaining free ends across
  `link_max_gap` using the same cost and the same hard κ forbid → resample and lightly smooth the
  final polylines (`scipy.ndimage.uniform_filter1d`, size 5, `mode="nearest"`), then **assert** the
  κ bound holds and, if a vertex still violates it, re-smooth that neighbourhood until it does.
  Rasterize each polyline with `half_width=1.0` for the returned masks.

- [ ] **Step 4: Run tests** — expected 5 passed.

- [ ] **Step 5: Run the full local test suite** — `python -m pytest -q` → all green.

---

## Task 12: Oracle benchmark — PySOAX baseline vs A

**Files:**
- Create: `scripts/run_oracle_eval.py`
- Modify: `docs/protocol.md` (append the results section)

**Interfaces:**
- Consumes: everything above. PySOAX is imported from
  `tulen:~/BIOCEV/code/microtubules/convnext_instance_seg/pysoax.py` (copy it into
  `third_party/pysoax.py` locally, unmodified, with a provenance header).

- [ ] **Step 1: Copy PySOAX locally**

```bash
mkdir -p third_party
scp prusek@tulen.utia.cas.cz:'~/BIOCEV/code/microtubules/convnext_instance_seg/pysoax.py' third_party/pysoax.py
```

- [ ] **Step 2: Write `scripts/run_oracle_eval.py`** — for each MT-34 frame: build the oracle mask,
  run (a) PySOAX with its tuned `PYSOAX_PARAMS`, (b) `instance_a`; score both with
  `centerline_f1` + all diagnostics; print per-split, per-source tables and the error attribution
  (fragmentation / junction-identity / bundle-recovery).

- [ ] **Step 3: Run on the VAL split only**

Run: `python scripts/run_oracle_eval.py --split val`
Expected: PySOAX's junction-identity rate is clearly below A's; A's `max_curvature` ≤ κ_max.

- [ ] **Step 4: Record the numbers in `docs/protocol.md`** as a new section "§17 Oracle
  instancing baseline" with the exact table.

---

## Task 13: Optuna tuning of A on oracle VAL

**Files:**
- Create: `scripts/tune_instancer.py`

- [ ] **Step 1: Write the tuner** — Optuna TPE over `merge_radius` (2–5), `window` (6–20),
  `w_theta` (0.2–3), `w_kappa` (0–30), `w_gap` (0–0.2), `c_open` (0.4–2.5), `min_length` (8–40),
  `link_max_gap` (0–25); objective = mean centerline-F1 on the **oracle VAL** frames; κ_max fixed
  at the Task 6 value. Fixed seed, `n_trials=120`, results to
  `data/enc_sensitivity_testset/instancer_tuning/`.

- [ ] **Step 2: Run** — `python scripts/tune_instancer.py --n-trials 120` (on tulen if slow).

- [ ] **Step 3: Persist the best params** to `src/instance/params_a.json` and load them as the
  default in `instance_a`.

- [ ] **Step 4: Re-run `scripts/run_oracle_eval.py --split val`** and confirm the improvement.

---

## Task 14: Instancer B — lifted (x, y, θ) graph

**Files:**
- Create: `src/instance/lifted.py`, `src/instance/instancer_b.py`,
  `tests/test_lifted.py`, `tests/test_instancer_b.py`

**Interfaces:**
- Produces:
  - `refine_theta(channels: np.ndarray, K_out: int) -> np.ndarray` — circular interpolation of the
    K=6 channel stack to `K_out` bins via the doubled-angle representation.
  - `build_lifted_graph(channels, kappa_max, ds=1.0, prob_thr=0.3) -> LiftedGraph` with
    `nodes: dict[(r, c, b) -> int]` and a SciPy CSR adjacency of costs.
  - `instance_b(channels, kappa_max, params=None) -> tuple[list[np.ndarray], list[np.ndarray]]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from instance.oracle import oracle_ori_channels
from instance.instancer_b import instance_b
from instance.lifted import refine_theta

KMAX = 0.30

def test_refine_theta_preserves_the_peak_orientation():
    ch = np.zeros((6, 4, 4), dtype=np.float32); ch[1] = 1.0       # bin 1 = 30..60 deg
    out = refine_theta(ch, K_out=18)
    peak_deg = (out[:, 2, 2].argmax() + 0.5) * (180.0 / 18)
    assert 25.0 < peak_deg < 65.0

def test_perpendicular_crossing_gives_two_instances():
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    ch = oracle_ori_channels([h, v], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 2

def test_wavy_microtubule_stays_ONE_instance_across_theta_bins():
    # A sine that sweeps its tangent through many bins -- the failure mode that
    # sank the earlier PER-BIN approach (F1 0.11). The joint graph must not shatter it.
    x = np.linspace(5.0, 145.0, 400)
    y = 75.0 + 30.0 * np.sin(2 * np.pi * x / 90.0)
    ch = oracle_ori_channels([np.stack([x, y], axis=1)], (150, 150), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    assert len(pls) == 1, f"wavy MT must not fragment across bins; got {len(pls)}"

def test_removing_one_theta_slice_leaves_the_pixel_for_the_other_mt():
    # Both crossing MTs must be recovered at FULL length -- the amodal win.
    h = np.array([[10.0, 50.0], [90.0, 50.0]])
    v = np.array([[50.0, 10.0], [50.0, 90.0]])
    ch = oracle_ori_channels([h, v], (100, 100), K=6, up=1.0)
    pls, _ = instance_b(ch, KMAX)
    lengths = sorted(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in pls)
    assert lengths[0] > 70.0, "the second MT must survive the crossing at full length"
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement.** Nodes = `(r, c, b)` for every pixel above `prob_thr` in bin `b`.
  Edges: from `(r, c, b)` step to the 8-neighbour closest to direction `θ_b` (and its two
  neighbouring directions), into bins `b′` with `|Δθ| ≤ κ_max · ds`; cost
  `−log(p + 1e-6) + λ (Δθ)² / ds`. Extraction: repeatedly run Dijkstra from the highest-scoring
  unused tip node, take the minimum-cost path to the best-scoring reachable tip, record it,
  **delete only the `(p, θ)` nodes on it** (plus θ-neighbours within one bin), and stop when no
  path exceeds `min_length`.

- [ ] **Step 4: Run tests** — expected 4 passed.

- [ ] **Step 5: Add B to `scripts/run_oracle_eval.py`** and re-run `--split val`.

---

## Task 15: Model masks — v4b on MT-34

**Files:**
- Create: `scripts/predict_v4b_mt34.py` (runs **on tulen**)

- [ ] **Step 1: Sync the benchmark to tulen**

```bash
ssh prusek@tulen.utia.cas.cz 'mkdir -p ~/mt_enc_exp/mt34/tif ~/mt_enc_exp/mt34_gt'
scp data/real/mt34_eval/tif/*.tif prusek@tulen.utia.cas.cz:'~/mt_enc_exp/mt34/tif/'
```

Then write the GT as `.npy` object arrays **pre-scaled by 1.5** (matching `alice_gt/`) and scp
them to `~/mt_enc_exp/mt34_gt/`.

- [ ] **Step 2: Write the predictor** — mirror `amodal_eval2.py`'s loading exactly:
  `zoom(norm01(tifffile.imread(t)), 1.5, order=1)` → `dino_seg.py` `SEG_MODE=ori`,
  `SEG_ARCH=base`, `SEG_WEIGHTS=dino_seg_ori_v4b.pth` → save the K=6 probability stack per frame
  as `.npz` into `~/mt_enc_exp/mt34_pred/`.

- [ ] **Step 3: Run on tulen** and copy the `.npz` files back (or keep evaluation on tulen).

- [ ] **Step 4: Sanity-check over-firing** — predicted fg% on MT-34 vs in-domain synth fg%; the
  ratio must stay below 3× (the project's acceptance gate). Record it.

---

## Task 16: A and B on model masks

**Files:**
- Modify: `scripts/run_oracle_eval.py` → add `--masks {oracle,model}`

- [ ] **Step 1: Run both instancers on the v4b channels, VAL split.**
- [ ] **Step 2: Report the oracle→model drop** per instancer and per diagnostic. This attributes
  the residual error to segmentation vs instancing.
- [ ] **Step 3: Record in `docs/protocol.md`.**

---

## Task 17: nnU-Net trained on synth, as a semantic-model candidate

**Files:**
- Create: `scripts/nnunet_prepare_synth.py`, `scripts/nnunet_eval_mt34.py` (both on tulen)

- [ ] **Step 1: Generate the synth training set** with the same generator settings as v4b
  (`gen_train.py --mask_hw 1.0 --calib calib_reg418_morph.json`), ~2000 frames.
- [ ] **Step 2: Convert to nnU-Net raw format** — `Dataset501_MTSynth`, 2D, one foreground class,
  `channel_names: {"0": "IRM"}`.
- [ ] **Step 3: Plan and preprocess** with the residual-encoder planner:
  `nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM -c 2d`.
  **Inspect the generated patch size and downsampling** — if the network downsamples more than 4×
  before the first skip, thin 2 px filaments will be coarsened (the v8/ASPP failure mechanism);
  if so, edit the plan to cap `n_conv_per_stage`/pool ops and record the change.
- [ ] **Step 4: Train** `nnUNetv2_train 501 2d 0 -p nnUNetResEncUNetMPlans` (single fold first).
- [ ] **Step 5: Evaluate on MT-34** with the same 1.5× convention: semantic tol1/tol2/tol5 and the
  over-firing fg% ratio, head-to-head with v4b. Then run instancer A on the nnU-Net mask.
- [ ] **Step 6: Decide and document.** If nnU-Net beats v4b on the extended benchmark it becomes
  the primary semantic model; otherwise it is reported as the baseline. Either way, write the
  numbers into `docs/protocol.md` and `docs/TODO.md`.

---

## Task 18: Final TEST scoring and documentation

- [ ] **Step 1: Freeze all hyperparameters.** No further tuning after this point.
- [ ] **Step 2: Score the TEST split once** — PySOAX / A / B × oracle / model masks, per-source
  (Alice / new-22) and pooled, with clean vs chimera and all diagnostics.
- [ ] **Step 3: Report the 19-frame fully-reviewed sensitivity number** alongside the 22-frame one.
- [ ] **Step 4: Report error rates split by `manual` vs `file` GT polylines** (agreement-bias
  quantification).
- [ ] **Step 5: Update the living docs** — `docs/TODO.md` (tick W9, add the calibration carve-out
  item), `docs/protocol.md` (§17 the full path), `docs/INSTANCE_SEGMENTATION_RESEARCH.md`
  (curvature-bounded matching + lifted tracing as the new recommended build),
  `docs/PAPER_PLAN.md` (the N2 claim is now measurable).
- [ ] **Step 6: Run the full test suite** — `python -m pytest -q` → all green.

---

## Self-Review

**Spec coverage:** §4 benchmark → Tasks 2, 3, 4, 6. §5 metrics/oracle → Tasks 7, 8, 12.
§6 instancer A → Tasks 5, 9, 10, 11, 13. §7 instancer B → Task 14. §8 model masks + nnU-Net →
Tasks 15, 16, 17. §9 tooling → Task 1; docs → Tasks 3, 12, 18. §10 risks → the leakage and
agreement-bias caveats land in Tasks 3 and 18; the wavy-MT risk has a dedicated test in Task 14;
the nnU-Net coarsening risk is Task 17 Step 3. All covered.

**Type consistency:** polylines are `(N, 2)` `(x=col, y=row)` `float64` everywhere;
masks are `bool` `(H, W)`; `oracle_*` take native `shape` plus `up`; `instance_a` and `instance_b`
both return `(polylines, masks)`. `window_tangent` returns `(theta, kappa)` and is consumed by
`ArmEnd(theta=…, kappa=…)`. κ is in **rad/px** throughout.

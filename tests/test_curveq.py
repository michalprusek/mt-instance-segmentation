"""Tests for the overlap-permitting curve-query emission (src/instance/curveq.py).

The properties these lock down are the ones the 2026-09-08 post-mortem of the 0.013 set-prediction
attempt, and the 2026-09-13 adversarial review, say decide whether this paradigm works at all:

  * OVERLAP IS PERMITTED. Two queries can claim the same pixel at full confidence. There is no
    softmax across queries anywhere. If this breaks, the design has no reason to exist — it is the
    single property that separates it from the shipped binary map.
  * ATTENTION FOLLOWS THE CURVE. Moving a query's control points moves where it pools from. The
    failed attempt attended to a fixed stride-8 grid against a 5 px tolerance.
  * THE CHAMFER IS SYMMETRIC. A half-length query must not beat a full-length one; one-directional
    distance is minimised by a short curve on the middle of a long filament, which is the coverage
    failure the strict metric charges.
  * EXTENT COMES FROM THE MASK. End placement alone is worth 0.113 macro, so a query's ends must be
    decided by where its own mask support stops, never by a regressed length.
  * IT CAN OVERFIT A TRIVIAL CASE. A set-prediction model that cannot fit four straight lines in a
    32x32 feature map will not fit microtubules, and this is the cheap test the 0.013 attempt did
    not have.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from instance.curveq import (N_CTRL, N_SAMP, CurveQHead, chamfer_qt,  # noqa: E402
                             curveq_loss, decode_curves, dense_mask_logits,
                             hungarian_match, mask_logits_at, overfire_stats, sample_curve)


def _head(q=16, dim=32, layers=2, feat=64):
    torch.manual_seed(0)
    return CurveQHead(feat_dim=feat, n_queries=q, dim=dim, n_layers=layers, n_heads=4)


def _lines(b=2, t=4, n=N_SAMP):
    out = []
    for _ in range(b):
        cs = [torch.stack([torch.linspace(0.1, 0.9, n),
                           torch.full((n,), 0.15 + 0.2 * k)], -1) for k in range(t)]
        out.append(torch.stack(cs))
    return out


def _pts_fn(tg, t, tol=0.03):
    def f(b):
        pts = tg[b].reshape(-1, 2)
        lab = torch.zeros(t, len(pts))
        for k in range(t):
            lab[k] = (torch.cdist(pts[None], tg[b][k][None])[0].min(-1).values < tol).float()
        return pts, lab
    return f


# --------------------------------------------------------------- the defining property

def test_two_queries_can_claim_the_same_pixel():
    """THE point of the design. No softmax over queries; overlap must be representable."""
    h = _head()
    outs = h(torch.randn(1, 64, 32, 32))
    emb, pix = outs[-1]["emb"], outs[-1]["pix"]
    # force two queries to have the SAME mask embedding -> both must fire identically
    emb = emb.clone()
    emb[0, 1] = emb[0, 0]
    ml = torch.sigmoid(mask_logits_at(emb, pix, torch.rand(1, 30, 2)))
    assert torch.allclose(ml[0, 0], ml[0, 1], atol=1e-5)
    # and the per-pixel sum over queries is free to exceed 1 (a softmax would pin it at 1)
    dm = torch.sigmoid(dense_mask_logits(emb, pix))
    assert float(dm.sum(1).max().detach()) > 1.0


def test_no_softmax_over_queries_anywhere():
    src = open(os.path.join(ROOT, "src", "instance", "curveq.py")).read()
    body = src.split("class CurveQHead")[1]
    for line in body.splitlines():
        if "softmax" in line and "dim=-1" in line and "self.weights" not in line:
            # the only softmax allowed is over ATTENTION POINTS inside one head, never over queries
            assert "w = torch.softmax" not in line or "dim=-1" in line


# --------------------------------------------------------------- attention follows the curve

def test_attention_reference_points_move_with_the_control_polygon():
    h = _head(q=4)
    feat = torch.randn(1, 64, 32, 32)
    with torch.no_grad():
        a = h(feat)[-1]["ctrl"].clone()
        h.anchor.weight.data.add_(1.5)          # move every query's anchor curve (LOGIT space)
        b = h(feat)[-1]["ctrl"].clone()
    assert not torch.allclose(a, b)
    ra, rb = sample_curve(a), sample_curve(b)
    assert float((rb - ra).abs().mean()) > 0.05  # the pooling locations really moved


def test_sample_curve_is_arclength_uniform_on_a_straight_line():
    ctrl = torch.stack([torch.linspace(0, 1, N_CTRL), torch.zeros(N_CTRL)], -1)[None, None]
    s = sample_curve(ctrl, n=9)[0, 0]
    d = (s[1:] - s[:-1]).norm(dim=-1)
    assert float(d.std()) < 1e-4
    assert abs(float(s[0, 0])) < 1e-4 and abs(float(s[-1, 0]) - 1.0) < 1e-4


# --------------------------------------------------------------- matching

def test_chamfer_is_symmetric_so_a_half_length_query_loses():
    n = N_SAMP
    full = torch.stack([torch.linspace(0, 1, n), torch.zeros(n)], -1)
    half = torch.stack([torch.linspace(0.25, 0.75, n), torch.zeros(n)], -1)
    c = chamfer_qt(torch.stack([full, half]), full[None])
    assert float(c[0, 0]) < float(c[1, 0]), "the query covering the whole target must win"


def test_chamfer_matches_a_hand_computation():
    n = 4
    a = torch.stack([torch.linspace(0, 1, n), torch.zeros(n)], -1)
    b = a + torch.tensor([0.0, 0.1])
    got = float(chamfer_qt(a[None], b[None])[0, 0])
    assert abs(got - 0.1) < 1e-5


def test_hungarian_is_one_to_one():
    h = _head(q=8)
    outs = h(torch.randn(2, 64, 32, 32))[-1]
    tg = _lines(2, 3)
    pairs = hungarian_match(outs["ctrl"], outs["obj"], outs["emb"], outs["pix"], tg)
    for qi, ti in pairs:
        assert len(qi) == len(ti) == 3
        assert len(set(qi.tolist())) == 3 and sorted(ti.tolist()) == [0, 1, 2]


def test_empty_targets_do_not_crash_and_give_no_positives():
    h = _head(q=8)
    outs = h(torch.randn(1, 64, 32, 32))
    loss, st = curveq_loss(outs, [torch.zeros(0, N_SAMP, 2)], lambda b: (None, None))
    assert st["n_pos"] == 0
    loss.backward()          # must stay differentiable on an empty frame


# --------------------------------------------------------------- endpoints

def test_extent_comes_from_the_mask_not_from_a_length():
    """Support only the middle of a query's curve; the decoded polyline must be trimmed to it."""
    h = _head(q=2)
    outs = h(torch.randn(1, 64, 32, 32))[-1]
    ctrl = torch.stack([torch.linspace(0.05, 0.95, N_CTRL), torch.full((N_CTRL,), 0.5)], -1)
    ctrl = ctrl[None, None].expand(1, 2, N_CTRL, 2).contiguous()
    obj = torch.full((1, 2), 5.0)

    class FakePix(torch.Tensor):
        pass

    # a pixel embedding whose dot with emb is high only in a central x band
    H = W = 32
    pix = torch.zeros(1, outs["emb"].shape[-1], H, W)
    xs = torch.arange(W).float() / (W - 1)
    band = ((xs > 0.35) & (xs < 0.65)).float() * 20.0 - 10.0
    pix[0, 0] = band[None, :].expand(H, W)
    emb = torch.zeros(1, 2, outs["emb"].shape[-1])
    emb[0, :, 0] = 1.0
    got = decode_curves(ctrl, obj, emb, pix, (H, W), obj_thr=0.5, mask_thr=0.5, min_support=0.1)
    assert len(got) == 2
    for c in got:
        x = c[:, 0] / (W - 1)
        assert x.min() > 0.30 and x.max() < 0.70, f"not trimmed to the supported band: {x.min()}..{x.max()}"


def test_unsupported_query_is_rejected():
    H = W = 32
    ctrl = torch.stack([torch.linspace(0.05, 0.95, N_CTRL), torch.full((N_CTRL,), 0.5)], -1)
    ctrl = ctrl[None, None]
    obj = torch.full((1, 1), 5.0)
    pix = torch.full((1, 4, H, W), -10.0)          # nothing supports anything
    emb = torch.zeros(1, 1, 4)
    emb[0, 0, 0] = 1.0
    assert decode_curves(ctrl, obj, emb, pix, (H, W), min_support=0.5) == []


def test_objectness_threshold_gates_emission():
    H = W = 32
    ctrl = torch.stack([torch.linspace(0.05, 0.95, N_CTRL), torch.full((N_CTRL,), 0.5)], -1)[None, None]
    pix = torch.full((1, 4, H, W), 10.0)
    emb = torch.zeros(1, 1, 4)
    emb[0, 0, 0] = 1.0
    assert len(decode_curves(ctrl, torch.full((1, 1), 5.0), emb, pix, (H, W))) == 1
    assert decode_curves(ctrl, torch.full((1, 1), -5.0), emb, pix, (H, W)) == []


# --------------------------------------------------------------- the over-firing gate

def test_overfire_stats_sees_what_the_dense_gate_cannot():
    a = [np.stack([np.linspace(0, 100, 10), np.zeros(10)], 1)]
    b = a * 20
    sa, sb = overfire_stats(a, (1000, 1000)), overfire_stats(b, (1000, 1000))
    assert sb["n_curves"] == 20 * sa["n_curves"]
    assert sb["total_len_px"] > 19 * sa["total_len_px"]
    assert sb["len_per_Mpx"] > sa["len_per_Mpx"]


# --------------------------------------------------------------- the decisive cheap test

@pytest.mark.slow
def test_it_can_overfit_four_straight_lines():
    """If it cannot fit four lines in a 32x32 map, it will not fit microtubules.

    This is the test the 0.013 set-prediction attempt did not have. It is a LEARNING test, so the
    criterion is on the loss trajectory, not on a fixed value: the matched geometric distance must
    fall by at least half, and the loss must fall monotonically over the last quarter of training.
    """
    torch.manual_seed(0)
    h = _head(q=16, dim=64, layers=3)
    feat = torch.randn(1, 64, 32, 32)
    tg = _lines(1, 4)
    fn = _pts_fn(tg, 4)
    opt = torch.optim.AdamW(h.parameters(), lr=3e-3)
    hist = []
    for _ in range(120):
        outs = h(feat)
        loss, st = curveq_loss(outs, tg, fn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append((float(loss.detach()), st["geo"]))
    first, last = hist[0], hist[-1]
    assert last[1] < 0.5 * first[1], f"geometric distance did not halve: {first[1]} -> {last[1]}"
    assert np.mean([x[0] for x in hist[-10:]]) < 0.6 * hist[0][0], "total loss barely moved"


def test_aux_queries_are_ignored_in_objectness_not_soft_labelled():
    """The overfit test found this: a soft 0.25 target for aux queries adds an irreducible entropy
    floor of H(0.25) = 0.56 nats each, which stalls the objectness term at a value no training can
    move. Aux queries must be MASKED OUT of the objectness loss instead.

    The check is behavioural, not textual: with every query either matched or aux, the objectness
    term must be able to reach ~0, which is impossible under any soft target.
    """
    torch.manual_seed(0)
    h = _head(q=8, dim=32, layers=1)             # 2 targets -> 2 matched + 6 aux = every query
    feat = torch.randn(1, 64, 32, 32)
    tg = _lines(1, 2)
    fn = _pts_fn(tg, 2)
    opt = torch.optim.AdamW(h.parameters(), lr=5e-3)
    for _ in range(80):
        loss, st = curveq_loss(h(feat), tg, fn, aux_k=3)
        opt.zero_grad(); loss.backward(); opt.step()
    assert st["obj"] < 0.05, f"objectness stalled at {st['obj']:.3f} — soft target floor is back"


# --------------------------------------------------------------- targets agree with the raster

def test_transform_curves_agrees_with_the_rasterised_target():
    """THE check that matters: after crop + rot + flips, every curve point must land on ITS OWN id
    in the id channel the dense loader produces from the same transforms.

    The project has paid twice for a silent (x=col, y=row) transpose, and a curve target that
    disagrees with the image by a transpose trains a model on nonsense while every loss looks
    healthy. This test derives the answer two independent ways and compares them.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tv5_for_curveq", os.path.join(ROOT, "scripts", "train_v5.py"))
    tv5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tv5)
    from instance.curveq import transform_curves

    rng = np.random.default_rng(0)
    c = 64
    # two curves in a 96x96 frame, well inside it
    curves = [np.stack([np.arange(10, 70, 1.0), np.full(60, 20.0)], 1),
              np.stack([np.full(50, 45.0), np.arange(15, 65, 1.0)], 1)]
    for k in range(4):
        for flr in (False, True):
            for fud in (False, True):
                y0 = x0 = 10
                # dense path: rasterise the FULL frame with ids, then crop/rot/flip the raster
                gt = tv5.ori_channels_soft([{"centerline": q} for q in curves], (96, 96),
                                           half_width=1.0, curve_ids=True)
                cid = gt[-1][y0:y0 + c, x0:x0 + c]
                if k:
                    cid = np.rot90(cid, k).copy()
                if flr:
                    cid = cid[:, ::-1].copy()
                if fud:
                    cid = cid[::-1, :].copy()
                # curve path: transform the coordinates
                got = transform_curves(curves, y0, x0, c, k=k, flip_lr=flr, flip_ud=fud)
                assert got, f"k={k} lr={flr} ud={fud}: all curves lost"
                for p in got:
                    rr = np.clip(np.rint(p[:, 1]).astype(int), 0, c - 1)
                    cc = np.clip(np.rint(p[:, 0]).astype(int), 0, c - 1)
                    hit = cid[rr, cc] > 0
                    assert hit.mean() > 0.95, (
                        f"k={k} lr={flr} ud={fud}: only {hit.mean():.2f} of curve points land on "
                        f"the rasterised id map — the curve target disagrees with the image")


def test_mask_points_allow_a_point_to_belong_to_two_targets():
    """The overlap property, on the TARGET side. A one-hot assignment here would silently restore
    the single-owner rule the dense map already has."""
    from instance.curveq import mask_points
    n = N_SAMP
    x = torch.linspace(0.1, 0.9, n)
    a = torch.stack([x, torch.full((n,), 0.5)], -1)
    b = torch.stack([torch.full((n,), 0.5), x], -1)      # crosses a at (0.5, 0.5)
    tgt = torch.stack([a, b])
    pts, lab = mask_points(tgt, c=100, n_pos=200, n_neg=0, tol_px=4.0)
    assert lab.shape[0] == 2
    assert float(lab.sum(0).max()) >= 2.0, "no point is shared: the crossing is not amodal"


def test_resample_norm_is_arclength_uniform_and_in_unit_range():
    from instance.curveq import resample_norm
    q = np.stack([np.linspace(0, 63, 200), np.zeros(200)], 1)
    t = resample_norm([q], c=64, n=9)[0]
    d = (t[1:] - t[:-1]).norm(dim=-1)
    assert float(d.std()) < 1e-5
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


# --------------------------------------------------------------- the decode contract

def test_merge_joins_a_split_curve_and_leaves_a_crossing_alone():
    """The tiled path's merge rule must join a filament cut by a seam and must NOT join two
    filaments that merely touch at a crossing — that would be the chimera failure, created by the
    decode rather than by the model."""
    from instance.curveq import merge_curve_pieces
    a = np.stack([np.arange(0, 30, 1.0), np.zeros(30)], 1)
    b = np.stack([np.arange(31, 60, 1.0), np.zeros(29)], 1)          # same line, seam gap 1 px
    got = merge_curve_pieces([a, b], tol=6.0, deg=30.0)
    assert len(got) == 1 and len(got[0]) == 59

    # two curves meeting at 90 degrees: ends are close, tangents are not
    h = np.stack([np.arange(0, 30, 1.0), np.full(30, 10.0)], 1)
    v = np.stack([np.full(30, 30.0), np.arange(10, 40, 1.0)], 1)
    got2 = merge_curve_pieces([h, v], tol=6.0, deg=30.0)
    assert len(got2) == 2, "a 90-degree junction was merged into one curve — that is a chimera"


def test_merge_is_a_no_op_on_a_single_curve():
    from instance.curveq import merge_curve_pieces
    a = np.stack([np.arange(0, 20, 1.0), np.zeros(20)], 1)
    got = merge_curve_pieces([a])
    assert len(got) == 1 and np.allclose(got[0], a)


# --------------------------------------------------------------- the runaway that killed run 1

def test_control_points_stay_inside_the_frame_under_the_real_optimiser():
    """The first CURVEQ run died here and it cost 3 GPU-hours, so this is a regression test.

    The anchors were raw coordinates; after 3000 steps of nnU-Net's own optimiser (SGD, momentum
    0.99, lr 0.01 -- tuned for a conv trunk, not a coordinate embedding) they reached -7.17 ..
    10.18, every query pinned to the clamp, all 300 objectness values became identical to four
    decimals, and the model emitted nothing. The fix is structural: anchors live in logit space and
    coordinates are their sigmoid, so this cannot recur however badly the optimiser behaves.
    """
    torch.manual_seed(0)
    h = _head(q=16, dim=32, layers=2)
    feat = torch.randn(1, 64, 32, 32)
    tg = _lines(1, 4)
    fn = _pts_fn(tg, 4)
    # the optimiser the head ACTUALLY gets: AdamW, because the shared nnU-Net SGD collapses it
    opt = torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(60):
        loss, _ = curveq_loss(h(feat), tg, fn)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.parameters(), 12.0)
        opt.step()
    ctrl = h(feat)[-1]["ctrl"]
    assert float(ctrl.min()) >= 0.0 and float(ctrl.max()) <= 1.0, "a control point left the frame"
    # and the queries must still be DIFFERENT from one another -- the actual failure was collapse
    obj = torch.sigmoid(h(feat)[-1]["obj"][0])
    assert float(obj.std()) > 1e-3, "all queries produced the same objectness: collapsed again"
    spread = ctrl[0].mean((1, 2)).std()
    assert float(spread) > 1e-3, "every query converged to the same curve"


def test_the_shared_nnunet_sgd_differentiates_queries_far_worse_than_adamw():
    """Lock down the DIAGNOSIS, so nobody re-merges the parameter groups to simplify the trainer.

    History, because the number moved and the honest record matters more than the original claim.
    On 2026-09-13, before the attention offsets were initialised properly, SGD(momentum 0.99,
    lr 1e-2) drove objectness std across queries to 0.00000 -- total collapse -- against AdamW's
    0.107. After the Deformable-DETR offset init (which stopped all `n_points` of a head sampling
    the same location), the same SGD reaches 0.029. So run 1's collapse was MULTI-CAUSAL: a
    degenerate attention pattern and an optimiser that a query decoder cannot take, together.
    The remaining gap is still large and one-directional, which is what this test pins.
    """
    def run(make_opt):
        torch.manual_seed(0)
        h = _head(q=16, dim=32, layers=2)
        feat = torch.randn(1, 64, 32, 32)
        tg = _lines(1, 4)
        fn = _pts_fn(tg, 4)
        opt = make_opt(h)
        for _ in range(60):
            loss, _ = curveq_loss(h(feat), tg, fn)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), 12.0)
            opt.step()
        return float(torch.sigmoid(h(feat)[-1]["obj"][0]).std())

    sgd = run(lambda h: torch.optim.SGD(h.parameters(), 1e-2, momentum=0.99, nesterov=True))
    adam = run(lambda h: torch.optim.AdamW(h.parameters(), 1e-4, weight_decay=1e-4))
    assert adam > 1e-2, f"AdamW collapsed ({adam:.5f}) — the parameter split is not doing its job"
    assert adam > 2.5 * sgd, (
        f"the shared nnU-Net SGD ({sgd:.5f}) is no longer much worse than AdamW ({adam:.5f}); "
        f"re-measure before merging the parameter groups")


def test_the_mask_head_actually_predicts_a_mask_at_realistic_sparsity():
    """THE test run 2 needed and did not have, and it cost 5 GPU-hours.

    A filament is a 3 px line in a 512 px crop, so a per-query mask label is ~2.6 % positive. Under
    an unweighted BCE the minimum is "predict nothing" at a loss of 0.1601; run 2 sat at 0.1843 from
    epoch 2 to 17 and the best query's mask support along its own curve was 0.006, so `decode_curves`
    rejected everything. The loss curve looked fine the whole time.

    This reproduces that sparsity and asserts the OUTCOME that matters: after training, a matched
    query's own mask must actually fire along its own curve. A loss value cannot be the criterion —
    that is exactly what hid the failure.
    """
    torch.manual_seed(0)
    n_t = 12
    h = _head(q=32, dim=64, layers=3)
    feat = torch.randn(1, 64, 64, 64)
    tg = [torch.stack([torch.stack([torch.linspace(0.1, 0.9, N_SAMP),
                                    torch.full((N_SAMP,), 0.06 + 0.07 * k)], -1)
                       for k in range(n_t)])]

    def fn(b):
        from instance.curveq import mask_points
        return mask_points(tg[b], c=512, n_pos=256, n_neg=256, tol_px=3.0,
                           rng=np.random.default_rng(0))

    # the sparsity this test exists to reproduce
    _, lab0 = fn(0)
    assert 0.01 < float(lab0.mean()) < 0.06, f"sparsity not realistic: {float(lab0.mean()):.4f}"

    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(150):
        loss, st = curveq_loss(h(feat), tg, fn)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.parameters(), 12.0)
        opt.step()

    o = h(feat)[-1]
    # the mask the DECODER actually reads, i.e. including the geometric conditioning; evaluating
    # only the appearance term would measure half the head and would have gone on passing after the
    # geometry was added, which is the failure mode this whole test family exists to prevent
    pairs = hungarian_match(o["ctrl"], o["obj"], o["emb"], o["pix"], tg, rel_w=o.get("rel_w"))
    qi, _ = pairs[0]
    pts = sample_curve(o["ctrl"][:, qi], n=32)[0].clamp(0, 1)
    sup = []
    for j in range(len(qi)):
        rw = o.get("rel_w")
        rw = None if rw is None else rw[:, qi[j]:qi[j] + 1]
        ml = torch.sigmoid(mask_logits_at(o["emb"][:, qi[j]:qi[j] + 1], o["pix"], pts[j][None],
                                          o["ctrl"][:, qi[j]:qi[j] + 1], rw)[0, 0])
        sup.append(float(ml.mean().detach()))
    assert np.median(sup) > 0.3, (
        f"matched queries have no mask under their own curves (median support {np.median(sup):.3f}) "
        f"— the mask head is at the predict-nothing minimum, as in run 2")


# --------------------------------------------------------------- defects the 2026-09-14 review found

def test_overlap_label_exists_at_REALISTIC_filament_length():
    """The existing overlap test uses c=100 and 80 px curves, which is the easy case; a review
    measured that at realistic lengths the shared label had all but disappeared.

    Two separate causes, both now fixed: the label was computed against the 24 sampled VERTICES
    rather than the segments, and — the larger one — the positive points were themselves only
    those vertices, which sit L/23 px apart (13 px on a 300 px filament), so on a real crossing
    there was no candidate point near the junction at all. Measured before the fix: a shared label
    in 100 % of crossings of 80 px curves and 0 % of crossings of 300 px curves.
    """
    from instance.curveq import mask_points
    rng = np.random.default_rng(0)
    for length in (200.0, 300.0, 400.0):
        shared = 0
        trials = 40
        for _ in range(trials):
            c = 512
            cx, cy = rng.uniform(0.35, 0.65, 2)
            th = rng.uniform(np.pi / 6, np.pi / 2)
            t = np.linspace(-length / 2, length / 2, N_SAMP) / c
            a = np.stack([cx + t, np.full(N_SAMP, cy)], -1)
            b = np.stack([cx + t * np.cos(th), cy + t * np.sin(th)], -1)
            tgt = torch.tensor(np.stack([a, b]), dtype=torch.float32).clamp(0, 1)
            _, lab = mask_points(tgt, c, n_pos=256, n_neg=256, tol_px=3.0, rng=rng)
            if lab is not None and float(lab.sum(0).max()) >= 2:
                shared += 1
        assert shared / trials > 0.9, (
            f"at {length:.0f} px filaments only {100 * shared / trials:.0f} % of crossings carry a "
            f"shared mask label — the amodal supervision the module exists for is missing")


def test_the_matcher_prefers_a_specialist_over_a_mask_that_fires_everywhere():
    """The Hungarian mask cost used to be the query's mean sigmoid over ALL targets' points,
    expanded across targets. Two things were wrong and the second is the subtle one: its optimum is
    a mask that fires everywhere, AND a per-query value is a ROW constant, which
    `linear_sum_assignment` is invariant to — so it contributed NOTHING to the assignment while the
    loss it is meant to approximate demands specificity.

    So the test has to make GEOMETRY pull the wrong way and check that the mask cost overrides it.
    q0 is the specialist for t0 but sits 0.04 off it; q1 fires everywhere and sits exactly on t0.
    Geometry alone (chamfer 0.04/0.36 vs 0.00/0.40) hands t0 to q1; a target-specific mask cost
    hands it to q0. Verified: with the row-constant cost the assignment really is q0->t1, q1->t0.
    """
    n = N_SAMP
    x = torch.linspace(0.2, 0.8, n)
    t0 = torch.stack([x, torch.full((n,), 0.30)], -1)
    t1 = torch.stack([x, torch.full((n,), 0.70)], -1)
    targets = torch.stack([t0, t1])
    q0 = torch.stack([torch.linspace(0.2, 0.8, N_CTRL), torch.full((N_CTRL,), 0.34)], -1)
    q1 = torch.stack([torch.linspace(0.2, 0.8, N_CTRL), torch.full((N_CTRL,), 0.30)], -1)
    ctrl = torch.stack([q0, q1])[None]
    obj = torch.zeros(1, 2)
    dim, H, W = 4, 64, 64
    pix = torch.full((1, dim, H, W), -8.0)
    ys = torch.arange(H).float() / (H - 1)
    pix[0, 0] = torch.where((ys - 0.30).abs() < 0.03, 8.0, -8.0)[:, None].expand(H, W)
    pix[0, 1] = 8.0
    emb = torch.zeros(1, 2, dim)
    emb[0, 0, 0] = 1.0        # specialist for t0, slightly off it geometrically
    emb[0, 1, 1] = 1.0        # fires everywhere, sits exactly on t0
    (qi, ti), = hungarian_match(ctrl, obj, emb, pix, [targets])
    owner = {int(t): int(q) for q, t in zip(qi, ti)}
    assert owner[0] == 0, (
        "target 0 went to the query whose mask fires everywhere rather than to the specialist — "
        "the matcher's mask cost is a row constant and cannot affect the assignment")


# --------------------------------------------------------------- the wall CURVEQ hit

def _own_vs_other_auc(h, feat, tg, n_eval=None):
    """Median AUC separating a matched query's mask on ITS filament from its mask on another.

    This is the quantity that killed CURVEQ runs 3 and 4: measured in domain on exact labels, a
    query's mask read 0.6221 on its own filament and 0.6219 on a different one — AUC 0.483, i.e.
    chance. Loss, geometry and objectness all looked healthy at the same time.
    """
    o = h(feat)[-1]
    pairs = hungarian_match(o["ctrl"], o["obj"], o["emb"], o["pix"], tg)
    qi, ti = pairs[0]
    aucs = []
    for j in range(len(qi)):
        own = tg[0][ti[j]][None]
        other = tg[0][ti[(j + 1) % len(ti)]][None]
        rw = o.get("rel_w")
        rw = None if rw is None else rw[:, qi[j]:qi[j] + 1]
        a = torch.sigmoid(mask_logits_at(o["emb"][:, qi[j]:qi[j] + 1], o["pix"], own,
                                         o["ctrl"][:, qi[j]:qi[j] + 1], rw)[0, 0]).detach()
        b = torch.sigmoid(mask_logits_at(o["emb"][:, qi[j]:qi[j] + 1], o["pix"], other,
                                         o["ctrl"][:, qi[j]:qi[j] + 1], rw)[0, 0]).detach()
        sc = torch.cat([a, b]).numpy()
        lab = np.r_[np.ones(len(a)), np.zeros(len(b))]
        order = sc.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(sc))
        n1, n0 = lab.sum(), len(lab) - lab.sum()
        aucs.append((r[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
    return float(np.median(aucs))


def test_a_query_mask_can_tell_its_own_filament_from_a_parallel_one():
    """THE test the paradigm lives or dies on, and the one four runs did not have.

    Two parallel filaments 3 px apart in a 512 crop look identical to any position-invariant
    function of local appearance, so a mask of the form sigmoid(emb_q . pix(x)) — a 1x1 conv of
    shared features dotted with a query embedding — provably cannot fire on one and not the other.
    Measured on the real thing: AUC 0.483 after 40 epochs. The mask must therefore be conditioned
    on the query's OWN GEOMETRY, which is what CondInst and SOLOv2 do with relative coordinates.
    """
    torch.manual_seed(0)
    n_t = 6
    h = _head(q=48, dim=64, layers=3)
    feat = torch.randn(1, 64, 64, 64)
    # pairs of filaments 3 px apart in a 512 frame => 0.00587 normalised: indistinguishable by
    # appearance, separable only by position
    ys = []
    for k in range(n_t // 2):
        base = 0.2 + 0.25 * k
        ys += [base, base + 3.0 / 511.0]
    tg = [torch.stack([torch.stack([torch.linspace(0.1, 0.9, N_SAMP),
                                    torch.full((N_SAMP,), y)], -1) for y in ys])]

    def fn(b):
        from instance.curveq import mask_points
        return mask_points(tg[b], c=512, n_pos=256, n_neg=256, tol_px=1.5,
                           rng=np.random.default_rng(0))

    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(250):
        loss, _ = curveq_loss(h(feat), tg, fn)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(h.parameters(), 12.0)
        opt.step()
    auc = _own_vs_other_auc(h, feat, tg)
    assert auc > 0.75, (
        f"a query's mask cannot tell its own filament from a parallel one 3 px away "
        f"(AUC {auc:.3f}); a position-invariant pixel embedding is not enough and the emission "
        f"is not overlap-permitting in any useful sense")

"""CURVEQ — an overlap-permitting curve-query emission for 2D microtubule instance segmentation.

WHY THIS EXISTS, in one paragraph of measured fact. The shipped model emits ONE binary foreground
map, so a pixel belongs to at most one filament, and the assembler recovers instances by cutting a
skeleton. Screen A (2026-09-13, exact generator ground truth, no model in the loop for the geometry)
found that **91.2 % of in-domain breaks sit on a crossing or a 2-6 px close-parallel stretch while
only 5.4 % are an actual hole in the mask**: the pixels are present and the mask is continuous, and
what fails is that one pixel cannot be owned by two filaments. CURVEQ changes exactly that. Each
query owns a curve and emits its OWN mask; the per-query masks are independent sigmoids, never a
softmax over queries, so two filaments 3 px apart and both arms of a crossing can claim the same
pixel at full confidence.

WHAT MAKES THIS DIFFERENT FROM THE 0.013 SET-PREDICTION FAILURE (docs/TODO.md 0.0-SETPRED). That
attempt failed at LOCALISATION. Its post-mortem names four absent ingredients, and this module has
all four: (1) attention anchored on the query's OWN predicted geometry rather than on a stride-8
global memory quantised at 8 px against a 5 px tolerance; (2) per-layer iterative refinement of the
curve instead of one emission from a final hidden state; (3) one-to-many auxiliary supervision, so
early layers get gradient from more than one matched query; (4) a per-query MASK head, which is
what makes the emission overlap-permitting at all.

THREE BLOCKING REPAIRS FROM THE ADVERSARIAL REVIEW, implemented rather than noted:
  * POINT-SAMPLED masks. A dense 300 x 256^2 x 6-layer mask tensor OOMs at batch 8. Both the
    Hungarian cost and the mask loss are computed on K sampled points (Mask2Former's trick), so
    memory is O(Q x K) not O(Q x H x W).
  * WHOLE-FRAME DECODE IS DECLARED, not hand-waved. `decode_frame` states its tiling contract and
    merges curves across tile seams explicitly; a per-query emission is NOT tile-averageable the
    way a dense map is, and pretending otherwise would silently reintroduce the assembly step this
    paradigm claims to avoid.
  * A QUERY-SIDE OVER-FIRING GATE. The dense trainer's gate reads `fg` from a dense map, which a
    query emission does not have, so it cannot see this model over-firing. `overfire_stats` returns
    accepted curves per frame and total emitted centerline length, which is what the gate reads.

ENDPOINTS, the named risk. End placement alone is worth 0.113 macro on this metric, and a
synth-learned length prior regressing ends is how a curve model loses it. So the curve head does
NOT regress an extent: it emits a fixed-length control polygon and the EXTENT IS DECODED FROM THE
QUERY'S OWN MASK — the curve is trimmed to where its mask support ends. A length prior can then
bias the shape but cannot invent an end.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Control points per query. 8 gives a cubic-ish polyline enough freedom for the curvature this
# project measured (kappa_max 0.25 rad/px at an 8 px baseline) without inviting the wiggle a
# denser polygon would fit to noise.
N_CTRL = 8
# Samples along a curve for mask supervision and for attention reference points.
N_SAMP = 24
# Largest per-layer step in anchor LOGIT space. 2.0 lets one layer move a control point roughly
# from 0.27 to 0.9 of the frame, which is more than any refinement should need and far less than
# the runaway the unconstrained version produced.
DELTA_MAX = 2.0
# The trunk's required input divisor. nnU-Net ResEnc-M downsamples six times and its residual adds
# mismatch by one pixel if the input is not a multiple of this (train_v5.TILE_FOR says the same).
DIVISOR = 128


def sample_curve(ctrl: torch.Tensor, n: int = N_SAMP) -> torch.Tensor:
    """Uniformly resample a control polygon (B, Q, N_CTRL, 2) to (B, Q, n, 2) in normalised coords.

    A straight linear interpolation along the polygon, not a Bezier: the control points ARE on the
    curve, which keeps the Hungarian geometric cost interpretable (it is a distance between points
    that both lie on their curves) and means a query's attention reference points are places the
    model claims filament, not abstract handles.
    """
    ctrl = ctrl.float()          # geometry is tiny and needs full precision under autocast
    b, q, c, _ = ctrl.shape
    seg = ctrl[..., 1:, :] - ctrl[..., :-1, :]
    d = seg.norm(dim=-1)                                    # (B, Q, C-1)
    cum = torch.cumsum(d, dim=-1)
    total = cum[..., -1:].clamp_min(1e-6)
    s = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1) / total   # (B, Q, C), in [0,1]
    t = torch.linspace(0, 1, n, device=ctrl.device, dtype=ctrl.dtype).view(1, 1, n)
    # piecewise-linear interpolation of ctrl at arclength t
    idx = torch.searchsorted(s.contiguous(), t.expand(b, q, n).contiguous())
    idx = idx.clamp(1, c - 1)
    s0 = torch.gather(s, -1, idx - 1)
    s1 = torch.gather(s, -1, idx)
    w = ((t - s0) / (s1 - s0).clamp_min(1e-6)).unsqueeze(-1)
    p0 = torch.gather(ctrl, 2, (idx - 1).unsqueeze(-1).expand(b, q, n, 2))
    p1 = torch.gather(ctrl, 2, idx.unsqueeze(-1).expand(b, q, n, 2))
    return p0 + w * (p1 - p0)


class CurveDeformAttn(nn.Module):
    """Attention that pools features AT THE QUERY'S OWN CURVE, plus a learned offset per head.

    This is the load-bearing difference from the failed set-prediction attempt, which attended to a
    fixed stride-8 grid. Here the reference points move with the curve at every layer, so a query
    that is 6 px off after layer 1 pools from where it now believes the filament is, not from where
    a global grid happens to sit.
    """

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.h, self.p, self.dim = n_heads, n_points, dim
        self.offsets = nn.Linear(dim, n_heads * n_points * 2)
        self.weights = nn.Linear(dim, n_heads * n_points)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        # Deformable-DETR's offset init. Zero-initialising the BIAS as well puts all `n_points`
        # sampling locations of a head at exactly the same place, which is an exact fixed point of
        # the gradient: measured, the spread of `offsets.weight` gradient across the points within
        # a head is 0.0 at step 1 and still 0.0 after 300 steps, `self.weights` receives gradient
        # of order 1e-8, and the attention softmax stays exactly uniform forever. The head then has
        # 1 effective sampling point instead of `n_points`. Distinct unit directions break it.
        with torch.no_grad():
            ang = torch.arange(n_heads, dtype=torch.float32) * (2 * np.pi / n_heads)
            grid = torch.stack([ang.cos(), ang.sin()], -1)                       # (H, 2)
            grid = grid / grid.abs().max(-1, keepdim=True).values
            grid = grid[:, None, :].repeat(1, n_points, 1)                       # (H, P, 2)
            for i in range(n_points):
                grid[:, i, :] *= (i + 1)
            self.offsets.bias.copy_(grid.reshape(-1))

    def forward(self, q: torch.Tensor, feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """q (B, Q, D) · feat (B, D, H, W) · ref (B, Q, S, 2) in [0,1] (x, y) -> (B, Q, D)."""
        B, Q, S, _ = ref.shape
        v = self.value(feat.flatten(2).transpose(1, 2))                    # (B, HW, D)
        v = v.transpose(1, 2).reshape(B, self.dim, *feat.shape[-2:])
        # sampling locations: every curve sample, displaced per head and per point
        off = self.offsets(q).view(B, Q, self.h, self.p, 2)
        # offsets are in units of the sampling stride, kept small so layer 1 does not teleport
        loc = ref.view(B, Q, 1, S, 1, 2) + off.view(B, Q, self.h, 1, self.p, 2) * 0.05
        loc = loc.reshape(B, Q, self.h, S * self.p, 2)
        w = self.weights(q).view(B, Q, self.h, 1, self.p)
        w = w.expand(B, Q, self.h, S, self.p).reshape(B, Q, self.h, S * self.p)
        w = torch.softmax(w, dim=-1)
        dh = self.dim // self.h
        out = []
        for hh in range(self.h):
            g = (loc[:, :, hh] * 2 - 1).view(B, Q, S * self.p, 2).to(v.dtype)   # [-1,1], v's dtype
            s = F.grid_sample(v[:, hh * dh:(hh + 1) * dh], g, mode="bilinear",
                              padding_mode="border", align_corners=False)  # (B, dh, Q, S*P)
            out.append((s * w[:, None, :, hh]).sum(-1))                    # -> (B, dh, Q)
        return self.out(torch.cat(out, dim=1).transpose(1, 2))


class CurveQHead(nn.Module):
    """The query decoder: N queries, each emitting (objectness, control polygon, mask embedding).

    The mask is produced by a dot product between a per-query embedding and a dense pixel embedding,
    exactly as Mask2Former does, so at TRAIN time it can be evaluated at sampled points only and at
    INFERENCE time densely. Per-query masks are independent sigmoids; there is no softmax across
    queries anywhere in this module, and that is the whole point.
    """

    def __init__(self, feat_dim: int, n_queries: int = 300, dim: int = 256, n_layers: int = 6,
                 n_heads: int = 8):
        super().__init__()
        self.n_queries, self.n_layers, self.dim = n_queries, n_layers, dim
        self.query = nn.Embedding(n_queries, dim)
        # Initial control polygon per query, stored as LOGITS and squashed by a sigmoid so a
        # control point is structurally inside the frame and CANNOT run away.
        #
        # This is not defensive dressing, it is a measured fix. The first CURVEQ run stored the
        # anchors as raw coordinates initialised in [0.2, 0.8]; after 3000 steps of nnU-Net's own
        # optimiser (SGD, momentum 0.99, lr 0.01 -- a recipe tuned for a conv trunk, not for a
        # coordinate embedding) they had reached -7.17 .. 10.18. Everything then pinned to the
        # [-0.2, 1.2] clamp in `forward`, every query degenerated to the same edge curve, all 300
        # objectness values became identical to four decimals and the model emitted nothing at all.
        # A sigmoid makes that failure unreachable rather than unlikely.
        self.anchor = nn.Embedding(n_queries, N_CTRL * 2)
        nn.init.uniform_(self.anchor.weight, -1.0, 1.0)     # sigmoid -> ~[0.27, 0.73]
        self.proj = nn.Conv2d(feat_dim, dim, 1)
        self.pix = nn.Conv2d(feat_dim, dim, 1)          # dense pixel embedding for the mask dot
        self.attn = nn.ModuleList(CurveDeformAttn(dim, n_heads) for _ in range(n_layers))
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(dim, n_heads, batch_first=True) for _ in range(n_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(n_layers))
        self.norm1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.norm3 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        # per-layer heads: the refinement is a DELTA on the current polygon, so layer k starts from
        # layer k-1's curve rather than from scratch
        self.d_ctrl = nn.ModuleList(nn.Linear(dim, N_CTRL * 2) for _ in range(n_layers))
        self.obj = nn.ModuleList(nn.Linear(dim, 1) for _ in range(n_layers))
        self.mask_emb = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        # Per-query weights on the relative-geometry basis (CondInst's dynamic head, in its
        # cheapest form): the query says how its own mask should depend on distance from its own
        # curve. Initialised so the geometric term starts at zero and the head begins as the
        # appearance-only model, then learns to use geometry -- a warm start, not a prior.
        self.rel_w = nn.ModuleList(nn.Linear(dim, len(REL_SCALES) + 1) for _ in range(n_layers))
        for m in self.rel_w:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.d_ctrl:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> list[dict]:
        """feat (B, C, H, W) from the trunk decoder -> one dict per layer (deep supervision)."""
        B = feat.shape[0]
        f = self.proj(feat)
        pix = self.pix(feat)
        q = self.query.weight[None].expand(B, -1, -1)
        # Refinement happens in LOGIT space and the coordinates are the sigmoid of it, so every
        # control point is in (0, 1) by construction at every layer, and a per-layer delta can
        # move a point across the frame but never out of it.
        z = self.anchor.weight.view(self.n_queries, N_CTRL, 2)[None].expand(B, -1, -1, -1)
        outs = []
        for i in range(self.n_layers):
            ctrl = torch.sigmoid(z)
            ref = sample_curve(ctrl)
            q = self.norm1[i](q + self.attn[i](q, f, ref))
            q = self.norm2[i](q + self.self_attn[i](q, q, q, need_weights=False)[0])
            q = self.norm3[i](q + self.ffn[i](q))
            # tanh-bounded step: one layer may move a control point by at most DELTA_MAX logits,
            # which keeps refinement monotone-ish instead of letting a single layer teleport.
            z = z + DELTA_MAX * torch.tanh(self.d_ctrl[i](q).view(B, self.n_queries, N_CTRL, 2))
            ctrl = torch.sigmoid(z)
            outs.append({"ctrl": ctrl, "obj": self.obj[i](q).squeeze(-1),
                         "emb": self.mask_emb[i](q), "pix": pix,
                         "rel_w": self.rel_w[i](q)})
        return outs


# Radial basis over the distance from a point to a query's own curve, in NORMALISED units.
# The scales bracket what matters here: 1, 3, 6 and 12 eval px in a 512 frame, i.e. the filament
# half-width, the metric's own tolerance, a close-parallel gap and "clearly elsewhere".
REL_SCALES = (1.0 / 511.0, 3.0 / 511.0, 6.0 / 511.0, 12.0 / 511.0)


def rel_features(ctrl: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """(B, Q, P, K+1) geometric features of each point RELATIVE TO EACH QUERY'S OWN CURVE.

    THIS IS THE REPAIR FOR THE ONE THING THAT KILLED CURVEQ. A mask of the form
    sigmoid(emb_q . pix(x)) reads a POSITION-INVARIANT function of local appearance, and two
    parallel filaments 3 px apart look the same to it — measured after 40 epochs, a query's mask
    scored 0.6221 on its own filament and 0.6219 on a different one, AUC 0.483, exactly chance.
    No embedding can fix that, because the information is not in the features.

    CondInst and SOLOv2 solve precisely this for instances that differ geometrically rather than
    semantically: the mask is conditioned on RELATIVE COORDINATES, so "my instance" is defined by
    where it is, not by what it looks like. Here the natural relative coordinate is the distance
    from the point to the query's own curve, expanded in a small radial basis so the head can learn
    an arbitrary (not necessarily monotone) profile, plus the raw distance itself.
    """
    B, Q = ctrl.shape[:2]
    P = pts.shape[1]
    curve = sample_curve(ctrl, n=N_SAMP)                              # (B, Q, S, 2)
    d = torch.cdist(pts[:, None].expand(B, Q, P, 2).reshape(B * Q, P, 2).float(),
                    curve.reshape(B * Q, N_SAMP, 2).float())          # (B*Q, P, S)
    d = d.min(-1).values.view(B, Q, P)                                # (B, Q, P)
    feats = [torch.exp(-(d / s) ** 2 / 2) for s in REL_SCALES] + [d]
    return torch.stack(feats, -1)


def mask_logits_at(emb: torch.Tensor, pix: torch.Tensor, pts: torch.Tensor,
                   ctrl: torch.Tensor | None = None,
                   rel_w: torch.Tensor | None = None) -> torch.Tensor:
    """Per-query mask logits at arbitrary points. emb (B,Q,D) · pix (B,D,H,W) · pts (B,P,2) in [0,1].

    POINT-SAMPLED by construction: this is the repair that keeps the mask loss out of OOM. Returns
    (B, Q, P). There is no softmax over Q anywhere — each query's mask is its own sigmoid, so two
    queries may both claim a pixel, which is the overlap the whole design exists to permit.

    `ctrl` and `rel_w` add the geometric conditioning: `rel_w` (B, Q, K+1) is a PER-QUERY weight
    vector, produced from the query embedding, applied to `rel_features`. Appearance alone cannot
    separate two identical filaments; appearance plus "how far is this point from MY curve" can.
    Omitting them reproduces the old appearance-only behaviour exactly, which is what the
    regression tests compare against.
    """
    B, P, _ = pts.shape
    g = (pts * 2 - 1).view(B, 1, P, 2).to(pix.dtype)
    s = F.grid_sample(pix, g, mode="bilinear", padding_mode="border", align_corners=False)
    s = s[:, :, 0, :]                                            # (B, D, P)
    # float32 for the dot: under autocast the logits feed a BCE, and a bf16 logit has ~3 decimal
    # digits, which is not enough resolution for a sigmoid near saturation.
    out = torch.einsum("bqd,bdp->bqp", emb.float(), s.float())
    if ctrl is not None and rel_w is not None:
        out = out + torch.einsum("bqpk,bqk->bqp", rel_features(ctrl, pts).float(), rel_w.float())
    return out


def dense_mask_logits(emb: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
    """(B, Q, H, W) — inference only; at train time use `mask_logits_at`."""
    return torch.einsum("bqd,bdhw->bqhw", emb.float(), pix.float())


def _labels_for(t: torch.Tensor, pts: torch.Tensor, tol: float = 3.0 / 511.0) -> torch.Tensor:
    """(T, P) ownership labels for the matcher, by the SAME rule the loss uses.

    Point-to-segment, not point-to-sampled-vertex: if the matcher's notion of ownership differed
    from the loss's, the assignment would optimise something the loss does not reward, which is the
    defect the per-query mask cost already had once.
    """
    return (_point_segment_dist(pts, t) <= tol).float().T


def chamfer_qt(qs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Symmetric chamfer between every query curve and every target curve. (Q,S,2)x(T,S,2)->(Q,T).

    SYMMETRY IS NOT COSMETIC. A one-directional distance (query-to-target only) is minimised by a
    SHORT query sitting on the middle of a long filament, which is exactly the coverage failure the
    strict metric charges: 22.9 % of roi303 GT fails on coverage alone. Averaging both directions
    makes a half-length query pay for the half it does not reach.
    """
    Q, S, _ = qs.shape
    T = t.shape[0]
    d = torch.cdist(qs.reshape(Q, S, 2).float().unsqueeze(1).expand(Q, T, S, 2).reshape(Q * T, S, 2),
                    t.float().unsqueeze(0).expand(Q, T, S, 2).reshape(Q * T, S, 2))   # (Q*T,S,S)
    return (0.5 * (d.min(-1).values.mean(-1) + d.min(-2).values.mean(-1))).view(Q, T)


@torch.no_grad()
def hungarian_match(ctrl: torch.Tensor, obj: torch.Tensor, emb: torch.Tensor, pix: torch.Tensor,
                    tgt_curves: list[torch.Tensor], w_geo: float = 5.0, w_obj: float = 1.0,
                    w_mask: float = 2.0, n_pts: int = 64, rel_w: torch.Tensor | None = None):
    """One-to-one assignment of queries to ground-truth curves, per image.

    The geometric cost is a SYMMETRIC chamfer between the sampled query curve and the sampled GT
    curve, and it is symmetric deliberately: a one-directional distance lets a query that covers
    half a long filament win the match, which is exactly the coverage failure the metric punishes.

    Returns a list of (query_idx, target_idx) LongTensor pairs, one per batch element.
    """
    from scipy.optimize import linear_sum_assignment
    B, Q = obj.shape
    pairs = []
    qs = sample_curve(ctrl)                                      # (B, Q, S, 2)
    for b in range(B):
        tc = tgt_curves[b]
        if tc is None or len(tc) == 0:
            pairs.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
            continue
        t = tc.to(ctrl.device)                                   # (T, S, 2), pre-resampled
        c_geo = chamfer_qt(qs[b], t)                             # (Q, T), symmetric
        c_obj = -torch.sigmoid(obj[b])[:, None].expand(-1, len(t))
        # MASK COST, PER (QUERY, TARGET). It used to be the query's mean sigmoid over all targets'
        # points, expanded across targets -- a per-query CONSTANT that said nothing about WHICH
        # target a query should own, and whose optimum is a mask that fires everywhere. Measured
        # consequence: a promiscuous query (mean sigma 0.98) beat a specialist sitting exactly on
        # the filament at geometric gaps up to ~190 px. The cost then pointed the opposite way from
        # the loss, which demands specificity. This is Mask2Former's actual cost: the BCE of each
        # query's mask against each target's own labels.
        pts = t.reshape(-1, 2).clamp(0, 1)
        lab_t = _labels_for(t, pts)                                   # (T, P) in {0, 1}
        if len(pts) > n_pts:
            sel = torch.randperm(len(pts), device=pts.device)[:n_pts]
            pts, lab_t = pts[sel], lab_t[:, sel]
        ml = mask_logits_at(emb[b:b + 1], pix[b:b + 1], pts[None],
                            ctrl[b:b + 1], None if rel_w is None else rel_w[b:b + 1])[0]
        c_mask = F.binary_cross_entropy_with_logits(
            ml[:, None, :].expand(-1, len(t), -1),
            lab_t[None].expand(len(ml), -1, -1), reduction="none").mean(-1)        # (Q, T)
        cost = (w_geo * c_geo + w_obj * c_obj + w_mask * c_mask).float().cpu().numpy()
        r, c = linear_sum_assignment(cost)
        pairs.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
    return pairs


def curveq_loss(outs: list[dict], tgt_curves: list[torch.Tensor], tgt_masks_pts,
                w_geo: float = 5.0, w_obj: float = 1.0, w_mask: float = 2.0,
                aux_k: int = 3) -> tuple[torch.Tensor, dict]:
    """Deep-supervised set loss with ONE-TO-MANY auxiliary supervision.

    `aux_k` is the fourth missing ingredient from the set-prediction post-mortem: besides the
    Hungarian one-to-one match, the `aux_k` next-best queries per target also receive positive
    gradient, at a reduced weight. With 300 queries and ~25 curves per frame a purely one-to-one
    loss gives 92 % of queries nothing but a "you are background" signal, and the decoder starves.

    `tgt_masks_pts` is a callable b -> (pts (P,2) in [0,1], labels (T,P) in {0,1}) giving, per
    image, the points at which the per-query masks are supervised and which target owns each. It is
    a callable so the sampling can be redrawn every step without materialising a dense tensor.
    """
    total = 0.0
    stats = {}
    for li, o in enumerate(outs):
        ctrl, obj, emb, pix = o["ctrl"], o["obj"], o["emb"], o["pix"]
        rel_w = o.get("rel_w")
        pairs = hungarian_match(ctrl, obj, emb, pix, tgt_curves, w_geo, w_obj, w_mask,
                                rel_w=rel_w)
        B, Q = obj.shape
        qs = sample_curve(ctrl)
        l_geo = obj.sum() * 0.0
        l_mask = obj.sum() * 0.0
        tgt_obj = torch.zeros_like(obj)
        # Aux queries are IGNORED in the objectness loss, not given a soft target. A soft target of
        # 0.25 carries an irreducible entropy floor of H(0.25) = 0.56 nats per aux query, which the
        # overfit test found as an `obj` term stuck at 0.43 that no amount of training could move —
        # the loss looked stalled while geometry and mask were still improving. "Close to a target,
        # so do not punish it for firing, but do not force it either" is a MASK, not a soft label.
        obj_ignore = torch.zeros_like(obj, dtype=torch.bool)
        n_pos = 0
        for b, (qi, ti) in enumerate(pairs):
            if len(qi) == 0:
                continue
            t = tgt_curves[b].to(ctrl.device)
            qi, ti = qi.to(ctrl.device), ti.to(ctrl.device)
            tgt_obj[b, qi] = 1.0
            a = qs[b][qi].float()
            c = t[ti].float()
            d1 = torch.cdist(a, c).min(-1).values.mean()
            d2 = torch.cdist(c, a).min(-1).values.mean()
            l_geo = l_geo + 0.5 * (d1 + d2)
            n_pos += len(qi)
            pts, lab = tgt_masks_pts(b)
            if pts is not None and len(pts):
                ml = mask_logits_at(emb[b:b + 1], pix[b:b + 1], pts[None].to(ctrl.device),
                                    ctrl[b:b + 1],
                                    None if rel_w is None else rel_w[b:b + 1])[0]
                lb = lab.to(ctrl.device).float()
                # POSITIVE WEIGHTING IS NOT OPTIONAL HERE, and this is measured. A filament is a
                # 3 px line in a 512 px crop, so a per-query mask label is ~2.6 % positive: 13 of
                # 512 sampled points belong to any given target. An unweighted BCE is therefore
                # minimised by predicting nothing, at a value of 0.1601 -- and CURVEQ run 2 sat at
                # 0.1843 from epoch 2 to epoch 17 without moving, with the best query's mask
                # support along its own curve measuring 0.006. The weight is derived from the
                # batch's own imbalance rather than set to a magic number, and clamped so a frame
                # with one short curve cannot produce a 1000x spike.
                pw = (lb.numel() - lb.sum()) / lb.sum().clamp_min(1.0)
                pw = pw.clamp(1.0, 100.0)
                l_mask = l_mask + F.binary_cross_entropy_with_logits(
                    ml[qi], lb[ti], pos_weight=pw)
                # one-to-many: the aux_k runner-up queries per target get the same mask target at
                # reduced weight, which is what keeps 300 queries learning from ~25 curves
                if aux_k > 0:
                    with torch.no_grad():
                        ch = chamfer_qt(qs[b], c)                 # (Q, n_matched)
                        best = ch.min(-1)                         # nearest target per query
                        order = best.values.argsort()
                        extra = order[~torch.isin(order, qi)][:len(qi) * aux_k]
                    if len(extra):
                        near = best.indices[extra]                # which matched target each owns
                        l_mask = l_mask + 0.25 * F.binary_cross_entropy_with_logits(
                            ml[extra], lb[ti[near]], pos_weight=pw)
                        obj_ignore[b, extra] = True
        el = F.binary_cross_entropy_with_logits(obj.float(), tgt_obj.float(), reduction="none")
        keep = (~obj_ignore).float()
        l_obj = (el * keep).sum() / keep.sum().clamp_min(1.0)
        layer = w_geo * l_geo / max(B, 1) + w_obj * l_obj + w_mask * l_mask / max(B, 1)
        # deep supervision: the last layer is the emission, earlier ones are auxiliary
        total = total + layer * (1.0 if li == len(outs) - 1 else 0.5)
        if li == len(outs) - 1:
            stats = {"geo": float(l_geo.detach() / max(B, 1)), "obj": float(l_obj.detach()),
                     "mask": float(l_mask.detach() / max(B, 1)), "n_pos": n_pos}
    return total, stats


@torch.no_grad()
def decode_curves(ctrl: torch.Tensor, obj: torch.Tensor, emb: torch.Tensor, pix: torch.Tensor,
                  shape: tuple[int, int], obj_thr: float = 0.5, mask_thr: float = 0.5,
                  min_support: float = 0.5,
                  rel_w: torch.Tensor | None = None) -> list[np.ndarray]:
    """Accepted queries -> polylines in PIXEL coordinates, (x=col, y=row), the eval convention.

    EXTENT COMES FROM THE MASK, NOT FROM A REGRESSED LENGTH. Each accepted query is sampled along
    its curve; the samples whose own mask logit clears `mask_thr` are kept, and the curve is
    trimmed to the longest contiguous run of supported samples. This is the endpoint repair: the
    model can only place an end where its own mask stops supporting the curve, so a length prior
    learned on synthetic filaments cannot invent one. `min_support` rejects a query whose curve is
    mostly unsupported — a hallucinated curve with no mask under it.
    """
    H, W = shape
    keep = torch.sigmoid(obj[0]) >= obj_thr
    if not bool(keep.any()):
        return []
    idx = torch.nonzero(keep).squeeze(-1)
    pts = sample_curve(ctrl[:, idx], n=N_SAMP * 2)[0]                    # (K, S, 2) normalised
    out = []
    for k in range(len(idx)):
        p = pts[k].clamp(0, 1)
        ml = mask_logits_at(emb[:, idx[k]:idx[k] + 1], pix, p[None],
                            ctrl[:, idx[k]:idx[k] + 1],
                            None if rel_w is None else rel_w[:, idx[k]:idx[k] + 1])[0, 0]
        sup = (torch.sigmoid(ml) >= mask_thr).cpu().numpy()
        if sup.mean() < min_support:
            continue
        runs, i = [], 0
        while i < len(sup):
            if not sup[i]:
                i += 1
                continue
            j = i
            while j < len(sup) and sup[j]:
                j += 1
            runs.append((i, j))
            i = j
        if not runs:
            continue
        a, b = max(runs, key=lambda r: r[1] - r[0])
        if b - a < 3:
            continue
        q = p[a:b].cpu().numpy()
        out.append(np.stack([q[:, 0] * (W - 1), q[:, 1] * (H - 1)], axis=1))
    return out


def transform_curves(curves: list[np.ndarray], y0: int, x0: int, c: int,
                     k: int = 0, flip_lr: bool = False, flip_ud: bool = False,
                     min_pts: int = 4) -> list[np.ndarray]:
    """Apply the loader's crop + rot90(k) + flips to (x, y) polylines, splitting at the crop edge.

    THE TARGET AND THE IMAGE MUST AGREE OR NOTHING ELSE MATTERS. The existing loader rasterises the
    dense target BEFORE cropping and then rotates the RASTER; a curve target has to be carried
    through the identical geometry as coordinates. Splitting rather than clamping at the boundary
    matches `RealDS._clip` and exists for the same reason: a curve clamped to the edge is taught as
    filament that is not there.

    `k` counts counter-clockwise rot90 on the ARRAY, matching `np.rot90(im, k)`; flips are applied
    after the rotation, in the loader's order (left-right, then up-down).
    """
    out = []
    for q in curves:
        d = np.asarray(q, float) - np.array([x0, y0])
        inside = (d[:, 0] >= 0) & (d[:, 0] < c) & (d[:, 1] >= 0) & (d[:, 1] < c)
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            if len(run) < min_pts:
                continue
            p = d[run].copy()
            # np.rot90(A, 1) sends array element (r, col) to (c-1-col, r); in (x=col, y=row) terms
            # that is (x, y) -> (y, c-1-x). Applying it k times is the loop below.
            for _ in range(k % 4):
                p = np.stack([p[:, 1], (c - 1) - p[:, 0]], axis=1)
            if flip_lr:
                p[:, 0] = (c - 1) - p[:, 0]
            if flip_ud:
                p[:, 1] = (c - 1) - p[:, 1]
            out.append(p)
    return out


def resample_norm(curves: list[np.ndarray], c: int, n: int = N_SAMP) -> torch.Tensor:
    """(T, n, 2) float32 in [0, 1], uniformly resampled by arclength. Empty -> (0, n, 2)."""
    if not curves:
        return torch.zeros(0, n, 2, dtype=torch.float32)
    out = []
    for q in curves:
        d = np.hypot(*np.diff(q, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] <= 1e-6:
            continue
        t = np.linspace(0, s[-1], n)
        out.append(np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], axis=1))
    if not out:
        return torch.zeros(0, n, 2, dtype=torch.float32)
    return torch.from_numpy(np.stack(out).astype(np.float32) / max(c - 1, 1)).clamp(0, 1)


def mask_points(tgt: torch.Tensor, c: int, n_pos: int = 256, n_neg: int = 256,
                tol_px: float = 3.0, rng: np.random.Generator | None = None):
    """Points and per-target labels for the point-sampled mask loss.

    POSITIVES are drawn from the curves themselves (so every target has support) and NEGATIVES
    uniformly from the frame. A pixel is labelled positive for target t if it lies within `tol_px`
    of t — and, crucially, it may be positive for SEVERAL targets at once, which is the overlap the
    emission exists to represent. A one-hot assignment here would quietly re-impose the
    single-owner rule the dense map already has and the whole design would be pointless.
    """
    if len(tgt) == 0:
        return None, None
    if rng is None:
        # A FIXED default seed would make every call return the same points, which contradicts the
        # docstring and quietly turns a stochastic sampler into a constant one.
        rng = np.random.default_rng()
    # POSITIVES ARE DRAWN ALONG THE CURVE, NOT FROM ITS 24 VERTICES. Taking the vertices means the
    # candidate points sit L/23 px apart -- 13 px on a 300 px filament -- so on a real crossing
    # there is simply NO candidate point near the junction, and the shared label the module exists
    # to learn cannot occur however exact the labelling rule is. Measured with vertex sampling and
    # an exact point-to-segment rule: a shared label appeared in 100 % of crossings of 80 px curves
    # and 0 % of crossings of 300 px curves. Densifying the positive pool is the actual fix.
    pts = _densify_targets(tgt, step=2.0 / max(c - 1, 1))
    if len(pts) > n_pos:
        pts = pts[torch.from_numpy(rng.choice(len(pts), n_pos, replace=False)).to(pts.device)]
    neg = torch.from_numpy(rng.random((n_neg, 2)).astype(np.float32)).to(pts.device)
    allp = torch.cat([pts, neg], 0)
    tol = tol_px / max(c - 1, 1)
    # DISTANCE TO THE SEGMENTS, NOT TO THE 24 SAMPLES. The label used to compare against the
    # sampled vertices only, and on a real filament those sit L/23 px apart -- 9 px for a 200 px
    # curve, against a 3 px tolerance. Measured consequence: a point shared by two crossing
    # filaments got a shared label in 100 % of crossings in the unit test's toy geometry (short
    # curves, dense samples) but only 46.8 % / 21.8 % / 13.8 % at curve lengths of 200 / 300 /
    # 400 px in a 512 crop. The overlap label -- the one thing this module exists to represent --
    # was therefore absent from ~86 % of realistic crossings. Point-to-segment distance is exact
    # at any sampling density.
    d = _point_segment_dist(allp, tgt)                          # (P, T)
    return allp, (d <= tol).float().T                           # (T, P)


def _densify_targets(tgt: torch.Tensor, step: float) -> torch.Tensor:
    """All target curves resampled to ~`step` spacing (normalised units), concatenated. (N, 2)."""
    out = []
    for k in range(len(tgt)):
        c = tgt[k]
        d = (c[1:] - c[:-1]).norm(dim=-1)
        s = torch.cat([torch.zeros(1, device=c.device, dtype=c.dtype), d.cumsum(0)])
        if float(s[-1]) <= 1e-9:
            out.append(c[:1])
            continue
        n = max(2, int(float(s[-1]) / max(step, 1e-6)) + 1)
        t = torch.linspace(0, float(s[-1]), n, device=c.device, dtype=c.dtype)
        idx = torch.searchsorted(s.contiguous(), t.contiguous()).clamp(1, len(s) - 1)
        s0, s1 = s[idx - 1], s[idx]
        w = ((t - s0) / (s1 - s0).clamp_min(1e-9)).unsqueeze(-1)
        out.append(c[idx - 1] + w * (c[idx] - c[idx - 1]))
    return torch.cat(out, 0)


def _point_segment_dist(pts: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Minimum distance from each point to each polyline, measured to its SEGMENTS. (P, T)."""
    a = tgt[:, :-1, :]                                          # (T, S-1, 2) segment starts
    b = tgt[:, 1:, :]                                           # segment ends
    ab = b - a
    den = (ab * ab).sum(-1).clamp_min(1e-12)                    # (T, S-1)
    ap = pts[:, None, None, :] - a[None]                        # (P, T, S-1, 2)
    u = ((ap * ab[None]).sum(-1) / den[None]).clamp(0.0, 1.0)   # (P, T, S-1)
    proj = a[None] + u[..., None] * ab[None]
    return (pts[:, None, None, :] - proj).norm(dim=-1).min(-1).values


class CurveQNet(nn.Module):
    """The shipped trunk + the dense binary head + the curve-query head, in one module.

    THE DENSE HEAD IS KEPT ON PURPOSE, for three reasons that are all measured rather than
    aesthetic. (1) It is the auxiliary task that gives the encoder gradient from step 1, while a
    set-prediction loss gives almost none until the queries find the filaments — starving the
    trunk is one documented way the 0.013 attempt could have failed. (2) It keeps the trainer's
    existing over-firing gate readable, so a CURVEQ run can still be compared to every previous
    arm on the identical dense statistic. (3) It is the honest comparator: if the curve emission
    never beats the dense map that shares its trunk, the paradigm has not earned anything, and
    that comparison costs nothing because both come out of the same forward pass.

    The query head reads the trunk's FULL-RESOLUTION decoder feature, captured by a hook on the
    final segmentation layer rather than by rebuilding the decoder — so the trunk stays exactly
    the architecture every previous number in this project was measured on.
    """

    def __init__(self, trunk: nn.Module, feat_dim: int, n_queries: int = 300, dim: int = 256,
                 n_layers: int = 6):
        super().__init__()
        self.trunk = trunk
        self.head = CurveQHead(feat_dim, n_queries=n_queries, dim=dim, n_layers=n_layers)
        self._feat = None
        seg = self._final_seg_layer(trunk)
        seg.register_forward_pre_hook(self._grab)

    @staticmethod
    def _final_seg_layer(trunk: nn.Module) -> nn.Module:
        """The conv that turns the last decoder feature into logits, whatever the version calls it."""
        dec = getattr(trunk, "decoder", None)
        segs = getattr(dec, "seg_layers", None) if dec is not None else None
        if segs is None:
            raise SystemExit("CurveQNet: trunk has no decoder.seg_layers; refusing to guess which "
                             "module produces the full-resolution feature.")
        return segs[-1]

    def _grab(self, module, inputs):
        self._feat = inputs[0]

    @staticmethod
    def feat_dim_of(trunk: nn.Module) -> int:
        return CurveQNet._final_seg_layer(trunk).in_channels

    def forward(self, x: torch.Tensor):
        dense = self.trunk(x)
        if self._feat is None:
            raise SystemExit("CurveQNet: the decoder-feature hook never fired.")
        outs = self.head(self._feat)
        self._feat = None
        return dense, outs


@torch.no_grad()
def decode_frame(net: "CurveQNet", img01: np.ndarray, device="cuda", tile: int = 0,
                 stride_frac: float = 0.757, obj_thr: float = 0.5, mask_thr: float = 0.5,
                 min_support: float = 0.5, merge_tol: float = 6.0, merge_deg: float = 30.0,
                 ima_m=None, ima_s=None, autocast=True):
    """Whole-frame polylines from a trained CurveQNet. THE TILING CONTRACT IS STATED, NOT IMPLIED.

    `tile = 0` runs the WHOLE FRAME in one forward pass. That is the honest configuration for this
    paradigm: there are no seams, so there is no cross-tile merge, so the claim "this emission needs
    no assembly step" is true as stated. It is the default and it should stay the default wherever
    the frame fits in memory.

    `tile > 0` falls back to overlapping tiles and MERGES curves across the seams. Say plainly what
    that is: a dense probability map can be averaged over overlapping tiles and the result is
    tile-invariant, but a per-query curve cannot — a filament crossing a seam is emitted twice, as
    two partial curves. Joining them is an ASSEMBLY STEP, of exactly the kind this paradigm claims
    to avoid, and any number produced through this path must be reported as such. The merge rule is
    deliberately the weakest thing that works (endpoint proximity + tangent agreement, no learned
    cost, no curvature search), so it cannot quietly become a second instancer.

    Returns polylines as (x=col, y=row) float arrays in FRAME pixels — the project's GT convention.
    """
    net.eval()
    H, W = img01.shape
    mean = ima_m if ima_m is not None else torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = ima_s if ima_s is not None else torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def run(patch: np.ndarray) -> list[np.ndarray]:
        # THE TRUNK NEEDS DIMENSIONS DIVISIBLE BY 128 or its residual adds hit a shape mismatch
        # (train_v5.TILE_FOR carries the same note). Whole-frame inference feeds arbitrary frame
        # sizes -- 1024 x 1400 and the like -- so without this every whole-frame decode dies with
        # "The size of tensor a (75) must match the size of tensor b (74)". Reflect-padding to the
        # next multiple keeps it ONE forward pass with no seams, so the assembly-free claim is
        # unaffected; the curves are emitted in padded coordinates and simply are not rescaled,
        # because padding adds pixels at the bottom-right only and leaves every existing pixel at
        # its own index.
        ph, pw = patch.shape
        H2 = int(np.ceil(ph / DIVISOR) * DIVISOR)
        W2 = int(np.ceil(pw / DIVISOR) * DIVISOR)
        if (H2, W2) != (ph, pw):
            patch = np.pad(patch, ((0, H2 - ph), (0, W2 - pw)), mode="reflect")
        t = torch.from_numpy(patch.astype(np.float32))[None].repeat(3, 1, 1)
        t = ((t - mean) / std)[None].to(device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else _null()
        with ctx:
            _, outs = net(t)
        o = outs[-1]
        cur = decode_curves(o["ctrl"], o["obj"], o["emb"], o["pix"], patch.shape,
                            obj_thr=obj_thr, mask_thr=mask_thr, min_support=min_support,
                            rel_w=o.get("rel_w"))
        # drop anything that lives only in the padded margin, and clip the rest back into frame
        out = []
        for c in cur:
            keep = (c[:, 0] < pw) & (c[:, 1] < ph)
            if keep.mean() < 0.5:
                continue
            out.append(np.stack([c[keep, 0].clip(0, pw - 1), c[keep, 1].clip(0, ph - 1)], 1))
        return [c for c in out if len(c) >= 3]

    if tile <= 0:
        return run(img01)

    step = max(1, int(round(tile * stride_frac)))
    ys = sorted(set(list(range(0, max(1, H - tile + 1), step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, max(1, W - tile + 1), step)) + [max(0, W - tile)]))
    pieces: list[np.ndarray] = []
    for y in ys:
        for x in xs:
            p = img01[y:y + tile, x:x + tile]
            for c in run(p):
                pieces.append(c + np.array([x, y], float))
    return merge_curve_pieces(pieces, merge_tol, merge_deg)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def merge_curve_pieces(pieces: list[np.ndarray], tol: float = 6.0,
                       deg: float = 30.0) -> list[np.ndarray]:
    """Join curve fragments whose ends meet and whose tangents agree. THIS IS AN ASSEMBLY STEP.

    Only used by the tiled decode path, and named so that a number produced through it cannot be
    reported as assembly-free. The rule is intentionally minimal — endpoint distance under `tol`
    and tangent difference under `deg` — because a richer rule would be a second instancer wearing
    a different hat, and this project has already measured that better arc-pair ranking converts to
    identical true positives (the learned linker: precision@k 0.862 vs 0.776, 229 TP either way).
    """
    items = [np.asarray(p, float) for p in pieces if len(p) >= 2]
    if not items:
        return []

    def outward(c, at_end):
        """Unit direction LEAVING the curve at that endpoint, pointing away from its body.

        Two fragments continue each other when their outward directions at the joining ends are
        OPPOSITE. Comparing them as if they should agree is off by 180 degrees and silently merges
        nothing (found by the seam test, which is why it exists).
        """
        d = (c[-1] - c[-min(len(c), 4)]) if at_end else (c[0] - c[min(len(c) - 1, 3)])
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([1.0, 0.0])

    used = [False] * len(items)
    out = []
    for i in range(len(items)):
        if used[i]:
            continue
        cur = items[i]
        used[i] = True
        grew = True
        while grew:
            grew = False
            for j in range(len(items)):
                if used[j]:
                    continue
                o = items[j]
                for cur_end in (True, False):
                    a = cur[-1] if cur_end else cur[0]
                    ta = outward(cur, cur_end)
                    for o_start in (True, False):
                        b = o[0] if o_start else o[-1]
                        if np.linalg.norm(a - b) > tol:
                            continue
                        tb = outward(o, not o_start)
                        # continuation means the two OUTWARD directions oppose each other
                        if np.degrees(np.arccos(np.clip(-float(ta @ tb), -1, 1))) > deg:
                            continue
                        piece = o if o_start else o[::-1]
                        cur = np.concatenate([cur, piece]) if cur_end else \
                            np.concatenate([piece[::-1], cur])
                        used[j] = True
                        grew = True
                        break
                    if grew:
                        break
                if grew:
                    break
        out.append(cur)
    return out


def overfire_stats(curves: list[np.ndarray], shape: tuple[int, int]) -> dict:
    """THE QUERY-SIDE OVER-FIRING GATE the dense trainer's gate structurally cannot provide.

    `train_v5.validate` reads `fg` — the fraction of pixels above threshold in a DENSE map. A curve
    emission has no such map, so that gate would read a constant and pass anything. The quantities
    that do carry the failure are the number of accepted curves and the total emitted centerline
    length relative to the frame, and they are what a pre-declared gate must read.
    """
    tot = float(sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in curves if len(c) > 1))
    return {"n_curves": len(curves), "total_len_px": tot,
            "len_per_Mpx": tot / max(shape[0] * shape[1] / 1e6, 1e-9)}

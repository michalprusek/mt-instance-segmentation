"""Centerline-boundary Dice (cbDice), adapted to this project's K-channel orientation head.

Why this loss and not more clDice
---------------------------------
The measured bottleneck (protocol 20) is that the predicted mask has the right coverage, width,
localisation and branch topology, and is nevertheless **shattered**: 2043 connected components
and 4302 endpoints against an oracle's 294 and 968, for the same 494 microtubules. clDice at
weight 0.1 is already in the loss and does not prevent it; raising it to 0.5 collapses training
to all-foreground (protocol history).

cbDice (Shi et al., MICCAI 2024) differs from clDice in exactly the way that matters here: it
weights the topological precision/sensitivity terms by the **local radius** taken from the
distance transform, which the authors introduce to balance "branch growth and fracture impacts"
across vessel calibres. A one-pixel break in a thin filament and a one-pixel break in a thick
one cost the same under clDice; under cbDice the thin one -- ours, always -- costs more.

Adaptation
----------
The reference implementation assumes nnU-Net's layout: a softmax over classes with channel 0 as
background. This head is different -- K=6 **independent sigmoid** orientation channels, where a
crossing puts the two filaments in different channels. The reduction to a single foreground
probability is therefore ``sigmoid(logits).amax(1)``, which is exactly what the existing
``soft_cldice`` in ``dino_seg`` already does, so the two losses see the same foreground.

The distance transforms sit behind a threshold and carry no gradient in the reference either;
they are computed with SciPy on CPU because monai is not installed here. Gradients flow through
the probability maps that those distance maps weight, which is the same path as upstream.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _edt(binary: torch.Tensor) -> torch.Tensor:
    """Euclidean distance transform of a (B, H, W) binary tensor, on CPU, without gradient.

    Behind a threshold in the reference implementation too, so nothing differentiable is lost.
    """
    arr = binary.detach().to("cpu").numpy().astype(bool)
    out = np.stack([distance_transform_edt(a) for a in arr]).astype(np.float32)
    return torch.from_numpy(out).to(binary.device)


def soft_erode(p: torch.Tensor) -> torch.Tensor:
    return torch.min(-F.max_pool2d(-p, (3, 1), 1, (1, 0)),
                     -F.max_pool2d(-p, (1, 3), 1, (0, 1)))


def soft_skel(p: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Morphological soft skeleton, identical to the one already used by soft_cldice."""
    p1 = F.max_pool2d(soft_erode(p), 3, 1, 1)
    skel = F.relu(p - p1)
    for _ in range(iters):
        p = soft_erode(p)
        p1 = F.max_pool2d(soft_erode(p), 3, 1, 1)
        d = F.relu(p - p1)
        skel = skel + F.relu(d - skel * d)
    return skel


def _weights(mask_prob: torch.Tensor, skel_prob: torch.Tensor, differentiable: bool):
    """cbDice's radius-normalised weight maps.

    Returns ``(dist_map_norm * mask, skel_R_norm * mask, I_norm * skel)``. ``I_norm`` is the
    subtraction-based inverse radius: a THIN skeleton point weighs more than a thick one, which
    is the whole reason this loss is a better fit than clDice for filaments that are two pixels
    across and break at one.
    """
    mask = (mask_prob > 0.5).float()
    skel = (skel_prob > 0.5).float()
    with torch.no_grad():
        dist = _edt(mask)
        dist = dist * mask
        skel_radius = dist * skel
        dist_norm = torch.zeros_like(dist)
        skel_r_norm = torch.zeros_like(dist)
        i_norm = torch.zeros_like(dist)
        for b in range(dist.shape[0]):
            r = skel_radius[b]
            r_max = torch.clamp(r.max(), min=1.0)
            r_min = torch.clamp(r[r > 0].min() if (r > 0).any() else r.max(), min=1.0)
            d = torch.clamp(dist[b], max=r_max)
            dist_norm[b] = d / r_max
            skel_r_norm[b] = r / r_max
            i_norm[b] = (r_max - r + r_min) / r_max         # 2-D form
        i_norm = i_norm * skel
    if differentiable:
        return dist_norm * mask_prob, skel_r_norm * mask_prob, i_norm * skel_prob
    return dist_norm * mask, skel_r_norm * mask, i_norm * skel


def _combine(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac, bc = a * c, b * c
    out = bc.clone()
    m = (a != 0) & (b == 0)
    out[m] = ac[m]
    return out


def soft_cbdice(logits: torch.Tensor, target: torch.Tensor, iters: int = 8,
                smooth: float = 1.0) -> torch.Tensor:
    """cbDice loss for a K-channel orientation head. Lower is better; returns ``1 - cbDice``.

    ``logits``: (B, K, H, W) raw outputs. ``target``: (B, K, H, W) binary orientation channels.
    Both are reduced to one foreground map by ``amax`` over channels, matching ``soft_cldice``.
    """
    p = torch.sigmoid(logits).amax(1)                     # (B, H, W)
    t = target.amax(1)

    skel_p_soft = soft_skel(p.unsqueeze(1), iters).squeeze(1)
    with torch.no_grad():
        skel_t = soft_skel(t.unsqueeze(1), iters).squeeze(1)
        skel_p_hard = (skel_p_soft > 0.5).float()
    skel_p = skel_p_hard * p                              # gradient enters here

    q_vl, q_slvl, q_sl = _weights(t, skel_t, differentiable=False)
    q_vp, q_spvp, q_sp = _weights(p, skel_p, differentiable=True)

    w_tprec = ((q_sp * q_vl).sum() + smooth) / (_combine(q_spvp, q_slvl, q_sp).sum() + smooth)
    w_tsens = ((q_sl * q_vp).sum() + smooth) / (_combine(q_slvl, q_spvp, q_sl).sum() + smooth)
    cb = 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens + 1e-8)
    return 1.0 - cb

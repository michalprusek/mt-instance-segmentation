"""Curve set prediction for microtubule instances: DETR-style, no assembly step.

WHY this architecture, given what the project already measured. The mask factorial (2026-08-25)
showed that with v5H's foreground 310 of 463 microtubules are *attainable* and only 227 are
achieved -- 83 are lost purely in assembly -- and the false-positive audit showed 84.3 % of
"false positives" are fragments of real annotated microtubules. Both numbers name the same
failure: skeletonise -> arc graph -> match -> walk turns one microtubule into several objects.

A set predictor has no such step. Each query emits a WHOLE curve, so a microtubule cannot be
fragmented by construction, and two predicted curves may overlap freely -- which is why this
escapes both earlier failures: per-pixel embeddings merge close parallels (they compete for the
same pixel) and point-cloud graph cuts collapse 2D crossings (degree-2 cap). Neither mechanism
exists when curves are objects rather than pixel assignments.

The supervision this needs -- an exact polyline per instance -- is precisely what the synthetic
generator already returns for free (`generate_frame` -> instances[i]["centerline"]), and what real
data cannot provide.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_curve(p: torch.Tensor, k: int) -> torch.Tensor:
    """Resample an (N,2) polyline to exactly k points, uniformly in ARCLENGTH.

    Arclength rather than vertex index: human and generator polylines carry wildly uneven vertex
    spacing, so index-uniform sampling would put most control points wherever the tracer happened
    to click and the L1 loss would be dominated by annotation habits rather than by geometry.
    """
    if p.shape[0] == 1:
        return p.repeat(k, 1)
    d = torch.linalg.norm(p[1:] - p[:-1], dim=1)
    s = torch.cat([torch.zeros(1, device=p.device, dtype=p.dtype), torch.cumsum(d, 0)])
    total = s[-1]
    if total <= 0:
        return p[:1].repeat(k, 1)
    t = torch.linspace(0, float(total), k, device=p.device, dtype=p.dtype)
    idx = torch.clamp(torch.searchsorted(s, t, right=True) - 1, 0, len(s) - 2)
    s0, s1 = s[idx], s[idx + 1]
    w = torch.where((s1 - s0) > 0, (t - s0) / (s1 - s0), torch.zeros_like(t))
    return p[idx] + w[:, None] * (p[idx + 1] - p[idx])


def curve_l1(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Direction-invariant mean L1 between (..., K, 2) curves.

    A microtubule has no canonical direction: the same filament may be stored start-to-end or
    end-to-start. Charging the forward distance only would hand the model two contradictory
    gradients for one curve and drive it toward the average of both -- a straight line through the
    middle. Taking the min over the curve and its reverse removes the ambiguity.
    """
    fwd = (pred - tgt).abs().mean(dim=(-1, -2))
    bwd = (pred - tgt.flip(-2)).abs().mean(dim=(-1, -2))
    return torch.minimum(fwd, bwd)


class PositionalEncoding2D(nn.Module):
    """Fixed sine positional encoding over a feature map."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        half = self.dim // 2
        freq = torch.exp(torch.arange(half // 2, device=x.device, dtype=torch.float32)
                         * (-math.log(10000.0) / max(1, half // 2)))
        yy = torch.arange(h, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]
        xx = torch.arange(w, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]
        pe_y = torch.cat([yy.sin(), yy.cos()], 1)[:, None, :].expand(h, w, half)
        pe_x = torch.cat([xx.sin(), xx.cos()], 1)[None, :, :].expand(h, w, half)
        pe = torch.cat([pe_y, pe_x], -1).permute(2, 0, 1)[None]
        return pe.expand(b, -1, -1, -1)


class CurveSetNet(nn.Module):
    """Backbone feature map -> transformer decoder -> (objectness, K control points) per query."""

    def __init__(self, backbone: nn.Module, backbone_ch: int = 32, dim: int = 256,
                 n_queries: int = 400, n_points: int = 16, n_layers: int = 6, n_heads: int = 8,
                 stride: int = 8):
        super().__init__()
        self.backbone = backbone
        self.n_queries, self.n_points = n_queries, n_points
        self.proj = nn.Conv2d(backbone_ch, dim, kernel_size=stride, stride=stride)
        self.pe = PositionalEncoding2D(dim)
        self.query = nn.Embedding(n_queries, dim)
        layer = nn.TransformerDecoderLayer(dim, n_heads, dim_feedforward=4 * dim,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.obj_head = nn.Linear(dim, 1)
        self.pt_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                     nn.Linear(dim, n_points * 2))

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(img)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        f = self.proj(feat)
        f = f + self.pe(f)
        b = f.shape[0]
        mem = f.flatten(2).transpose(1, 2)
        q = self.query.weight[None].expand(b, -1, -1)
        h = self.dec(q, mem)
        obj = self.obj_head(h).squeeze(-1)
        pts = self.pt_head(h).view(b, self.n_queries, self.n_points, 2).sigmoid()
        return obj, pts

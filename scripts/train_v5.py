#!/usr/bin/env python3
"""Foreground training, v5. RUNS ON TULEN / KAJMAN.

Three measured defects are fixed here; each is a separate flag so each can be attributed.

1. **The generator labelled filaments below the noise floor.** Synthetic labelled microtubules
   sat at median SNR 0.50 against 1.45 for human-annotated real ones, and coverage is 97.5 %
   above SNR 1 versus 63 % below it -- so roughly a third of training positives carried no
   learnable signal. Fixed in the DATA by ``--calib`` (contrast x3); this script only has to
   consume it. See docs/F1_PUSH_PLAN.md.
2. **The shallow full-resolution branch is dead weight that shatters the mask.** Ablating it
   changes instance-F1 by +0.017 (95 % CI [-0.032, +0.068], p=0.31) while removing 68 % of the
   mask's connected components, and its 29 px receptive field cannot integrate evidence along a
   375 px filament. ``--arch resunet`` replaces it with a downsampling encoder-decoder, whose
   receptive field is hundreds of pixels and which is the topology nnU-Net already beat v4b
   with on this same synthetic data.
3. **Selection ran on proxies.** Section 17p is the cost: the continuity proxy improved 25 %
   and downstream F1 did not move. Validation here scores the TARGET -- strict centerline-F1
   after instancing -- and still refuses any checkpoint that fails the over-firing gate,
   because the continuity metrics remain gameable by flooding the frame.

The recipe additions (cosine schedule with warmup, AMP, EMA, rotation augmentation) each
answer a measured defect: every surviving loss curve was still descending at its cutoff, late
epochs oscillate to 12-15 components per microtubule, and the K-bin orientation head previously
saw only the four flip-induced bin permutations.

    PYTHONPATH=src:third_party python scripts/train_v5.py --arch resunet \
        --calib /home/prusek/mt_enc_exp/calib_reg418_morph_snr3.json --epochs 40
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, grey_dilation, zoom
from skimage.morphology import disk

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "synth"))

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fg_quality import foreground_quality, mean_properties  # noqa: E402
from photo_aug import photometric_augment  # noqa: E402

DEV = "cuda"
K_ORI = 6
TILE, STRIDE = 518, 392
UP = 1.5
KAPPA_MAX = 0.25
IMA_M = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMA_S = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------------------------------- architecture
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
            nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
            nn.ReLU(True))

    def forward(self, x):
        return self.b(x)


class ResUNet(nn.Module):
    """Plain residual U-Net at full resolution, K orientation channels + a binary union head.

    The binary head is not cosmetic. The union target is ONE connected component per
    microtubule while the per-channel orientation targets average 4.35 pieces, so reading
    foreground off max-over-K asks the network to reconstruct a connected object from sparse,
    bin-switching supervision. nnU-Net's binary head on this same synthetic data already
    produced 40 % fewer mask components at HIGHER foreground.
    """

    def __init__(self, base=32, depth=5, out_ori=K_ORI):
        super().__init__()
        chs = [base * min(2 ** i, 8) for i in range(depth)]
        self.enc = nn.ModuleList()
        cin = 3
        for c in chs:
            self.enc.append(ConvBlock(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(chs[i - 1] * 2, chs[i - 1]))
        self.head_ori = nn.Conv2d(chs[0], out_ori, 1)
        self.head_fg = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, s], 1))
        return torch.cat([self.head_ori(x), self.head_fg(x)], 1)


#: nnU-Net's own 2D ResEnc-M plan for Dataset501_MTSynth, read out of
#: ``nnUNet_results/.../plans.json`` rather than guessed. The 8 stages are the point: seven
#: successive /2 downsamplings give the bottleneck a receptive field spanning the whole 512 px
#: patch, which is what the segmenter audit said was missing (the v4b high-resolution branch
#: reaches 29 px and cannot integrate evidence along a 375 px filament). The plain 5-stage
#: U-Net tried first capped at 256 channels and lost to v4b.
NNUNET_PLAN = dict(
    n_stages=8,
    features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
    kernel_sizes=[[3, 3]] * 8,
    strides=[[1, 1]] + [[2, 2]] * 7,
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6, 6],
    n_conv_per_stage_decoder=[1] * 7,
)


def build_nnunet(out_ch: int, deep_supervision: bool = True, in_ch: int = 3):
    """nnU-Net's ResEnc-M topology with a MULTI-LABEL head.

    nnU-Net's own framework cannot train this target: its head assigns each pixel exactly one
    class, so it structurally cannot express a crossing pixel belonging to two microtubules at
    once (the note is in scripts/nnunet_prepare_synth.py, and it is why the existing nnU-Net
    run is binary-only). Borrowing the architecture and supervising it with per-channel
    sigmoids keeps the topology that measured best on this data while restoring the amodal
    orientation channels that instancer B needs and that the K-channel convention depends on.
    """
    import torch.nn as _nn
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=in_ch, num_classes=out_ch, conv_op=_nn.Conv2d,
        conv_bias=True, norm_op=_nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=_nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision, **NNUNET_PLAN)


def build_model(arch: str, out_ch: int = K_ORI + 1, in_ch: int = 3):
    """``out_ch`` exists so legacy checkpoints stay loadable.

    Everything trained before this script has a K=6 head; this one adds the binary union
    channel. Hard-coding K+1 would make every earlier checkpoint unloadable with a shape error,
    and re-deriving their numbers under the corrected per-model thresholding is precisely the
    work the calibration finding created.
    """
    if arch == "nnunet":
        return build_nnunet(out_ch, in_ch=in_ch)
    if arch == "resunet":
        return ResUNet(out_ori=out_ch - 1 if out_ch > K_ORI else out_ch)
    from dino_seg import DinoSeg                       # the v4b architecture, unchanged
    m = DinoSeg()
    if out_ch != m.head.out_channels:
        m.head = nn.Conv2d(m.head.in_channels, out_ch, 1)
    return m


# --------------------------------------------------------------------------- data
def heat_channel(instances, shape, sigma: float) -> np.ndarray:
    """Gaussian centerline heatmap ``max_i exp(-d_i^2 / 2 sigma^2)`` = ``exp(-d_min^2 / 2 sigma^2)``
    where ``d_min`` is the Euclidean distance to the nearest 1-px rasterised centerline.

    The HEAT arm's localisation target (docs/TODO.md 0.0-CAMPAIGN, entries 19/20; research reports
    R2 / #3): a target whose MAXIMUM sits on the centerline by construction and whose value between
    two parallels at gap g is exp(-(g/2)^2 / 2 sigma^2) -- at sigma = 1.2 px that is 0.17 for
    g = 4.5 px and 0.04 for g = 6 px, so the dip is built into the label; a 3 px binary band has
    none (two bands 4.5 px apart share no gap at all at the training scale). Decoded by ridge
    non-maximum suppression, not by a threshold. ``max`` over instances (not a sum) keeps the value
    at 1 through crossings on both filaments.
    """
    from scipy.ndimage import distance_transform_edt
    from instance.geometry import resample
    cl = np.zeros(tuple(shape), bool)
    for ins in instances:
        c = np.asarray(ins["centerline"], float)
        if len(c) < 2:
            continue
        pts = resample(c, ds=0.4)
        cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, shape[1] - 1)
        rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, shape[0] - 1)
        cl[rr, cc] = True
    if not cl.any():
        return np.zeros(tuple(shape), np.float32)
    d = distance_transform_edt(~cl)
    return np.exp(-0.5 * (d / float(sigma)) ** 2).astype(np.float32)


# ARM 53 (2026-09-16): the target band's shape. "square" = the inherited 3x3 dilation of the rasterised
# centerline, whose pixel count per unit length depends on the curve's angle to the grid (3.0 at 0/90
# deg, ~3.5-4.0 at 30-60 deg) and is preserved by the rot90/flip augmentation group; "euclid" = every
# pixel within TARGET_BAND_RADIUS eval px (exact Euclidean distance) of the resampled centerline,
# which is orientation-isotropic. Set from --target-band; unflagged runs stay bit-identical.
TARGET_BAND = "square"
TARGET_BAND_RADIUS = 1.5     # 3 px wide at every angle except the measure-zero exact diagonal (pixel centres at sqrt(2))


def ori_channels_soft(instances, shape, K=K_ORI, half_width=1.0, heat_sigma: float = 0.0,
                      curve_ids: bool = False):
    """Orientation-keyed centerline target with SOFT two-bin assignment and sub-pixel stamping.

    ``gen_train.ori_channels`` stamps ``int(round(x))`` into the HARD bin of the local tangent.
    Measured consequence: argmax-bin correctness falls from 0.90 to 0.58 within one degree of a
    bin boundary, and each microtubule's per-channel supervision is chopped into 4.35 pieces.
    Splitting each sample between its two nearest bins by angular distance removes the
    discontinuity at no training cost. The union over channels is unchanged.

    Channel layout of the returned array: ``K`` orientation channels, then the binary union
    foreground, then an instance-COUNT map. The count map is supervision bookkeeping, not a
    prediction target -- it tells the crossing term which pixels are genuinely amodal, which
    cannot be recovered by thresholding the soft orientation channels (a single filament lying
    exactly between two bins puts 0.5 in each).
    """
    from instance.geometry import resample, segment_angles
    ch = np.zeros((K,) + tuple(shape), np.float32)
    fg = np.zeros(tuple(shape), np.float32)
    cnt = np.zeros(tuple(shape), np.float32)
    # Per-curve identity, 0 = background, j = the j-th instance. Needed only by the worst-point
    # term (--minpool-weight); it is built lazily so an unflagged run is bit-identical.
    cid = np.zeros(tuple(shape), np.float32) if curve_ids else None
    width = np.pi / K
    for j, ins in enumerate(instances, start=1):
        cl = np.asarray(ins["centerline"], float)
        if len(cl) < 2:
            continue
        pts = resample(cl, ds=0.4)
        seg = segment_angles(pts)
        ang = np.concatenate([seg, seg[-1:]]) % np.pi
        pos = ang / width - 0.5                       # bin centres at integer positions
        b0 = np.floor(pos).astype(int)
        w1 = pos - b0
        cc = np.clip(np.rint(pts[:, 0]).astype(int), 0, shape[1] - 1)
        rr = np.clip(np.rint(pts[:, 1]).astype(int), 0, shape[0] - 1)
        if TARGET_BAND == "euclid" and int(round(half_width)) == 1:
            # exact Euclidean band: candidate pixels = the raster dilated 5x5, kept if their centre is
            # within TARGET_BAND_RADIUS of the nearest resampled point; that point's soft bins apply
            from scipy.spatial import cKDTree
            one = np.zeros(tuple(shape), bool)
            one[rr, cc] = True
            cand = binary_dilation(one, structure=np.ones((5, 5), bool))
            cy, cx = np.nonzero(cand)
            dist, idx = cKDTree(pts).query(np.stack([cx, cy], 1).astype(float))
            keep = dist <= TARGET_BAND_RADIUS
            cy, cx, idx = cy[keep], cx[keep], idx[keep]
            for bb, ww in (((b0) % K, 1.0 - w1), ((b0 + 1) % K, w1)):
                np.maximum.at(ch, (bb[idx], cy, cx), ww[idx].astype(np.float32))
            fg[cy, cx] = 1.0
            if cid is not None:
                cid[cy, cx] = float(j)
            cnt += binary_dilation(one, structure=np.ones((3, 3), bool))
            continue
        for bb, ww in (((b0) % K, 1.0 - w1), ((b0 + 1) % K, w1)):
            np.maximum.at(ch, (bb, rr, cc), ww.astype(np.float32))
        fg[rr, cc] = 1.0
        if cid is not None:
            cid[rr, cc] = float(j)
        one = np.zeros(tuple(shape), np.float32)
        one[rr, cc] = 1.0
        cnt += binary_dilation(one > 0, structure=np.ones((3, 3), bool))
    # WIDTH OF THE TARGET. `half_width` is in eval-frame pixels: the 1 px rasterised centerline is
    # dilated by a disk of that radius, so half_width=1.0 gives the 3 px line this project has
    # always trained against, and 2.0 gives 5 px.
    #
    # It used to be accepted and then IGNORED -- the width was a hard-coded 3x3 dilation and the
    # argument reached nothing. Any sweep of it would have compared a model against itself and
    # recorded a guaranteed null. Found 2026-09-01; the 2026-08-26 thin-target result came from a
    # different code path (gen_train.ori_channels) and still stands, but it cannot be reproduced
    # through this flag, and only a THINNER target was ever tried. A real microtubule is 3-6 px
    # wide in the 1.5x eval frame while the target is 3, so the thicker direction is untested.
    # half_width is in eval-frame pixels:
    #   0  -> the bare 1 px rasterised centerline, the THINNEST target there is
    #   1  -> the inherited 3 px line (a full 3x3 SQUARE, not skimage's cross-shaped disk(1)),
    #         which is what every number this project has measured used
    #   >=2 -> a disk of that radius
    r = max(int(round(half_width)), 0)
    # optional localisation channel, appended LAST so every existing index (K ori, K_ORI = union
    # foreground, K_ORI + 1 = instance count) is unchanged; heat_sigma = 0 appends nothing.
    extra = [heat_channel(instances, shape, heat_sigma)[None]] if heat_sigma > 0 else []
    if r == 0:
        ch = ch.copy()
        tail = extra + ([cid[None]] if cid is not None else [])
        return np.concatenate([ch, (fg > 0).astype(np.float32)[None], cnt[None]] + tail, 0)
    if TARGET_BAND == "euclid" and r == 1:
        # the band was built exactly above; no dilation
        tail = extra + ([cid[None]] if cid is not None else [])
        return np.concatenate([ch, (fg > 0).astype(np.float32)[None], cnt[None]] + tail, 0)
    fp = np.ones((3, 3), bool) if r == 1 else disk(r)
    ch = np.stack([grey_dilation(ch[k], footprint=fp) for k in range(K)])
    fg = binary_dilation(fg > 0, structure=fp).astype(np.float32)
    if cid is not None:
        # The id map is dilated by the SAME footprint as the union target, so every foreground
        # pixel the BCE sees carries an owner. Where two curves' bands overlap the higher id
        # wins; a crossing pixel therefore belongs to exactly one curve's worst-point set, which
        # is the conservative choice (it cannot double-count a crossing as two separate dips).
        cid = grey_dilation(cid, footprint=fp)
    tail = extra + ([cid[None]] if cid is not None else [])
    return np.concatenate([ch, fg[None], cnt[None]] + tail, 0)


def apply_field_stop(im, gt, rng, K=K_ORI, sat_range=(1.3, 5.0)):
    """Composite an octagonal FIELD STOP, and drop the labels outside it.

    Task-586 frames were acquired through an octagonal aperture: a central imaged region inside
    a hard transition into a SATURATED surround (see mt_bench.fov, which detects it). The
    generator composites onto real EMPTY-FIELD backgrounds, which carry no stop, so the model
    has never had a reason to learn that a field boundary is not a filament -- and it duly fires
    on it: 102 of 367 false positives on TEST are the rim, 70.6 % of them within 20 px of the
    aperture, where 0.0 % of true positives lie. Masking at inference recovers ~16 % of that
    mass at a ground-truth-safe erosion and no more, so the honest fix is to put the stop in the
    training data.

    Labels outside the aperture are removed rather than kept: the instrument has no data there,
    and the annotators drew none.

    Geometry and photometry are measured off the 19 MT-34 frames that have a stop, not guessed:
    the detected interior covers ~91 % of the frame, and the surround sits at **2.5-5x the
    interior median** (e.g. 37 -> 187, 4563 -> 11783). Re-measured 2026-09-16 on the 76 roi303
    VAL frames that carry one (55 % of the block): surround / deep-interior median p10/p50/p90 =
    1.45 / 2.0 / 3.5, no dark ring inside the edge (ring depth p50 1.02), a slight bright rim in
    the last 3 px (1.21) from the blur of the surround. The surround range is therefore widened
    to 1.3-5x -- the domain-randomisation direction, covering both blocks -- and the geometry
    kept. SPARSE35 and every arm so far were trained with ``--fov-prob 0`` (the flag never
    appeared in a launcher), so the model met its first field stop on the benchmark: the montage
    of its off-annotation predictions (2026-09-16) shows the aperture edge in a large share. It is applied to the RAW render, before
    the percentile stretch, because that is where it sits in reality -- a bright surround drags
    the frame's percentiles and squashes the imaged interior, which is a distortion the model
    has to cope with at inference and would never see if the stop were pasted on afterwards.
    """
    h, w = im.shape
    cy = h * (0.5 + 0.06 * (rng.random() - 0.5) * 2)
    cx = w * (0.5 + 0.06 * (rng.random() - 0.5) * 2)
    apo = min(h, w) * rng.uniform(0.50, 0.54)           # apothem -> ~83-97 % of the frame inside
    phi = rng.uniform(0, np.pi / 4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    inside = np.ones((h, w), bool)
    for k in range(8):                                  # octagon = 8 half-planes
        a = phi + k * np.pi / 4
        inside &= ((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)) <= apo
    if inside.all():                                    # aperture larger than the crop
        return im.astype(np.float32), gt
    # A hard edge with a short ramp: the transition ring is exactly the gradient the segmenter
    # mistakes for a filament, so it has to be present, not idealised away.
    ramp = gaussian_filter(inside.astype(np.float32), rng.uniform(0.8, 2.5))
    interior = float(np.median(im[inside]))
    sat = interior * rng.uniform(*sat_range)            # roi303 1.45-3.5, MT-34 2.5-5 (measured)
    out = im * ramp + sat * (1.0 - ramp)
    gt = gt * inside[None, :, :].astype(gt.dtype)
    return out.astype(np.float32), gt


def rotate_pair(img, gt, theta_deg):
    """ARM 54 (2026-09-17): rotate a synthetic frame AND its target stack by an arbitrary angle.
    Image bilinear with reflect padding (what the rotation probe does to real frames); target
    channels nearest-neighbour so the union stays binary and the count map integer. The K
    orientation channels are keyed to the ORIGINAL tangent bins and are NOT re-binned, so this
    is only valid for heads that do not read them (--head binary); main() enforces that.
    """
    from scipy.ndimage import rotate as _rot
    im = _rot(np.asarray(img, np.float32), theta_deg, reshape=False, order=1, mode="reflect")
    g = np.stack([_rot(gt[k], theta_deg, reshape=False, order=0, mode="constant", cval=0.0) for k in range(gt.shape[0])])
    return im.astype(np.float32), g.astype(gt.dtype)


class OnlineDS(torch.utils.data.Dataset):
    """Fresh synthetic frame per item, with rotation augmentation.

    Rotation matters here specifically: the K-bin orientation head previously saw only the four
    orientation permutations that flips induce, so its bins were unevenly sampled. A rotation
    by a multiple of 180/K degrees permutes the bins exactly, which keeps the target derivable
    without re-rasterising.
    """

    def __init__(self, cfg, bg_paths, gen_fn, crop=518, gen=640, epoch_len=5000, rot=True,
                 fov_prob=0.0, target_half_width=1.0, tirf_prob=0.0, photo_aug=0.0,
                 scale_aug=None, scale_prob=0.7, lowres_prob=0.0, heat_sigma=0.0,
                 curve_ids=False, curve_targets=False, rot_aug_prob=0.0):
        self.cfg, self.bg, self.gen_fn = cfg, bg_paths, gen_fn
        self.heat_sigma = heat_sigma          # > 0 appends the HEAT localisation channel
        self.curve_ids = curve_ids            # True appends the per-curve id channel LAST
        # True ALSO returns the exact generator centerlines, carried through the identical crop /
        # rotation / flip as coordinates (--head curveq). The generator already has them; nothing
        # new is rendered, and with the flag off the loader's return type is unchanged.
        self.curve_targets = curve_targets
        self.crop, self.gen, self.epoch_len, self.rot = crop, gen, epoch_len, rot
        self.fov_prob = fov_prob
        self.target_half_width = target_half_width
        self.rot_aug_prob = rot_aug_prob                  # ARM 54: arbitrary-angle rotation, 0 = off (bit-identical)
        # Geometric SCALE augmentation of the RENDERED frame (Holm entries 17/18, declared
        # 2026-09-04): the image and the polylines are rescaled by s ~ LogUniform(scale_aug)
        # before the target is rasterised, so the target keeps its width in OUTPUT pixels while
        # the filament's apparent width scales with s. Measured need: synthetic filaments train
        # at 4.0-4.8 px FWHM, real ones are 3.5 px native and 5.5 px in the 1.5x eval frame
        # (docs/research/2026-09-02-scale-and-gap-audit.md). `lowres_prob` adds nnU-Net's
        # low-resolution simulation (nearest downsample by U(0.6, 1), cubic upsample back) on the
        # image only. None / 0.0 draw nothing and are bit-identical to earlier runs.
        self.scale_aug = tuple(scale_aug) if scale_aug else None
        self.scale_prob = scale_prob
        self.lowres_prob = lowres_prob
        # Photometric nuisance augmentation (synth/photo_aug.py): gamma, contrast, noise, blur,
        # illumination field, clipped stretch + quantisation. Measured need 2026-09-04: the
        # network's confidence is a FRAME property (per-frame optimal threshold 0.30-0.9995,
        # per-frame oracle threshold worth +0.048). 0.0 draws nothing and is bit-identical.
        self.photo_aug = photo_aug
        # Fraction of frames rendered with the TIRF appearance instead of IRM. The morphology is
        # sampled identically either way -- only the renderer differs -- so this teaches the
        # network a second modality without touching what a microtubule IS. 0.0 reproduces the
        # previous behaviour exactly, so unflagged runs stay bit-identical.
        self.tirf_prob = tirf_prob

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        import random
        rng = np.random.default_rng((((wid + 1) * 1000003) * (i + 1)) ^ random.getrandbits(48))
        g = self.gen
        bg = None
        for _ in range(8):
            b = np.asarray(Image.open(self.bg[int(rng.integers(len(self.bg)))]).convert("F"),
                           np.float32)
            while b.ndim > 2:
                b = b[..., 0]
            if b.shape[0] >= g and b.shape[1] >= g:
                bg = b
                break
        if bg is None:
            bg = np.asarray(Image.fromarray(b).resize((max(g, b.shape[1]), max(g, b.shape[0]))),
                            np.float32)
        H, W = bg.shape
        y0, x0 = int(rng.integers(0, H - g + 1)), int(rng.integers(0, W - g + 1))
        bg = bg[y0:y0 + g, x0:x0 + g]
        if self.tirf_prob and rng.random() < self.tirf_prob:
            img, inst, gm = self.gen_fn(bg, rng, self.cfg, modality="tirf")
            # Drop microtubules that carry no fluorophore at all. They are invisible in TIRF, so a
            # target that still claims them is unlearnable and can only push the model towards
            # firing on empty background. The generator marks them; nothing is marked in IRM, and
            # nothing is marked when sparse labelling is off, so both stay bit-identical.
            inst = [z for z in inst if not z.get("tirf_unlabelled")]
        else:
            img, inst, gm = self.gen_fn(bg, rng, self.cfg)
        nb = gm.get("neighbours") if isinstance(gm, dict) else None   # ARM 65: (t-1, t+1) or None
        if self.scale_aug is not None and rng.random() < self.scale_prob:
            lo_s, hi_s = self.scale_aug
            sc = float(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))))
            order = int(rng.choice([1, 3]))
            img = zoom(np.asarray(img, np.float32), sc, order=order)
            inst = [{**z, "centerline": np.asarray(z["centerline"], float) * sc} for z in inst]
        gh, gw = img.shape
        if self.lowres_prob and rng.random() < self.lowres_prob:
            f = float(rng.uniform(0.6, 1.0))
            small = zoom(np.asarray(img, np.float32), f, order=0)
            up = zoom(small, (gh / small.shape[0], gw / small.shape[1]), order=3)
            img = np.pad(up, ((0, max(0, gh - up.shape[0])), (0, max(0, gw - up.shape[1]))),
                         mode="edge")[:gh, :gw]
        gt = ori_channels_soft(inst, (gh, gw), half_width=self.target_half_width,
                               heat_sigma=self.heat_sigma, curve_ids=self.curve_ids)
        if self.fov_prob and rng.random() < self.fov_prob:
            # BEFORE the stretch: a bright surround drags the frame's percentiles and squashes
            # the interior, which is exactly what happens to the real task-586 frames.
            img, gt = apply_field_stop(img, gt, rng)
        if self.rot_aug_prob and rng.random() < self.rot_aug_prob:
            # ARM 54: the rot90/flip group below preserves any grid-keyed response; an arbitrary
            # angle (U(0, 90) is enough, the group supplies the rest) is the only augmentation that
            # can make the response isotropic. Half the samples stay un-resampled on purpose so the
            # crisp-noise competence real frames need is kept.
            img, gt = rotate_pair(img, gt, float(rng.uniform(0.0, 90.0)))
        lo, hi = np.percentile(img, [1, 99])
        im = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        _inv = rng.random() < .5
        if _inv:
            im = 1.0 - im                                     # IRM polarity is ambiguous

        c = self.crop
        if gh < c or gw < c:                       # a downscaled frame smaller than the crop
            pad = ((0, max(0, c - gh)), (0, max(0, c - gw)))
            im = np.pad(im, pad, mode="reflect")
            gt = np.pad(gt, ((0, 0),) + pad, mode="constant")
            gh, gw = im.shape
        y = int(rng.integers(0, gh - c + 1))
        x = int(rng.integers(0, gw - c + 1))
        im, gt = im[y:y + c, x:x + c], gt[:, y:y + c, x:x + c]
        _k = _flr = _fud = 0
        # k*90 degree rotations and flips; both permute the orientation bins exactly.
        if self.rot:
            k = int(rng.integers(4))
            _k = k
            if k:
                im = np.rot90(im, k).copy()
                gt = np.rot90(gt, k, axes=(1, 2)).copy()
                # rot90 maps theta -> theta + k*90deg, i.e. bin b -> b + k*K/4 (K=6 -> 1.5, so
                # only k=2 is exact); K=6 is exact for 180deg, which is identity on [0,pi).
                if k % 2 == 1:
                    gt[:K_ORI] = np.roll(gt[:K_ORI], K_ORI // 2, axis=0)
        if rng.random() < .5:
            _flr = 1
            im = im[:, ::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, :, ::-1].copy()
        if rng.random() < .5:
            _fud = 1
            im = im[::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, ::-1, :].copy()
        # LAST, after every geometric draw: the augmentation consumes random numbers, and placing
        # it here keeps the crop / rotation / flip sequence -- and therefore the TARGET -- identical
        # between a photo_aug=0 and a photo_aug=1 item drawn from the same seed. That is what the
        # effect test in tests/test_photo_aug.py checks.
        if self.photo_aug > 0:
            im = photometric_augment(im, rng, self.photo_aug)
        t = torch.from_numpy(im)[None].repeat(3, 1, 1)
        if nb is not None:
            def _same(a):
                # ARM 65: its own percentile stretch, as the wrapper does per frame; t's polarity, crop, rotation and flips
                l0, h0 = np.percentile(a, [1, 99])
                a = np.clip((a - l0) / (h0 - l0 + 1e-6), 0, 1).astype(np.float32)
                if _inv:
                    a = 1.0 - a
                a = a[y:y + c, x:x + c]
                if _k:
                    a = np.rot90(a, _k)
                if _flr:
                    a = a[:, ::-1]
                if _fud:
                    a = a[::-1]
                return np.ascontiguousarray(a)
            t = torch.from_numpy(np.stack([_same(nb[0]), np.ascontiguousarray(im), _same(nb[1])]))
        t = (t - IMA_M) / IMA_S
        if self.curve_targets:
            from instance.curveq import resample_norm, transform_curves
            cur = transform_curves([np.asarray(z["centerline"], float) for z in inst],
                                   y, x, c, k=_k, flip_lr=bool(_flr), flip_ud=bool(_fud))
            return (t, torch.from_numpy(np.ascontiguousarray(gt)), resample_norm(cur, c))
        return t, torch.from_numpy(np.ascontiguousarray(gt))


def collate_curveq(batch):
    """Keep the per-image curve targets RAGGED; stack only the image and the dense target.

    Every image has a different number of microtubules, so the default collate would try to stack
    (T, N_SAMP, 2) tensors of different T and fail. Padding to a fixed T and masking would work too
    but would put fake curves in front of the Hungarian matcher, which is the kind of quiet defect
    this project keeps paying for; a list is unambiguous.
    """
    ims = torch.stack([b[0] for b in batch])
    gts = torch.stack([b[1] for b in batch])
    return ims, gts, [b[2] for b in batch]


class RealDS(torch.utils.data.Dataset):
    """Crops of REAL annotated frames, with the identical target, augmentation and output contract
    as :class:`OnlineDS`.

    Why this exists. Every foreground this project has trained was fed SYNTHETIC frames, because
    until the ROI-303 block was built (2026-08-28) there was not enough clean human annotation to
    train on -- and the synthetic-to-real gap is the documented wall. 14 123 human-drawn curves over
    281 frames changes that premise. The architecture is unchanged (the same nnU-Net ResEnc-M the
    synth arms use); only the data source differs, so an A/B against them measures the data and
    nothing else.

    Preprocessing is ``zoom(norm01(img), 1.5)`` -- normalise THEN upscale, which is the order
    eval_v5 uses. The reverse order (upscale then stretch) gives a slightly different histogram,
    and a training distribution that does not match evaluation is a silent handicap.

    Crops are rejected and redrawn while they contain no annotation at all. Real frames are up to
    99.7 % background, so uniform crops would train almost entirely on empty coverslip; the
    synthetic generator never had that problem because it always draws filaments into the field.
    The rejection is capped so a genuinely sparse frame still contributes its empty regions.
    """

    def __init__(self, frames, crop=518, epoch_len=5000, rot=True, target_half_width=1.0,
                 min_fg_tries=8):
        self.frames = frames                  # list of (image [H,W] float32, polylines at 1.5x)
        self.crop, self.epoch_len, self.rot = crop, epoch_len, rot
        self.target_half_width = target_half_width
        self.min_fg_tries = min_fg_tries

    def __len__(self):
        return self.epoch_len

    @staticmethod
    def _clip(polys, y0, x0, c):
        """Polylines restricted to a crop, SPLIT at the boundary rather than clamped.

        ori_channels_soft clips indices into the array, so a curve leaving the crop would be
        smeared along its edge and taught as foreground that is not there."""
        out = []
        for q in polys:
            d = np.asarray(q, float) - np.array([x0, y0])
            inside = (d[:, 0] >= 0) & (d[:, 0] < c) & (d[:, 1] >= 0) & (d[:, 1] < c)
            if not inside.any():
                continue
            # contiguous runs of inside points, so a curve that re-enters gives two pieces
            idx = np.flatnonzero(inside)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
                if len(run) >= 2:
                    out.append(d[run])
        return out

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        import random
        rng = np.random.default_rng((((wid + 1) * 1000003) * (i + 1)) ^ random.getrandbits(48))
        c = self.crop
        im_full, polys = self.frames[int(rng.integers(len(self.frames)))]
        H, W = im_full.shape
        if H < c or W < c:
            pad = ((0, max(0, c - H)), (0, max(0, c - W)))
            im_full = np.pad(im_full, pad, mode="reflect")
            H, W = im_full.shape
        sub = []
        for _ in range(self.min_fg_tries):
            y = int(rng.integers(0, H - c + 1))
            x = int(rng.integers(0, W - c + 1))
            sub = self._clip(polys, y, x, c)
            if sub:
                break
        im = im_full[y:y + c, x:x + c].astype(np.float32)
        gt = ori_channels_soft([{"centerline": q} for q in sub], (c, c),
                               half_width=self.target_half_width)

        if rng.random() < .5:
            im = 1.0 - im                                     # IRM polarity is ambiguous
        if self.rot:
            k = int(rng.integers(4))
            if k:
                im = np.rot90(im, k).copy()
                gt = np.rot90(gt, k, axes=(1, 2)).copy()
                if k % 2 == 1:
                    gt[:K_ORI] = np.roll(gt[:K_ORI], K_ORI // 2, axis=0)
        if rng.random() < .5:
            im = im[:, ::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, :, ::-1].copy()
        if rng.random() < .5:
            im = im[::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, ::-1, :].copy()
        t = torch.from_numpy(np.ascontiguousarray(im))[None].repeat(3, 1, 1)
        t = (t - IMA_M) / IMA_S
        return t, torch.from_numpy(np.ascontiguousarray(gt))


def load_real_frames(data_dir, split):
    """Real annotated frames in the EVAL frame: normalise, then upscale, exactly as eval_v5 does."""
    import glob as _glob
    from mt_bench.cvat_import import read_frame_h5
    out = []
    for p in sorted(_glob.glob(os.path.join(data_dir, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) != split:
            continue
        polys = [np.asarray(q, float) * UP for q in fr["polylines"] if len(q) >= 2]
        if not polys:
            continue
        im = zoom(norm01(np.asarray(fr["image"], float)), UP, order=1).astype(np.float32)
        out.append((im, polys))
    return out


class _TwoOpt:
    """Two optimisers behind one interface, so the scheduler and the loop stay unchanged.

    `param_groups` concatenates both, which is what `LambdaLR` iterates over; each group keeps its
    own `initial_lr`, so the poly/cosine factor multiplies the right base learning rate for the
    trunk and for the head.
    """

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)


def _cq_epoch_line() -> str:
    """Per-epoch curve-loss components, and the ONE number that says whether the queries live.

    The first CURVEQ run printed only a total loss, and a total loss falling from 10.76 to 9.48
    looked healthy while every query had in fact collapsed to the same output. Diagnosing that cost
    a probe and 3 GPU-hours. `n_pos` (matched queries per step) and the geometry term make the
    collapse visible from the log alone.
    """
    r = globals().get("_CQ_RUN")
    if not r or not r.get("n"):
        return ""
    n = r["n"]
    out = (f"  [cq geo={r.get('geo', 0) / n:.4f} obj={r.get('obj', 0) / n:.4f} "
           f"mask={r.get('mask', 0) / n:.4f} matched/step={r.get('n_pos', 0) / n:.1f}]")
    globals()["_CQ_RUN"] = {}
    return out


class _TwoSched:
    """One scheduler per optimiser, stepped together.

    `LambdaLR` type-checks its argument, so a two-optimiser wrapper cannot be handed to it
    directly. Giving each real optimiser its own `LambdaLR` is both simpler and more correct: each
    keeps its own `initial_lr`, so the same schedule factor multiplies 1e-2 for the trunk and 1e-4
    for the query head rather than one base leaking into the other.
    """

    def __init__(self, opt, fn):
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, fn) for o in opt.opts]

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


# --------------------------------------------------------------------------- losses
def dice(p, t, e=1.0):
    p = torch.sigmoid(p)
    i = (p * t).sum((2, 3))
    return (1 - (2 * i + e) / (p.sum((2, 3)) + t.sum((2, 3)) + e)).mean()


def minpool_term(logits, cid, temp: float = 1.0, min_px: int = 8):
    """BCE on the WORST point of each ground-truth curve, not on the average pixel.

    Measured 2026-09-11 with `scripts/diag_model_only.py`, no instancer in the loop: the median
    per-curve MEAN predicted probability is 0.95-0.99 on every block, in domain and on real, while
    the median per-curve MINIMUM is 0.94 in domain and 0.51-0.87 on real data, falling with curve
    length. The strict metric needs 95 % (or 80 %) of a curve's LENGTH above threshold, so what
    decides a true positive is the weakest point on the filament -- and the per-pixel mean of
    BCE + Dice is almost indifferent to it: one dip in a 300 px curve is 3 % of that curve's
    pixels and a millionth of the frame.

    This term gives every curve equal weight and reads only its weak end of the distribution:

        m_c = sum_i w_i * l_i ,  w = softmax(-l_i / temp)  over the pixels of curve c
        L   = mean_c BCE_with_logits(m_c, 1)

    `temp` is in LOGIT units. As temp -> 0 this is the hard minimum (gradient to a single pixel
    per curve); at temp = 1 the weight halves for every logit 1 above the curve's weakest, so a
    handful of pixels carry the term while the gradient stays smooth. Curves with fewer than
    `min_px` pixels in the crop are skipped: they are boundary fragments, not filaments.

    Applied at FULL resolution only. The deep-supervision heads see max-pooled targets, where an
    id map would have to be downsampled by an arbitrary rule and a 2 px dip disappears anyway.
    """
    B, _, H, W = logits.shape
    l = logits[:, 0].reshape(B, -1).float()
    c = cid[:, 0].reshape(B, -1).long()
    n = int(c.max().item())
    if n <= 0:
        return logits.sum() * 0.0                      # no curve in this batch: exact zero
    g = n + 1                                          # slot 0 of each sample is background
    idx = (c + torch.arange(B, device=l.device)[:, None] * g).reshape(-1)
    lf = l.reshape(-1)
    z = -lf / max(temp, 1e-3)

    def seg(src, reduce, init):
        o = torch.full((B * g,), init, device=l.device, dtype=torch.float32)
        return o.scatter_reduce(0, idx, src, reduce=reduce, include_self=True)

    zmax = seg(z, "amax", -1e30)                        # stabilise the softmax per curve
    e = torch.exp(z - zmax[idx])
    den = seg(e, "sum", 0.0)
    num = seg(e * lf, "sum", 0.0)
    cnt = seg(torch.ones_like(lf), "sum", 0.0)

    keep = cnt >= float(min_px)
    keep = keep & (torch.arange(B * g, device=l.device) % g != 0)   # drop the background slot
    if not bool(keep.any()):
        return logits.sum() * 0.0
    m = num[keep] / den[keep].clamp_min(1e-20)
    return F.binary_cross_entropy_with_logits(m, torch.ones_like(m))


def crossing_term(logits, ori_tgt, count):
    """Penalise collapsing a crossing to a single orientation.

    Measured: where two ground-truth filaments overlap in different bins, BOTH channels exceed
    threshold only 14.3 % of the time and the weaker bin averages p=0.145 against 0.718. The
    per-channel BCE is dominated by single-filament pixels and is nearly indifferent to the
    weaker orientation, so nothing in the previous objective punished the collapse that the
    K channels exist to prevent.

    ``count`` is the per-pixel number of ground-truth instances, which is why it is carried
    through the target tensor: with soft two-bin assignment the number of ACTIVE channels no
    longer identifies an amodal pixel.
    """
    with torch.no_grad():
        amodal = count >= 2                                    # (B,H,W)
        active = ori_tgt > 0.2
    n = amodal.sum()
    if n == 0:
        return logits.sum() * 0.0
    p = torch.sigmoid(logits[:, :K_ORI]).clamp(1e-6, 1 - 1e-6)
    weak = torch.where(active, p, torch.ones_like(p)).amin(1)  # weakest supervised orientation
    return (-torch.log(weak) * amodal).sum() / n


# --------------------------------------------------------------------------- validation
def scramble_overlap(ov):
    """CONTROL: destroy the multiplicity target's registration with the image, keep its statistics.

    An extra output head is also plain multi-task regularisation, so a map-side effect from the
    OVERLAP arm does not by itself show that MULTIPLICITY supervision is what helped. Rolling the
    target by a large random offset per sample preserves its density, its morphology and its
    positive rate exactly, and destroys only the correspondence with the image. If the effect
    survives this, it was regularisation.
    """
    h, w = ov.shape[-2], ov.shape[-1]
    sh = (int(torch.randint(64, max(65, h - 64), (1,)).item()),
          int(torch.randint(64, max(65, w - 64), (1,)).item()))
    return torch.roll(ov, shifts=sh, dims=(-2, -1))


def junction_weight(cnt: "torch.Tensor", radius: float, w_j: float) -> "torch.Tensor":
    """Per-pixel loss weight: ``w_j`` inside ``radius`` of a crossing, 1 elsewhere.

    The crossing seed is the generator's own multiplicity channel (``cnt >= 2``): at
    ``half_width=1.0`` a pixel is covered by two centerline rasters only if those centerlines
    pass within ~2 px of each other, which is the research note's "genuinely cross or coincide"
    condition (docs/research/2026-09-04-junctions-and-crossings.md §2.1) — two parallels 3+ px
    apart stay class 1 on each ridge. Dilation is a square max-pool rather than a disc: it
    over-covers by ~27 % in area, which is inside the coverage budget measured before the arm
    (13 % of foreground at r=16 on dense synthetic frames, ceiling 40 %).

    ``radius`` is in pixels OF THE TENSOR PASSED IN, so a deep-supervision level must pass its
    own down-scaled radius.
    """
    j = (cnt >= 2).float()
    r = int(round(radius))
    if r > 0:
        j = F.max_pool2d(j, kernel_size=2 * r + 1, stride=1, padding=r)
    return 1.0 + (float(w_j) - 1.0) * j


def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


#: Tile size per architecture. The DINO trunk wants 518 (its /14 patch grid); nnU-Net's ResEnc
#: has seven /2 stages, so its input must be divisible by 128 or the residual adds hit a
#: shape mismatch on the way back up. This is a property of the network, not a global constant.
TILE_FOR = {"dino": 518, "resunet": 518, "nnunet": 512}


class PriorBank:
    """Frozen classical filter bank at NATIVE resolution, concatenated to the network's input.

    A deep encoder discards structure narrower than its stride, and a microtubule is 3-6 px wide in
    the 1.5x eval frame. The network currently has to rediscover a ridge detector from raw intensity
    at every scale; a Frangi/LoG/structure-tensor bank hands it one for free, computed at full
    resolution, and it is exactly the operator a filament calls for. Borrowed from AutoSeg's
    ``hyperbank_bank`` -- pure torch, so it runs on the GPU inside the training step rather than in
    the dataloader, and it is FROZEN (no gradient, fixed scales).

    SCALES ARE ABSOLUTE PIXELS AND MUST STAY ABOVE 1.0. The bank reads sigma <= 1.0 as a FRACTION of
    min(H, W), which for a 1536 px frame means a 1536-px kernel rather than a 1-px one. The finest
    absolute Frangi scale usable here is therefore just over 1, and 1.5-6 px brackets a microtubule.
    """

    def __init__(self, device, autoseg="/disk1/prusek/active-segmenter"):
        if autoseg not in sys.path:
            sys.path.insert(0, autoseg)
        from active_segmenter.segment.hyperbank_bank import HyperBank
        sig = (1.5, 2.0, 3.0, 4.0, 6.0)
        self.m = HyperBank(frangi_sigmas=sig, sauvola_windows=(7, 15, 51),
                           struct_sigmas=(2.0, 8.0), gradmag_sigmas=(1.5, 3.0, 6.0),
                           use_log=True, log_sigmas=sig).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        # `feature_maps`, NOT `forward`. HyperBank is the whole segmenter from AutoSeg's earlier
        # work -- its forward runs a 1x1 head and returns ONE logit channel. Calling it would have
        # handed the network a single pre-mixed score instead of the filter bank, and the run would
        # have looked fine: the smoke test printed "1 native-resolution channels" and nothing else
        # complained.
        with torch.no_grad():
            self.n_ch = int(self.m.feature_maps(
                torch.rand(1, 1, 64, 64, device=device)).shape[1])
        if self.n_ch < 8:
            raise SystemExit(f"the prior bank produced only {self.n_ch} channels; expected a few "
                             f"dozen. Refusing to train on a bank that is not a bank.")

    @torch.no_grad()
    def __call__(self, gray):
        """gray: [B,1,H,W] in [0,1] -> [B,C,H,W] priors at native resolution."""
        return self.m.feature_maps(gray).float()


def with_priors(t, bank):
    """Prepend the classical priors to an ImageNet-normalised 3-channel batch.

    The grayscale is recovered exactly from channel 0 rather than passed separately, so the bank
    always sees the same pixels the network does -- a second path could drift out of step."""
    if bank is None:
        return t
    gray = (t[:, :1] * IMA_S[0].to(t.device) + IMA_M[0].to(t.device)).clamp(0, 1)
    return torch.cat([t, bank(gray).to(t.dtype)], 1)


@torch.no_grad()
def predict(model, img01, tile: int = TILE, stride: int | None = None, bank=None, fusion: str = "uniform"):
    """Tiled inference. ``fusion="uniform"`` is the historical path (probability mean, stride
    0.757 tile) and stays bit-identical. ``fusion="gauss"`` is the seam fix measured as G256_a on
    2026-10-05: Gaussian-weighted LOGIT mean at stride tile/2 -- the uniform mean lets the
    context-starved tile border pull filaments under the cut within ~8 px of a seam."""
    assert fusion in ("uniform", "gauss"), fusion
    TILE = tile
    gauss = fusion == "gauss"
    STRIDE = stride if stride is not None else (tile // 2 if gauss else int(round(tile * 392 / 518)))
    H, W = img01.shape
    ys = list(range(0, max(1, H - TILE + 1), STRIDE)) or [0]
    xs = list(range(0, max(1, W - TILE + 1), STRIDE)) or [0]
    if ys[-1] != max(0, H - TILE):
        ys.append(max(0, H - TILE))
    if xs[-1] != max(0, W - TILE):
        xs.append(max(0, W - TILE))
    acc = cnt = None
    for y in ys:
        for x in xs:
            t = img01[y:y + TILE, x:x + TILE]
            th, tw = t.shape
            tt = torch.from_numpy(t.astype(np.float32))[None].repeat(3, 1, 1)
            tt = ((tt - IMA_M) / IMA_S)[None].to(DEV)
            tt = with_priors(tt, bank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(tt)
            # Unwrap repeatedly: deep supervision returns a list, and --head curveq returns
            # (dense_list, query_outputs). Taking [0] once would hand a LIST to sigmoid. This
            # path decodes the DENSE head of a curveq model, which is deliberate: it keeps the
            # trainer's existing over-firing gate and every historical comparison readable.
            while isinstance(o, (tuple, list)):
                o = o[0]
            if gauss:
                o = np.clip(o.float()[0].cpu().numpy(), -30.0, 30.0)
                wy = np.arange(th) - (th - 1) / 2
                wx = np.arange(tw) - (tw - 1) / 2
                g = np.exp(-(wy[:, None] ** 2 + wx[None] ** 2) / (2 * (TILE / 8) ** 2))
                g = np.maximum(g, g.max() * 1e-3)
            else:
                o = torch.sigmoid(o.float())[0].cpu().numpy()
                g = 1.0
            if acc is None:
                acc = np.zeros((o.shape[0], H, W))
                cnt = np.zeros((H, W))
            acc[:, y:y + th, x:x + tw] += o[:, :th, :tw] * g
            cnt[y:y + th, x:x + tw] += g
    if gauss:
        return 1 / (1 + np.exp(-acc / cnt[None]))
    return acc / np.maximum(cnt, 1)[None]


def validate(model, frames, params, thr, use_fg_head: bool, tile: int = TILE, bank=None,
             prob_channel: int = -1, up: float = UP):
    """Strict centerline-F1 through the instancer -- the metric the project is judged on.

    ``prob_channel >= 0`` decodes that output channel alone (0 = the existence head of a
    ``--head heat`` model, so its selection gate reads the same quantity as a binary arm's)."""
    model.eval()
    rows, fgq = [], []
    for fr in frames:
        im01 = norm01(np.asarray(fr["image"], float))
        x = im01 if abs(up - 1.0) < 1e-9 else zoom(im01, up, order=1)
        ch = predict(model, x, tile, bank=bank)
        if abs(up - UP) > 1e-9:
            # The METRIC frame is always 1.5x (GT is scaled by UP below), exactly as
            # eval_v5.py does it: only the INFERENCE resolution changes, so a map predicted
            # at another scale is resampled onto the metric frame before it is scored.
            from eval_v5 import eval_frame_shape, resample_to_eval
            ch = resample_to_eval(ch, eval_frame_shape(im01.shape))
        ori = ch[:K_ORI]
        if prob_channel >= 0:
            prob = ch[prob_channel]
        else:
            prob = ch[K_ORI] if (use_fg_head and ch.shape[0] > K_ORI) else ori.max(0)
        mask = prob > thr
        _, masks = instance_a(mask, KAPPA_MAX, params=params, channels=ori, prob=prob)
        gt = [np.asarray(q, float) * UP for q in fr["polylines"]]
        r = centerline_f1(masks, gt, tol=5.0, length_coverage=0.95, precision_coverage=0.95)
        r["fg"] = float(mask.mean())
        rows.append(r)
        fgq.append(foreground_quality(mask, fr["polylines"], up=UP))
    model.train()
    agg = aggregate_benchmark(rows)
    props = mean_properties(fgq)
    agg["fg"] = float(np.mean([r["fg"] for r in rows]))
    agg["cc_per_gt"] = props.get("cc_per_gt", float("nan"))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resunet", choices=["resunet", "dino", "nnunet"])
    ap.add_argument("--crop", type=int, default=0,
                    help="0 = the architecture's own tile size (TILE_FOR)")
    ap.add_argument("--opt", default="auto", choices=["auto", "adamw", "sgd"],
                    help="auto = nnU-Net's SGD recipe for --arch nnunet, AdamW otherwise")
    ap.add_argument("--deep-sup", type=int, default=4,
                    help="decoder outputs supervised (nnunet only); deeper ones are below the "
                         "resolution at which a 2 px filament still exists")
    ap.add_argument("--calib", default="/home/prusek/mt_enc_exp/calib_reg418_morph_snr3.json")
    ap.add_argument("--overlap-scramble", action="store_true",
                    help="CONTROL: roll the multiplicity target by a random offset per sample, so "
                         "the head gets an auxiliary task with identical statistics and no "
                         "information. Default off, so every previous run is bit-identical.")
    ap.add_argument("--overlap-weight", type=float, default=1.0,
                    help="weight of the overlap head's BCE (--head overlap)")
    ap.add_argument("--overlap-pos-weight", type=float, default=40.0,
                    help="pos_weight for the overlap logit. Overlap pixels are ~0.1 % of a frame "
                         "(measured: 76 crossing-seed px against 3530 foreground px on a live "
                         "generator sample), so an unweighted BCE is minimised by predicting zero.")
    ap.add_argument("--jw-radius", type=float, default=0.0,
                    help="CROSSING-REGION SUPERVISION (research note 2026-09-04 rec. #1): "
                         "radius in training px around crossing pixels (cnt>=2) inside which the "
                         "BCE is weighted by --jw-weight. 0 = off (bit-identical to before). "
                         "Set from the MEASURED failure distance: the weak arm sits 8-20 EVAL px "
                         "from the junction = 5-13 native px, so 14 covers it with margin.")
    ap.add_argument("--jw-weight", type=float, default=6.0,
                    help="weight inside the junction region (research note: w_J = 6)")
    ap.add_argument("--cq-queries", type=int, default=300,
                    help="--head curveq: number of curve queries. 300 against ~25 curves per "
                         "synthetic frame; the one-to-many auxiliary supervision is what keeps "
                         "the rest from starving.")
    ap.add_argument("--cq-dim", type=int, default=128,
                    help="--head curveq: decoder width. MEASURED on tulen at crop 512, batch 8, "
                         "300 queries, 6 layers: dim 256 needs 24.9 GB and 2.03 s/step, dim 128 "
                         "needs 15.6 GB and 1.45 s/step. 128 is the default because it fits BOTH "
                         "GPUs and costs ~2x the dense baseline's 0.72 s/step rather than ~2.8x.")
    ap.add_argument("--cq-layers", type=int, default=6)
    ap.add_argument("--cq-dense-weight", type=float, default=1.0,
                    help="weight of the retained DENSE binary loss. It is the auxiliary task that "
                         "gives the encoder gradient from step 1 while the queries are still "
                         "finding the filaments, and it keeps the trainer's existing over-firing "
                         "gate readable. 0 disables it and is NOT recommended.")
    ap.add_argument("--cq-geo-weight", type=float, default=5.0)
    ap.add_argument("--cq-mask-weight", type=float, default=2.0)
    ap.add_argument("--cq-head-lr", type=float, default=1e-4,
                    help="AdamW learning rate for the QUERY HEAD only; the trunk keeps nnU-Net's "
                         "SGD. Measured: the shared SGD(momentum 0.99, lr 1e-2) drives objectness "
                         "std across queries to 0.00000 -- total collapse -- while AdamW 1e-4 "
                         "gives 0.107 on the identical data and seed.")
    ap.add_argument("--cq-aux-k", type=int, default=3,
                    help="one-to-many supervision: this many runner-up queries per target also get "
                         "mask gradient, and are IGNORED in the objectness loss (a soft target "
                         "there adds an irreducible entropy floor -- found by the overfit test).")
    ap.add_argument("--minpool-weight", type=float, default=0.0,
                    help="WORST-POINT TERM (measured 2026-09-11, scripts/diag_model_only.py): "
                         "weight of a BCE on the soft-MINIMUM predicted logit along each GT "
                         "curve, added to the per-pixel BCE+Dice. The per-curve MEAN probability "
                         "is 0.95-0.99 in every domain while the per-curve MINIMUM is 0.94 in "
                         "domain and 0.51-0.87 on real data -- and the metric is decided by the "
                         "minimum, because it needs 80-95 % of a curve's LENGTH above threshold. "
                         "0 = off and bit-identical to every previous run (the id channel is not "
                         "even built).")
    ap.add_argument("--minpool-temp", type=float, default=1.0,
                    help="softmin temperature in LOGIT units; -> 0 is the hard minimum")
    ap.add_argument("--minpool-min-px", type=int, default=8,
                    help="curves with fewer target pixels inside the crop are skipped (boundary "
                         "fragments, not filaments)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch-len", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=8.0,
                    help="positive class weight in the BCE term. 8.0 is v4b's inherited "
                         "value and the default, so an unflagged run is bit-identical to "
                         "every previous one; it has never been tuned.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--val-data", default="data/real/mt34_eval")
    ap.add_argument("--val-n", type=int, default=8, help="VAL frames scored per validation")
    ap.add_argument("--params", default="src/instance/params_a_model_synthtuned.json")
    ap.add_argument("--fg-head", action="store_true", help="decode from the binary union head")
    ap.add_argument("--heat-sigma", type=float, default=1.2,
                    help="sigma (px, training scale) of the Gaussian centerline heatmap regressed "
                         "by --head heat")
    ap.add_argument("--heat-weight", type=float, default=2.0,
                    help="weight of the heatmap L1 term relative to BCE+Dice of the existence head")
    ap.add_argument("--head", default="amodal",
                    choices=["amodal", "binary", "heat", "overlap", "curveq"],
                    help="heat = binary existence head PLUS a Gaussian centerline heatmap head "
                         "(2 channels; entries 19/20 of the campaign). "
                         "overlap = binary existence PLUS a second logit for 'two or more "
                         "instances cover this pixel' (the research note's multiplicity head, "
                         "campaign arm 34). Justified by the 2026-09-08 in-domain measurement: "
                         "59.5 % of failing crossings have a CONNECTED mask with both arms "
                         "present and no dip, and 57 % of them survive a perfect mask, so the "
                         "decoder is missing INFORMATION, not confidence. "
                         "binary = union foreground ONLY, no orientation channels. This is the "
                         "arm that separates 'the amodal head costs F1' from 'my training of it "
                         "is weaker than nnU-Net's own framework' -- the two are confounded in "
                         "every comparison against the framework-trained binary run.")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--synth-fg", type=float, default=0.016)
    ap.add_argument("--fov-prob", type=float, default=0.0,
                    help="fraction of training frames given a synthetic octagonal field stop. "
                         "Real task-586 frames have one and the segmenter fires on its rim; "
                         "the background pool has none, so without this the model never sees a "
                         "field boundary that is not a filament.")
    ap.add_argument("--temporal-p-single", type=float, default=-1.0,
                    help="ARM 65: >= 0 renders (t-1, t, t+1) items through synth/mt_clip.py, this share as (t, t, t); frame t and its target stay byte-identical to the still recipe. Off by default.")
    ap.add_argument("--tirf-prob", type=float, default=0.0,
                    help="fraction of training frames rendered as TIRF (bright filaments on a dark "
                         "field, additive, no interference halo) instead of IRM; 0.0 = IRM only")
    ap.add_argument("--priors", action="store_true",
                    help="prepend a frozen classical filter bank (Frangi / LoG / structure tensor "
                         "/ Sauvola / gradient, all at NATIVE resolution) to the network's input. A "
                         "deep encoder discards structure narrower than its stride and a "
                         "microtubule is 3-6 px wide, so this hands it the ridge detector it would "
                         "otherwise have to rediscover from raw intensity.")
    ap.add_argument("--autoseg", default="/disk1/prusek/active-segmenter",
                    help="where the classical bank is imported from")
    ap.add_argument("--init-from", default="",
                    help="initialise from this checkpoint instead of scratch. The point is the "
                         "synthetic-then-real recipe: MEASURED 2026-08-31, a model trained on 141 "
                         "real frames from ONE lab scores 0.357 across other labs against 0.433 for "
                         "the domain-randomised synthetic arm, so real data alone generalises WORSE "
                         "-- but it is the only data that shows what a real microtubule looks like. "
                         "Pre-train wide on synthetic, then adapt on real.")
    ap.add_argument("--real-data", default="",
                    help="train on REAL annotated frames from this directory instead of synthetic "
                         "ones. Everything else -- architecture, loss, optimiser, schedule, "
                         "checkpoint gate -- is unchanged, so an A/B against a synthetic arm "
                         "measures the DATA and nothing else.")
    ap.add_argument("--real-split", default="val",
                    help="which split of --real-data to TRAIN on. Never 'test'.")
    ap.add_argument("--rot-aug-prob", type=float, default=0.0,
                    help="ARM 54 (2026-09-17): probability of rotating a synthetic training frame (image bilinear, "
                         "targets nearest) by a uniform angle in [0, 90) deg before the crop and the rot90/flip "
                         "group. Binary head only (orientation channels are not re-binned). 0 = off, bit-identical")
    ap.add_argument("--target-band", default="square", choices=["square", "euclid"],
                    help="ARM 53 (2026-09-16): shape of the 3 px training target band. square = the inherited 3x3 "
                         "dilation (pixel count per unit length 3.0 at 0/90 deg vs ~3.5-4.0 at 30-60 deg, invariant "
                         "under rot90/flip); euclid = exact 1.5 px Euclidean band, orientation-isotropic")
    ap.add_argument("--paint-aa", action="store_true",
                    help="ARM 52 (2026-09-16): render filaments with the orientation-isotropic stamp "
                         "(GenConfig.paint_aa) instead of the rounded 0.5-px offset stamp, whose peak "
                         "depends on the filament's angle to the pixel grid by 20-40 %%; the stamp is "
                         "renormalised so the orientation-averaged peak equals the shipped one")
    ap.add_argument("--tirf-label-sparse", action="store_true",
                    help="render TIRF filaments as a chain of DISCRETE fluorophore puncta rather "
                         "than a continuous line -- what a real TIRF microtubule looks like at a "
                         "low labelling ratio. The GT mask stays continuous, so the model is "
                         "taught to bridge the dark gaps instead of ending an instance at one. "
                         "Off by default: every IRM number measured so far used the continuous "
                         "renderer.")
    ap.add_argument("--photo-aug", type=float, default=0.0,
                    help="strength of the photometric nuisance augmentation (synth/photo_aug.py): "
                         "0 = off and bit-identical to every earlier run; 1.0 = the declared "
                         "recipe (gamma, contrast/brightness, additive noise, blur, low-frequency "
                         "illumination, clipped stretch + 16-256 level quantisation), each applied "
                         "with its own probability and followed by the evaluation's [1,99] stretch.")
    ap.add_argument("--scale-aug", default="",
                    help="'lo,hi': rescale the RENDERED frame and its polylines by "
                         "s ~ LogUniform(lo, hi) before the target is rasterised (target width "
                         "stays 3 px in output pixels; the filament's apparent width scales). "
                         "Empty = off and bit-identical. Declared arm SCALE uses 0.8,1.5.")
    ap.add_argument("--scale-prob", type=float, default=0.7,
                    help="fraction of frames that get a scale draw (else s = 1)")
    ap.add_argument("--lowres-prob", type=float, default=0.0,
                    help="nnU-Net-style low-resolution simulation on the image only: nearest "
                         "downsample by U(0.6, 1), cubic upsample back. 0 = off.")
    ap.add_argument("--target-half-width", type=float, default=1.0,
                    help="half-width the TRAINING TARGET is rasterised at. 1.0 is the inherited "
                         "value and the default, so an unflagged run is unchanged. Measured "
                         "2026-08-26 (scripts/parallel_screen.py, oracle masks, real VAL): a "
                         "1 px target lifts bundle recovery 0.820 -> 0.950, junction identity "
                         "0.919 -> 0.964 and F1 0.876 -> 0.906, because a thick target spends the "
                         "gap between two close parallels and skeletonising a fused band yields "
                         "ONE centerline instead of two.")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls weight init and augmentation draws. Repeat runs differing "
                         "ONLY in this to get the run-to-run variance every claim needs: the "
                         "project records ~0.02 training noise and no arm here has ever been "
                         "repeated, so single-run differences below that are uninterpretable.")
    args = ap.parse_args()
    if not args.crop:
        args.crop = TILE_FOR[args.arch]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _random
    _random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    sys.path.insert(0, "/home/prusek/mt_enc_exp/synth")
    from gen_train import build_cfg
    from mt_generator import generate_frame
    cfg = build_cfg(json.load(open(args.calib))["best_params"], mask_hw=1.0)
    # BIOCEV lives on tulen's disk only; kajman gets the copy on the shared home. Same list and
    # same order as train_temporal.py, so both trainers draw from the identical pool.
    bgs = []
    # MT_BG_DIR (2026-10-28): a host-local copy of the same 428 frames for hosts other than tulen,
    # where the BIOCEV path is a symlink into tulen's /disk2. Unset = the old search, unchanged.
    for d in tuple(filter(None, (os.environ.get("MT_BG_DIR"),))) + (
              "/home/prusek/mt_enc_exp/irm_backgrounds_v2",
              "/home/prusek/BIOCEV/datasets/microtubules/IRM_backgrounds_v2"):
        bgs = sorted(glob.glob(os.path.join(d, "*.tif")))
        if bgs:
            break
    if not bgs:
        raise SystemExit("no IRM backgrounds found")
    print(f"backgrounds: {len(bgs)} from {os.path.dirname(bgs[0])}", flush=True)
    # Optional GenConfig overrides carried in the calib file, so a morphology correction is a
    # DATA artefact with provenance rather than a flag someone has to remember to pass.
    _cal = json.load(open(args.calib))
    if _cal.get("gen_overrides"):
        cfg = dataclasses.replace(cfg, **_cal["gen_overrides"])
        print(f"gen_overrides: {_cal['gen_overrides']}", flush=True)
    if args.photo_aug > 0:
        print(f"photo_aug={args.photo_aug:g}  (synth/photo_aug.py recipe, applied after every "
              f"geometric draw; the target is untouched)", flush=True)
    if args.head == "heat":
        print(f"head=heat: existence (BCE pos_weight {args.pos_weight:g} + Dice) + Gaussian "
              f"centerline heatmap sigma={args.heat_sigma:g} px, weighted L1 x {args.heat_weight:g}; "
              f"selection gate decodes channel 0", flush=True)
    scale_aug = tuple(float(v) for v in args.scale_aug.split(",")) if args.scale_aug else None
    if scale_aug:
        print(f"scale_aug=LogUniform{scale_aug} p={args.scale_prob:g}  lowres_prob={args.lowres_prob:g}"
              f"  (rendered frame + polylines rescaled BEFORE the target is rasterised)", flush=True)
    if args.rot_aug_prob > 0:
        if args.head != "binary":
            raise SystemExit("--rot-aug-prob is defined for --head binary only (orientation channels are not re-binned)")
        print(f"rot_aug_prob={args.rot_aug_prob}  (arbitrary-angle rotation of image + target, U(0, 90) deg)")
    if args.target_band != "square":
        global TARGET_BAND
        TARGET_BAND = args.target_band
        print(f"target_band={TARGET_BAND}  (radius {TARGET_BAND_RADIUS} eval px, orientation-isotropic band)")
    if args.paint_aa:
        cfg = dataclasses.replace(cfg, paint_aa=True)
        print(f"paint_aa=True  (orientation-isotropic filament stamp, gain {cfg.paint_aa_gain})")
    if args.tirf_label_sparse:
        cfg = dataclasses.replace(cfg, tirf_label_sparse=True)
        print(f"tirf_label_sparse=True  (density prior 10^{cfg.tirf_label_density_log10} "
              f"fluorophores/px, amplitude CV {cfg.tirf_label_amp_cv})", flush=True)
    if args.real_data:
        if args.real_split == "test":
            raise SystemExit("refusing to train on a TEST split")
        frames = load_real_frames(args.real_data, args.real_split)
        if not frames:
            raise SystemExit(f"no {args.real_split} frames with annotation under {args.real_data}")
        n_curves = sum(len(p) for _, p in frames)
        print(f"REAL training data: {len(frames)} frames, {n_curves} annotated curves "
              f"from {args.real_data} [{args.real_split}]", flush=True)
        print("  the synthetic generator is NOT used in this arm", flush=True)
        ds = RealDS(frames, crop=args.crop, epoch_len=args.epoch_len,
                    target_half_width=args.target_half_width)
    else:
        gen_fn = generate_frame
        if args.temporal_p_single >= 0:
            if scale_aug or args.lowres_prob or args.fov_prob or args.rot_aug_prob or args.photo_aug:
                raise SystemExit("--temporal-p-single does not combine with scale, lowres, field stop, arbitrary rotation or photo augmentation")
            import functools
            from mt_clip import clip_triplet
            gen_fn = functools.partial(clip_triplet, p_single=args.temporal_p_single)
            print(f"ARM 65: temporal input, p_single={args.temporal_p_single}", flush=True)
        ds = OnlineDS(cfg, bgs, gen_fn, crop=args.crop, epoch_len=args.epoch_len,
                      target_half_width=args.target_half_width,
                      fov_prob=args.fov_prob, tirf_prob=args.tirf_prob,
                      photo_aug=args.photo_aug, scale_aug=scale_aug,
                      scale_prob=args.scale_prob, lowres_prob=args.lowres_prob,
                      heat_sigma=(args.heat_sigma if args.head == "heat" else 0.0),
                      curve_ids=args.minpool_weight > 0,
                      curve_targets=args.head == "curveq",
                      rot_aug_prob=args.rot_aug_prob)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers, drop_last=True,
        persistent_workers=True, prefetch_factor=3,
        collate_fn=collate_curveq if args.head == "curveq" else None)

    frames = []
    for p in sorted(glob.glob(os.path.join(args.val_data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == "val" and fr["polylines"]:
            frames.append(fr)
    frames = frames[:args.val_n]
    params = json.load(open(args.params))
    thr = params.get("prob_thr", 0.35)
    print(f"arch={args.arch} calib={os.path.basename(args.calib)} "
          f"VAL={len(frames)} frames (selection only; training data is {'REAL ' + args.real_data if args.real_data else 'synthetic'})", flush=True)

    # "curveq" belongs in this map: its dense auxiliary head is BINARY, so only channel 0 is ever
    # supervised (see `o_ex = ...[:, :1]` in both loss branches). Before this line was fixed it fell
    # through to K_ORI + 1 and the trunk carried SIX DEAD OUTPUT CHANNELS -- no gradient, no effect
    # on the decoder feature the query head taps (that is the INPUT to this conv), but six unused
    # rows in a 1x1 convolution and a checkpoint whose shape nothing else expects. The CQ arm
    # launched 2026-09-13 13:38 is running with the old 7-channel trunk; it is functionally
    # identical for that reason, and `scripts/eval_curveq.py` reads the width from the checkpoint
    # rather than assuming it, so both shapes score.
    out_ch = {"binary": 1, "heat": 2, "overlap": 2, "curveq": 1}.get(args.head, K_ORI + 1)
    ov_w = torch.tensor(float(args.overlap_pos_weight), device=DEV)
    bank = None
    if args.priors:
        bank = PriorBank(DEV, autoseg=args.autoseg)
        print(f"classical prior bank: {bank.n_ch} native-resolution channels "
              f"(frozen) prepended to the input -> {3 + bank.n_ch} input channels", flush=True)
    model = build_model(args.arch, out_ch=out_ch, in_ch=3 + (bank.n_ch if bank else 0))
    if args.head == "curveq":
        if args.arch != "nnunet":
            raise SystemExit("--head curveq taps decoder.seg_layers, which only the nnunet trunk "
                             "has; refusing to guess the feature tap for another architecture.")
        from instance.curveq import CurveQNet
        model = CurveQNet(model, CurveQNet.feat_dim_of(model), n_queries=args.cq_queries,
                          dim=args.cq_dim, n_layers=args.cq_layers)
        print(f"head=curveq: {args.cq_queries} queries, dim {args.cq_dim}, {args.cq_layers} "
              f"layers, overlap-permitting per-query masks (no softmax over queries); the dense "
              f"binary head is KEPT as the auxiliary task and the honest comparator", flush=True)
    model = model.to(DEV)
    if args.init_from:
        _sd = torch.load(args.init_from, map_location=DEV)
        _missing = model.load_state_dict(_sd, strict=False)
        print(f"initialised from {args.init_from}", flush=True)
        if getattr(_missing, "missing_keys", None):
            print(f"  WARNING: {len(_missing.missing_keys)} tensors were NOT loaded and stay "
                  f"random -- e.g. {_missing.missing_keys[:3]}", flush=True)
        if getattr(_missing, "unexpected_keys", None):
            print(f"  {len(_missing.unexpected_keys)} tensors in the checkpoint were unused",
                  flush=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    train_p = [p for p in model.parameters() if p.requires_grad]
    steps = args.epochs * (args.epoch_len // args.batch)
    warm = max(1, steps // 50)
    use_sgd = args.opt == "sgd" or (args.opt == "auto" and args.arch == "nnunet")
    if use_sgd:
        # nnU-Net's optimiser, unchanged: SGD with very high momentum and a poly schedule.
        # It is not interchangeable with AdamW here -- the ResEnc encoder is 8 stages deep and
        # this is the recipe the architecture's own defaults were tuned against.
        lr0 = args.lr if args.lr != 3e-4 else 1e-2
        opt = torch.optim.SGD(train_p, lr0, momentum=0.99, nesterov=True, weight_decay=3e-5)

        def lr_at(s):
            return (1 - s / max(1, steps)) ** 0.9

    if args.head == "curveq":
        # THE TRUNK KEEPS ITS RECIPE; THE QUERY HEAD DOES NOT SHARE IT. nnU-Net's SGD (momentum
        # 0.99, lr 1e-2) is tuned for an 8-stage conv encoder and it COLLAPSES a DETR-style query
        # decoder: measured on the head alone, 60 steps, identical data and seed --
        #   SGD(0.99) lr 1e-2  -> objectness std across queries 0.00000  (every query identical)
        #   SGD(0.9)  lr 1e-3  -> 0.087
        #   AdamW     lr 1e-4  -> 0.107   (the DETR-family standard)
        #   AdamW     lr 3e-4  -> 0.074
        # The first CURVEQ run used the shared SGD and duly collapsed: all 300 objectness values
        # equal to four decimals, zero curves emitted, 3 GPU-hours spent. Separate parameter
        # groups keep the trunk bit-compatible with every previous arm while the head gets the
        # optimiser its architecture needs.
        head_ids = {id(p) for p in model.head.parameters()}
        trunk_p = [p for p in train_p if id(p) not in head_ids]
        head_p = [p for p in train_p if id(p) in head_ids]
        if use_sgd:
            opt = _TwoOpt(
                torch.optim.SGD(trunk_p, lr0, momentum=0.99, nesterov=True, weight_decay=3e-5),
                torch.optim.AdamW(head_p, args.cq_head_lr, weight_decay=1e-4))
        else:
            opt = _TwoOpt(torch.optim.AdamW(trunk_p, lr0, weight_decay=1e-4),
                          torch.optim.AdamW(head_p, args.cq_head_lr, weight_decay=1e-4))
        print(f"curveq optimiser split: trunk {'SGD(0.99)' if use_sgd else 'AdamW'} lr {lr0:g} | "
              f"query head AdamW lr {args.cq_head_lr:g} (the shared nnU-Net SGD collapses it)",
              flush=True)
    else:
        lr0 = args.lr
        opt = torch.optim.AdamW(train_p, lr0, weight_decay=1e-4)

        def lr_at(s):
            if s < warm:
                return s / warm
            return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = (_TwoSched(opt, lr_at) if isinstance(opt, _TwoOpt)
             else torch.optim.lr_scheduler.LambdaLR(opt, lr_at))
    print(f"optimiser={'SGD(nnU-Net)' if use_sgd else 'AdamW'} lr0={lr0:g} steps={steps}",
          flush=True)
    pos_w = torch.tensor(float(args.pos_weight), device=DEV)
    # Channel index of the per-curve id map: K orientation channels, K_ORI = union foreground,
    # K_ORI + 1 = instance count, [K_ORI + 2 = heat, only for --head heat], then the id map LAST.
    CID_CH = K_ORI + 2 + (1 if args.head == "heat" else 0)
    if args.minpool_weight > 0:
        if not isinstance(ds, OnlineDS):
            raise SystemExit("--minpool-weight needs the synthetic loader: RealDS builds no "
                             "curve ids, and indexing that channel would read the count map.")
        probe = ds[0][1]
        if probe.shape[0] != CID_CH + 1:
            raise SystemExit(f"--minpool-weight: expected {CID_CH + 1} target channels, the "
                             f"loader produced {probe.shape[0]}; refusing to guess which is the "
                             f"id map.")
        print(f"worst-point term: weight {args.minpool_weight:g}, softmin temp "
              f"{args.minpool_temp:g} (logit units), curves >= {args.minpool_min_px} px, "
              f"id channel {CID_CH}", flush=True)
    history, step = [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.arch == "dino":
            model.dino.eval()
        tot = 0.0
        for batch in dl:
            cq_tgt = None
            if args.head == "curveq":
                im, gt, cq_tgt = batch
                cq_tgt = [t.to(DEV, non_blocking=True) for t in cq_tgt]
            else:
                im, gt = batch
            im, gt = im.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            im = with_priors(im, bank)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(im)
                cq_outs = None
                if args.head == "curveq":
                    # CurveQNet returns (dense, per-layer query outputs); the dense half then goes
                    # through the IDENTICAL binary path below, so the auxiliary task is literally
                    # the baseline's loss and a curveq run stays comparable to every dense arm.
                    out, cq_outs = out
                # Soft orientation targets are used as-is -- that is the point of the two-bin
                # assignment; the union head keeps a hard target.
                tgt = (gt[:, K_ORI:K_ORI + 1]
                       if args.head in ("binary", "heat", "overlap", "curveq")
                       else gt[:, :K_ORI + 1])
                # HEAT: channel 0 is the existence head (BCE + Dice exactly as --head binary),
                # channel 1 regresses the Gaussian centerline heatmap (gt index K_ORI + 2) with a
                # positive-weighted L1 -- the mean over a 98 %-background frame would otherwise
                # be minimised by predicting zero everywhere.
                heat_tgt = gt[:, K_ORI + 2:K_ORI + 3] if args.head == "heat" else None

                def _heat_loss(o_heat, t_heat):
                    wgt = 1.0 + float(args.pos_weight) * t_heat
                    return (wgt * (torch.sigmoid(o_heat.float()) - t_heat).abs()).sum() / wgt.sum()

                if isinstance(out, (tuple, list)):
                    # Deep supervision. Targets are MAX-pooled, not averaged: a 2 px filament
                    # averaged to a quarter resolution falls below any threshold and the deep
                    # heads would be trained on a blank target.
                    outs = list(out)[:max(1, args.deep_sup)]
                    w = np.array([0.5 ** i for i in range(len(outs))])
                    w = w / w.sum()
                    loss = 0.0
                    for oi, (o, wi) in enumerate(zip(outs, w)):
                        t = tgt
                        th = heat_tgt
                        if o.shape[-2:] != tgt.shape[-2:]:
                            f = tgt.shape[-1] // o.shape[-1]
                            t = F.max_pool2d(tgt, f, f)
                            th = F.max_pool2d(heat_tgt, f, f) if heat_tgt is not None else None
                        o_ex = o[:, :1] if args.head in ("heat", "overlap", "curveq") else o
                        if args.jw_radius > 0:
                            cnt_l = gt[:, K_ORI + 1:K_ORI + 2]
                            r_l = args.jw_radius
                            if o.shape[-2:] != tgt.shape[-2:]:
                                f_l = tgt.shape[-1] // o.shape[-1]
                                cnt_l = F.max_pool2d(cnt_l, f_l, f_l)
                                r_l = args.jw_radius / f_l
                            wmap = junction_weight(cnt_l, r_l, args.jw_weight)
                            bce_el = F.binary_cross_entropy_with_logits(
                                o_ex, t, pos_weight=pos_w, reduction="none")
                            bce = (wmap * bce_el).sum() / (wmap.sum() * o_ex.shape[1])
                        else:
                            bce = F.binary_cross_entropy_with_logits(o_ex, t, pos_weight=pos_w)
                        loss = loss + wi * (bce + dice(o_ex, t))
                        if th is not None:
                            loss = loss + wi * args.heat_weight * _heat_loss(o[:, 1:2], th)
                        if args.head == "overlap":
                            ov = (gt[:, K_ORI + 1:K_ORI + 2] >= 2).float()
                            if args.overlap_scramble:
                                ov = scramble_overlap(ov)
                            if o.shape[-2:] != tgt.shape[-2:]:
                                ov = F.max_pool2d(ov, tgt.shape[-1] // o.shape[-1],
                                                  tgt.shape[-1] // o.shape[-1])
                            loss = loss + wi * args.overlap_weight * F.binary_cross_entropy_with_logits(
                                o[:, 1:2], ov, pos_weight=ov_w)
                    if args.head == "amodal":
                        loss = loss + 0.5 * crossing_term(outs[0], gt[:, :K_ORI],
                                                          gt[:, K_ORI + 1])
                    if args.minpool_weight > 0:
                        loss = loss + args.minpool_weight * minpool_term(
                            outs[0][:, :1], gt[:, CID_CH:CID_CH + 1],
                            args.minpool_temp, args.minpool_min_px)
                else:
                    o_ex = out[:, :1] if args.head in ("heat", "overlap") else out
                    if args.head == "curveq":
                        o_ex = out[:, :1]
                    if args.jw_radius > 0:
                        wmap = junction_weight(gt[:, K_ORI + 1:K_ORI + 2],
                                               args.jw_radius, args.jw_weight)
                        bce_el = F.binary_cross_entropy_with_logits(
                            o_ex, tgt, pos_weight=pos_w, reduction="none")
                        bce = (wmap * bce_el).sum() / (wmap.sum() * o_ex.shape[1])
                    else:
                        bce = F.binary_cross_entropy_with_logits(o_ex, tgt, pos_weight=pos_w)
                    loss = (bce + dice(o_ex, tgt))
                    if heat_tgt is not None:
                        loss = loss + args.heat_weight * _heat_loss(out[:, 1:2], heat_tgt)
                    if args.head == "overlap":
                        ov = (gt[:, K_ORI + 1:K_ORI + 2] >= 2).float()
                        if args.overlap_scramble:
                            ov = scramble_overlap(ov)
                        loss = loss + args.overlap_weight * F.binary_cross_entropy_with_logits(
                            out[:, 1:2], ov, pos_weight=ov_w)
                    if args.head == "amodal":
                        loss = loss + 0.5 * crossing_term(out, gt[:, :K_ORI], gt[:, K_ORI + 1])
                    if args.minpool_weight > 0:
                        loss = loss + args.minpool_weight * minpool_term(
                            out[:, :1], gt[:, CID_CH:CID_CH + 1],
                            args.minpool_temp, args.minpool_min_px)
                if cq_outs is not None:
                    from instance.curveq import curveq_loss, mask_points
                    _c = im.shape[-1]
                    _rngs = np.random.default_rng(step)

                    def _pts(b, _t=cq_tgt, _c=_c, _r=_rngs):
                        return mask_points(_t[b], _c, rng=_r)

                    l_cq, cq_st = curveq_loss(cq_outs, cq_tgt, _pts,
                                              w_geo=args.cq_geo_weight,
                                              w_mask=args.cq_mask_weight,
                                              aux_k=args.cq_aux_k)
                    loss = args.cq_dense_weight * loss + l_cq
                    cq_run = globals().setdefault("_CQ_RUN", {})
                    for kk, vv in cq_st.items():
                        cq_run[kk] = cq_run.get(kk, 0.0) + float(vv)
                    cq_run["n"] = cq_run.get("n", 0) + 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_p, 12.0)
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                d = 0.999 if step > 1000 else 0.99
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += float(loss)
        print(f"epoch {ep}/{args.epochs} loss={tot / max(1, len(dl)):.4f} "
              f"lr={sched.get_last_lr()[0]:.2e}"
              + (_cq_epoch_line() if args.head == "curveq" else ""), flush=True)

        if ep % args.val_every == 0 or ep == args.epochs:
            agg = validate(ema, frames, params, thr,
                           args.fg_head and args.head == "amodal", TILE_FOR[args.arch], bank=bank,
                           # A 2-channel head must be validated on its EXISTENCE channel: at
                           # -1 `validate` takes ori.max(0) over ch[:K_ORI], which for out_ch=2
                           # is max(existence, multiplicity) -- a different quantity from the
                           # 1-channel baselines, and not comparable to them. Found after the
                           # first OVERLAP run, whose log is the union and is withdrawn.
                           prob_channel=(0 if args.head in ("heat", "overlap") else -1))
            gate = agg["fg"] <= 3 * args.synth_fg
            ck = os.path.join(args.ckpt_dir, f"ep{ep:03d}.pth")
            torch.save({k: v for k, v in ema.state_dict().items() if not k.startswith("dino.")},
                       ck)
            agg.update({"epoch": ep, "ckpt": ck, "loss": tot / max(1, len(dl)),
                        "gate": bool(gate)})
            history.append(agg)
            print(f"  VAL instance-F1={agg['mean_f1']:.4f} tp={agg['total_tp']} "
                  f"fp={agg['total_fp']} fn={agg['total_fn']} fg={100 * agg['fg']:.2f}% "
                  f"cc/gt={agg['cc_per_gt']:.2f} gate={'PASS' if gate else 'FAIL'}", flush=True)
            with open(args.report, "w") as fh:
                json.dump({"history": history, "arch": args.arch, "calib": args.calib}, fh,
                          indent=1)

    ok = [h for h in history if h["gate"]]
    if not ok:
        print("NO checkpoint passed the over-firing gate -- nothing selected.", flush=True)
        raise SystemExit(2)
    best = max(ok, key=lambda h: h["mean_f1"])
    model.load_state_dict(torch.load(best["ckpt"], map_location=DEV), strict=False)
    torch.save(model.state_dict(), args.out)
    print(f"\nselected epoch {best['epoch']} instance-F1={best['mean_f1']:.4f} -> {args.out}",
          flush=True)
    with open(args.report, "w") as fh:
        json.dump({"history": history, "selected": best, "arch": args.arch,
                   "calib": args.calib, "out": args.out}, fh, indent=1)


if __name__ == "__main__":
    main()

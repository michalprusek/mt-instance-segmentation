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

sys.path.insert(0, "/home/prusek/mt_enc_exp/scripts")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party"))

os.environ.setdefault("SEG_MODE", "ori")
os.environ.setdefault("SEG_BACKBONE", "dinov2")
os.environ.setdefault("SEG_INPUT", "raw")
os.environ.setdefault("SEG_ARCH", "base")

from instance.instancer_a import instance_a  # noqa: E402
from instance.metrics import aggregate_benchmark, centerline_f1  # noqa: E402
from mt_bench.cvat_import import read_frame_h5  # noqa: E402
from mt_bench.fg_quality import foreground_quality, mean_properties  # noqa: E402

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


def build_nnunet(out_ch: int, deep_supervision: bool = True):
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
        input_channels=3, num_classes=out_ch, conv_op=_nn.Conv2d,
        conv_bias=True, norm_op=_nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=_nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision, **NNUNET_PLAN)


def build_model(arch: str, out_ch: int = K_ORI + 1):
    """``out_ch`` exists so legacy checkpoints stay loadable.

    Everything trained before this script has a K=6 head; this one adds the binary union
    channel. Hard-coding K+1 would make every earlier checkpoint unloadable with a shape error,
    and re-deriving their numbers under the corrected per-model thresholding is precisely the
    work the calibration finding created.
    """
    if arch == "nnunet":
        return build_nnunet(out_ch)
    if arch == "resunet":
        return ResUNet(out_ori=out_ch - 1 if out_ch > K_ORI else out_ch)
    from dino_seg import DinoSeg                       # the v4b architecture, unchanged
    m = DinoSeg()
    if out_ch != m.head.out_channels:
        m.head = nn.Conv2d(m.head.in_channels, out_ch, 1)
    return m


# --------------------------------------------------------------------------- data
def ori_channels_soft(instances, shape, K=K_ORI, half_width=1.0):
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
    width = np.pi / K
    for ins in instances:
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
        for bb, ww in (((b0) % K, 1.0 - w1), ((b0 + 1) % K, w1)):
            np.maximum.at(ch, (bb, rr, cc), ww.astype(np.float32))
        fg[rr, cc] = 1.0
        one = np.zeros(tuple(shape), np.float32)
        one[rr, cc] = 1.0
        cnt += binary_dilation(one > 0, structure=np.ones((3, 3), bool))
    fp = np.ones((3, 3), bool)
    ch = np.stack([grey_dilation(ch[k], footprint=fp) for k in range(K)])
    fg = binary_dilation(fg > 0, structure=fp).astype(np.float32)
    return np.concatenate([ch, fg[None], cnt[None]], 0)


def apply_field_stop(im, gt, rng, K=K_ORI):
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
    interior median** (e.g. 37 -> 187, 4563 -> 11783). It is applied to the RAW render, before
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
    sat = interior * rng.uniform(2.5, 5.0)
    out = im * ramp + sat * (1.0 - ramp)
    gt = gt * inside[None, :, :].astype(gt.dtype)
    return out.astype(np.float32), gt


class OnlineDS(torch.utils.data.Dataset):
    """Fresh synthetic frame per item, with rotation augmentation.

    Rotation matters here specifically: the K-bin orientation head previously saw only the four
    orientation permutations that flips induce, so its bins were unevenly sampled. A rotation
    by a multiple of 180/K degrees permutes the bins exactly, which keeps the target derivable
    without re-rasterising.
    """

    def __init__(self, cfg, bg_paths, gen_fn, crop=518, gen=640, epoch_len=5000, rot=True,
                 fov_prob=0.0, target_half_width=1.0):
        self.cfg, self.bg, self.gen_fn = cfg, bg_paths, gen_fn
        self.crop, self.gen, self.epoch_len, self.rot = crop, gen, epoch_len, rot
        self.fov_prob = fov_prob
        self.target_half_width = target_half_width

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
        img, inst, _ = self.gen_fn(bg, rng, self.cfg)
        gt = ori_channels_soft(inst, (g, g), half_width=self.target_half_width)
        if self.fov_prob and rng.random() < self.fov_prob:
            # BEFORE the stretch: a bright surround drags the frame's percentiles and squashes
            # the interior, which is exactly what happens to the real task-586 frames.
            img, gt = apply_field_stop(img, gt, rng)
        lo, hi = np.percentile(img, [1, 99])
        im = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        if rng.random() < .5:
            im = 1.0 - im                                     # IRM polarity is ambiguous

        c = self.crop
        y = int(rng.integers(0, g - c + 1))
        x = int(rng.integers(0, g - c + 1))
        im, gt = im[y:y + c, x:x + c], gt[:, y:y + c, x:x + c]
        # k*90 degree rotations and flips; both permute the orientation bins exactly.
        if self.rot:
            k = int(rng.integers(4))
            if k:
                im = np.rot90(im, k).copy()
                gt = np.rot90(gt, k, axes=(1, 2)).copy()
                # rot90 maps theta -> theta + k*90deg, i.e. bin b -> b + k*K/4 (K=6 -> 1.5, so
                # only k=2 is exact); K=6 is exact for 180deg, which is identity on [0,pi).
                if k % 2 == 1:
                    gt[:K_ORI] = np.roll(gt[:K_ORI], K_ORI // 2, axis=0)
        if rng.random() < .5:
            im = im[:, ::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, :, ::-1].copy()
        if rng.random() < .5:
            im = im[::-1].copy()
            gt = np.concatenate([gt[:K_ORI][::-1], gt[K_ORI:]])[:, ::-1, :].copy()
        t = torch.from_numpy(im)[None].repeat(3, 1, 1)
        t = (t - IMA_M) / IMA_S
        return t, torch.from_numpy(np.ascontiguousarray(gt))


# --------------------------------------------------------------------------- losses
def dice(p, t, e=1.0):
    p = torch.sigmoid(p)
    i = (p * t).sum((2, 3))
    return (1 - (2 * i + e) / (p.sum((2, 3)) + t.sum((2, 3)) + e)).mean()


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
def norm01(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


#: Tile size per architecture. The DINO trunk wants 518 (its /14 patch grid); nnU-Net's ResEnc
#: has seven /2 stages, so its input must be divisible by 128 or the residual adds hit a
#: shape mismatch on the way back up. This is a property of the network, not a global constant.
TILE_FOR = {"dino": 518, "resunet": 518, "nnunet": 512}


@torch.no_grad()
def predict(model, img01, tile: int = TILE, stride: int | None = None):
    TILE = tile
    STRIDE = stride if stride is not None else int(round(tile * 392 / 518))
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(tt)
            if isinstance(o, (tuple, list)):
                o = o[0]
            o = torch.sigmoid(o.float())[0].cpu().numpy()
            if acc is None:
                acc = np.zeros((o.shape[0], H, W))
                cnt = np.zeros((H, W))
            acc[:, y:y + th, x:x + tw] += o[:, :th, :tw]
            cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)[None]


def validate(model, frames, params, thr, use_fg_head: bool, tile: int = TILE):
    """Strict centerline-F1 through the instancer -- the metric the project is judged on."""
    model.eval()
    rows, fgq = [], []
    for fr in frames:
        ch = predict(model, zoom(norm01(np.asarray(fr["image"], float)), UP, order=1), tile)
        ori = ch[:K_ORI]
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
    ap.add_argument("--head", default="amodal", choices=["amodal", "binary"],
                    help="binary = union foreground ONLY, no orientation channels. This is the "
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
    for d in ("/home/prusek/mt_enc_exp/irm_backgrounds_v2",
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
    ds = OnlineDS(cfg, bgs, generate_frame, crop=args.crop, epoch_len=args.epoch_len,
                  target_half_width=args.target_half_width,
                  fov_prob=args.fov_prob)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                     drop_last=True, persistent_workers=True, prefetch_factor=3)

    frames = []
    for p in sorted(glob.glob(os.path.join(args.val_data, "*.h5"))):
        fr = read_frame_h5(p)
        if str(fr["attrs"].get("split")) == "val" and fr["polylines"]:
            frames.append(fr)
    frames = frames[:args.val_n]
    params = json.load(open(args.params))
    thr = params.get("prob_thr", 0.35)
    print(f"arch={args.arch} calib={os.path.basename(args.calib)} "
          f"VAL={len(frames)} frames (selection only; training data is synthetic)", flush=True)

    out_ch = 1 if args.head == "binary" else K_ORI + 1
    model = build_model(args.arch, out_ch=out_ch).to(DEV)
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
    else:
        lr0 = args.lr
        opt = torch.optim.AdamW(train_p, lr0, weight_decay=1e-4)

        def lr_at(s):
            if s < warm:
                return s / warm
            return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"optimiser={'SGD(nnU-Net)' if use_sgd else 'AdamW'} lr0={lr0:g} steps={steps}",
          flush=True)
    pos_w = torch.tensor(float(args.pos_weight), device=DEV)
    history, step = [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.arch == "dino":
            model.dino.eval()
        tot = 0.0
        for im, gt in dl:
            im, gt = im.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(im)
                # Soft orientation targets are used as-is -- that is the point of the two-bin
                # assignment; the union head keeps a hard target.
                tgt = (gt[:, K_ORI:K_ORI + 1] if args.head == "binary"
                       else gt[:, :K_ORI + 1])
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
                        if o.shape[-2:] != tgt.shape[-2:]:
                            f = tgt.shape[-1] // o.shape[-1]
                            t = F.max_pool2d(tgt, f, f)
                        loss = loss + wi * (
                            F.binary_cross_entropy_with_logits(o, t, pos_weight=pos_w)
                            + dice(o, t))
                    if args.head != "binary":
                        loss = loss + 0.5 * crossing_term(outs[0], gt[:, :K_ORI],
                                                          gt[:, K_ORI + 1])
                else:
                    loss = (F.binary_cross_entropy_with_logits(out, tgt, pos_weight=pos_w)
                            + dice(out, tgt))
                    if args.head != "binary":
                        loss = loss + 0.5 * crossing_term(out, gt[:, :K_ORI], gt[:, K_ORI + 1])
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
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        if ep % args.val_every == 0 or ep == args.epochs:
            agg = validate(ema, frames, params, thr,
                           args.fg_head and args.head != "binary", TILE_FOR[args.arch])
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

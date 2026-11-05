"""Short synthetic clips around one still frame, for the temporal input arm (docs/TODO.md 2026-11-04 08:10).

``mt_sequence.py`` renders from a shared appearance seed but has drifted from what the trainer draws:
it drops instances (which desynchronises the per-instance appearance draws), draws the NMIX noise
regime afresh per frame, and moves filaments only. This module keeps one property above all:

    The frame t and everything drawn from ``rng`` are BYTE-IDENTICAL to ``generate_frame``.

Neighbours use a snapshot of the generator state taken before t's render, so they replay t's
appearance, and a jumped stream for everything else, so ``rng`` does not advance and the item's crop
and flips do not move. Training on t alone is therefore exactly the still recipe. Every instance
stays in the list in every frame, even off-frame, because the appearance draws are per instance in
list order; a filament is never dropped or killed here.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from mt_generator import GenConfig, flatten_background, render_irm, render_tirf, sample_scene
from mt_sequence import arclength, extend_path, resample_window

REGIMES = ("static", "gliding", "dynamic")


@dataclass
class ClipConfig:
    """Priors of the motion between frames. Widened to cover, not fitted: the frame interval of the real videos is mostly unknown."""
    regime_probs: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    glide_speed_px: Tuple[float, float] = (0.5, 16.0)   # per clip, log-uniform, px per frame
    glide_scatter: float = 0.25                         # per filament, log-normal around the clip's speed
    stuck_frac: Tuple[float, float] = (0.0, 0.5)        # per clip, share of filaments that do not glide
    grow_px: Tuple[float, float] = (0.5, 3.0)           # a dynamic tip, per frame
    shrink_px: Tuple[float, float] = (2.0, 9.0)
    jitter_px: Tuple[float, float] = (0.0, 0.5)         # per clip, sd of a filament's wobble per frame
    shift_scale_px: Tuple[float, float] = (0.0, 4.0)    # per clip, s; the whole frame steps by U(0, s) whole px per frame
    min_length_px: float = 8.0                          # a shrinking filament stops here; it never dies


def freeze_noise(cfg: GenConfig, rmeta: dict, modality: str) -> GenConfig:
    """The config that redraws t's noise regime and levels exactly, leaving only the realisation free."""
    if modality == "tirf":
        return dataclasses.replace(cfg, tirf_poisson_gain_range=(rmeta["poisson_gain"],) * 2,
                                   tirf_read_noise_range=(rmeta["read_noise"],) * 2)
    fixed = {"noise_alt_prob": 0.0}
    if rmeta.get("noise_alt"):
        fixed.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.noise_alt.items()})
    fixed["poisson_gain_range"] = (rmeta["poisson_gain"],) * 2
    fixed["read_noise_range"] = (rmeta["read_noise"],) * 2
    if "noise_span_rel" in rmeta:
        fixed["noise_span_rel_range"] = (rmeta["noise_span_rel"],) * 2
    if "noise_corr_frac" in rmeta:
        fixed["noise_corr_frac_range"] = (rmeta["noise_corr_frac"],) * 2
        fixed["noise_corr_sigma_range"] = (rmeta["noise_corr_sigma"],) * 2
    if "noise_impulse_frac" in rmeta:
        fixed["noise_impulse_frac_range"] = (rmeta["noise_impulse_frac"],) * 2
    return dataclasses.replace(cfg, **fixed)


def move_image(img: np.ndarray, d) -> np.ndarray:
    """Whole-pixel shift by ``d = (dy, dx)``, edges replicated: a point at p lands at p + d. Noise keeps its statistics."""
    dy, dx = int(d[0]), int(d[1])
    if dy == 0 and dx == 0:
        return img
    H, W = img.shape
    y = np.clip(np.arange(H) - dy, 0, H - 1)
    x = np.clip(np.arange(W) - dx, 0, W - 1)
    return img[y][:, x]


class _Filament:
    """A window of arclength on a path extended beyond the filament, as in ``mt_sequence``, but the middle of the path is the filament."""

    def __init__(self, cl: np.ndarray, margin: float, rng, cfg: GenConfig):
        self.n = len(cl)
        self.cl = cl
        full = extend_path(cl, rng, cfg, margin, where="both") if margin > 0 and self.n >= 2 else cl
        self.path, self.s = full, arclength(full)
        off = (len(full) - self.n) // 2
        self.s0 = float(self.s[off])
        self.s1 = float(self.s[off + self.n - 1])

    def at(self, s0: float, s1: float) -> np.ndarray:
        return resample_window(self.path, self.s, s0, s1, self.n)


def _walks(fil: _Filament, regime: str, n: int, sign: int, speed: float, rng, ccfg: ClipConfig) -> List[np.ndarray]:
    """The filament's centerline at steps 1..n in one time direction."""
    out, s0, s1 = [], fil.s0, fil.s1
    for _ in range(n):
        if regime == "gliding":
            s0, s1 = s0 + sign * speed, s1 + sign * speed
        elif regime == "dynamic":
            grow = [rng.random() < 0.5, rng.random() < 0.5]
            d0 = -rng.uniform(*ccfg.grow_px) if grow[0] else rng.uniform(*ccfg.shrink_px)
            d1 = rng.uniform(*ccfg.grow_px) if grow[1] else -rng.uniform(*ccfg.shrink_px)
            s0, s1 = float(np.clip(s0 + d0, fil.s[0], fil.s[-1])), float(np.clip(s1 + d1, fil.s[0], fil.s[-1]))
            if s1 - s0 < ccfg.min_length_px:
                s0, s1 = s0 - d0, s1 - d1
        out.append(fil.at(s0, s1) if regime != "static" else fil.cl)
    return out


def generate_clip(background: np.ndarray, rng, cfg: GenConfig, n_before: int = 1, n_after: int = 1,
                  modality: str = "irm", ccfg: Optional[ClipConfig] = None, regime: Optional[str] = None):
    """-> ``(images, per_frame, meta)``, frame t at index ``n_before``; ``images[n_before]`` and ``per_frame[n_before]`` are exactly ``generate_frame``'s."""
    ccfg = ccfg or ClipConfig()
    instances, cond = sample_scene(background.shape, rng, cfg)
    render = render_tirf if modality == "tirf" else render_irm
    flat_sigma = None
    if cfg.bg_flatten_prob > 0 and rng.random() < cfg.bg_flatten_prob:
        flat_sigma = float(np.exp(rng.uniform(*np.log(cfg.bg_flatten_sigma_range))))
        background = flatten_background(background, flat_sigma)
    appearance = copy.deepcopy(rng)
    img, rmeta = render(instances, cond, background, rng, cfg)
    if flat_sigma is not None:
        rmeta["bg_flatten_sigma"] = flat_sigma
    for i, ins in enumerate(instances):
        ins["polarity_base"] = 1.0 if modality == "tirf" else -1.0
        ins["inst_id"] = i
    meta = {"modality": modality, "regime": cond["regime"], "wavy": cond["wavy"], "hairpin_active": cond["hairpin_active"],
            "orient_kappa": cond["orient_kappa"], "n_instances": len(instances), **rmeta}
    if n_before == 0 and n_after == 0:
        return [img], [instances], meta

    side = np.random.Generator(rng.bit_generator.jumped())   # does not advance rng
    motion = regime if regime is not None else REGIMES[int(side.choice(3, p=ccfg.regime_probs))]
    base = float(np.exp(side.uniform(*np.log(ccfg.glide_speed_px))))
    stuck = float(side.uniform(*ccfg.stuck_frac))
    jitter = float(side.uniform(*ccfg.jitter_px))
    scale = float(side.uniform(*ccfg.shift_scale_px))
    frozen = freeze_noise(cfg, rmeta, modality)
    steps = max(n_before, n_after)
    back, ahead = [], []
    for ins in instances:
        cl = np.asarray(ins["centerline"], float)
        speed, sgn = 0.0, 1 if side.random() < 0.5 else -1
        if motion == "gliding" and side.random() >= stuck:
            speed = base * float(np.exp(side.normal(0.0, ccfg.glide_scatter)))
        margin = speed * (steps + 1) if motion == "gliding" else (max(ccfg.grow_px) * (steps + 1) if motion == "dynamic" else 0.0)
        fil = _Filament(cl, margin, side, cfg)
        kind = "gliding" if speed > 0 else ("dynamic" if motion == "dynamic" else "static")
        back.append(_walks(fil, kind, n_before, -sgn, speed, side, ccfg))
        ahead.append(_walks(fil, kind, n_after, sgn, speed, side, ccfg))

    def step() -> np.ndarray:
        size = side.uniform(0.0, scale) if scale > 0 else 0.0
        angle = side.uniform(0.0, 2 * np.pi)
        return np.array([size * np.sin(angle), size * np.cos(angle)])

    def frames(walks, n):
        out, d = [], np.zeros(2)
        for k in range(n):
            d = d + step()
            whole = np.rint(d).astype(int)                                    # (dy, dx)
            inst = []
            for ins, w in zip(instances, walks):
                wobble = side.normal(0.0, jitter, 2) if jitter > 0 else np.zeros(2)
                inst.append({"centerline": w[k] + wobble, "inst_id": ins["inst_id"], "polarity_base": ins["polarity_base"]})
            noise = np.random.default_rng(int(side.integers(2 ** 63)))
            im, _ = render(inst, cond, background, copy.deepcopy(appearance), frozen, noise_rng=noise)
            for z in inst:
                z["centerline"] = z["centerline"] + whole[::-1]                # centerlines are (x, y)
            out.append((move_image(im, whole), inst))
        return out

    before, after = frames(back, n_before), frames(ahead, n_after)
    images = [x[0] for x in before[::-1]] + [img] + [x[0] for x in after]
    per_frame = [x[1] for x in before[::-1]] + [instances] + [x[1] for x in after]
    meta.update(n_frames=len(images), motion=motion, glide_speed=base, stuck_frac=stuck, jitter=jitter, shift_scale=scale, centre=n_before)
    return images, per_frame, meta


def clip_triplet(background: np.ndarray, rng, cfg: GenConfig, modality: str = "irm", p_single: float = 0.35,
                 ccfg: Optional[ClipConfig] = None):
    """The trainer's ``gen_fn``: frame t, its instances, and ``meta["neighbours"]`` = (t-1, t+1), or None for a ``(t, t, t)`` item."""
    single = np.random.Generator(rng.bit_generator.jumped(2)).random() < p_single
    if single:
        imgs, per, meta = generate_clip(background, rng, cfg, 0, 0, modality, ccfg)
        meta["neighbours"] = None
        return imgs[0], per[0], meta
    imgs, per, meta = generate_clip(background, rng, cfg, 1, 1, modality, ccfg)
    meta["neighbours"] = (imgs[0], imgs[2])
    return imgs[1], per[1], meta

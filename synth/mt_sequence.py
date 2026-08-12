"""Synthetic microtubule VIDEO: sequences with exact, free cross-frame correspondence.

The generator has only ever produced stills, so nothing in this project has been able to train
or measure a tracker. This adds time at the seam the generator already has -- ``sample_scene``
samples morphology, ``render_irm`` paints it -- by evolving the morphology between renders.

The representation that makes all three regimes one mechanism
-------------------------------------------------------------
Every microtubule is a **window of arclength [s0, s1] on a longer fixed 2D path**. Motion is
then a statement about the window, not about the pixels:

* **static** -- window fixed; the filament is immobilised. Only sub-pixel jitter and the
  frame-global stage drift move it. This is the regime that catches a tracker inventing motion.
* **gliding** -- both ends advance by the same ``v*dt``: the filament slides along its own
  contour, which is exactly what a motor-propelled filament does, and its length is conserved
  automatically. The path *ahead* of the leading tip is where it is going next, which is why
  the path is sampled longer than the filament.
* **dynamic** -- the two ends move independently under two-state switching (growth/shrinkage
  with catastrophe and rescue), so the body stays put while the tips explore.

``n_frames=1`` reproduces ``generate_frame`` bit for bit -- pinned by a test. A separate
"video path" that drifts from the still path is the failure this project has already seen once
in its deployed wrapper, and it is not worth repeating.

Appearance across a sequence
----------------------------
Frame-level appearance (contrast, illumination NA, base height, PSF width) and the dirt specks
are drawn from a **per-sequence seed replayed identically every frame**, while sensor noise
comes from a per-frame stream. So the field looks like the same field, the dirt stays put, and
only the noise flickers -- which is the whole reason temporal fusion can repair a dropout.
Keeping the point count per instance fixed across the sequence is what makes the replay exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from mt_generator import (GenConfig, rasterize_mask, render_irm, render_tirf,
                          sample_scene)


@dataclass
class MotionConfig:
    """Per-frame motion priors. Speeds are in PIXELS PER FRAME, not physical units.

    Defaults sit in the single-digit-pixel range the real acquisitions were reported to be in.
    They are a prior, not a measurement: with no cross-frame ground truth available (CVAT
    project 31 turned out to hold static crop boxes, not filament tracks) nothing here has been
    calibrated against real video. Treat every velocity number produced with these defaults as
    synthetic-only until that changes.
    """
    # --- static -------------------------------------------------------------------------
    static_jitter_px: float = 0.25          # sub-pixel wobble of an immobilised filament
    # --- gliding ------------------------------------------------------------------------
    glide_speed_px: tuple = (1.5, 8.0)      # per-MT speed, sampled once and held
    glide_speed_log_std: float = 0.25       # log-normal scatter around that speed
    glide_reverse_prob: float = 0.05        # occasionally a filament runs the other way
    # --- dynamic instability -------------------------------------------------------------
    grow_px: tuple = (0.5, 3.0)             # tip growth rate while in the growing state
    shrink_px: tuple = (2.0, 9.0)           # shrinkage is faster than growth (real MTs)
    p_catastrophe: float = 0.10             # growth -> shrinkage, per tip per frame
    p_rescue: float = 0.25                  # shrinkage -> growth
    min_length_px: float = 40.0             # below this a shrinking filament disappears
    # --- whole-field --------------------------------------------------------------------
    drift_px_std: float = 0.6               # stage drift, SHARED by every filament in a frame
    drift_correlated: bool = True           # a drift that persists in direction, not white noise


@dataclass
class Track:
    """One microtubule through a sequence: a fixed path plus a moving window on it."""
    inst_id: int
    path: np.ndarray                        # (M, 2) the underlying 2D path, (x, y)
    s: np.ndarray                           # (M,) arclength along ``path``
    s0: float                               # window start
    s1: float                               # window end
    n_pts: int                              # points to resample the window to; FIXED per track
    speed: float = 0.0                      # px/frame along the contour (gliding)
    tip_state: tuple = ("grow", "grow")     # (tail, head) states in the dynamic regime
    alive: bool = True
    history: List[float] = field(default_factory=list)   # arclength advance per frame

    def centerline(self) -> np.ndarray:
        return resample_window(self.path, self.s, self.s0, self.s1, self.n_pts)


def arclength(path: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_window(path: np.ndarray, s: np.ndarray, s0: float, s1: float,
                    n_pts: int) -> np.ndarray:
    """The piece of ``path`` between arclengths s0 and s1, as ``n_pts`` evenly spaced points.

    A FIXED point count (rather than a fixed spacing) is deliberate: it keeps every per-instance
    appearance draw in the renderer aligned frame to frame, so the same microtubule keeps the
    same look while it grows or glides.
    """
    s0 = float(np.clip(s0, s[0], s[-1]))
    s1 = float(np.clip(s1, s[0], s[-1]))
    if s1 - s0 < 1e-6:
        s1 = min(s[-1], s0 + 1e-6)
    q = np.linspace(s0, s1, max(int(n_pts), 2))
    return np.stack([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])], axis=1)


def extend_path(path: np.ndarray, rng, cfg: GenConfig, extra_len: float,
                where: str = "both") -> np.ndarray:
    """Continue a centerline beyond its ends with the same tangent-angle statistics.

    A gliding filament advances onto path it has not occupied yet, and a growing tip does the
    same, so the underlying path has to exist before the motion does. The continuation is a 2D
    worm-like-chain walk seeded from the existing end tangent -- the same process
    ``sample_centerline`` uses for the gliding regime -- rather than a straight line, which
    would make every filament eventually glide off in a dead-straight direction and give the
    tracker a much easier problem than reality does.
    """
    if extra_len <= 0:
        return path
    ds = max(float(cfg.step_px), 0.5)
    n_add = int(np.ceil(extra_len / ds))
    if n_add < 1:
        return path
    lp = max(float(np.mean(cfg.lp_gliding_px_range)), 300.0)

    def grow(anchor: np.ndarray, theta0: float) -> np.ndarray:
        dtheta = rng.normal(0.0, np.sqrt(2.0 * ds / lp), n_add)
        theta = theta0 + np.cumsum(dtheta)
        step = np.stack([np.cos(theta), np.sin(theta)], axis=1) * ds
        return anchor + np.cumsum(step, axis=0)

    out = path
    if where in ("both", "head"):
        t = path[-1] - path[-2]
        head = grow(path[-1], float(np.arctan2(t[1], t[0])))
        out = np.concatenate([out, head], axis=0)
    if where in ("both", "tail"):
        t = path[0] - path[1]
        tail = grow(path[0], float(np.arctan2(t[1], t[0])))
        out = np.concatenate([tail[::-1], out], axis=0)
    # A light smooth over the joins only; the walk itself must keep its statistics.
    out[:, 0] = gaussian_filter1d(out[:, 0], 1.0)
    out[:, 1] = gaussian_filter1d(out[:, 1], 1.0)
    return out


def init_tracks(instances, regime: str, rng, cfg: GenConfig, mcfg: MotionConfig,
                n_frames: int) -> List[Track]:
    """Turn the still frame's instances into tracks, extending each path to cover the motion."""
    tracks = []
    for i, ins in enumerate(instances):
        cl = np.asarray(ins["centerline"], dtype=float)
        n_pts = len(cl)
        speed = 0.0
        margin = 0.0
        if regime == "gliding":
            speed = float(rng.uniform(*mcfg.glide_speed_px)
                          * np.exp(rng.normal(0.0, mcfg.glide_speed_log_std)))
            if rng.random() < mcfg.glide_reverse_prob:
                speed = -speed
            margin = abs(speed) * (n_frames + 1)
        elif regime == "dynamic":
            margin = max(mcfg.grow_px) * (n_frames + 1)

        path = extend_path(cl, rng, cfg, margin, where="both") if margin > 0 else cl
        s = arclength(path)
        # The original filament sits in the middle of the extended path.
        span = arclength(cl)[-1]
        s0 = (s[-1] - span) / 2.0 if margin > 0 else 0.0
        tracks.append(Track(inst_id=i, path=path, s=s, s0=s0, s1=s0 + span, n_pts=n_pts,
                            speed=speed,
                            tip_state=("grow" if rng.random() < 0.5 else "shrink",
                                       "grow" if rng.random() < 0.5 else "shrink")))
    return tracks


def _switch(state: str, rng, mcfg: MotionConfig) -> str:
    if state == "grow":
        return "shrink" if rng.random() < mcfg.p_catastrophe else "grow"
    return "grow" if rng.random() < mcfg.p_rescue else "shrink"


def step_tracks(tracks: List[Track], regime: str, rng, mcfg: MotionConfig) -> None:
    """Advance every track by one frame, in place."""
    for tr in tracks:
        if not tr.alive:
            continue
        if regime == "gliding":
            adv = tr.speed
            tr.s0 += adv
            tr.s1 += adv
            tr.history.append(float(adv))
            # Ran off the end of its own path: it has left the field of view.
            if tr.s1 >= tr.s[-1] or tr.s0 <= tr.s[0]:
                tr.alive = False
        elif regime == "dynamic":
            tail, head = tr.tip_state
            tail, head = _switch(tail, rng, mcfg), _switch(head, rng, mcfg)
            d_tail = (-rng.uniform(*mcfg.grow_px) if tail == "grow"
                      else rng.uniform(*mcfg.shrink_px))
            d_head = (rng.uniform(*mcfg.grow_px) if head == "grow"
                      else -rng.uniform(*mcfg.shrink_px))
            tr.s0 = float(np.clip(tr.s0 + d_tail, tr.s[0], tr.s[-1]))
            tr.s1 = float(np.clip(tr.s1 + d_head, tr.s[0], tr.s[-1]))
            tr.tip_state = (tail, head)
            tr.history.append(float(d_head - d_tail))
            if tr.s1 - tr.s0 < mcfg.min_length_px:
                tr.alive = False
        else:                                   # static
            tr.history.append(0.0)


def generate_sequence(background: np.ndarray, rng, cfg: GenConfig, n_frames: int = 1,
                      mcfg: Optional[MotionConfig] = None, modality: str = "irm"):
    """A synthetic movie with exact correspondence.

    Returns ``(images, per_frame_instances, meta)``. ``per_frame_instances[k]`` is a list of
    ``{centerline, mask, inst_id}``; **``inst_id`` is the correspondence ground truth** -- the
    same integer means the same microtubule, and an id absent from a frame has left or died.

    ``n_frames=1`` is byte-identical to ``mt_generator.generate_frame`` for the same seed.
    """
    mcfg = mcfg or MotionConfig()
    instances, cond = sample_scene(background.shape, rng, cfg)
    render = render_tirf if modality == "tirf" else render_irm
    regime = cond["regime"]

    if n_frames <= 1:
        img, rmeta = render(instances, cond, background, rng, cfg)
        for ins in instances:
            ins["polarity_base"] = 1.0 if modality == "tirf" else -1.0
        for i, ins in enumerate(instances):
            ins["inst_id"] = i
        meta = {"modality": modality, "regime": regime, "n_frames": 1,
                "n_instances": len(instances), **_cond_meta(cond), **rmeta}
        return [img], [instances], meta

    tracks = init_tracks(instances, regime, rng, cfg, mcfg, n_frames)

    # Stage drift: one vector for the whole field, shared by every filament. A tracker that
    # reports it as motility is wrong in the way that matters for a motility assay, so the
    # generator must produce it.
    drift = np.zeros(2)
    drift_dir = rng.normal(0, 1, 2)
    drift_dir /= np.linalg.norm(drift_dir) + 1e-9

    # Appearance is replayed from one seed every frame; only the noise stream advances.
    appearance_seed = int(rng.integers(0, 2**31 - 1))
    H, W = background.shape

    images, per_frame, offsets = [], [], []
    for k in range(n_frames):
        if k > 0:
            step_tracks(tracks, regime, rng, mcfg)
            drift = (drift + mcfg.drift_px_std * drift_dir if mcfg.drift_correlated
                     else mcfg.drift_px_std * rng.normal(0, 1, 2))

        frame_inst = []
        for tr in tracks:
            if not tr.alive:
                continue
            cl = tr.centerline() + drift
            if regime == "static" and mcfg.static_jitter_px > 0:
                cl = cl + rng.normal(0, mcfg.static_jitter_px, 2)
            mask = rasterize_mask(cl, (H, W), cfg.mask_half_width)
            if not mask.any():
                continue
            frame_inst.append({"centerline": cl, "mask": mask, "inst_id": tr.inst_id,
                               "polarity_base": 1.0 if modality == "tirf" else -1.0})

        img, rmeta = render(frame_inst, cond, background,
                            np.random.default_rng(appearance_seed), cfg, noise_rng=rng)
        images.append(img)
        per_frame.append(frame_inst)
        offsets.append({tr.inst_id: (tr.history[-1] if tr.history else 0.0) for tr in tracks})

    meta = {"modality": modality, "regime": regime, "n_frames": n_frames,
            "n_instances": len(tracks), "drift_px_per_frame": float(mcfg.drift_px_std),
            "arclength_offsets": offsets,
            "speeds": {tr.inst_id: tr.speed for tr in tracks},
            **_cond_meta(cond), **rmeta}
    return images, per_frame, meta


def _cond_meta(cond) -> dict:
    return {"wavy": cond["wavy"], "hairpin_active": cond["hairpin_active"],
            "orient_kappa": cond["orient_kappa"]}

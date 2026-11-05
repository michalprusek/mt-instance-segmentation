#!/usr/bin/env python3
"""ARM 65 (docs/TODO.md 2026-11-04 08:10): the temporal flag of train_v5.py and the TIRF noise levels in meta. Anchored on exact strings, refuses to apply twice. Flag off: no new draw, every existing recipe is bit-identical.
    python patch_temporal.py path/to/train_v5.py path/to/mt_generator.py"""
import sys
T, G = sys.argv[1:3]; s = open(T).read(); g = open(G).read()
if "temporal_p_single" in s: sys.exit("train_v5.py already patched")
def rep(text, a, b):
    if text.count(a) != 1: sys.exit(f"anchor not unique / missing: {a!r}")
    return text.replace(a, b)
# the generator: render_tirf reports the levels it drew, no effect on the image or the draws
g = rep(g, """    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1),
                         noise_rng if noise_rng is not None else rng,
                         cfg.tirf_poisson_gain_range, cfg.tirf_read_noise_range)
    img = lo + np.clip(z, 0, 1) * span
    return img, {"inverted": False, "detach_active": detach_active}
""", """    tmeta = {"inverted": False, "detach_active": detach_active}
    z = _shot_read_noise(np.clip((img - lo) / span, 0, 1),
                         noise_rng if noise_rng is not None else rng,
                         cfg.tirf_poisson_gain_range, cfg.tirf_read_noise_range, meta=tmeta)
    img = lo + np.clip(z, 0, 1) * span
    return img, tmeta
""")
# the trainer's item: t's own path is untouched, neighbours follow its stretch-free decisions
s = rep(s, """            img, inst, _ = self.gen_fn(bg, rng, self.cfg, modality="tirf")
""", """            img, inst, gm = self.gen_fn(bg, rng, self.cfg, modality="tirf")
""")
s = rep(s, """        else:
            img, inst, _ = self.gen_fn(bg, rng, self.cfg)
        if self.scale_aug is not None and rng.random() < self.scale_prob:
""", """        else:
            img, inst, gm = self.gen_fn(bg, rng, self.cfg)
        nb = gm.get("neighbours") if isinstance(gm, dict) else None   # ARM 65: (t-1, t+1) or None
        if self.scale_aug is not None and rng.random() < self.scale_prob:
""")
s = rep(s, """        if rng.random() < .5:
            im = 1.0 - im                                     # IRM polarity is ambiguous

        c = self.crop
        if gh < c or gw < c:                       # a downscaled frame smaller than the crop
""", """        _inv = rng.random() < .5
        if _inv:
            im = 1.0 - im                                     # IRM polarity is ambiguous

        c = self.crop
        if gh < c or gw < c:                       # a downscaled frame smaller than the crop
""")
s = rep(s, """        t = torch.from_numpy(im)[None].repeat(3, 1, 1)
        t = (t - IMA_M) / IMA_S
        if self.curve_targets:
""", """        t = torch.from_numpy(im)[None].repeat(3, 1, 1)
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
""")
s = rep(s, """    ap.add_argument("--tirf-prob", type=float, default=0.0,
""", """    ap.add_argument("--temporal-p-single", type=float, default=-1.0,
                    help="ARM 65: >= 0 renders (t-1, t, t+1) items through synth/mt_clip.py, this share as (t, t, t); frame t and its target stay byte-identical to the still recipe. Off by default.")
    ap.add_argument("--tirf-prob", type=float, default=0.0,
""")
s = rep(s, """    else:
        ds = OnlineDS(cfg, bgs, generate_frame, crop=args.crop, epoch_len=args.epoch_len,
""", """    else:
        gen_fn = generate_frame
        if args.temporal_p_single >= 0:
            if scale_aug or args.lowres_prob or args.fov_prob or args.rot_aug_prob or args.photo_aug:
                raise SystemExit("--temporal-p-single does not combine with scale, lowres, field stop, arbitrary rotation or photo augmentation")
            import functools
            from mt_clip import clip_triplet
            gen_fn = functools.partial(clip_triplet, p_single=args.temporal_p_single)
            print(f"ARM 65: temporal input, p_single={args.temporal_p_single}", flush=True)
        ds = OnlineDS(cfg, bgs, gen_fn, crop=args.crop, epoch_len=args.epoch_len,
""")
open(T, "w").write(s); open(G, "w").write(g); print("patched", T, G)

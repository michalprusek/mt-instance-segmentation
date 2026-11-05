# XCOMP — crossing-gated X-completion: design, adversarial verification, and a self-predicted null

Produced 2026-09-08 by a 10-agent workflow: three independent designs (crossing-gated growth, the
matched-filter test itself, growth-as-pairing), each attacked twice — once by a reviewer told to
default to `fatal=true`, once by an implementation reviewer — then synthesised into one spec.

Written AFTER the day's diagnosis that on real data 97.1 % of crossing failures are a broken mask
and the assembler is innocent, and AFTER the multiplicity head was validated as a detector
(overlap AP 0.778 in domain / 0.332 on real) but measured worth only +0.013 strict F1 when
consumed inside the arc graph.

**Read the first section before anything else: the spec predicts its own pre-registered falsifier
fires with probability better than one in two, and says so unprompted.**

---

I have what I need — the pool numbers, the code facts, and advisor calibration. Writing the spec.

## XCOMP — X-completion by crossing-gated mask growth

### Bottom line, before anything else

**This will not move strict F1 past the 0.041 seed span, and it will not clear the paired same-dump resolution either (0.008 macro≥10 GT, 0.004 micro).** My honest chain gives deployed junction identity **+0.030 central (band +0.019 to +0.045)** — below the +0.04 bar every one of the three designs set for itself — and strict F1 **+0.001 to +0.005**. Worse, the one pre-registered falsifier below is predicted to land at +0.063 to +0.090 against a kill line of +0.08, so **it fires with probability better than one in two**. I am specifying it anyway, and Section 10 says what for: arms B/C/D on one cached dump are the only experiment that can test the claim this project could newly make — that crossing evidence buys a missing arm without paying the 17 px overshoot — and a clean null there is itself the publishable reading. Build Stage 0. Build Stage 1 only if Stage 0 survives.

---

### 1. The operation, in one paragraph

A **free arc end** of the predicted mask is grown along a curvature-bounded Hermite path onto the **centroid of an existing arity-3 junction**, converting that junction into a 4-arm X. Nothing else is ever grown: no T-completion, no arc-interior acceptor, no open-field extension, no end-to-end pairing. The growth is committed only when the image, read with a steerable second-derivative-of-Gaussian filter and scored against rotation decoys drawn in the same frame, beats a per-frame empirical null. This is the intersection of the three designs' strongest parts: XGROW's Tier-1 X-completion and density-capped damage budget, Design 3's two-anchor geometry with composite curvature and corridor guards, Design 2's decoy null and steerable filter.

**Why this specific operation and no other.** `_link_free_ends` builds `ArmEnd`s from *chains* and pairs chain ends to chain ends. A junction centroid is not a chain end and can never be a linker target. So free-end→junction-centroid is the one join the deployed pipeline structurally cannot make — verified in `src/instance/instancer_a.py:189-221`. It is also the join that produces the paying topology: a 4-arm X resolves at 0.993 on real data against 0.561 for 3-arm and 0.333 for the two-junctions-sharing-an-arc case.

---

### 2. The linker-redundancy objection (XGROW verdict 1, fatal) — repaired by the pool definition

The objection was that Tier-1 candidates are a subset of what `_link_free_ends` already attempts on a looser gate, so the credited baseline (broken-class identity 0.224) is inflated by selection.

**The repair is definitional, and it needs no separate pre-check.** The addressable pool is defined as *crossings where exactly one filament fails identity*. If the linker could join that filament's two stubs, it already did, and the filament's `identity` is `True` — so it is **not in the failure set**. The pool is linker-unreachable by construction.

One residual overlap survives and is accepted, not hidden: pairs the linker **attempted and rejected on geometric cost** (`c_open_link` 2.31856, `match_junction` includes a pair only when `cost < 2*c_open_link`). Growth may legitimately overrule those, because it reads image evidence the linker never does — `bridge_thr` 0.0712 fired 0 of 1286 cuts, so the linker's evidence term rejects nothing and its 91% closure is purely geometric. **Declared report:** of accepted growth events, the fraction whose donor and acceptor the baseline already placed in one chain. Above 0.30, the arm is re-doing the linker's work through a more expensive door and must be recorded as such.

---

### 3. The overshoot non-sequitur (Design 2 verdict 1, fatal) — repaired by requiring the completed topology

The objection was correct and lethal to the framing all three designs used: landing on a topological target says nothing about where the *annotator* stopped. A free end at an annotation end whose real filament continues faintly toward a neighbour 20 px away is licensed, passes, and paints a textbook overshoot.

**Repair, in three parts:**

1. **The acceptor must be an arity-≥3 junction whose existing arms already contain a through-pair.** Formally, at candidate time: among the acceptor junction's existing `ArmEnd`s there must exist a pair `(b1,b2)` with `pair_cost(b1,b2, gap_len=None, ...)` giving `kappa_implied <= 0.25` and `cost < 2*c_open` — filament B demonstrably passes through — **and** a remaining arm `a_near` with which the *donor, as it will exist after painting*, is likewise admissible. A filament terminus has no such structure waiting for it. This is the mechanism; the landing requirement alone was never one.
2. **T-completion is dropped entirely.** Arity-3 outcomes are the chimera generator (0.333/0.561 identity) and carried only 17% of XGROW's predicted value while requiring a 5× stricter bar. Cut, not deferred.
3. **Gate precision is measured, not asserted.** Every licensed candidate in all 139 VAL frames is enumerated; accepted paths are split **on-GT** (path within tol 5 px of an annotated centerline over ≥95% of its scored length) vs **off-GT**. **Declared failure rule: off/on > 0.2 fails the arm regardless of identity.** This is the number D2-1 correctly said no probe measured.

**Residual, accepted with stated cost:** a true crossing between an annotated and an *unannotated* filament. Growth there is physically correct, completes properly, and is charged at full price — the htw block already put 82% of excess precision on unannotated filaments. No gate can see this. It is measured as `off_other` and folded into the harm reads.

---

### 4. The both-filaments factor (Design 3 verdict 1, fatal) — accepted, and now inside the pool definition

`diag_crossing_map.py` sets `rec["identity"] = ok_all` with `ok_all &= ok` over **both** filaments. One repair fixes a crossing only when the partner is already carried. All three designs applied the leverage 0.898 to the whole missing-arm pool; only Design 3's `mask_connected` line halved, and it omitted the same factor from identity, its declared primary.

**Measured on the cached dump** (`data/enc_sensitivity_testset/crossing_diag_native/crossing_diag.json`, 6 frames, thr 0.98, n=1083, identity 0.2179):

| quantity | value |
|---|---|
| failures | 847 (0.782) |
| `fail_arm_missing` | 526 (0.621 of failures) |
| ... both filaments fail identity | 0.749 |
| ... both filaments not `found` | 0.451 |
| **... exactly one filament identity-OK (partner intact)** | **0.251** |
| P(a filament fails identity) | 0.6454 — independence predicts both-fail 0.4166, measured 0.5088, so failures are **correlated** and this fraction will not rise much at a better operating point |

**Accepted cost:** 74.9% of arm-missing failures need *two* successful repairs at one crossing and are out of scope for a single-event operator. Rather than carry an `r²` term, the pool is defined to exclude them.

---

### 5. The gate sited where the evidence is not (Design 3 verdict 2, fatal) — repaired

Design 3 read the multiplicity map in a 5 px disc **at the donor tip**, which is by construction 8–20 px from the crossing. The head fires where two centerlines are within ~2 px — a band `_absorb_crossing_bridges` itself puts at `~4*half_width/sin(alpha)`, i.e. 3–4 px at a steep crossing. The gate could not reach the evidence it gated on.

**Repair:** G2 reads `max(overlap)` in a 5 px disc **at the acceptor junction centroid**, plus `max(overlap)` along the candidate path. Never at the donor tip. `q` is fixed on `data/synth_eval_s35` (in-domain AP 0.778, exact multiplicity labels) at the component-level precision-0.80 operating point via `scripts/diag_overlap_ap.py`, and is **never re-tuned on real** — on real (AP 0.332) it is a floor, and the site-density cap does the work.

**Both gate readings are pre-declared**, because Design 3 declared only one:
- **B ≈ D** (ungated equals gated) → the crossing gate is decoration; publishable negative about the multiplicity head's usefulness.
- **B ≈ A while D ≠ A** → the gate is too strict and suppressed the arm; this is the silent failure Design 3 had no reading for.

---

### 6. Stage 0 — THE pre-registered falsifier

**One number, measurable on cached maps, no growth code, ~2 CPU-hours on tulen.**

`scripts/probe_xcomplete.py`, reusing `diag_crossing_map.py`'s crossing enumeration.

**Pool enumeration.** For each GT crossing on roi303 VAL against the **OVERLAP ep040** 2-channel dump (channel 0 = existence, thr 0.97), classify:
```
pool(crossing) := (exactly one filament fails identity)
              AND (that filament's failing side has a predicted free arc end
                   within 26.0 eval px of the crossing point)
              AND (a junction of arity >= 3 exists within 8.0 eval px of the crossing)
              AND (not crossing["connected"])
```

**The ceiling read.** For each pooled crossing, paint the **GT centerline of the failing filament** from that free end to the crossing point into the mask at `half_width=1.0` (GT used purely as a measuring stick, never as an input to the deployed method), re-run `instance_a` unchanged, and read junction identity **over all 2243 crossings** so the number is comparable to the baseline.

> **KILL IF Δ junction identity < +0.08**, paired on the same dump.

**Arithmetic behind the threshold.** The declared deployed bar is +0.04; the most optimistic pass rate for a marginal matched filter is 0.5; so a ceiling below +0.08 cannot pay for the build. **Predicted ceiling: +0.063 to +0.090.** Derivation: the X-completable class measures **156/1083 = 0.1440** on the cached dump ("exactly one filament fails identity, both `found`, mask disconnected"); scaling by 0.765 for the better operating point (failures fall from 0.782 to 0.598 of crossings) gives 0.110 × 2243 ≈ 247 crossings; discounting for the arity-≥3 and 26 px reach requirements gives **157–224**; ceiling = P × 0.898 / 2243 = **+0.063 to +0.090**.

**What it means if it fires.** The X-completable population, routed through *this* assembler, cannot pay for any local mask repair — because the repaint is `r = 1`, an upper bound no gate, filter or geometry can exceed. Post-hoc local repair of the probability map is then closed for this campaign, and the only remaining routes are the survey's extension formulation (17 px overshoot at full price) or the map itself. That is a real result, and it costs 2 CPU-hours.

**Census emitted free by the same pass** (reported, never a second falsifier): donor-tip-to-junction gap distribution in eval px; donor arc length distribution (against `min_length*0.5 = 22.3716`, the pre-`_link_free_ends` cull); acceptor junction arity histogram; per-side `before`/`after` coverage separately; same- vs opposite-polarity split.

---

### 7. Stage 1 — the operator

**New module `src/instance/growth.py`**, numpy/scipy/networkx only:

```python
grow_x_completions(mask, image_eval, prob, overlap, params, kappa_max)
    -> (mask_grown: np.ndarray[bool], events: list[dict], stats: dict)
```

Internals: `_free_ends(g)` · `_acceptor_junctions(g, overlap, params)` · `_hermite_pencil(p_a, t_a, p_b, t_b)` · `_steer_dog(image_eval, sigma_h)` · `_score_path(Ixx, Ixy, Iyy, path, sign)` · `_decoy_null(...)` · `_composite_kappa_ok(...)` · `_junction_admissible(...)` · `_corridor_ok(...)` · `_resolve(cands)` · `_paint(mask, paths)`.

**Hook.** `instance_a` gains one keyword `image=None`. At the top:
```
g = build_arc_graph(mask, ...)
if image is not None and overlap is not None and p["grow_alpha"] > 0:
    mask, events, stats = grow_x_completions(mask, image, prob, overlap, p, kappa_max)
    g = build_arc_graph(mask, ...)          # second skeletonisation: the point
```
Everything downstream is byte-for-byte unchanged. `image=None` reproduces today exactly — the same convention `channels`, `prob`, `overlap` and `linker` already follow, with the same bit-identity guard tests.

**Frame and filter placement.** Everything runs in the **eval frame** (`img_eval = zoom(img01, 1.5, order=1)`, 1/99-percentile normalisation, identical at every call site). Design 2 argued for the native image because the zoom adds noise correlation. That cost is real and is **charged by the decoy null rather than modelled** — the decoys are generated in the same zoomed frame with the same interpolation, so the added correlation raises the bar rather than inflating the statistic. In exchange, the MC power table transfers exactly (it assumed σ_f = 1.49 in the same units as σ_h) and no coordinate conversion can go wrong.

#### Constants (all eval-frame px)

| symbol | value | source |
|---|---|---|
| donor arc min length | 10.0 | shorter stubs have no usable tangent |
| donor tangent window | **12.0** | *not* the deployed 28.4061 — a 12 px stub has no 28 px window |
| growth heading | `window_tangent(poly, which, 12.0)[0] + pi` | the function returns the **inward** tangent |
| max path arclength `L_max` | 26.0 | 8–20 px missing arm + one step |
| min scored gap | **12.0** | 2σ_h trim each end leaves m ≥ 4 |
| max path/chord | 1.4 | no loops |
| path sampling `ds` | 1.0 | |
| Hermite pencil λ | {0.4, 0.6, 0.8, 1.0, 1.2} × chord | 5 members, max-over-family paid by the null |
| filter σ_h | **2.0** | measured MC: z = 3.24 (m=4) / 3.70 (m=7); σ_h=1.5 gives only 2.41/2.84 |
| filament σ_f | 1.49 | 3.5 px FWHM / 2.355 |
| trim | 2σ_h = 4.0 each end | removes parent-body and acceptor-body leakage |
| ridge tolerance | transverse max over {−1, 0, +1} | null takes the same max |
| lateral competition | v ∈ {−4…+4}; reject if argmax \|v\| ≥ 2 | bundle guard, applied to decoys identically |
| decoys | 16 per candidate, rotated about the donor tip by ±[45°, 135°] | same λ pencil, same trim, same m, same lateral test |
| length bins for τ | {12–16, 16–21, 21–26} | null scale depends on m; rotation preserves m exactly |
| α | expected decoy accepts ≤ **0.25/frame** | 139 chimeras ≈ −0.02 strict F1 = the whole predicted gain |
| acceptor gate disc | 5.0 at the **junction centroid** | never the donor tip |
| site area / density cap | ≥ 4 px component; **9 sites per 1e6 eval px**, top-K by peak multiplicity | 3× the measured 3.00/1e6 eval annotated crossing density |
| corridor radius | 2.5, excluding first/last 3.0 px and the 7 px crossing disc | parallel-bundle guard |
| contact orientation | ≥ 20° **at the contact point**, not only at the site | repairs XGROW-1's parallel-closure hole |
| redundancy | ≥ 60% of the **scored interior** (post-trim) outside the current mask | post-trim, so short gaps are not self-rejected |
| paint | disc dilation, `half_width = 1.0`, `oracle_instance_masks(..., up=1.0)` on the eval shape | |
| caps | one growth per donor free end, one per acceptor junction; global `networkx.max_weight_matching` | |

#### Hard rejects, in order, before any image is read

1. Chord < 12.0 or path arclength > 26.0 or path/chord > 1.4.
2. **Composite curvature:** donor's last 12 px + path + acceptor arm's first 12 px, resampled to `KAPPA_DS = 8.0`, `max_abs_curvature(...) <= 0.25`. Checking the path alone is not enough — a smooth curve can kink at both joins, and the joins are where a wrong pairing shows.
3. **Junction admissibility** (repairs D3-2's exempted acceptor): run `pair_cost` between the donor-as-painted and each existing `ArmEnd` at the acceptor junction **at the true span** (`gap_len` = actual arc distance, **not** the floored `gap_floor = 9.5808`), and require at least one with `kappa_implied <= 0.25` and `cost < 2*c_open`. A painted arm with no admissible continuation is a chimera generator, not a recovered arm. Separately require an existing through-pair `(b1, b2)` at the same junction (Section 3.1).
4. Corridor rule; redundancy rule; contact-orientation ≥ 20°.

#### The statistic

Three convolutions per frame give every orientation exactly by steering:
`r_theta = -(sin²t·Ixx − 2·sin t·cos t·Ixy + cos²t·Iyy)`, with `t` the path normal. A second derivative has zero DC and annihilates a linear ramp, so slow IRM illumination drift and the fitted heavy-tail dirt drop out by filter shape, not by an estimated background.

**Polarity** is anchored, not assumed: `sigma_0 = sign(median c(s))` over the donor's last 8 px, where the mask is confident; the statistic is one-sided on `sigma_0 * r`. Not `|sum|`, which folds the null and doubles its tail mass. Measured on the cached dump, `sign(contrast_near) != sign(contrast_far)` on the same arm in **13.4%** of not-found arms — **accepted as controlled false negatives**, which is the safe direction. Samples within ±3 px of a multiplicity site are zero-weighted: opposite-polarity filaments physically cancel at the overlap (identity 0.082 vs 0.233 same-polarity on this dump), so absent evidence there is predicted by the physics.

**Null.** No analytic null — lag-1 ACF +0.19 inflates a 20 px mean's variance by ~1.47 and kurtosis reaches 122. The null is the **procedure's own**: 16 rotation decoys per candidate about the same tip in the same frame, paying every maximisation the real test pays (λ pencil, transverse max, lateral test, trim, m). This is Design 2's construction and it **repairs XGROW verdict 2's objection** that background anchors at `prob < 0.05`, ≥ 6 px from any mask, sample cold empty field while growth happens in warm sub-threshold halo — a null so permissive that XGROW's "commit rate vs α×attempts" pre-check could not have failed. Standardise per frame by decoy median and 1.4826·MAD within the length bin, pool across VAL (~1e5 draws), read the empirical quantile.

**Accepted cost — the ceded band.** With a 4.0 px trim at each end, gaps of **8–12 eval px carry fewer than 4 scored samples and are handed to the existing linker untested**. The measured missing-arm window is 8–20 px, so a real fraction of the target population is ceded by construction. The census in Stage 0 reports the gap distribution; if it is dominated by < 12 px, `r` collapses and Stage 1 should not be built even on a passing ceiling.

---

### 8. Predicted effect, with the arithmetic

Baseline for the paired read is **OVERLAP ep040** (identity 0.4213, mask_connected 0.2675, broken 0.7325), **not** the SPARSE35 numbers 0.402/0.252 quoted in the brief — a different existence channel, and the confound is worth ~0.01 on its own. Deltas below apply to either.

Let `P` = pool size (157–224 crossings, central 190) and `r` = P(a licensed crossing is actually recovered) = gate recall × filter pass × geometry pass ≈ **0.3–0.5, central 0.4** (from the corrected MC at σ_h = 2.0: z = 3.24 at m = 4, 3.70 at m = 7, against τ ≈ 3.0–3.5).

**Junction identity.** Ceiling P × 0.898 / 2243 = +0.063 to +0.090; deployed × r:
```
+0.063 x 0.3 = +0.019      +0.090 x 0.5 = +0.045      central 0.076 x 0.4 = +0.030
```
**0.402 → ~0.432** (0.4213 → ~0.451). **Below the +0.04 bar all three designs set.** Instrument: McNemar on 2243 paired binary outcomes, clustered by the 139 frames, not by crossing. Second mechanism report: the arity histogram at gated junctions before/after — the fraction reaching arity 4 is what says whether growth built an X or a blob.

**mask_connected.** In the pool the partner is intact and therefore connected, so one repair flips the crossing's flag: P × r / 2243 = 157×0.3/2243 to 224×0.5/2243 = **+0.021 to +0.050, central +0.034**. **0.252 → ~0.286.** Requires re-running `diag_crossing_map.py` on the *grown* mask (one-line change). Reported with the caveat that it is partly gameable — closing a 1 px void raises it without changing topology.

**Strict F1.** Measured identity→F1 conversion in this project is ~0.03 macro per identity unit (threshold drop: +0.093 identity → +0.003 macro; OVERLAP ep040: +0.0196 identity → −0.0013 matched macro). Growth adds ~30× less foreground than a threshold drop and adds true centerline *length*, which is what the 95%-coverage rule charges for, so credit 1–5× the naive rate:
```
+0.030 x 0.03 = +0.0009      +0.030 x 0.15 = +0.0045
```
**VAL 0.520 → 0.521–0.525.** At 0.80/0.80, +0.002 to +0.010. **An order below the 0.041 seed span**, readable only as a per-frame paired Wilcoxon over 139 frames on one cached dump. Declared as a **non-regression check, not the read.**

**Harm channels, predicted before the read.** overshoot FPs +0 to +2%; `off_other` +1 to +3%; chimeras +2 to +5%. **Declared failure rule: if the FP rise exceeds the TP rise, the arm fails regardless of identity.** Plus the off-GT/on-GT gate-precision rule from Section 3.3.

---

### 9. The read

Four arms, one cached OVERLAP ep040 dump, paired per frame, no seed noise:

| arm | configuration |
|---|---|
| **A** | baseline: thr 0.97, deployed v5H params, unchanged linker |
| **B** | + growth |
| **C** | + per-frame threshold drop **matched on added foreground to B** — mandatory, not optional; without it the read repeats the arm-29 confound and rediscovers the already-deployed +0.093 |
| **D** | + growth with G2 removed (ungated) — tests whether the crossing gate is load-bearing |

**Void check first:** if B adds zero foreground, or no site survives the density cap on any frame, the read is **VOID, not negative**. Report `n_sites` vs `K` per frame.

**Success criterion, pre-declared:** identity beats **arm C** by ≥ +0.04 on paired crossings, **AND** strict F1 is non-inferior to A by more than −0.010, **AND** off-GT/on-GT ≤ 0.2, **AND** FP rise ≤ TP rise. Beating arm A alone on identity is not a result — it is the deployed threshold effect rediscovered.

---

### 10. Plumbing corrections — two verdicts got these wrong

- **The cached `crossing_diag.json` is 6 frames, thr 0.98, n = 1083, identity 0.2179, mask_connected 0.1727** — not the 139-frame read-35 instrument. Every split I report from it is **directional, not exact**. It also **lacks `d_junction`, `d_free_end`, `topo`, `mask_on`, `n_arms_junction`**, so Design 3 verdict 2's claim that these "ARE already recorded" is false for this dump, and both verdicts' "zero-cost pre-check on the cached dump" does not exist as described.
- `diag_crossing_map.py` must additionally record the polyline index and arc position (`c["i"]`, `c["s_i"]` are in scope), **`before` and `after` separately** rather than only `found = before or after`, and `d_free_end` of the failing filament's own component. Then run against the OVERLAP dump on tulen.
- **`scripts/crossing_report.py:82` does `np.load(npz)["prob"].max(axis=0)` with no `--channel` flag.** On a 2-channel `--head overlap` dump this silently maxes existence with multiplicity — the identical bug already caught once in `validate()`. It must gain `--channel` before it is used as the junction-identity instrument. (`diag_crossing_map.py` already has the flag.)
- **`eval_v5.py`'s cache tuple is `(ch, keep, gt)` — no image on the `--pred-dir` path.** Add a 4th element. "Three call sites already hold `fr["image"]` and zoom by 1.5" is **one**, not three: only `diag_crossing_map.py` builds `img_eval`; `crossing_report.py` never touches `fr["image"]`.
- `window_tangent` returns the tangent pointing **into** the body — grow along θ+π. A sign error here grows silently back along the parent.
- `oracle_instance_masks(polylines, shape, half_width, up)` multiplies polylines by `up` and takes the **native** shape; growth works in the eval frame, so pass `up=1.0` with the eval shape (the precedent is `instancer_a.py:377`).
- `w_ori = 2.6027` is **inert as deployed**: with `channels=prob[None]` the ori profile is length-1 and `_ori_mismatch`'s cosine is identically 1. Do not attribute any change to it.

**Unit tests required:** bit-identical output when `image` or `overlap` is None (mirroring the existing `overlap=None` tests); `kappa_max` never violated on the composite over randomised donor/acceptor pairs; the corridor rule rejects a stroke painted beside a 3 px parallel; a synthetic X with one arm erased is completed; a straight filament with no acceptor does **not** grow; the `(x=col, y=row)` convention gets its own regression test — that convention has caused two silent bugs in this repo already.

---

### 11. Cost

**Stage 0:** ~150 lines reusing `diag_crossing_map.py`'s enumeration, ~4 h to write, ~2 CPU-hours to run, **0 GPU-hours**. Nothing else is built until the +0.08 ceiling clears.

**Stage 1 (only if it clears):** `src/instance/growth.py` ~350 lines; one kwarg + 6-line hook in `instance_a`; `image=` plumbing at 2 call sites; 6 unit tests. ~2 CPU-days including review. Runtime: 3 convolutions + 2 skeletonisations + ~50 candidates × 5 λ × 17 decoy-and-real paths × ~20 samples × 9 lateral offsets per frame ≈ 5–10 s/frame; 4 arms × 139 frames ≈ 1 h wall on tulen's 32 cores. Needs the 2-channel OVERLAP ep040 dump on roi303 VAL — no new inference, no retraining.

**Prerequisite not costed here:** `q` calibrated on `data/synth_eval_s35` for component-level site precision 0.80 (~1 CPU-hour).

---

### 12. What this is worth doing for, given the honest prediction

1. **It is the only experiment that tests the project's new claim.** Arms B vs D on one dump measure whether crossing evidence buys a missing arm without paying the 17 px overshoot. A clean null on D vs B — the gate is decoration — is publishable as a bound on what the multiplicity head is good for, and it is not obtainable any other way.
2. **The Stage 0 ceiling converts "the assembler is innocent" from an oracle-*mask* statement into a bound on what any local mask repair can deliver through it.** Today's 0.835 oracle-mask figure conflates "topology repair helps" with "a perfect map helps". The repaint separates them for 2 CPU-hours.
3. **If the falsifier fires — which I predict at better than even odds — it closes the post-hoc local-repair family** and points the effort back where the measurement already says the gap lives: the map. That is worth 2 CPU-hours to learn cheaply rather than after 2 CPU-days of building.

**Weakest point, stated plainly.** The pool is thin and it is thin for a structural reason: the multiplicity head reads the same image evidence whose absence broke the mask, and at opposite-polarity crossings — the worst class — the two filaments physically cancel exactly where the damage is worst. Every number above scales linearly in `P` and `r`, and `P` is measured only on a 6-frame dump at a weaker operating point. Stage 0 exists to find that out for two CPU-hours instead of two CPU-days.
# Making trace cells emerge: a metric, and three task paradigms

**Started:** 2026-08-11 · **Branch:** `sdu/object-into-hidden-state`
**Goal:** produce object-trace cells in the pRNN hidden state — units with a place field at a
former object's location (`trace-cells-spatial-tuning.png`). Prior work established they do
**not** appear under the standard OMT paradigm; this document tests task paradigms designed to
create the demand that would produce them.

Prerequisites: [`README_object_experiments.md`](README_object_experiments.md) for the index,
[`compaction.md`](compaction.md) for the condensed state.

---

## 1. Why the standard paradigm cannot produce them

From `exp_object_trace_cells_2026-07-30.md` and `exp_object_into_hidden_state_2026-08-01.md`:

- Object memory is **decoder-localised** — `W_out`, not the recurrent dynamics (readout
  +0.0625 vs dynamics +0.0157 gain-corrected, 9/9 runs).
- The place code is **unchanged** by exposure (r ≈ 0.98) in **three** reference frames.
- **Root cause — redundancy.** The object sits at a fixed cell, so its presence is a
  deterministic function of the agent's position, and `h` already encodes position. Nothing
  needs to be remembered, and adjusting a linear readout is the cheaper gradient direction.
- Freezing `W_out` and scaling `W_in` (`lr_trials=[2,0,8]`) raises *encoding* in `h`
  (0.6947 → 0.7214) but not *memory*: information decays ~2× per masked step.

**Design principle for everything below:** make the object's past presence informative about a
future observation that **position alone does not predict**.

---

## 2. The metric (deliverable 1)

`scripts/trace_metric.py`.

### 2.1 Definition

For unit `u` and candidate cell `c`:

```
field_gain   g(u,c)  = mean rate in a radius-2 disc at c  /  unit's mean rate over valid bins
trace score  dg(u,c) = g_post(u,c) − g_pre(u,c)
```

`g` is **scale-free**: a unit whose rate doubles everywhere scores zero change. This is the
"change in receptive-field strength at the object location" the goal asks for.

### 2.2 Within-unit null

`dg(u,c)` is evaluated at **all 172 walkable cells**; the unit's score is the object cell's
**percentile within its own distribution**. This controls for global rate change, for the
unit's own map structure, and partly for location-specific drift.

A unit is an **object cell** when that percentile > 95. Under the null the expected fraction
is 5%, so the population test is binomial against 0.05.

### 2.3 Location-control matrix — the decisive test

The within-unit null **cannot** see drift shared *across* units at one location. That is
precisely what produced the spurious `(14,7)` result in the earlier work. So score every
candidate location under every run:

```
excess(L) = frac(L | object at L) − mean over M≠L of frac(L | object at M)
```

Real object coding is positive; pure location drift cancels.

### 2.4 Validation — report with every result

| control | method | result |
|---|---|---|
| **negative** | odd vs even probe trajectories of the *same* net (two independent map estimates) | frac = **0.0601**, p = 0.174 ✓ |
| **positive** | inject a Gaussian bump of known amplitude into 10% of units | **100% recall at amplitude 0.01** (1% of a unit's mean rate) |

Scoring a map against *itself* is degenerate (`dg ≡ 0`) and is not a null — the negative
control must use two independent estimates.

The positive control is what converts a null result into a bound: **an effect as small as a
field 1% of mean rate in 10% of units would have been detected.**

### 2.5 Result on the pre-existing data — no object cells

```
                          scored at:
run with object at:   (7,11)    (14,7)     (7,2)
        (7,11)        0.0321    0.0902    0.0200
        (14,7)        0.0140    0.1002    0.0501
         (7,2)        0.0581    0.0721    0.0341

excess:               −0.004    +0.019    −0.001
```

Column `(14,7)` is elevated **regardless of where the object was** — drift, not coding. The
metric removes the artifact that fooled the earlier object-modulation analysis, and returns
**no object cells** in any condition.

---

## 3. Scenario A — object identity varies, location fixed

**Rationale.** Four colours (`green/blue/purple/yellow`) re-rolled per **episode** at a fixed
cell (7,11). Position still says "something is here" but no longer says **which**. Predicting
the colour therefore requires remembering having seen it — a genuine within-episode memory
demand — while keeping the allocentric location structure a trace cell needs. This is also the
closest analogue here to the actual NOR paradigm (familiar vs novel object).

**Implementation.** `tasks.otc.colors=[...]` in `tasks/otc/`. The colour is sampled per env
per batch, and each env runs one 256-step episode per batch, so identity is **fixed within an
episode** and varies across them. Toggling goes through `new_obj_color` on the environment;
the banked observation wrapper re-keys off the grid fingerprint at `reset()`, so each colour
gets its own cached bank (verified: 4 distinct fingerprints).

**Runs.** 2 arms × 3 seeds × 3000 trajectories: `A_norm` (`lr_trials=[2,2,2]`) and `A_frz`
(`[2,0,8]`). wandb project `curious-george-otc`.

### Results — NULL

Fraction of units called object cells at (7,11), final checkpoint, scored against the
pre-exposure baseline:

| condition | n | frac object cells | p_binom |
|---|---:|---|---:|
| A identity, `lr_trials=[2,2,2]` | 3 | 0.0294 ± 0.0105 | 0.756 |
| A identity, `lr_trials=[2,0,8]` | 3 | 0.0180 ± 0.0057 | 1.000 |
| ref: fixed green, `[2,2,2]` | 1 | 0.0281 | 0.994 |
| ref: fixed green, `[2,0,8]` | 1 | 0.0401 | 0.871 |

Null expectation 0.05; measured negative control 0.0601. **Every condition is at or below
chance** — randomising object identity produces no object cells, and does not even differ from
the fixed-green reference.

Given the positive control (100% recall at a field 1% of a unit's mean rate in 10% of units),
this is a real bound, not an absence of power: **an effect that small would have been seen.**

Worth noting the fractions sit slightly *below* 0.05 rather than at it. With a within-unit
percentile null the object cell is one of 172 candidates, so ties and the discreteness of the
percentile push the realised rate a little under nominal; the negative control (0.0601)
brackets it from the other side. Neither direction is significant.

---

## 4. Scenario D — sparsity pressure on the weight change

**Rationale.** The change `[2,0,8]` induces is **distributed** (`W` moves ~13% globally)
rather than localised, which is why no single unit looks like an object cell. Biological place
fields are sparse. Pressuring the exposure-phase weight change to be small should force it to
be concentrated where it actually reduces loss.

**Implementation note — an important dead end.** `predNet.sparsity` (`f`) is **init-only**:
it is consumed as `norm.ppf(f)` to set the bias at construction, so it has no effect on a
loaded checkpoint. The available lever is therefore **per-param-group weight decay**, scaled
during exposure exactly as `lr_trials` scales learning rate — added as `tasks.otc.wd_trials`.

This is an L2 prior, not L1, so it pressures the change to be *small* rather than *sparse*.
That distinction is real and limits how much this scenario can be expected to do; it is the
closest available lever without modifying the pRNN.

**Runs.** `wd_trials=[20,0,20]` and `[100,0,100]` on top of `lr_trials=[2,0,8]`.
**Reduced to n=1 per strength (a go/no-go screen)** because of the overnight time budget —
see `compaction.md`. Justified by A being decisively null at n=3 and by D being the weaker
lever; seeds get added only if the screen shows anything.

### Results — NULL, but the lever did not do what the scenario requires

| condition | frac object cells | p_binom | dW | dW_in |
|---|---:|---:|---:|---:|
| wd ×20, `[2,0,8]` | 0.0140 | 1.000 | 0.181 | 0.397 |
| wd ×100, `[2,0,8]` | 0.0240 | 0.999 | 0.464 | 0.804 |
| ref: no wd change | 0.0401 | 0.871 | 0.165 | 0.742 |

Null 0.05; negative control 0.0601. No object cells. But the diagnostic matters more than the
score:

| condition | map corr vs baseline | frac units silent | mean \|h\| |
|---|---:|---:|---:|
| baseline | 1.0000 | 0.002 | 0.2469 |
| wd ×20 | **0.7349** | 0.002 | 0.2124 |
| wd ×100 | **0.7395** | 0.002 | 0.1702 |
| ref | 0.9741 | 0.002 | 0.2353 |

**`dW` *grows* with weight decay** (0.165 → 0.181 → 0.464), the place code collapses from
r = 0.97 to r = 0.73, mean activity falls, and the silent-unit fraction does not move at all.
L2 decay pulls every weight toward zero — a large change *away* from the trained solution —
rather than concentrating the change where it helps.

**So D is not a clean negative: it is an untestable hypothesis with the available lever.**
"Sparsity pressure on `h`" needs an L1 penalty or an activity regulariser on the hidden state.
`predNet.sparsity` (`f`) is init-only (`norm.ppf(f)` sets the bias at construction) so it
cannot be applied to a loaded checkpoint, and adding an L1 term means changing the pRNN, which
this work is constrained not to do. **The sparsity hypothesis remains open.**

---

## 5. Scenario C — occlusion (only if A and D fail)

**Rationale.** The remaining reason the network never needs memory: with
`see_through_walls=True` and a non-blocking floor tile, the object is either visible (and thus
in the input) or reconstructible from position. A **blocking** `Ball`/`Box` with
`see_through_walls=False` makes the object genuinely leave view, so holding it is the only way
to predict what is behind the wall.

**Two implementation findings that shaped the design.**

1. **A blocking object does not occlude.** `Ball` and `Box` have `can_overlap() = False` but
   `see_behind() = True`, and `gen_obs_grid` only runs `process_vis` when
   `see_through_walls` is False. So occlusion has to come from the room's own wall, which is
   why C needs its own baseline.
2. **The observation bank would have silently corrupted C.** It was keyed on `grid.encode()`
   alone, which does not capture occlusion — it would have served non-occluded observations
   for an occluded env. The fingerprint now gains an `-occl` suffix when occlusion is on,
   appended only in that case so all previously cached banks keep their filenames.

**Choosing the object cell — and a bug caught before launch.** C was initially armed with
(13,10), which is **inside the wall block and not walkable**. Replaced after measuring, for
each candidate, how often the object itself is hidden by the wall:

| cell | viewpoints seeing it, no occlusion | with occlusion | hidden |
|---|---:|---:|---:|
| (12,7) | 171 | 149 | 12.9% |
| (13,7) | 150 | 128 | 14.7% |
| **(14,7)** | **129** | **107** | **17.1%** |
| (13,5) | 144 | 136 | 5.6% |
| (7,11) | 178 | 168 | 5.6% |
| (12,3), (7,2), (3,3) | 150 | 150 | 0.0% |

C uses **(14,7)**. Note the honest limit: even the best cell is hidden only **17%** of the
time it would otherwise be seen. The L-room has a single interior wall, so it cannot hide much
— this bounds how strong a memory demand C can create in this environment.

**Baseline.** Measured 0.52 s/trajectory, so the original 79,679-trajectory baseline would
take 11.6 h. The occluded baseline is capped at 3.5 h (~24k trajectories) to leave room for
the exposure runs. This does not confound the metric, which is a within-lineage pre-vs-post
comparison — but the shorter baseline **must be checked for a usable place code** before any
C result is trusted.

### Results — the first positive signal, but underpowered at n=3

**Gate first: does the shortened occluded baseline have a usable place code?** Yes.
31,199 trajectories (vs 79,679 for the original non-occluded baseline):

| | occluded baseline (31k) | original baseline (80k) |
|---|---|---|
| SI median / max | 0.690 / 2.069 | 0.759 / 2.541 |
| silent units | 1/500 | 1/500 |
| valid bins | 172/196 | 172/196 |

Slightly weaker as expected from a third of the training, but clearly a place code.

**Metric re-validated on the occluded probe** (a new probe had to be built — the existing one
is non-occluded): negative control **0.0421, p = 0.819**; positive control 100% recall from
amplitude 0.01.

**Result — object at (14,7), 3 seeds:**

| seed | frac object cells | p_binom | map corr vs baseline |
|---|---:|---:|---:|
| 5200 | 0.0561 | 0.293 | 0.9762 |
| 5201 | **0.1423** | **<0.001** | 0.9698 |
| 5202 | **0.0982** | **<0.001** | 0.9656 |
| **mean** | **0.0989 ± 0.0352** | | |

**Location control — the effect is specific to the object's cell:**

| | **(14,7)** | (12,7) | (13,5) | (7,11) | (7,2) | (3,3) | (2,5) | (4,7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean frac | **0.0989** | 0.0441 | 0.0387 | 0.0307 | 0.0261 | 0.0301 | 0.0347 | 0.0354 |

Object location **0.0989** vs **0.0343** over seven controls — **excess +0.0646**, about 3×,
against a measured negative control of 0.0421. The second-highest cell is (12,7), the
neighbouring occluded cell, which is what a spatially localised effect should look like.

### Result at n=8 — SIGNIFICANT, and spatially graded

| | **(14,7)** | (12,7) | (13,5) | (7,11) | (7,2) | (3,3) | (2,5) | (4,7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean frac (n=8) | **0.0892** | 0.0433 | 0.0463 | 0.0248 | 0.0293 | 0.0293 | 0.0293 | 0.0276 |

```
object location            0.0892
mean of 7 controls         0.0329
EXCESS                    +0.0563
paired t across seeds      t = 5.55,  p = 0.001
(14,7) is the highest column in 8 of 8 seeds  -> (1/8)^8 ~ 6e-8 by chance
```

**The effect is spatially graded**: 0.089 at the object, ~0.045 at the two cells about 2 bins
away, ~0.028 at the five distant cells — which sits at the measured negative control (0.0421)
and below nominal chance (0.05). That gradient is the shape a genuine place-field-like effect
should have, and no previous condition in this project produced one.

![scenario C gradient](../outputs/trace/fig_scenarioC_gradient.png)

### The remaining confound, and the control for it

(12,7) and (13,5) sit in the **same occluded region** as (14,7), so the gradient could reflect
"the occluded region changes more" rather than "the object's location changes more". This is
the same trap that produced the spurious (14,7) result in the non-occluded analysis, so it
must be closed before the effect is claimed.

**Control queued:** occluded exposure with the object at **(12,7)** instead, 3 seeds. If the
effect is object-driven the peak must move to (12,7); if it is regional drift, (14,7) stays
highest regardless. This is `location_control_matrix` applied within the occluded env.

---

## 6. Sequential displacement — the Tsao/Moser paradigm

**Design.** `(7,11) → (7,2) → (4,7) → REMOVED`, 1000 trajectories per phase, checkpoints at
every phase boundary, **n=8 seeds run in parallel on Mila**. (14,7) is deliberately excluded —
it has produced spurious effects in three independent analyses.

Must be **across** episodes, not within: the pRNN state resets every `seqdur=256` steps, so a
cross-phase trace can only live in the weights — which is also where the biological
across-trial trace lives.

Checkpoint provenance is asserted in-job: `slurm/otc_seq.sh` pins `CUR_CKPT_DIR` and fails on
a sha256 mismatch against `c1e43a6b…`, the exact baseline every earlier result used. (Mila's
`.env` carries a *relative* path that resolves inside `$SLURM_TMPDIR` and does not exist
there — the first launch of these 8 jobs would have crashed on it.)

### Result — complete null

```
                   (7,11)          (7,2)          (4,7)
ph0 obj(7,11)   0.0283±0.015   0.0446±0.019   0.0346±0.016
ph1 obj(7,2)    0.0243±0.008   0.0386±0.020   0.0338±0.017
ph2 obj(4,7)    0.0230±0.008   0.0378±0.011   0.0441±0.012
ph3 REMOVED     0.0276±0.007   0.0371±0.015   0.0316±0.017
```

**Nothing forms during exposure.** Each location during its own phase: 0.0283 / 0.0386 /
0.0441 — all *below* the 0.05 null, (7,11) significantly so (t=−3.81, p=0.0066). With no
field forming there is nothing to trace, which settles the question by itself.

**No trace.** During-minus-after: +0.0033 (p=0.61), +0.0011 (p=0.89), +0.0125 (p=0.23).

**No leakage.** (4,7) before the object arrives 0.0342 vs during 0.0441, **p=0.14**. The
shared-readout generalisation inferred earlier from the contaminated (14,7) row does **not**
appear on a clean cell; that claim is withdrawn.

### What the visualisation shows

`scripts/seq_figures.py` produces three panels. The interpretable one is the **object-centred**
prediction map: the green change averaged over every in-view timestep, with each 7×7 patch
rolled so the object's view cell sits at the centre. An object-locked effect appears at the
centre; viewpoint-specific noise averages away.

On the *old* 3-phase run it showed a tight centred bump that decayed after the object left —
i.e. a real trace in the **readout**. Two of those three rows were clean; the (14,7) row was
not, and the leakage read off it is retracted per the n=8 result above.

The `fields` panel shows units **do** reorganise their receptive fields substantially across
phases — but not at the object. That is the null made visible.

### The readout side — transfer, not trace (n=8)

Green change at the object's own view cell, averaged over every in-view timestep with each
7x7 patch rolled so the object sits at the centre (`scripts/seq_figures.py`, `predcentred`):

```
phase              (7,11)           (7,2)           (4,7)
ph0 obj(7,11)   +0.0192±0.029   +0.0449±0.024   +0.0520±0.029
ph1 obj(7,2)    +0.0051±0.017   +0.0490±0.011   +0.0270±0.010
ph2 obj(4,7)    -0.0096±0.022   +0.0209±0.009   +0.0445±0.008
ph3 REMOVED     -0.0151±0.018   +0.0183±0.009   +0.0205±0.009
```

**It does not persist.** During its own phase vs after the object leaves:
(7,2) 0.0490 → 0.0196, drop 0.0295, **p=0.0007**; (4,7) 0.0445 → 0.0205, drop 0.0239,
**p=0.0043**; (7,11) 0.0192 → −0.0066, p=0.081. **Transfer, not trace.**

**And the readout generalises almost completely across locations.** (4,7) before the object
ever arrives: **+0.0395 ± 0.011 (t=9.57, p<1e-4)**, against +0.0445 while it is there — so
**~89% of the apparent object signal at a location comes from exposure somewhere else.**

`W_out` has one row per view cell applied to every hidden state, so it learns "boost green at
this object-centred view offset", not "there is an object at (4,7)". That is the mechanism
behind the whole project: nothing location-specific has to be learned, so the hidden state
never changes, while the pixel-space metric always finds something.

Note this leakage is in the **readout**; the hidden-state leakage test was null (p=0.14).
Two different measurements of two different things — do not conflate them.

## 7. Standing methodological cautions

Carried from the earlier work; all were paid for at least once.

1. **Pooling hides effects.** Three separate times, averaging across the variable the mechanism
   runs on turned a real effect into a flat null. Always split by mask phase, by viewing
   condition, and by location before concluding anything.
2. **Never quote an effect size from n=1.** A "+142%" result became non-significant at n=10.
3. **Check a job has finished** before reading `max(steps)` — reading step 0 of a running job
   once produced a fabricated "catastrophic" result.
4. **Implausibly exact agreement is a bug signal**, not a finding.

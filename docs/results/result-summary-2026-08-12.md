# Object and trace cells in the pRNN: what we found, and what each figure shows

**Written:** 2026-08-12 · **Branch:** `sdu/object-into-hidden-state`
**Scope:** the whole object/trace-cell investigation, 2026-07-30 → 2026-08-12.

**The goal was** to produce *object-trace cells* in the pRNN hidden state `h` — units that grow a
place field where a novel object used to be, as in Tsao, Moser & Moser 2013
([`ref-trace-cells.png`](../ref-trace-cells.png), doi 10.1016/j.cub.2013.01.036).

**They do not appear.** Ten designs have now been run and every hidden-state result is null. This
document explains the mechanism that predicts all of them, and links every claim to a figure.

### How to read the evidence markers

| marker | meaning |
|---|---|
| **[cache]** | I re-derived the number in this session from the named cache. Confirmed. |
| **[doc]** | Recorded in a repo doc; I did not re-derive it (no cache survives, or it needs GPU). |
| **[live]** | Measured in this session from wandb or from source. Confirmed. |
| ⚠️ | I could not verify it, or it disagrees with what I measured. Details in §7. |

Figures under `outputs/trace/` and `outputs/moser/` were regenerated in this session from their
caches and all open. Figures under `outputs/summary/` are new
(`scripts/summary_figures.py`). Every figure here either **compares against a
baseline/control** or **shows spatial tuning**; none is a bare time series.

---

## Section 0 — The mechanism: the object lives in `W_out`, position-independently

**Lead with this. It predicts every other result in this document.**

### Question

If the object is demonstrably learned — the network's pixel predictions change — why does no
hidden-state measurement ever see it?

### Why this design

The decisive measurement is **where** the change lives, not **how big** it is. Two designs make
that separable:

- A **gain-corrected chimaera swap**. Build networks from `base dynamics + trained W_out` and the
  reverse, and see which reproduces the effect. Gain correction is not optional: `‖h‖` is ~9%
  smaller in trained nets while `‖W_out‖` is ~2% larger, so a raw transplant over-drives one
  chimaera and under-drives the other — **both errors biasing toward "the readout carries it"**
  ([doc], `exp_object_trace_cells_2026-07-30.md` §3.3). Correcting it moved the headline from
  6.6× to 4.0×.
- **Scoring a cell before the object ever goes there.** This is the design that identifies the
  mechanism rather than just its location, and it only became possible with the sequential-
  displacement runs (§3), where (4,7) is used late.

### Method

Sequential displacement `(7,11) → (7,2) → (4,7) → REMOVED`, n=8 seeds. At each phase, measure the
green-channel prediction change at each candidate cell's own view cell, averaged over every
in-view timestep, with each 7×7 patch rolled so the object sits at the patch centre
(`scripts/seq_figures.py`, `predcentred` mode). Caches: `outputs/trace/seq4_readout.npy`,
`outputs/trace/seq4_matrix.npy`.

### Result

![sequential hidden vs readout](../../outputs/summary/fig_seq_hidden_vs_readout.png)

**Figure 0.1** — `outputs/summary/fig_seq_hidden_vs_readout.png`. Same runs, two measurements.
Left: hidden state, every cell below the 0.05 chance rate. Right: readout, and the dashed box is
the mechanism.

The readout at (4,7) is already elevated **before the object has ever been placed there**:

```
(4,7) readout, mean of ph0+ph1, BEFORE the object arrives:  +0.0395   t=9.57, p=2.9e-5  [cache]
(4,7) readout at ph2, WHILE the object is there:            +0.0445                     [cache]
```

So **~89% of the apparent "object signal" at a location is generalisation from exposure
somewhere else.** `W_out` is `Linear(500 → 147, no bias) → Sigmoid` — one row per view cell,
applied to every hidden state — so what it learns is "boost green at this object-centred view
offset", not "there is an object at (4,7)".

Supporting facts:

- Object memory is decoder-localised: readout **+0.0625 ± 0.0130** vs dynamics
  **+0.0157 ± 0.0104**, gain-corrected, readout > dynamics in **9/9** runs, ratio 4.0×
  ([doc], `exp_object_trace_cells_2026-07-30.md` §4.1).
- The place code is unchanged by exposure: median per-unit rate-map correlation **r ≈ 0.98**
  from 400 trajectories onward, across 3 object locations ([doc], same doc §4.2).

![rate-map difference under frozen-Wout training](../../outputs/trace/fig_otc_maps_diff.png)

**Figure 0.2** — `outputs/trace/fig_otc_maps_diff.png`. Spatial tuning, as a control on Figure
0.1. Rate-map difference for the eight units the object decoder weights most heavily, `[2,0,8]`
minus normal training. The `+` marks the object cell (7,11). The differences are real and
substantial — and they are **everywhere except the object**.

### Interpretation

The object's location is already linearly decodable from `h`, because `h` is a place code and the
object is a deterministic function of position. Gradient descent has no reason to touch the
recurrent weights: adjusting a linear map is the cheaper direction. `h` is **redundant** with the
object, not ignorant of it. Consequences, which are exactly what the rest of this document
observes:

- nothing location-specific has to be learned → **every hidden-state metric is null**;
- the readout does change → **the pixel-space metric always finds something**;
- the readout tracks the *present* object and collapses when it moves → **transfer, not trace**.

---

## Section 1 — L-room + novel object: the baseline OMT

### Question

After novel-object exposure at a fixed cell, does `h` hold a spatial trace at the object's
location?

### Why this design

Three object locations × 3 seeds rather than one, because the L-room's geometry is not uniform
and a single location cannot separate "object effect" from "this cell is special". That choice
paid for itself: **(14,7) turned out to be a geometry artifact** and would have been reported as
a positive from a one-location design.

The scoring uses a **within-run null** — the same statistic at all 172 walkable cells, reporting
the object cell's percentile — rather than pooling seeds. Pooling manufactures apparent
clustering when the object sits in a structurally high-traffic spot
([doc], `exp_object_trace_cells_2026-07-30.md` §3.4).

### Method

`tasks/omt/`, object at `(7,11)`, `(14,7)`, `(7,2)`, 3 seeds each, 3000 trajectories, checkpoints
every 200. Probe: one trajectory from every (walkable cell, head direction) pair = **688
trajectories × 256 steps**, ~828 samples per spatial bin, maps 14×14 occupancy-masked to 172/196
valid bins. A `RandomActionAgent` generates actions independently of the network, so the same
trajectories are valid for every checkpoint — checkpoints are **replayed, not re-collected**, and
any map difference is in the weights. Determinism gate: replaying one checkpoint twice is
bitwise identical.

### Result

![place fields, baseline](../../outputs/trace/fig_tuned_units_main_train.png)

**Figure 1.1** — `outputs/trace/fig_tuned_units_main_train.png`. The positive control for the
whole method: the pRNN has clean, well-localised place fields. Any null below is a real bound,
not a broken measurement.

![trace across 3 locations](../../outputs/trace/fig_trace_3loc.png)

**Figure 1.2** — `outputs/trace/fig_trace_3loc.png`. **The main null, as spatial tuning.** Rows
are the units most modulated at (7,11); columns are pre-exposure and then three separate exposure
runs. The white `+` marks (7,11). Nothing appears there, in any column. Shared colour scale per
row.

![exposure timeline](../../outputs/trace/fig_exposure_timeline_14_7.png)

**Figure 1.3** — `outputs/trace/fig_exposure_timeline_14_7.png`. The same null across 16
checkpoints of exposure, so it is not a matter of having stopped too early.

Numbers, all [doc] from `exp_object_trace_cells_2026-07-30.md` §4.3:

- Allocentric Δ object-modulation, own location **0.021** vs off-diagonal **0.028** — the wrong
  way round.
- Within-run percentile of the object cell oscillates 10.5–79.1 across checkpoints, **never
  sustained above 95**.
- Object-vector frame: the object's own SI change is **+0.0005, +0.0001, −0.0174**.

Null in **three** reference frames: allocentric maps, object-location modulation, object-vector.

![behaviour](../../outputs/trace/fig_behavior_n3.png)

**Figure 1.4** — `outputs/trace/fig_behavior_n3.png`. The control that shows the exposure *did*
something. Behaviour is object-directed at (7,11) (+45.7 percentile) and (7,2) (+38.7, 3/3
seeds), and **not** at (14,7) (−33.7), whose baseline percentile was already 69 *before* exposure
([doc] §4.4). The agent approaches the object; its hidden state does not change.

### Interpretation

Exposure writes the object into the linear map from `h` to the prediction and leaves `h`'s spatial
geometry alone. Behaviour is object-directed because the curiosity reward is prediction MSE, which
depends on the readout — the policy's input is `h`, which never changes:

```
W_out → obs_pred → prediction MSE → curiosity reward → PPO → policy
h ──────────────────────────────────────────────────────────────^   (policy INPUT, unchanged)
```

Searching for allocentric object-trace *place fields* is the wrong target in this architecture:
there is nothing in the rate maps to decay after removal.

---

## Section 2 — Freeze `W_out`, scale `W_in` (`lr_trials=[2,0,8]`)

**The only manipulation in the whole project that ever moved `h`.**

### Question

If `W_out` is the cheap storage site, does removing it force the object into `h`?

### Why this design

Two rationales, and the second is the one that matters most methodologically.

1. **The intervention.** `lr_trials` is a per-optimizer-group LR multiplier over
   `[W, W_out, W_in]` — an existing config knob, so the `prnn` package did not have to change.
   `[2,0,8]` freezes the readout (removing the cheap route) and widens the input pathway 4×.
   The `[8,0,0]` control — recurrent matrix at 4×, `W_in` frozen — isolates *which* pathway does
   the work.
2. **The analysis must split by input-mask phase.** `thRNN_5win` is a `MaskedRNN` with
   `inMask = [True, False×5]`: the observation reaches the network **only 1 timestep in 6**, while
   `outMask` is all-True so it must predict at every step. ph0 is input-driven; ph1–ph5 are driven
   by actions and recurrence alone. **Pooling them is a documented past failure** — it averaged a
   real ph1 effect with near-chance ph3–ph5 and produced a flat ~0.530 across ten conditions,
   from which "the memory half is structurally impossible with RL-side changes" was reported to
   the user **twice** before being caught
   ([doc], `exp_object_into_hidden_state_2026-08-01.md` §7.3). Encoding and memory are different
   questions and the mask phase is the variable the mechanism runs on.

### Method

A logistic probe reads object presence out of `h`, restricted to timesteps where the object
location is inside the 7×7 view, **split by mask phase**. Train/test split is by trajectory, so
nothing leaks; each number averages 3 splits; chance = 0.5. Paired object-present/object-absent
probes with **byte-identical trajectories** — possible because the object is a non-blocking
`FloorBright` tile, so a fixed action sequence gives identical `agent_pos`/`agent_dir` (verified;
16.2% of timesteps differ in observation). `scripts/trace_presence_decoder.py`,
`scripts/otc_figures.py`.

### Result

Regenerated live this session from `outputs/trace/otc_results.npz`; the table printed by
`scripts/otc_figures.py plot` matches the recorded one exactly **[cache]**:

```
              ph0      ph1      ph2      ph3      ph4      ph5
[2,2,2]     0.6947   0.5834   0.5413   0.5242   0.5147   0.5087    (n=10)
[2,0,8]     0.7214   0.5938   0.5459   0.5247   0.5146   0.5095    (n=3)
baseline    0.6787   0.5781   0.5378   0.5210   0.5131   0.5066    (n=1)
```

Welch t-test, `[2,0,8]` vs `[2,2,2]` fixedpos: **ph0 t=14.44, p<1e-5; ph1 t=7.28, p=3e-5** [doc].
Every `[2,0,8]` seed exceeds every fixedpos seed.

![phases](../../outputs/trace/fig_otc_phases.png)

**Figure 2.1** — `outputs/trace/fig_otc_phases.png`. Decoding by mask phase, every condition
against the pre-exposure baseline. Right panel is the same data above chance on a log axis: decay
is close to exponential, **~2× per masked step**, reaching ~0.51 by ph5 **in every condition**.
`[2,0,8]` separates from the pack at ph0–ph1 and has converged with it by ph4.

![encoding](../../outputs/trace/fig_otc_encoding.png)

**Figure 2.2** — `outputs/trace/fig_otc_encoding.png`. ph0 per condition with every training seed
plotted against the n=10 control. This is the comparison that makes the effect an effect rather
than a single lucky run.

![tradeoff](../../outputs/trace/fig_otc_tradeoff.png)

**Figure 2.3** — `outputs/trace/fig_otc_tradeoff.png`. Left: encoding is **bought** — the
conditions with the highest ph0 decoding have the *lowest* object contrast. Right: ph0 against
`‖ΔW_in‖/‖W_in‖`, where `[8,0,0]` sits at x=0 with baseline-level decoding. That is the cleanest
single piece of evidence that **`W_in` is the causal pathway**, not the recurrent matrix.

Figure 0.2 above is the spatial-tuning view of the same intervention: the encoding gain is real
and is not localised at the object.

### Interpretation

`[2,0,8]` is **established for encoding and null for memory**. `h` encodes the object more
strongly; it does not hold it longer. Information decays ~2× per masked step and is near chance
within three, in every condition — including when replayed with `trainNoiseMeanStd=(0,0)`, which
raises ph0 ~2 points and leaves the memory phases unchanged: **the memory was never written, not
erased** [doc]. Randomising object position to remove the position shortcut moves things in the
predicted direction but does not survive correction (pooled ph1–5 p=0.092 at n=10 per arm) [doc].

Cost: the gain trades prediction accuracy, object contrast **+0.019 vs +0.078** for normal
training.

---

## Section 3 — Sequential displacement A→B→C→removed (n=8)

### Question

The Tsao/Moser paradigm proper: move the object between locations, then remove it, and look for a
field that persists at a departed location.

### Why this design

- **Across episodes, not within.** The pRNN state resets every `seqdur=256` steps, so a
  cross-phase trace can only live in the weights — which is also where the biological
  across-trial trace lives.
- **(14,7) is deliberately excluded.** It had produced spurious effects in three independent
  analyses by this point (§1, §4). This is the first design that gets a *clean* test of readout
  leakage, and it retracted an earlier claim that had been read off the contaminated (14,7) row.
- **n=8 seeds**, after "+142% at n=1" became non-significant at n=10 earlier in the project.
- Checkpoint provenance is asserted in-job: `slurm/otc_seq.sh` pins `CUR_CKPT_DIR` and fails on a
  sha256 mismatch against `c1e43a6b…`, the exact baseline every earlier result used.

### Method

`(7,11) → (7,2) → (4,7) → REMOVED`, 1000 trajectories per phase, checkpoints at every phase
boundary, 8 seeds in parallel. Hidden state scored with `scripts/trace_metric.py`: field gain
`g(u,c)` = mean rate in a radius-2 disc at `c` / the unit's mean rate (scale-free), trace score
`dg = g_post − g_pre`, and the unit's score is the object cell's **percentile among all 172
walkable cells**. Object cell if percentile > 95, so chance is 5% by construction.

Metric validation, reported with every result [doc]: negative control (odd vs even probe
trajectories, two independent map estimates) **0.0601, p=0.174**; positive control **100% recall
at a field 1% of a unit's mean rate in 10% of units**. That sensitivity is what makes each null a
bound rather than a shrug.

### Result

Figure 0.1 is the primary figure for this section — reproduced from
`outputs/trace/seq4_matrix.npy` and `seq4_readout.npy`. Every statistic below was re-derived from
those caches in this session and **matches the recorded values exactly [cache]**.

**Hidden state — nothing forms during exposure at all.** Each location during its own phase:

```
(7,11) 0.0283   t=-3.81  p=0.0066   <- significantly BELOW the 0.05 chance rate
(7,2)  0.0386   t=-1.49  p=0.18
(4,7)  0.0441   t=-1.36  p=0.22
```

**No trace after departure**, during-minus-after: +0.0033 (p=0.61), +0.0011 (p=0.89),
+0.0125 (p=0.23). With no field forming there is nothing to trace, which settles the question by
itself.

**No hidden-state leakage:** (4,7) before the object arrives 0.0342 vs 0.0441 during, **p=0.14**.

**Readout — transfer, not trace.** During its own phase vs after the object leaves:

```
(7,2)  0.0490 -> 0.0196   drop 0.0295   t=5.71  p=0.0007
(4,7)  0.0445 -> 0.0205   drop 0.0239   t=4.15  p=0.0043
(7,11) 0.0192 -> -0.0066  drop 0.0258   t=2.04  p=0.081
```

![sequential fields](../../outputs/trace/fig_seq_fields_3058.png)

**Figure 3.1** — `outputs/trace/fig_seq_fields_3058.png`. Spatial tuning across the four phases.
Units **do** reorganise their receptive fields substantially between phases — but not at the
object. That is the null made visible: the maps are not frozen, they simply do not care where the
object is.

![object-centred readout](../../outputs/trace/fig_seq_predcentred_3058.png)

**Figure 3.2** — `outputs/trace/fig_seq_predcentred_3058.png`. The object-centred prediction map:
green change averaged over every in-view timestep, each 7×7 patch rolled so the object's view cell
is at the centre. An object-locked effect appears at the centre; viewpoint-specific noise averages
away. This is the readout-side measurement behind the right panel of Figure 0.1.

### Interpretation

The hidden state never codes the object, so there is no trace to look for. The readout *does*
change, and it does **not** persist after departure — but ~89% of what it shows at any location
was already there before the object arrived (§0). Two different measurements of two different
things; conflating them is a documented failure mode in this project.

⚠️ **A claim was retracted here.** The shared-readout generalisation had earlier been inferred
from the (14,7) row of a 3-phase run. On a clean cell the *hidden-state* leakage test is null
(p=0.14), and that earlier claim is withdrawn. The readout leakage in §0 is a separate,
independently confirmed measurement.

---

## Section 4 — Occlusion (scenario C): a false positive, and the control that caught it

**This is the most transferable methodological lesson in the project.**

### Question

With `see_through_walls=True` and a non-blocking floor tile, the object is either visible or
reconstructible from position. If it genuinely leaves view, is holding it the only way to predict
what is behind the wall?

### Why this design — and two implementation findings that shaped it

- **A blocking object does not occlude.** `Ball` and `Box` have `can_overlap() = False` but
  `see_behind() = True`, and `gen_obs_grid` only runs `process_vis` when `see_through_walls` is
  False. **So occlusion was implemented via `see_through_walls=False` — the room's own walls
  became opaque — NOT by adding an occluding object.** The object itself remained a traversable
  `FloorBright` tile throughout. This matters for interpreting the result and is easy to
  misremember.
- **The observation bank would have silently corrupted this.** It was keyed on `grid.encode()`
  alone, which does not capture occlusion — it would have served non-occluded observations to an
  occluded env. An `-occl` fingerprint suffix was added, appended only in that case so previously
  cached banks keep their filenames.
- **The cell was chosen by measurement, and a bug was caught pre-launch.** C was initially armed
  with (13,10), which is **inside the wall block and not walkable**. Candidates were then scored
  by how often the object is hidden: (14,7) is the best available at **17.1%**. Note the honest
  limit — the L-room has a single interior wall, so even the best cell is hidden only 17% of the
  time it would otherwise be seen, which bounds how strong a memory demand C can create.

### Method

Occluded baseline capped at 3.5 h (31,199 trajectories vs 79,679), gated first on having a usable
place code: SI median 0.690 vs 0.759, 1/500 silent units, 172/196 valid bins — weaker as expected
from a third of the training, but clearly a place code [doc]. Metric re-validated on a
purpose-built occluded probe: negative control **0.0421, p=0.819**; positive control 100% recall
from amplitude 0.01.

### Result — it looked strong

![scenario C gradient](../../outputs/trace/fig_scenarioC_gradient.png)

**Figure 4.1** — `outputs/trace/fig_scenarioC_gradient.png`. Object-cell fraction by location,
n=8 seeds, against both the 0.05 chance line and the measured 0.0421 negative control [doc]:

```
object location (14,7)     0.0892
mean of 7 control cells    0.0329
EXCESS                    +0.0563     paired t=5.55, p=0.001
(14,7) highest in 8 of 8 seeds  ->  (1/8)^8 ~ 6e-8 by chance
```

And it was **spatially graded** — 0.089 at the object, ~0.045 at the two cells about 2 bins away,
~0.028 at the five distant cells. That gradient is the shape a genuine place-field-like effect
should have, and no previous condition in this project produced one.

### Result — the location control killed it

![occlusion location control](../../outputs/summary/fig_occlusion_control.png)

**Figure 4.2** — `outputs/summary/fig_occlusion_control.png`. **The decisive figure.** Rows are
where the object actually was; columns are the cell scored; shared colour scale across both
panels; black outline marks the run's own object cell.

Left panel, on pre-existing non-occluded data [doc, `exp_trace_cell_scenarios_2026-08-11.md`
§2.5]: cell **(14,7) is the highest column in every row**, whatever the object position.

```
                          scored at:
run with object at:   (7,11)    (14,7)     (7,2)
        (7,11)        0.0321    0.0902    0.0200
        (14,7)        0.0140    0.1002    0.0501
         (7,2)        0.0581    0.0721    0.0341

excess:               -0.004    +0.019    -0.001
```

Right panel, the occluded control: moving the object to (12,7) leaves **(14,7) still highest,
0.0835 vs 0.0441** ([doc], `compaction.md` line 61; ⚠️ see §7 — this summary line is the only
record, the per-seed table is not in the repo).

### Interpretation

The peak does not follow the object, so the effect is **location drift, not object coding**. The
within-unit percentile null controls for a unit's own map structure but **cannot see drift shared
across units at one location** — which is exactly what (14,7) has. The spatial gradient was real
and still meant nothing, because (12,7) and (13,5) sit in the same occluded region as (14,7):
"the occluded region changes more" produces the same gradient as "the object's location changes
more".

**Two rules any successor must follow:**

1. **The peak must move with the object.** Score every candidate location under every run. A
   within-unit null is not sufficient.
2. **Never use (14,7) as an object location in the L-room.** It has produced spurious effects in
   three independent analyses and reads elevated before the object ever arrives.

---

## Section 5 — Symmetric room + Moser sequence (2026-08-12)

### Question

Every failure above has the same shape: the object is *predicted* but never *needed*. The L-room
has an L-shaped wall plus a triangle, a plus and an x, so the agent localises from geometry alone.
Make the room symmetric so the object is the **only** disambiguating cue — the one design where a
trace cell is mechanistically *required* rather than merely permitted.

### Why this design

- **`MiniGrid-SquareRoom-v0`, four-fold symmetric.** Verified directly in observation space, not
  assumed: six (cell, heading) pairs against their 90-degree-rotated equivalents give
  **max|obs diff| = 0** for all six [doc].
- **Trained FROM SCRATCH**, not from the L-room checkpoint, so no geometry knowledge carries over.
- **Object cells and trace cells tested as separate populations.** The paper is explicit that they
  are independent: *"the latter cells generally did not respond to the object when it was
  present."* So the trace criterion requires a field at some earlier `P_j`, **and** no field at
  the current `P_k`, **and** no field at `P_j` in session 0 before any object existed.
- **Sessions chain as separate processes**, each loading the previous checkpoint, so weights, both
  optimizers and the frame counter carry over while the environment changes.
- **Pipeline negative control, run before launch:** on 40-episode toy checkpoints the metric
  returns 1.2% object and 0.2% trace cells, both under the 5% chance rate — no spurious positives
  from an untrained net [doc].

### Method

`no object → 6 positions → no object`. Session 0 no object, 8000 episodes; sessions 1–6 objects at
(5,4) (10,6) (6,11) (12,3) (3,9) (11,12), 2000 each; session 7 no object, 2000. Seed 1, n=1.
`scripts/moser_sessions.py`, `moser_analysis.py`, `moser_figures.py`.

### Result — object cells: NULL

![moser panel](../../outputs/moser/fig_moser_panel.png)

**Figure 5.1** — `outputs/moser/fig_moser_panel.png`. **Spatial tuning, the direct replication
attempt of [`ref-trace-cells.png`](../ref-trace-cells.png).** Units × sessions rate maps, white ring
= object position, O/T tags, shared colour scale within each row. *(This already does everything a
separate example-unit figure would, so I did not duplicate it.)* Read the columns: the fields move
between sessions, and they do not track the ring.

Object-cell fractions per session, **[cache]** re-read from `outputs/moser/moser_summary.json`,
matching the recorded table:

| session | object | object cells | p vs 5% [doc] |
|---|---|---|---|
| 1 | (5,4) | 1.4% | 1.00 |
| 2 | (10,6) | 2.0% | 1.00 |
| 3 | (6,11) | 2.8% | 0.99 |
| 4 | (12,3) | 4.2% | 0.82 |
| 5 | (3,9) | 2.4% | 1.00 |
| 6 | (11,12) | **6.4%** | **0.09** |

Never above chance. The single-position 5% null is correct here.

![moser gain](../../outputs/moser/fig_moser_gain.png)

**Figure 5.2** — `outputs/moser/fig_moser_gain.png`. The population-level control: mean field gain
at the *current* object position vs at *departed* positions vs all other walkable cells. The
current-object line sits mostly **below** the all-cells baseline and crosses it only at s4 and s6
— i.e. the population does not put its fields where the object is.

### Result — trace cells: NULL, and this was reported wrong first

Raw counts rise monotonically to **17.0% at session 7 (p<1e-4)** against a flat 5% line, which
looked like the result the whole project was after. It is not.

**The trace criterion is a cumulative OR over every previously-used object position, so the number
of chances to score a hit grows with session index** (1 at s2, 6 at s7). A flat 5% null is simply
the wrong null.

![trace null](../../outputs/summary/fig_moser_trace_null.png)

**Figure 5.3** — `outputs/summary/fig_moser_trace_null.png`. The correction. Observed counts
against an **empirical null** — the identical criterion applied to six positions the object never
occupied, 400 random draws. The null reproduces the entire rising pattern with no object ever
present, and sits **above** the observed count at every session.

```
session         s2      s3      s4      s5      s6      s7
observed      2.0%    3.2%    5.0%   10.2%    9.2%   17.0%   [cache, moser_summary.json]
empirical null 3.7%    6.6%   10.6%   12.1%   15.4%   20.3%   [doc]
p              0.70    0.89    0.92    0.64    0.91    0.81
```

⚠️ Note that `outputs/moser/fig_moser_counts.png` plots the raw counts against the **flat 5%
line** — it is the version that produced the retracted claim. Use Figure 5.3 instead.

### Result — why the room did not bind

![quadrant decode](../../outputs/summary/fig_quadrant_decode.png)

**Figure 5.4** — `outputs/summary/fig_quadrant_decode.png`. In a four-fold symmetric room,
decoding the agent's quadrant from `h` should be at chance (0.25) unless something breaks the
symmetry, and the object is the only candidate. Trajectory-level CV, so no within-trajectory
leakage. Result: **~84% with the object, ~84% without it, delta ~0.000 at every session** [doc].

The hidden state localises fine with no object at all. **Position information is reaching `h` by a
route that does not pass through the visual scene** — the remaining candidate being trajectory
history: each probe trajectory starts from a fixed cell with a fixed initial hidden state, and the
net integrates from there. Removing visual landmarks does not remove that.

⚠️ A claim of sRSA rising 0.036 → 0.60 was **retracted**: sessions are cumulative training
(8k → 22k gradient steps), so that rise is training length, not the object.

### ⚠️ The confound: the policy collapsed, and sessions 4–6 are uninformative

![policy collapse](../../outputs/summary/fig_policy_collapse.png)

**Figure 5.5** — `outputs/summary/fig_policy_collapse.png`. Policy entropy and occupancy entropy
across all eight sessions, against the L-room reference band. Fetched live from wandb
(`blake-richards/curious-george`, runs `moser-s{0..7}-*`; reference
`pRNN_curious_26-07-23-10-06-25`) and cached to `outputs/summary/wandb_entropy.npz` **[live]**.

The square room **does not begin degenerate**. Session 0 sits inside the L-room reference band and
is indistinguishable from it. It **collapses during training**, from session 4:

```
                      s0     s1     s2     s3  |  s4     s5     s6     s7
policy entropy       1.49   1.35   0.86   0.95 | 0.28   0.23   0.19   0.24   [live, mean of last 100 logged pts]
occupancy entropy    6.10   6.46   6.93   6.68 | 5.19   4.96   4.72   5.13   [live, same]
L-room reference     policy entropy p10-p90 = 1.31-1.56;  occupancy p10-p90 = 6.02-6.76  [live]
```

Maximum policy entropy is **log2(4) = 2.0 bits** — confirmed at source this session: the action
space is `Discrete(4)` (`curious_george/envs/factory.py:173`), and the logged value is in bits,
not nats (`curious_george/rl/update/losses.py:70` sets
`policy_entropy_bits = policy_entropy / _LOG2`, accumulated at
`curious_george/rl/update/updater.py:155` and surfaced as `policy_entropy` at
`curious_george/training/logging.py:52`) **[live]**. So s0's 1.49 bits is 75% of maximum and s6's
0.19 is under 10%.

**Why this matters scientifically:** object positions for sessions 4, 5 and 6 are **(12,3), (3,9),
(11,12)** — exactly the sessions with collapsed exploration. If the agent rarely visited them, no
object representation could form there whatever the architecture allows. **Those sessions' nulls
are UNINFORMATIVE, not negative**, and the trace counts that depend on them inherit the problem.

It is specific to this room, not systemic: the L-room reference holds its band across its entire
logged history, and the square room collapses well before the end of session 4.

**Likely cause and the cheap fix:** `Configs/algo/ppo.yaml:15` has **`entropy_coef: 0.0`**
[live] — nothing resists policy collapse. In the L-room the curiosity reward is spatially
structured and keeps the policy responsive; in a symmetric room the reward landscape is far more
degenerate, so PPO can drift to a deterministic policy with no gradient pulling it back.
`Configs/performance/ultra.yaml:42` already sets `entropy_coef: 0.01` [live] — the perf work had
reached the same conclusion from a different direction.

**Before re-running anything in the symmetric room:** set `rl.entropy_coef > 0` and gate on
`loc_entropy` staying flat across sessions. That gate is cheap and it is now a precondition, not
an optional check.

### Interpretation

The symmetric-room reasoning was **right about the shape of the problem and wrong about the escape
route**. Removing visual landmarks did not make the object necessary, because the network
localises by trajectory history instead. The design closed the geometry route and left the history
route wide open.

The result is also **n=1 and confounded** in its second half. Treat sessions 1–3 as a genuine
(if underpowered) null and sessions 4–6 as uninformative.

One framing question worth raising before more GPU: Tsao/Moser's LEC cells show *little* spatial
modulation in an empty field, whereas these pRNN units have mean SI ~0.9 — they are strongly
spatial, i.e. place-cell-like. We may be looking for LEC phenomenology in an architecture closer
to MEC/hippocampus, in which case **the target itself, not the room, is what needs rethinking**.

---

## Section 6 — What we would trust, and what we would not

| experiment | verdict | reliability | why |
|---|---|---|---|
| **Mechanism: object in `W_out`, ~89% position-independent** (§0) | ESTABLISHED | **High** | n=8, t=9.57, re-derived from cache this session; independently corroborated by the gain-corrected 9/9 chimaera swap and by r≈0.98 map stability |
| **`h` unchanged under standard OMT** (§1) | ESTABLISHED | **High** | 3 locations × 3 seeds, null in three reference frames, strong positive control (100% recall at 1% field) |
| **`[2,0,8]` raises encoding in `h`** (§2) | ESTABLISHED | **High** | n=3 vs n=10, every treated seed exceeds every control seed, p<1e-5; `[8,0,0]` control isolates `W_in` |
| **`[2,0,8]` does not create memory** (§2) | NULL | **High** | consistent across all 13 conditions; noise-free replay shows it was never written |
| **Sequential displacement: no trace in `h`** (§3) | NULL | **High** | n=8; nothing forms during exposure at all, so there is nothing to trace; every statistic re-derived from cache |
| **Readout is transfer, not trace** (§3) | ESTABLISHED | **Medium-High** | n=8, p=0.0007/0.0043; but rests on one probe design and one environment |
| **Occlusion produces object cells** (§4) | **FALSE POSITIVE** | **Do not trust** | killed by its own location control; retained here only as the methodological lesson |
| **The (12,7) occlusion control itself** (§4) | control | **Medium** | ⚠️ n=3, and only a one-line summary survives in the repo — no per-seed table (§7) |
| **Symmetric room, object cells** (§5) | NULL | **Low-Medium** | n=1; sessions 1–3 usable, sessions 4–6 confounded by policy collapse |
| **Symmetric room, trace cells** (§5) | NULL | **Low** | n=1, and the sessions with the most "chances" are exactly the confounded ones |
| **Quadrant decode ~84% without the object** (§5) | ESTABLISHED | **Medium-High** | clean design, trajectory-level CV, delta ~0.000 at all six sessions; but n=1 run and I did not re-run it |
| **Policy collapse from session 4** (§5) | ESTABLISHED | **High** | fetched live from wandb this session; unambiguous and large |
| **Scenario A (object identity varies)** | NULL | **Medium** | n=3, at/below chance, but a weaker manipulation |
| **Scenario D (weight-decay sparsity)** | **UNTESTABLE** | n/a | L2 shrinks all weights: place code collapses r 0.97→0.73 and `dW` *grows* with decay. The lever does not do what the scenario needs; the sparsity hypothesis remains **open** |

---

## The shape of the problem

Every design in this project has tried to make the object *necessary*, and every one has closed
**at most one** of the routes by which the network can localise without it. There are now two
demonstrated routes:

| room | how the net localises without the object | evidence |
|---|---|---|
| **L-room** | **geometry** — an L-shaped wall plus a triangle, a plus, an x | the whole §1–§4 null series; `h` never needs the object because position is readable off the walls |
| **square room** | **trajectory history** — dead reckoning from a fixed start | Figure 5.4: quadrant decode ~84% with the object *absent*, delta ~0.000 |

The symmetric room removed the geometry and left the history untouched, so the object stayed
optional and the result was null again — for a *new* reason, which is the actual progress. And
underneath both sits the §0 mechanism: even when the object *is* learned, `W_out` learns it as an
object-centred view offset applied to every hidden state, so nothing location-specific ever has to
enter `h`.

That gives a clear ordering for what to try next, and it is the first time the ordering is
mechanistic rather than analogical:

1. **Multi-environment training.** The one manipulation that attacks the *history* route without
   new machinery: the same integrated trajectory maps to a different absolute position in a
   different room, so dead reckoning cannot resolve position and the net must identify *which
   room it is in* from what it sees. That is exactly the pressure to bind a visual feature to a
   place.
2. **Break path integration directly** — teleport the agent mid-episode, and/or randomise the
   initial hidden state per trajectory. Cheap pre-check before building anything: re-run
   `scripts/moser_decode_quadrant.py` with the start cell randomised and hidden from the decoder.
   If accuracy collapses toward 0.25, history is confirmed as the route.
3. **Combine (1) or (2) with `lr_trials=[2,0,8]`.** The room removes the localisation route;
   freezing `W_out` removes the absorbing route. Both halves of the mechanism, closed at once —
   and `[2,0,8]` has never been combined with a room where the object is actually needed.
4. **Make the object task-relevant.** Nothing in the current objective requires knowing where the
   object is. Curiosity pays for *visiting*, and the net never learns the object anyway (predicts
   0.56 against a target of 0.99 after 3000 trajectories), so the bounty is permanent and never
   forces a representational change.

And before any of it: **`rl.entropy_coef > 0`, gated on `loc_entropy` staying flat.**

---

## Section 7 — Verification notes: what I could not confirm

Recorded so the numbers above can be trusted differentially.

1. **⚠️ Per-session entropy values disagree with `compaction.md` §8.** That table records
   `policy entropy 1.54 1.35 0.94 0.98 0.25 0.31 0.17 0.23` and
   `loc entropy 6.04 6.58 6.84 6.59 5.15 5.29 4.55 5.11`. Fetching the histories from wandb, I
   **could not reproduce those exact values under any aggregation I tried** — last value, mean,
   median, or mean of the last 100 logged points. The last-100 mean is closest (e.g. s4 0.281 vs
   0.25; s2 0.856 vs 0.94) but does not match. The aggregation is not stated in the doc. The
   figures and the numbers in §5 use **my own fetch with the aggregation stated inline**. The
   qualitative conclusion — healthy through s3, collapsed from s4 — is unaffected and confirmed.
2. **⚠️ The L-room "1.30–1.53 over 80,000 gradient steps" claim.** The band is essentially
   confirmed: I measure p10–p90 = **1.31–1.56**. The step count I could **not** confirm — the
   reference run's wandb `_step` reaches 110,341, and I did not verify that `_step` equals
   gradient steps. I have therefore not restated "80k" anywhere above.
3. **⚠️ `compaction.md` cites `losses.py:51` for the nats→bits conversion.** The actual line is
   **`losses.py:70`** (and `:98` for the second branch). Corrected above. Minor, but it is a
   citation someone will follow.
4. **⚠️ The occlusion (12,7) control has no per-seed record.** "0.0835 vs 0.0441" appears in the
   repo **only** as the one-line summary at `compaction.md` line 61. The three run directories
   exist (`outputs/C_at127-otc-p1.0-fixedpos-c1-0811-*`) but no result cache does, so I could not
   re-derive it. It is load-bearing for the §4 retraction, so it is worth re-running and caching.
5. **Not re-derived this session, taken from repo docs [doc]:** the gain-corrected chimaera swap
   (+0.0625 vs +0.0157); r ≈ 0.98 map stability; the scenario-C n=8 gradient and its excess; the
   Moser empirical-null percentages; the quadrant-decode accuracies; the `max|obs diff| = 0`
   symmetry check; the metric's negative/positive controls. Each needs GPU replay or has no
   surviving cache.
6. **The Tsao/Moser 2013 reference itself** is quoted from `docs/sab_context/goal_2026-08-12.md`
   and `docs/ref-trace-cells.png`. I did not fetch the DOI to re-verify the quotation.
7. **The brief for this document stated the older `outputs/trace/*.png` had been deleted. They had
   not** — all 22 were present, intact, and newer than their caches. I regenerated them anyway via
   `scripts/trace_cell_figures.py` and `scripts/otc_figures.py plot|maps`; all seven and four
   respectively rebuilt cleanly, and the `otc_figures.py plot` console table reproduced the §2
   phase numbers exactly.

---

## Reproduction

```bash
# ALWAYS --no-sync: plain `uv run` re-syncs from uv.lock and silently uninstalls the
# editable minigrid, which breaks MiniGrid-SquareRoom-v0.

uv run --no-sync python scripts/trace_cell_figures.py        # -> outputs/trace/fig_{occupancy,tuned_units_*,si_weighting,trace_3loc,exposure_*,behavior_n3}.png
uv run --no-sync python scripts/otc_figures.py plot          # -> outputs/trace/fig_otc_{phases,encoding,tradeoff}.png
uv run --no-sync python scripts/otc_figures.py maps          # -> outputs/trace/fig_otc_maps{,_diff}.png
uv run --no-sync python scripts/summary_figures.py           # -> outputs/summary/*.png  (this document's new figures)
uv run --no-sync python scripts/summary_figures.py fetch_entropy   # refresh the wandb cache (needs network)
```

`scripts/summary_figures.py` is new in this session; nothing else was modified.

---

## Regenerating every figure in this document

Verified end-to-end on 2026-08-12 after the `scripts/` reorganisation: **17 of the 19
figures rebuild from committed code plus the caches under `outputs/`**. All commands need
`--no-sync` (the editable `minigrid` checkout is required for `MiniGrid-SquareRoom-v0`, and
plain `uv run` re-syncs it away).

```bash
uv run --no-sync python scripts/summary_figures.py            # the 5 outputs/summary/ figures
uv run --no-sync python scripts/trace/trace_cell_figures.py   # 7 figures in outputs/trace/
uv run --no-sync python scripts/otc_figures.py plot           # fig_otc_phases/_encoding/_tradeoff
uv run --no-sync python scripts/otc_figures.py maps           # fig_otc_maps/_maps_diff
uv run --no-sync python scripts/moser/moser_figures.py        # the 4 outputs/moser/ figures
uv run --no-sync python scripts/seq_figures.py \
    outputs/seq4/OTC_seq_5200_10340479/SEQ4-otc-p0.5-fixedpos-c1-0811-123058
```

Note on that last path: half the `outputs/seq4/OTC_seq_*` directories are empty rsync
artifacts, and the real run sits one level deeper. The `3058` in the figure names is the
timestamp tail of the run directory.

### Two figures that do NOT regenerate

| figure | why |
|---|---|
| `ref-trace-cells.png` | static reference image from Tsao/Moser 2013 — nothing to regenerate |
| `fig_scenarioC_gradient.png` | **no generating script exists.** It was produced by a throwaway in an earlier session and was never promoted into `scripts/`. The underlying claim (scenario C's spatial gradient) is therefore not reproducible from this repo — treat it as an illustration, not evidence. The scenario C result is retracted anyway, on the location control. |

This is the same class of gap as the `(12,7)` control number noted in the verification
appendix: a load-bearing artefact with no cached input and no script.

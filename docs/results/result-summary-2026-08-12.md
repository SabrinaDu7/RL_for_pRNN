# Object and trace cells in the pRNN: what we found, and what each figure shows

**Written:** 2026-08-12 · **Branch:** `sdu/object-into-hidden-state`
**Scope:** the whole object/trace-cell investigation, 2026-07-30 → 2026-08-12.

**The goal was** to produce *object-trace cells* in the pRNN hidden state `h` — units that grow a
place field where a novel object used to be, as in Tsao, Moser & Moser 2013
([`ref-trace-cells.png`](../ref-trace-cells.png), doi 10.1016/j.cub.2013.01.036).

**They do not appear.** Ten designs have now been run and every hidden-state result is null. This
document explains the mechanism that predicts all of them, and links every claim to a figure.

*Updated 2026-08-13: §3 gains the behavioural result that was missing — Figure 3.3 (the location
control) and Figure 3.4 (occupancy maps). Behaviour tracks the present object and abandons the
departed one, matching the readout and not the hidden state.*

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
every 200.

The map caches these figures are built from carry no record of which run made them; those runs have
no config on disk and no surviving wandb entry. Recovered from the data instead by replaying each
candidate and correlating against the cache
(`scripts/trace/verify_map_provenance.py`, cache `outputs/trace/map_provenance.json`) **[live]**:

```
outputs/mila_omt/omt-cur-dot-0730-125634  ->  m_7_11    r = 1.00000  (margin +0.023)
outputs/mila_omt/omt-cur-dot-0730-130252  ->  m_7_2     r = 1.00000  (margin +0.080)
outputs/mila_omt/omt-cur-dot-0730-130307  ->  m_14_7    r = 1.00000  (margin +0.023)
```

So `maps_all.npz` — and therefore Figures 1.2 and 1.4 — comes from `outputs/mila_omt/`, one run per
location. The script asserts its own pipeline first (baseline checkpoint vs cached `baseline`,
r = 1.00000) so that a wrong-run answer cannot be confused with a wrong-pipeline one.

Probe: one trajectory from every (walkable cell, head direction) pair = **688
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

**Figure 1.2** — `outputs/trace/fig_trace_3loc.png` (rebuilt 2026-08-13). **The main null, as
spatial tuning, in location-control form.** Three blocks, one per object location. Each block
selects the units most modulated at **its own** location and marks **its own** location with the
white `+`; the red-outlined panel is that block's own exposure run, the only column in which a
trace at that `+` could appear. Colour scale shared across a row. Nothing appears at any `+`,
including inside every red outline.

⚠️ **The previous version of this figure could not have shown a trace at (14,7) or (7,2).** It
selected units by Δ object-modulation at (7,11) only and drew the `+` at (7,11) in *every* column,
so columns 3–4 showed units chosen for one location, marked at another, in runs whose object was
somewhere else. The null is unchanged; the figure now scores each panel against the thing it is
about.

⚠️ **The ranking statistic selects the network's quietest units, and this is not stated anywhere.**
`object_modulation` is the bounded contrast `(near − far)/(near + far)`, so a small denominator
inflates Δ. Measured this session from `maps_all.npz` **[cache]**: population mean rate median
0.2168, and all nine selected units fall in the bottom 0–28th percentile of rate. Unit #429 (rate
0.0034, **0th percentile**) tops the list at two of the three locations, with an absolute rate
change in its disc of +0.006 against a population p95 of 0.150. Two of the nine (#438 −0.020,
#314 −0.006) got *dimmer* at their own object location and were still ranked "most modulated".
Changing the ranking statistic is a methodological choice and has not been made; the figure now at
least prints each row's Δmodulation so the selection is visible. The same defect affects Figure 3.1
— see there.

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
16.2% of timesteps differ in observation). `scripts/trace/trace_presence_decoder.py`,
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

The two arms are **location-matched**, which was assumed rather than known until 2026-08-13: the
`[2,0,8]` runs trained at (7,11) (`wandb/run-20260801_120212-8xck38gb/files/config.yaml:9`), and the
three `mila_omt_dense` runs in the control arm are all (7,11) too (§7.8) **[live]**. Had they mixed
locations, the comparison would have confounded condition with object position.

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

`(7,11) → (7,2) → (4,7) → REMOVED`, checkpoints at every phase boundary, 8 seeds in parallel.

![sequential phase environments](../../outputs/summary/fig_seq_phases.png)

**Figure 3.0** — `outputs/summary/fig_seq_phases.png` (`scripts/trace/seq_phase_figure.py`). The four
environments and what each phase cost. Every panel is the same room with the same agent start; the
object — the single bright-green tile — is the only difference. Per phase, derived in-script from the
run's phase directories and the composed config **[live]**:

```
1,000 trajectories (125 batches x 8)      256,000 environment steps
1,000 world-model gradient steps          4,000 PPO gradient steps
```

`predNet.batched_wm` is **False** for these runs, so the world model takes one gradient step per
episode segment (`curious_george/rl/update/world_model.py`), i.e. `frames/seqdur` = 8 per batch —
not one per batch.

⚠️ **Exposure is not matched to §1, and that bounds what this null can be compared against.** The
pre-exposure checkpoint has **79,999** world-model gradient steps (the reference run's wandb `trial`
counter, fetched this session — this also resolves the open question in §7.2: `_step` reaches
110,349 but is the wandb event counter, not gradient steps). So each phase here adds **+1.25%** more
world-model training, while §1's standard OMT gave a single location 3,000 gradient steps, **+3.75%**
— three times more per location. Neither null is obviously an exposure-duration artifact (§1 is null
at 3× and its Figure 1.3 timeline is null throughout), but the two are not exposure-matched.

⚠️ **The run directories are misnamed in two ways.** `SEQ4-otc-p0.5-…` takes `p0.5` from
`tasks.otc.presence_prob` (`tasks/otc/main_task.py:40`), but when `tasks.otc.sequence` is set,
`main_task.py:85` dispatches to `train_sequence`, which **hardcodes** `presence_prob=1.0` for object
phases and `0.0` for the removal phase (`tasks/otc/task.py:165`) and never reads the config value —
every SEQ4 phase had the object in *every* environment, not half. And `phase{n}_992` is the
trajectory count at the *start* of the last batch: `train` saves at
`(num_batches − 1) × trajs_per_batch`, so the checkpoint labelled 992 is written after all 1,000
trajectories. `phase{n}_0` is likewise written after 8 trajectories, not 0 — nothing reads it today
(`seq_figures._phases` takes each phase's max step), but it would mislead anyone using it as a phase
baseline. Hidden state scored with `scripts/trace/trace_metric.py`: field gain
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

**Figure 3.1** — `outputs/trace/fig_seq_fields_3058.png` (re-run 2026-08-13, now self-documenting).
Spatial tuning across the four phases. Units **do** reorganise their receptive fields substantially
between phases — but not at the object. That is the null made visible: the maps are not frozen,
they simply do not care where the object is.

**What "most changed" means** — the figure never said, and it matters. From
`scripts/seq_figures.py::fields`, field gain `g(u,c)` is the unit's mean rate in a radius-2 disc at
`c` over its mean rate everywhere, and the ranking statistic is

```
dg[u] = max over (phase p, object location c) of | g_p(u,c) − g_pre(u,c) |
```

So it is an **absolute** change, maximised over both phases and locations, and scored **only inside
the three discs** — it is not a global map-change measure and says nothing about the reorganisation
elsewhere that the caption points at. Re-running it prints the selection **[live]**:

```
  unit  signed dg  phase        at  mean rate  rate pctile
   429     -3.161      0    (7, 2)     0.0034           0%
   449     +2.969      0    (7, 2)     0.0839           7%
   448     +2.546      1    (7, 2)     0.1256          18%
   372     -2.515      1   (7, 11)     0.1288          20%
   226     -2.041      0   (7, 11)     0.1030          11%
   390     -1.977      0    (7, 2)     0.1055          12%
   368     +1.921      0    (4, 7)     0.0302           1%
   195     +1.904      0    (4, 7)     0.1097          13%
```

Three things follow, and they resolve the tension between the selection and the caption:

1. **4 of 8 selected changes are decreases.** The absolute value ranks a lost field identically to
   a gained one, so half these rows are units that got *weaker* at an object location.
2. **6 of 8 maxima land at a location that did not hold the object in that phase.** Phase 0's object
   is at (7,11), yet #429, #449 and #390 max out at (7,2) and #368/#195 at (4,7). Only #448
   (phase 1 at (7,2)) and #226 (phase 0 at (7,11)) peak where the object actually was. The
   statistic never asks, so most of what it ranks is drift.
3. **The selection is the network's quietest units** — median rate percentile 11%, and #429 sits at
   the 0th (rate 0.0034). `g` divides by the unit's own mean rate, so a near-silent unit posts a
   large `dg` from a tiny absolute change. This is the same defect as in Figure 1.2, which uses a
   different normalisation (`object_modulation`) and lands on the same low-rate tail — #429 tops
   both figures.

The null is unaffected: nothing appears at an object location under either ranking. But "the units
whose receptive fields changed most" was never what this figure showed, and a ranking statistic that
is scale-free by division is not safe on a population containing near-silent units. **Fixing it is a
methodological choice and has not been made.**

![object-centred readout](../../outputs/trace/fig_seq_predcentred_3058.png)

**Figure 3.2** — `outputs/trace/fig_seq_predcentred_3058.png`. The object-centred prediction map:
green change averaged over every in-view timestep, each 7×7 patch rolled so the object's view cell
is at the centre. An object-locked effect appears at the centre; viewpoint-specific noise averages
away. This is the readout-side measurement behind the right panel of Figure 0.1.


### Result — behaviour: follows the present object, abandons the departed one (added 2026-08-13)

This section originally reported hidden state and readout and **no behaviour**; the behavioural
probe existed only for §1. The gap is not closable from the logs — the 8 finished SEQ4 runs in
`blake-richards/curious-george-otc` log 27 keys and **not one is a position or an occupancy**
(no `loc_entropy`, no `subroom_ids`). Answered instead by rolling the trained policy at each
phase-end checkpoint (`outputs/seq4/OTC_seq_*/SEQ4-*/phase{0..3}_992`) in that phase's own
environment, scored with the same within-run percentile §1 uses. 8 seeds x 4 phases x 24 rollouts
x 256 steps. **[live]**

![sequential behaviour](../../outputs/summary/fig_seq_behavior.png)

**Figure 3.3** — `outputs/summary/fig_seq_behavior.png`
(`scripts/trace/seq_behavior_figure.py`, cache `outputs/trace/seq_behavior.npz`).

**Panel A, occupancy percentile** (rows: where the object WAS; columns: location scored;
50 = a typical cell of 172):

```
  phase                    (7,11)     (7,2)     (4,7)
  0: object at (7,11)       63.2 *     18.2      61.3
  1: object at (7,2)        36.4       86.4 *    69.3
  2: object at (4,7)        63.9       40.4      87.6 *
  3: REMOVED                70.7       20.2      44.5
```

The object's own location is the highest entry in its row in every phase that has an object.
Read alone that says behaviour follows the object — **and that is exactly the reasoning that
produced the (14,7) false positive in §4.**

**Panel B, the location control.** The statistic that counts is the EXCESS: the value at L when
the object is at L, minus the mean at L when it is anywhere else, so pure location bias cancels.
Paired across the 8 seeds:

```
  location  own phase  other phases    EXCESS  paired t        p
   (7, 11)       63.2          57.0      +6.2      0.76   0.4715
    (7, 2)       86.4          26.3     +60.1     11.76   0.0000
    (4, 7)       87.6          58.3     +29.2      5.38   0.0010
```

**Two of the three survive; (7,11) does not.** It reads ~63rd percentile whatever the object
does — a structurally high-traffic cell, the same class of trap as (14,7). So the honest claim is
not a blanket "behaviour follows the object": it follows the object at (7,2) and (4,7), and is
not measurable at (7,11).

**Panel C, the departure test.** Does the agent linger where the object last was? No:

```
(4,7) raw occupancy   0.213 -> 0.043   5.0x drop, p=0.0022, 8 of 8 seeds
(4,7) percentile      87.6  -> 44.5    t=5.25, p=0.0012
```

44.5 is a median cell. The location becomes unremarkable the moment the object leaves — **no
behavioural trace.**

![sequential occupancy maps](../../outputs/summary/fig_seq_occupancy.png)

**Figure 3.4** — `outputs/summary/fig_seq_occupancy.png`
(`scripts/trace/seq_occupancy_figure.py`, cache `outputs/trace/seq_occupancy.npz`). The same
result as spatial tuning rather than as a table: mean occupancy over 8 seeds, **one shared colour
scale across all four phases**, grey = the L-room's wall block. Green ring = the object's current
position; white rings = the other two candidates.

Read the panels left to right: the occupancy mass sits on the object in phases 1 and 2, and in
phase 3 it has left (4,7) entirely and redistributed toward the bottom-right. The fifth panel is
phase 2 minus phase 3 on a diverging scale — **red is what removal costs**, and it is concentrated
exactly on and around (4,7); blue is where that mass went. This is the figure to present: it makes
the "follows the present object, abandons the departed one" claim visible without asking anyone to
trust a percentile.

**Why this matters.** Three independent measurements now show one signature:

| measurement | object present | after it leaves |
|---|---|---|
| readout (§0, §3) | elevated | collapses, drop 0.024–0.030, p<0.005 |
| **behaviour (§3, this)** | **elevated** | **collapses, 5.0x, 8/8 seeds** |
| hidden state (§3) | never elevated | nothing to collapse |

This is the behavioural confirmation of "transfer, not trace", and it is what the §1
interpretation predicts: curiosity reward is prediction MSE, a function of the *readout*, while
the policy's input is `h` — which never changes.

**Note on reproducibility.** The first pass at this used a `collect_policy_rollouts` that seeded
only its start-direction rng, so action sampling, pRNN noise AND start placement all varied
between invocations — two runs gave (4,7) excess +32.5 and +17.5. Three generators had to be
seeded, not one: `torch.manual_seed` (action sampling and pRNN noise), `np.random.seed`, and
`env.env.reset(seed=)` for the gymnasium generator that owns `place_agent` and which neither of
the others reaches — the same trap documented in
`curious_george/evaluation/spatial.py`. **All numbers above are from the seeded version**, and
removing that noise strengthened the result rather than weakening it: (4,7) went from +17.5
(p=0.23) to +29.2 (p=0.0010), and the departure test from 7/8 seeds (p=0.082) to 8/8 (p=0.0012).
⚠️ Full bitwise determinism across back-to-back calls that REUSE one agent object is still not
established; the analysis builds a fresh net and agent per checkpoint, which is the case that
matters here. Full write-up: `probe-seq-behaviour-2026-08-13.md`.

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
`scripts/moser/moser_sessions.py`, `moser_analysis.py`, `moser_figures.py`.

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
| **Behaviour follows the present object and abandons it on removal** (§3) | ESTABLISHED | **Medium-High** | n=8 paired, seeded rollouts; the departure drop is 5.0x in 8/8 seeds (p=0.0022). The location control leaves (7,2) +60.1 (p<1e-4) and (4,7) +29.2 (p=0.0010) significant; (7,11) +6.2 (p=0.47) is not — it is high-traffic regardless |
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
   `scripts/moser/moser_decode_quadrant.py` with the start cell randomised and hidden from the decoder.
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
   `scripts/trace/trace_cell_figures.py` and `scripts/otc_figures.py plot|maps`; all seven and four
   respectively rebuilt cleanly, and the `otc_figures.py plot` console table reproduced the §2
   phase numbers exactly.
8. **Map-cache provenance: RESOLVED for the caches, and for the runs the figures name**
   (2026-08-13, `scripts/trace/verify_map_provenance.py`). Neither `maps_all.npz` nor
   `maps_dense.npz` records which run produced it, the `outputs/mila_omt*` runs have no config on
   disk (only `pN-*.pt` and `status.pt`), and no wandb entry survives for them — queried
   `blake-richards/curious-george` and `curious-george-otc`, where the only run dated 2026-07-30 is
   `prnn_curious_26-07-30-10-37-02`. Recovered by replay-and-correlate, with the pipeline asserted
   first (baseline checkpoint vs cached `baseline`, **r = 1.00000**) so a wrong-run answer could not
   be mistaken for a wrong-pipeline one:

   ```
   maps_all.npz    <- outputs/mila_omt/       125634 (7,11)   130252 (7,2)    130307 (14,7)
   maps_dense.npz  <- outputs/mila_omt_dense/ 165325 (7,11)   165916 (7,2)    165922 (14,7)
   all six diagonal matches r = 1.00000, margins +0.023 to +0.080
   ```

   The map route cannot reach the six dense runs no cached set was built from — their correlations
   are noise (margins 0.0003–0.019, three of them scoring *higher* against `baseline` than against
   any target). Those are recovered instead from the **readout**, where §0 says exposure writes the
   object: excess predicted green at each candidate over the pre-exposure baseline at the same cell.
   §0 also says that write is ~89% position-independent, so the route is reported against the three
   runs whose answer the map route already fixed — **positive control 3/3, every margin ≥ +0.039**
   (`--readout`, cache `outputs/trace/map_provenance_readout.json`):

   ```
   (7,11)  165325  172405  175326          three seeds per location, three locations;
   (14,7)  165922  172845  175805          the launch groups are LOCATION triples,
    (7,2)  165916  173016  180402          not seed triples
   ```

   ✅ **Independently corroborated for the three it overlaps.**
   `scripts/trace/trace_objvector_test.py:22-24` carries a `RUNS` dict mapping
   `(7,11)→165325, (14,7)→165922, (7,2)→165916` — recorded in code rather than in a config, which is
   why the config/wandb search missed it. It agrees with the replay recovery exactly. Two independent
   sources, so the mapping is settled rather than merely inferred; the other six runs are new.

   **This settles two things.** Figure 0.2's `[2,2,2] normal` column is `omt-cur-dot-0730-165325`,
   trained at **(7,11)**, matching its `[2,0,8]` and RANDPOS columns (confirmed independently from
   `wandb/run-20260801_120212-8xck38gb/files/config.yaml:9` and sibling configs) — the figure is
   correct as drawn. And `otc_figures.CONDITIONS["[2,2,2] fixedpos"]` names `165325, 172405, 175326`,
   which are **all (7,11)** — so §2's control arm is location-matched to its treated arm. That was an
   assumption with nothing behind it until now.

---

## Regenerating every figure in this document

Verified end-to-end on 2026-08-12 after the `scripts/` reorganisation: **17 of the 19
figures rebuild from committed code plus the caches under `outputs/`**.

(An earlier revision of this document required `uv run --no-sync`, because `minigrid` was
installed editable so that `MiniGrid-SquareRoom-v0` would resolve. That is obsolete: minigrid
is pinned from git again in `uv.lock`, so plain `uv run` is correct and `--no-sync` would now
give you a *stale* environment.)

```bash
uv run python scripts/summary_figures.py            # the 5 outputs/summary/ figures
uv run python scripts/trace/trace_cell_figures.py   # 7 figures in outputs/trace/
uv run python scripts/otc_figures.py plot           # fig_otc_phases/_encoding/_tradeoff
uv run python scripts/otc_figures.py maps           # fig_otc_maps/_maps_diff
uv run python scripts/moser/moser_figures.py        # the 4 outputs/moser/ figures
uv run python scripts/seq_figures.py \
    outputs/seq4/OTC_seq_5200_10340479/SEQ4-otc-p0.5-fixedpos-c1-0811-123058
uv run python scripts/trace/seq_phase_figure.py       # fig_seq_phases    (Figure 3.0)
uv run python scripts/trace/seq_behavior_figure.py    # fig_seq_behavior  (Figure 3.3)
uv run python scripts/trace/seq_occupancy_figure.py   # fig_seq_occupancy (Figure 3.4)
uv run python scripts/summary_figures.py fetch_entropy   # refresh the wandb cache (network)
```

Provenance of the map caches those figures read (§7.8) — asserts its own pipeline before reporting,
and its readout route reports its positive control with every result:

```bash
uv run python scripts/trace/verify_map_provenance.py --runs outputs/mila_omt/*
uv run python scripts/trace/verify_map_provenance.py --all --cache maps_dense.npz --step 2800
uv run python scripts/trace/verify_map_provenance.py --all --readout
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

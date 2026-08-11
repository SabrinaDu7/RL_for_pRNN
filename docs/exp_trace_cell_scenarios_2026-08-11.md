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

**Results:** _(pending)_

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

**Results:** _(pending)_

---

## 6. Standing methodological cautions

Carried from the earlier work; all were paid for at least once.

1. **Pooling hides effects.** Three separate times, averaging across the variable the mechanism
   runs on turned a real effect into a flat null. Always split by mask phase, by viewing
   condition, and by location before concluding anything.
2. **Never quote an effect size from n=1.** A "+142%" result became non-significant at n=10.
3. **Check a job has finished** before reading `max(steps)` — reading step 0 of a running job
   once produced a fabricated "catastrophic" result.
4. **Implausibly exact agreement is a bug signal**, not a finding.

# Compaction handoff — object / trace-cell work

**Read this first after a compaction.** Written 2026-08-11. Branch:
`sdu/object-into-hidden-state` (pushed). Everything below is committed.

---

## The overall goal

Make **object-trace cells** appear in the pRNN hidden state `h`: units with a place field at
the location where a novel object was, of the kind in `docs/trace-cells-spatial-tuning.png`.
**They do not currently appear.** Two completed investigations explain why; a third
(scenarios A/D/C) is in progress.

## Current standing instruction (the active goal)

1. ✅ **Establish a clear metric for trace/object-cell coding.** Done —
   `scripts/trace_metric.py`, validated (see below).
2. 🔄 **Test scenario A, then D.** A is running; D is next.
3. ⏳ **If both fail, launch scenario C**, which needs a fresh ~4 h `main_train` baseline.
   Run it on the **local GPU** — Mila needs an OTP the user cannot supply overnight.
4. Document methods + results extensively in `docs/` and wandb as we go.

The user checks back in the morning.

---

## What is already established (do not re-derive)

### Finding 1 — object memory is decoder-localised, not in the dynamics
Transplanting `W_out` alone from a trained net onto untrained dynamics reproduces the object
effect; the reverse does not. Gain-corrected: readout **+0.0625 ± 0.0130** vs dynamics
**+0.0157 ± 0.0104**, 9/9 runs, ratio 4.0×.
*Doc:* `exp_object_trace_cells_2026-07-30.md` §4.1.

### Finding 2 — the place code does not change under exposure
Median per-unit rate-map correlation to pre-exposure baseline: **r ≈ 0.98** across 3000
trajectories, all three object locations. Null in **three reference frames**: allocentric maps,
object-location modulation, object-vector (egocentric view frame).

### Finding 3 — why: redundancy
The object's location is a deterministic function of the agent's position, and `h` is already
a place code. So adjusting a linear readout is the cheaper gradient direction — `h` is
*redundant* with the object, not ignorant of it. **This is the root cause of everything.**

### Finding 4 — RL-side interventions raise encoding but not memory
`tasks.training.lr_trials=[2,0,8]` (freeze `W_out`, scale `W_in` 4×) raises linear
decodability of object presence from `h`: **0.6947 → 0.7214**, t=14.4, p<1e-5 (n=3 vs 10).
`W_in` is the lever — `[8,0,0]` (recurrent at 4×, `W_in` frozen) returns to baseline.
**But memory is unchanged**: information decays ~2× per masked step, near chance within three,
in all 13 conditions tested. *Doc:* `exp_object_into_hidden_state_2026-08-01.md`.

### Finding 5 — architecture fact that dictates all analysis
`thRNN_5win` has **`inMask = [True, False×5]`**: the observation reaches the net only **1
timestep in 6**, while it must predict at *every* step. So ph0 = input-driven encoding,
ph1–ph5 = memory. **Never pool them** — doing so hid a real effect for ten conditions.

### Finding 6 — environment constraints
Object is a **`FloorBright` floor tile**: non-blocking (identical trajectories with/without,
only 16.2% of timesteps differ in observation), `see_through_walls=True`, action space
`Discrete(4)` with **no interaction primitive**. It is a coloured patch of floor with no
behavioural consequence.

---

## THE METRIC (deliverable 1, complete)

`scripts/trace_metric.py`. For unit `u` and cell `c`:

    field_gain g(u,c) = mean rate in a radius-2 disc at c  /  unit's mean rate over valid bins
    trace score dg(u,c) = g_post(u,c) - g_pre(u,c)

**Within-unit null:** `dg` is computed at all 172 walkable cells; the unit's score is the
object cell's percentile in its own distribution. A unit is an **object cell** if that
percentile > 95. Population statistic = binomial test of frac > 0.05.

**Location-control matrix** (`location_control_matrix`) is the decisive test — the within-unit
null cannot see drift shared across units at one location:

    excess(L) = frac(L | object at L) - mean over M != L of frac(L | object at M)

**Validation, both required alongside any result:**
- **negative** (odd vs even probe trajectories, same net): frac = **0.0601**, p = 0.174 ✓
- **positive** (inject Gaussian bump into 10% of units): **100% recall at amplitude 0.01**,
  i.e. 1% of a unit's mean rate. Very sensitive.

**Result on existing data — no object cells anywhere:**

```
                          scored at:
run with object at:   (7,11)    (14,7)     (7,2)
        (7,11)        0.0321    0.0902    0.0200
        (14,7)        0.0140    0.1002    0.0501
         (7,2)        0.0581    0.0721    0.0341
excess:               -0.004    +0.019    -0.001
```

Column (14,7) is high regardless of where the object was → **drift, not coding**. The metric
correctly removes the (14,7) artifact that fooled an earlier analysis.

---

## Scenarios

**A — object IDENTITY varies, location fixed** *(running now)*.
4 colours (`green/blue/purple/yellow`) re-rolled per episode at (7,11). Position says
"something is here" but not "which"; only memory says which. Implemented as
`tasks.otc.colors=[...]` in `tasks/otc/`. Runs: `A_norm` (`lr_trials=[2,2,2]`) and `A_frz`
(`[2,0,8]`), 3 seeds each, 3000 trajectories, ~26 min per run.

**D — sparsity pressure on `h`** *(next)*. NOTE: `predNet.sparsity` (`f`) is **init-only**
(`norm.ppf(f)` for bias init), so it cannot be changed on a loaded checkpoint. The available
lever is **per-param-group weight decay**, scaled during exposure exactly like `lr_trials`.
Needs a `wd_trials` addition to `tasks/otc/task.py::train`.

**C — occlusion** *(only if A and D fail)*. Blocking `Ball`/`Box` **and**
`see_through_walls=False`, so the object genuinely leaves view and must be held. Requires a
fresh ~4 h `main_train` baseline. **Run locally** — no Mila OTP available overnight.

---

## Overnight plan and its timing constraint (2026-08-11 ~00:40)

Measured rate: **0.52 s/trajectory** (3000 trajs = 26 min on the RTX 4060).

The existing baseline is **79,679 trajectories** — reproducing it under occlusion would take
**11.6 h**, which does not fit before morning. So:

- Scenario **D was cut from 6 runs to 2** (one seed per weight-decay strength) as a go/no-go
  screen. Justified because A was decisively null at n=3 and D is a weak lever (L2, not L1).
  Seeds get added only if the screen shows anything.
- The occluded baseline then gets the rest of the night — roughly 6.5 h ≈ **45k trajectories**,
  about half the original baseline's training.

**Why the shorter baseline does not invalidate C:** the trace metric is a *within-lineage*
pre-vs-post comparison (occluded baseline → occluded exposure), so absolute training length
does not confound it. What must be checked is that the shorter baseline has a usable place
code at all — run the spatial-tuning analysis on it before trusting any C result.

## Key files

| path | what |
|---|---|
| `docs/README_object_experiments.md` | index: which doc answers what, where each method lives |
| `docs/exp_object_trace_cells_2026-07-30.md` | investigation 1 (characterisation) |
| `docs/exp_object_into_hidden_state_2026-08-01.md` | investigation 2 (interventions) |
| `scripts/trace_metric.py` | **the metric** + validation |
| `scripts/trace_probe.py` | fixed probe: collect once, replay per checkpoint |
| `scripts/trace_maps.py` | occupancy-masked rate maps, SI, nulls |
| `scripts/trace_presence_decoder.py` | presence decoding from `h`, **split by mask phase** |
| `scripts/otc_figures.py` | `collect` / `plot` / `maps` |
| `scripts/trace_cell_figures.py` | regenerates the six trace-cell figures |
| `tasks/otc/` | stochastic presence / random position / random colour |

Caches in `outputs/trace/`; all figures regenerate without retraining.

---

## Recurring failure mode — read before analysing anything

**Pooling.** Three separate times, averaging across the variable the mechanism runs on turned
a real effect into a flat null:
1. curiosity-reward map binned by *agent position* rather than conditioned on object visibility
2. pooling input-driven with memory-only timesteps
3. pooling the five input-mask phases

The third caused a "structurally impossible" conclusion reported to the user **twice** before
being caught. **Split by the mechanism's variable first.**

Other standing lessons: never quote an effect size from n=1 (a "+142%" result became
non-significant at n=10); check a job has *finished* before reading `max(steps)` (reading step
0 of a running job produced a fabricated "catastrophic" result); treat implausibly exact
agreement between two measurements as a bug signal.

---

## Gate

`uv run pytest` → **126 passed, 0 failed, 7 deselected**. Unchanged all session; keep it there.
Compute so far: ~11 h local RTX 4060, no cluster jobs.

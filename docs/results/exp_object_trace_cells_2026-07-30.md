# Object-trace cells in the pRNN hidden state

**Dates:** 2026-07-30 → 2026-08-01 · **Branch:** `sdu/omt-and-trace-cells`
**Status:** complete. Superseded for *interventions* by
`exp_object_into_hidden_state_2026-08-01.md`; this document is the characterisation.

*(Rewritten 2026-08-03 from a running log. Several claims in that log were retracted as the
work progressed; §7 lists them. The log is in git history at `e30113b` and earlier.)*

---

## 1. Summary

**The question:** after novel-object exposure, does the pRNN hidden state hold a spatial trace
at the object's location, like hippocampal object-trace cells
(`docs/trace-cells-spatial-tuning.png`)?

**The answer: no — and the reason is that the object is not stored in the hidden state at all.**

| finding | status | evidence |
|---|---|---|
| Object memory is **decoder-localised** (`W_out`, not the dynamics) | ESTABLISHED | readout +0.0625±0.0130 vs dynamics +0.0157±0.0104, 9/9 runs, ratio 4.0× |
| The place code is **unchanged** by exposure | ESTABLISHED | median per-unit map correlation r ≈ 0.98 across 3000 trajectories, 3 locations |
| No object tuning in **any** of three reference frames | ESTABLISHED | allocentric maps, object-location modulation, object-vector all null |
| Behaviour **is** object-directed | ESTABLISHED (2 of 3 locations) | occupancy percentile 18.6→94.2 at (7,11); 3/3 seeds at (7,2) |
| `(14,7)` apparent object effect is a **geometry artifact** | ESTABLISHED | baseline occupancy percentile 69 *before* exposure; change −33.7 |
| Mechanism: object in view → sustained prediction error → curiosity reward → approach | ESTABLISHED | 3/3 correspondence between error signal and behaviour |

---

## 2. Setup and environment facts

`tasks/omt/` exposes a pre-trained agent to a novel object at a fixed location, then probes
whether the pRNN still predicts the object once it is gone. Three object locations were used:
`(7,11)`, `(14,7)`, `(7,2)`, 3 seeds each, 3000 trajectories, checkpoints every 200.

Four properties of the environment constrain every conclusion below, all verified:

- **The object is a `FloorBright` floor tile** — its own docstring says "Colored floor tile the
  agent can walk over", `can_overlap()` returns `True`, and it renders as a flat filled
  rectangle. It **cannot change the agent's trajectory**: with a fixed action sequence, the
  object-present and object-absent probes have byte-identical `agent_pos` and `agent_dir`, and
  only **16.2%** of timesteps differ in observation.
- **`see_through_walls=True`** — the 7×7 egocentric view is not occlusion-masked, so the object
  is visible from most of the room. *(Confirmed from the constructor flag; the downstream
  effect is inferred from it, not traced through minigrid's `gen_obs_grid`.)*
- **The action space is `Discrete(4)`** (turn left, turn right, forward, stay). There is **no
  interaction primitive** — the agent cannot contact or investigate the object.
- **The object is a deterministic function of position**, so `h` — a place code — already
  implies it. This is the root cause of everything in §4.

---

## 3. Methods

### 3.1 The probe

A **probe** is a fixed, frozen-network measurement rollout used only to estimate rate maps.
Because it uses a `RandomActionAgent` whose actions are pre-generated independently of the
network, and observations depend only on the (fixed) environment, **the same trajectories are
valid for every checkpoint** — so a checkpoint is replayed, not re-collected, and any map
difference is in the weights.

- One trajectory from every (walkable cell, head direction) pair: **688 trajectories × 256
  steps**, ~176k pooled samples, **~828 samples per spatial bin**.
- Maps are 14×14 (`env.get_map_bins()`), **occupancy-masked**: 172/196 bins valid, exactly the
  walkable-cell count, so the L-room cut-out falls out of the data rather than being hardcoded.
- **Determinism gate:** `predict` injects fresh noise every call (two identical calls differ by
  ~0.4 in `h`), so `replay_checkpoint` seeds torch immediately before each forward. Replaying
  one checkpoint twice is **bitwise identical**.

`scripts/trace/trace_probe.py`, `scripts/trace/trace_maps.py`.

### 3.2 Spatial tuning statistics

- **Rate maps + Skaggs SI**, with a **trajectory-level shuffle null** (pair unit activity from
  trajectory *i* with positions from *j≠i*, preserving within-trajectory autocorrelation) and
  **split-half stability** across odd/even trajectories.
- **Null calibration measured:** false-positive rate of the 95th-percentile threshold on
  held-out shuffles = **0.058** against a 0.05 target.
- ⚠️ **The null does not discriminate here.** Null SI ≈ 0.006 against a real median of 0.759,
  so **499/500 units pass** and split-half r has median 0.995. Essentially every non-silent
  unit is spatially informative; "number of tuned cells" is not a useful headline. Rank by SI.
- **Bin weighting matters and my prediction about it was wrong.** Skaggs (occupancy-weighted)
  SI *penalises* fields in oversampled bins, because a high-weight peak raises the mean rate
  and shrinks the very contrast SI measures. So Skaggs favours *interior* fields and uniform
  weighting favours *edge* ones — the opposite of what I expected. Spearman 0.974 overall but
  only 23/32 top-32 overlap. `spatial_info(weighting=...)` supports both; report both.

### 3.3 Object-tuning measures (three reference frames)

1. **Allocentric object-location modulation** — (mean rate within radius 2 of the object cell −
   mean rate elsewhere)/(sum), with a **within-run null**: the same statistic computed at all
   172 walkable cells, reporting the object cell's percentile. This controls for the L-room's
   unequal accessibility without pooling seeds.
2. **Object-vector frame** — bin `h` by the object's position in the agent's 7×7 view
   (`view_frame_maps`), with `ctrl_locs` treated identically as controls.
3. **Readout-vs-dynamics swap** — build chimaeras from pre- and post-exposure checkpoints
   (`base dynamics + trained W_out`, and the reverse) and measure predicted green at the absent
   object location. Whichever chimaera reproduces the effect is where the information lives.
   **Gain-corrected**: `‖h‖` is ~9% smaller in trained nets with `‖W_out‖` ~2% larger, so a raw
   transplant over-drives one chimaera and under-drives the other — both errors biasing toward
   "the readout carries it". `W_out` is rescaled by the measured `‖h‖` ratio.

`scripts/trace/trace_readout_test.py`, `scripts/trace/trace_readout_gaincorrected.py`,
`scripts/trace/trace_objvector_test.py`.

### 3.4 Behaviour

On-policy rollouts in the object-**present** env at each checkpoint, 128 per point, scored as
the object location's **percentile among all 172 walkable cells** for two measures:
occupancy within radius 2, and time with the cell inside the 7×7 view. Within-run, so it does
not depend on pooling seeds — which matters, because pooling manufactures apparent clustering
when the object sits in a structurally high-traffic spot (§4.4). `scripts/trace/trace_behavior.py`.

### 3.5 Curiosity reward

The reward is per-step pRNN prediction MSE (`curious_george/rl/update/rewards.py`). Measured
two ways: binned by **agent position** (`trace_reward_map.py`) and conditioned on the object
being **in view** (`trace_reward_inview.py`). The second is the correct one — see §7.2.

---

## 4. Results

### 4.1 Object memory is in the readout, not the dynamics

Gain-corrected chimaera swap, 9 runs (3 locations × 3 seeds):

```
full effect     +0.0732 ± 0.0146
readout-only    +0.0625 ± 0.0130     ← reproduces ~85% of it
dynamics-only   +0.0157 ± 0.0104
readout > dynamics in 9/9 runs       sign test p ≈ 0.002,  ratio 4.0×
```

### 4.2 The place code does not move

Median per-unit rate-map correlation against the pre-exposure baseline:

| trajs | (7,11) | (14,7) | (7,2) |
|---:|---:|---:|---:|
| 0 | 0.841 | 0.910 | 0.916 |
| 400 | 0.980 | 0.981 | 0.979 |
| 2000 | 0.975 | 0.974 | 0.982 |
| 2800 | 0.980 | 0.960 | 0.957 |

An early transient over the first ~200–400 trajectories, then **r ≈ 0.98 for the remaining
2400+**. The transient coincides with `trainNovelObject`'s `lr_trials=2` learning-rate boost;
that is inferred from timing, not traced.

### 4.3 No object tuning in any frame

- **Allocentric:** Δ object-modulation diagonal (own location) **0.021** vs off-diagonal
  **0.028** — the wrong way round. Within-run percentile of the object cell among 172 cells
  oscillates between 10.5 and 79.1 across checkpoints, never sustained above 95.
- **Object-vector:** median view-frame SI change, object minus controls: **+0.0205, +0.0063,
  +0.0022**. The object's own SI is unchanged (+0.0005, +0.0001, −0.0174).

### 4.4 Behaviour is object-directed — at two of three locations

128 rollouts per point, trajectory 0 vs final checkpoint, 3 seeds each:

| object | mean Δ occupancy percentile | verdict |
|---|---:|---|
| (7,11) | **+45.7** | rises hard in 2/3 seeds (18.6→94.2, 13.4→90.7), falls in the third |
| (7,2) | **+38.7** | rises in **3/3** seeds from low baselines (4.7–25.0 at traj 0) |
| (14,7) | **−33.7** | **artifact** — baseline percentile 69 *before* exposure; gets *less* visited over training |

**In-view time is the better readout than occupancy** (rises in 7/9 runs vs 6/9, larger mean
change), as predicted from the object being a non-blocking, see-through-walls-visible tile.

### 4.5 The mechanism, end to end

The policy's input is `h` (`predict_single`), **not** the prediction. The readout reaches
behaviour only through the curiosity reward, which is prediction MSE:

```
W_out → obs_pred → prediction MSE → curiosity reward → PPO → policy
h ──────────────────────────────────────────────────────────────^   (policy INPUT, unchanged)
```

MSE ratio (object in view / not in view), minus the same for control locations:

| trajs | (7,11) | (14,7) | (7,2) |
|---:|---:|---:|---:|
| 0 | +0.443 | −0.096 | +0.230 |
| 200 | **+0.739** | +0.008 | +0.234 |
| 1400 | +0.451 | +0.028 | +0.349 |
| 2992 | +0.007 | −0.026 | +0.283 |

**3/3 correspondence with behaviour**: the two locations carrying an error signal show
attraction; `(14,7)`, which carries none, shows none. **The error never resolves** over 3000
trajectories — partial learning leaves residual error, curiosity keeps paying out, approach
persists. No lag needed.

---

## 5. Interpretation

Exposure writes the object into the **linear map from `h` to the prediction** and leaves `h`'s
spatial geometry alone. That single fact explains all of it: a clear pixel-space effect, real
object-directed behaviour (driven by the reward, which depends on the readout), and rate maps
frozen at r ≈ 0.98.

Why the dynamics never change: the object's location is already linearly decodable from `h`,
because `h` is a place code and the object is a fixed function of position. Adjusting a linear
map is the cheaper gradient direction. `h` is *redundant* with the object, not ignorant of it.

**Consequence for the original hypothesis:** searching for allocentric object-trace *place
fields* is the wrong target in this architecture. There is nothing in the rate maps to decay
after removal, so a removal phase would measure a flat line.

---

## 6. Bounds

- One environment (LRoom), one object type, three locations, 3 seeds each.
- The behavioural result is seed-dependent (2/3 locations, 6–7 of 9 runs).
- `(14,7)` should be excluded from pooled analyses, or the pooling should be replaced by the
  within-run null in §3.4.
- A **removal phase was never run** — `tasks/otc/` was built for it but the null results above
  made it uninformative. The trace-decay question is untested.
- Residual unexplained: 3 of 9 runs still exceed 100% readout share after gain correction.

---

## 7. Errors made, and what each cost

1. **"Global drift during exposure"** — retracted. The maps are nearly frozen (r ≈ 0.98); I had
   selected the handful of units that changed, precisely *because* they changed. *Cost: a
   wrong characterisation of the exposure phase, corrected one log entry later.*
2. **"Policy lags the world model"** — retracted. It rested on MSE **binned by agent position**,
   where error at the object cell peaks then collapses while occupancy climbs. It replicated at
   **0 of the other 2 locations**. The binning was wrong: the reward is generated at *viewing*
   positions, not at the object cell. Conditioning on visibility instead gives a sustained
   signal and a 3/3 match with behaviour. *Cost: a confidently-stated mechanism that was wrong.*
3. **Predicted occupancy weighting inflates corner-unit SI** — backwards. It *penalises* them
   (§3.2). *Cost: a wrong caveat attached to the tuned-cell ranking.*
4. **"Trained dynamics counteract the object signal"** — retracted. The >100% readout shares
   were a **gain mismatch** (`‖h‖` 9% smaller in trained nets), which also biased the headline
   effect size upward by ~12 points. *Cost: an invented mechanism; the corrected result is
   4.0× rather than 6.6×.*

**Standing lesson (this project's recurring failure mode):** *pooling*. Binning the reward map
by agent position, and later pooling input-driven with memory-only timesteps, each turned a
real effect into a flat null. Split by the variable the mechanism runs on before concluding
anything.

---

## 8. Reproduction

```bash
# probe (built once, reused for every checkpoint)
uv run python -c "from scripts.trace_probe import *"   # see build_probe/save_probe

# figures
uv run python scripts/otc_figures.py plot    # decoding + encoding + tradeoff
uv run python scripts/otc_figures.py maps    # spatial-tuning panels
```

| figure | shows |
|---|---|
| `outputs/trace/fig_tuned_units_main_train.png` | the pRNN has clean place fields — method works |
| `outputs/trace/fig_occupancy.png` | probe sampling and the occupancy mask |
| `outputs/trace/fig_trace_3loc.png` | nothing appears at the object, across 3 locations |
| `outputs/trace/fig_exposure_timeline_14_7.png` | same across 16 checkpoints |
| `outputs/trace/fig_behavior_n3.png` | behaviour is object-directed; (14,7) is the artifact |
| `outputs/trace/fig_si_weighting.png` | Skaggs vs uniform bin weighting |

Modules: `scripts/trace/trace_probe.py`, `trace_maps.py`, `trace_figure.py`, `trace_behavior.py`,
`trace_readout_test.py`, `trace_readout_gaincorrected.py`, `trace_objvector_test.py`,
`trace_reward_map.py`, `trace_reward_inview.py`.

Gate: `uv run pytest` → 126 passed, 0 failed, 7 deselected.

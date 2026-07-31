# Object-trace cells in the pRNN hidden state

**Started:** 2026-07-30
**Status:** planned — no experimental code written yet. Two enabling fixes committed
(`38e3732`, `c8fea19`).
**Branch:** `sdu/omt-and-trace-cells`

---

## 1. Purpose

The OMT/NOR task (`tasks/omt/`) exposes a pre-trained agent to a novel object at a known
location and asks, in **prediction space**, whether the pRNN still predicts the object once
it is gone. This experiment asks the **hidden-state** version, which is the one that maps
onto real hippocampal data.

In rodent hippocampus and LEC, some cells develop a field at the location of a familiar
object and **keep it for days after the object is removed** — object-trace cells. The
reference panel is `docs/trace-cells-spatial-tuning.png`: one cell's rate map across days 0–11,
field present and strong while the object is there, persisting after removal, gone by day 11.

**Questions:**

1. How many units in the pRNN hidden state are spatially tuned, and which?
2. After novel-object exposure, do any units acquire a field **at the object location**?
3. Once the object is removed and training continues, how long does that field survive?
4. Does the field track the *object* (moves when the object moves across runs) or the *room*?

**Deliverable:** the reference figure reproduced for the pRNN — rows = units, columns =
checkpoints spanning pre-exposure → object present → object removed, occupancy-masked rate
maps, shared colour scale — plus a decay curve and a matched behavioural timeline.

### Why the current code cannot answer this

- OMT has **no post-removal phase**: the object is present for the entire exposure phase and
  the probe is a frozen object-absent rollout (`tasks/omt/task.py::trainNovelObject`).
- Its metric is **pixel-space**, not hidden-state (`tasks/omt/metrics.py::quantify_object_learning`).
- There is **no rate-map machinery anywhere in `curious_george/`** (confirmed by grep; the
  only tuning-curve code is in the installed `prnn` package).

---

## 2. Findings that shaped the design

All confirmed by reading or running the code on 2026-07-30. Anything inferred is marked.

### 2.1 `h` has ONE theta window, not five

`thRNN_5win = partial(MaskedRNN, cell=LayerNormRNNCell, k=5)`
(`prnn/utils/Architectures.py:858`), but `MaskedRNN.__init__` converts `k` into
`inMask`/`outMask` boolean arrays and **never sets `self.k`** (`Architectures.py:707-762`).
`pRNN.forward` then does `k = 0 if not hasattr(self, "k")` (`Architectures.py:288-295`).

Verified by running `pN.predict` on `outputs/ckpts/pRNN_curious_26-07-23-10-06-25`:

```
pRNNtype: thRNN_5win | hasattr k: False
obs (1, 65, 147)  act (1, 64, 8)
h   (1, 64, 500)   min 0.0   frac exactly zero 0.548
```

**Consequences:** `torch.mean(h, dim=0)` throughout the codebase
(`predictiveNet.py:919`, `evaluation/spatial.py:53,141`) is a no-op; there is no per-theta-window
analysis available for this architecture. `h` is ReLU (≥ 0) and ~55% sparse, so rate-map
semantics are sound but the active-time threshold matters.

### 2.2 Task checkpoints could not be re-loaded — **fixed, `38e3732`**

Two independent blockers, both now fixed (see `tasks/README.md`, "Checkpoint interop"):

- Filename: steps were `pN-<count>.pt`; `get_ckpt_env_vars` looks for
  `predictiveNet_state.pt`. Now canonical, with `resolve_prnn_ckpt()` accepting either.
- `status.pt["optimizer_state"]` meant the **AC Adam** (1 param group, verified) from
  `main_train` and the **pRNN RMSprop** (4 param groups, verified) from a task.
  `setup_algo` loads that key into the AC Adam. The pRNN optimizer now has its own key.

This was a genuine defect, not a convention. The experiment design below still avoids a
checkpoint round-trip mid-experiment (phase A and B share one process), but the fix means
the trace runs can be chained off each other and resumed.

### 2.3 Object learning saturates in ~200 trajectories

From wandb `curious-george-omt/6o2h06kn` (= `omt-cur-dot-0730-125634`, object `[7,11]`,
seed 5200, 3000 trajectories):

| trajs | Goal Modulation | Goal − Ctrl |
|---:|---:|---:|
| 0 | −0.0004 | −0.0003 |
| 8 | −0.0301 | −0.0536 |
| 208 | **0.0501** | **0.0289** |
| 408 | 0.0548 | 0.0329 |
| 1008 | 0.0658 | 0.0434 |
| 2008 | 0.0453 | 0.0480 |
| 2408 | 0.0780 | 0.0663 |
| 2808 | 0.0515 | 0.0302 |

Total pRNN loss (3000 points, one per trajectory) falls `0.0107 → 0.0083` by traj 300 and is
flat at `0.0065–0.0070` from ~1300 on — but it is dominated by the room, not the object.

**Phase A of 3000 trajectories is ~15× longer than needed.** Also note the metric's noise is
comparable to its mean, because each point uses only 8 test trajectories
(`tasks.testing.trajs`). It is far too noisy to fit a decay curve to as sampled — the main
argument for the large probe in §3.1.

⚠️ That run predates `86f31be` and had `saving_interval=1000000`: it kept only first and last
checkpoints. Only post-`86f31be` runs have the 200-trajectory series.

### 2.4 The probe machinery already exists

`scripts/analysis_OMT.py::collect_eval_trajectories` (`:165-283`) runs **one trajectory per
(walkable cell, head direction)** = 172 × 4 = **688 trajectories** of 256 steps, and
`EvalTrajectoryConfig` has `include_hidden_states`. Existing output at
`outputs/data_cur_lroom_step1008_goal711/` confirms `hidden_states (688, 256, 500)` — ~176k
samples for a 14×14 map, ~900 per bin, versus the ~2k samples the OMT eval uses.
Persistence exists (`save_/load_eval_trajectories`, `:291-346`). **Reuse; do not rebuild.**

Those particular files are dated 2026-03-24 (pre-migration) and are **not** reusable as data —
the `bias_lr` change alone makes their dynamics non-comparable.

### 2.5 Environment facts

`minigrid/envs/Lroom.py::LEnv`: 16×16 grid, L-shaped interior (`Lwidth=10, Lheight=8`),
~172 walkable cells. Maps are **14×14** — verified: `env.get_map_bins()` →
`(14, 14, (0.5, 14.5, 0.5, 14.5))` (`prnn/utils/Shell.py:303`).

Two properties that matter more than they look:

- `see_through_walls=True`, so the 7×7 egocentric view is **not occlusion-masked**.
  *(Confirmed from the constructor flag; the downstream effect — minigrid skipping
  `process_vis` — is inferred from that flag, not traced.)*
- The novel object is a **`FloorBright` floor tile placed with `put_obj`** — non-blocking.
  The agent can stand on it and it does not distort trajectories. That removes the "was the
  field at the object or beside it" problem the rodent literature has.

Together these mean **prediction error is not localised to the object cell** — the object is
visible from most of the room, so the agent can extinguish the trace from a distance without
ever standing on it. This is why the primary behavioural readout is in-view time, not occupancy.

### 2.6 Dead config — **removed, `c8fea19`**

`tasks.training.resetOptimizer` / `continueTraining` were read by nothing in this repo.

### 2.7 Already fast enough

`BankedRGBPartialObsWrapper` is applied unconditionally for `pRNN+PO` inputs
(`curious_george/envs/factory.py:83-89`), so observation lookups are already banked. No
further speedup available from that direction.

---

## 3. Method

### 3.1 The probe: collect once, replay everywhere

A **probe** is a fixed, frozen-network measurement rollout whose only job is to estimate rate
maps. It is not training and it is not the OMT test trial.

The key simplification: the probe uses a `RandomActionAgent`, whose actions are pre-generated
independently of the network (`prnn/utils/agent.py:48`), and observations depend only on the
fixed object-absent env. **So the probe trajectories are identical for every checkpoint, every
arm, and every run.** Collect them once; per checkpoint just run a forward pass over the
stored `(obs, act)`.

This gives a perfectly matched probe by construction — which is what makes SI comparable
across checkpoints, since SI is sensitive to sample count and occupancy distribution — and
reduces per-checkpoint cost to one batched `predict`.

- One probe file: 688 trajectories × 256 steps, object-ABSENT LRoom, fixed seed.
- **Must pin the init hidden state.** `collect_eval_trajectories` calls
  `predict(..., randInit=True)` (`analysis_OMT.py:250-254`), drawing fresh noise per call.
  Pass an explicit fixed `state` (as `tasks/omt/task.py::_dual_net_predictions` does).
- Drop the first 20 steps per trajectory as onset transient (matches
  `evaluation/spatial.py:88`).
- Store only `obs`, `act`, `agent_pos`, `agent_dir`. Skip `obs_pred`/`obs_next` — they are
  2/3 of the existing 1.1 GB files.
- **Do not persist raw hidden states per checkpoint.** Persist derived products: rate maps
  `(500, 14, 14)`, occupancy, SI, null percentiles, per-unit modulation indices ≈ 1 MB per
  checkpoint. Keep raw `h` only for the handful of checkpoints used in the figure.

Checkpoints are 4 MB (3.2 pRNN + 0.8 status, measured), so they are cheaper than the data
they generate — hence checkpoint densely, probe offline, and keep the probe protocol a free
parameter that can be revised without re-running the cluster.

### 3.2 The training run: one process, phase A → phase B

New task `tasks/otc/` (copy `tasks/template_task.py`), built on `setup_task` / `train_phase`
directly rather than subclassing `ObjectMemoryTask` — whose `__init__` asserts the training
env contains the object (`tasks/omt/task.py:57-62`).

1. Load the `main_train` checkpoint via `CUR_CKPT_DIR`.
2. **Phase A** — train in object-PRESENT LRoom, ~600 trajectories, checkpoint every 40.
3. **Swap `comps.envs_train`** to object-absent copies. Both phases share one
   `TaskComponents`, so pRNN weights, AC weights and both optimizers carry over in memory —
   no checkpoint round-trip mid-experiment.
4. **Phase B** — continue training. Pilot length 1500 trajectories, checkpoint every 40.
5. Log training-rollout `agent_pos` / `agent_dir` every batch. This is the **only** thing
   that cannot be regenerated offline.

~53 checkpoints per run ≈ 210 MB.

### 3.3 Arms

| arm | phase B policy | question it answers |
|---|---|---|
| **primary** | curious AC keeps training | how fast does a curiosity-driven agent erase its own trace |
| **reference clock** | random action | decay per unit of *neutral* experience — the rat analogue |
| **drift null** | `tasks.control=True` (no object in phase A either) | how much do maps move under continued training regardless |

× 3 object locations (`[7,2]`, `[7,11]`, `[14,7]` — the set already in use), 3 seeds for the
pilot. A fully-frozen arm was considered and dropped: the probe's fixed trajectories already
remove most of the sampling noise it would have measured.

**Why the random-action arm is not optional.** After removal the pRNN still predicts the
object, so prediction error at the old location *goes up*, and the intrinsic reward
**attracts** the curious agent back. It then learns, error drops, and PPO reallocates. So the
curious arm measures "how fast does an agent that actively seeks the discrepancy erase it" —
a good question, but not the rat's, where decay is driven by ambient object-free experience.

**Predicted ordering, stated up front because it is the opposite of the initial intuition:**
behaviour is a *readout* of the trace (trace → error → behaviour), so behaviour should **lag**
the trace, not lead it. Behaviour abandoning the site while the trace persists would mean the
policy is stale relative to the world model — a result, not the default. A visitation *spike*
right after removal (the agent searching where the object used to be) is also a result, and
has a rodent analogue.

Note `weight_decay=3e-3` is on, so there is a background decay clock even at zero exposure.
"The trace never fades" is not an available outcome; the question is always the timescale
relative to the control arms.

### 3.4 Offline analysis

Lives in `scripts/` (notebook-friendly) while the criteria are being iterated on; moves to
`tasks/` once stable.

Per checkpoint, from the replayed probe:

**Rate maps + SI.** Reuse `pN.calculateSpatialMetrics(h_pool, pos_pool, env)`
(`prnn/utils/predictiveNet.py:801`) — the precomputed-activity entry point, already used this
way at `evaluation/spatial.py:148-156`. Binning from `env.get_map_bins()`.
⚠️ It **zeroes** SI for units active below threshold rather than masking
(`predictiveNet.py:855`), which puts a fake spike at exactly 0 in any SI distribution and
ranks silent units identically to untuned ones. Carry those as NaN in our layer.

**Occupancy masking.** A rate map bin visited twice is noise; a bin never visited is
meaningless. Compute the occupancy (sample count) map alongside the rate map and treat bins
below a minimum count (≈20) as missing — transparent in figures, **excluded from the
statistics**, since an undersampled bin inflates SI and can fake a field. Those are the white
patches in the reference panel. `plotTuningCurvePanel` already does the crude version
(`alpha = (sum(place_fields) > 0)`, `predictiveNet.py:1636-1650`) — mask only never-visited
bins; we want a real threshold.

**Tuned-cell criterion.** SI is positively biased: `h` is smooth in time (recurrent,
`ntimescale=2`) and so is position, so an untuned unit still scores SI > 0. With 500 units and
196 bins that bias is not small, and the count of "spatially tuned cells" is entirely set by
where the threshold goes. So:

- **Trajectory-level shuffle null**: pair unit activity from trajectory *i* with positions
  from trajectory *j ≠ i*, recompute SI, ~500–1000 times per unit → per-unit null. Tuned =
  real SI above the unit's own 95th percentile. This preserves within-trajectory rate,
  distribution and autocorrelation while destroying the position pairing. A plain row
  permutation would destroy the autocorrelation too and let almost every unit pass. A
  circular time-shift is the textbook null but fits badly here — the data is 688 independent
  256-step trajectories with state resets between them, so wrapping glues unrelated states
  together and 256 steps is short relative to the autocorrelation. Run it as a secondary null;
  the two should agree.
- **Split-half stability**: correlate odd- vs even-trajectory half-maps. Separate question —
  the shuffle asks "above chance", this asks "stable enough to plot and to compare across
  checkpoints". Keep both. `SpatialTuningAnalysis`'s `EV_space`
  (`SpatialTuningAnalysis.py:164-168`) is closer to split-half than to a significance test;
  use it as a cross-check against the old code, not as the criterion.

**Object-location modulation index** per unit, on the object-ABSENT map:
`(mean rate within radius r of the object cell − mean rate elsewhere) / sum`.

**Trace-cell definition:** spatially tuned post-exposure **AND** modulation index
significantly above *the same unit's* pre-exposure value **AND** still elevated ≥1 checkpoint
after removal. Unit identity is free here (same indices across checkpoints), which rodent work
does not get.

**Second reference frame.** Build the same maps in object-relative view coordinates via
`get_view_coords_batch` (`tasks/omt/metrics.py:87`). This separates two cell types that both
look like "object tuning": an *object-location* cell (fires when the agent is at the object
cell — the trace cell) and an *object-view / landmark-vector* cell (fires at a fixed bearing
and distance from the object). Given §2.5, the second class will be large and will masquerade
as diffuse spatial tuning. A vector trace surviving removal would be the more novel result.

### 3.5 Behaviour: a within-run null, no seed pooling

The hard part is showing behaviour is targeted at *this* location rather than generally
active. Previous approach pooled location statistics across many seeds, which is expensive.
Instead, build it as a map with a null — the same shape as the rate-map analysis:

1. For **every** occupiable cell (not just the object cell and the 4 `ctrl_locs`), compute
   S(cell) = fraction of training-rollout timesteps with that cell in the 7×7 view
   (`get_view_coords_batch`), plus occupancy within radius *r*.
2. Normalise each cell by the same statistic at the **pre-exposure checkpoint of the same
   run** → a ratio map. This absorbs the L-room's structurally unequal accessibility.
3. "Targeting" = the percentile rank of the object cell within the ratio distribution over
   all cells. **One run, one number, real null.**

Seeds then supply error bars, not the null — 3–5 rather than 21. The three object locations
remain the decisive test: the peak of the ratio map should **move with the object**.

**In-view time is the primary readout, not occupancy** — see §2.5.

---

## 4. Files

**New**

| path | purpose |
|---|---|
| `tasks/otc/main_task.py` | hydra entry; builds object-present and object-absent envs |
| `tasks/otc/task.py` | phase A → env swap → phase B, on `setup_task`/`train_phase` |
| `Configs/tasks/otc.yaml` | phase lengths, save cadence, arm switch |
| `scripts/trace_probe.py` | build/load the shared probe; replay a checkpoint → `h` |
| `scripts/trace_maps.py` | rate maps, occupancy mask, shuffle null, split-half, modulation index |
| `scripts/trace_behavior.py` | in-view/occupancy ratio maps + within-run null |
| `scripts/trace_figure.py` | the units × checkpoints panel + decay curve |
| `slurm/otc_task.sh` | modelled on `slurm/omt_task.sh` |

**Reused, not rewritten:** `scripts/analysis_OMT.py` (`collect_eval_trajectories`,
`save_/load_eval_trajectories`, `get_walkable_mask`), `curious_george/evaluation/task.py`
(`setup_task`, `train_phase`), `pN.calculateSpatialMetrics`,
`tasks/omt/metrics.py::get_view_coords_batch`, `curious_george/world_model/device.py`
(`eval_mode`, `on_device` — the generalised versions; the local copies in
`analysis_OMT.py:78-116` do not move `pN.state`).

**Not touched:** `tasks/omt/` beyond the checkpoint-interop fix, and the untracked
`tests/perf/` and `data/obs_bank/` files in the working tree.

---

## 5. Sequencing

1. **Probe + analysis on existing checkpoints.** Build the shared probe, replay the
   `main_train` checkpoint and one existing OMT checkpoint, produce rate maps and tuned-cell
   counts. Validates the whole analysis path before spending cluster time. Cross-check SI
   against `SpatialTuningAnalysis`'s `EV_space` on the same maps.
2. **`tasks/otc/` locally.** 100-trajectory phase A + 100-trajectory phase B; confirm the env
   swap, checkpoint cadence and behaviour logging.
3. **Pilot on Mila.** 1 object location × 1 seed × curious arm, phase A 600 / phase B 1500,
   checkpoint every 40. Measure the actual decay timescale offline.
4. **Set cadence from the pilot, then the full grid.** 3 locations × 3 arms × 3 seeds.
5. **Figure + update this document** with results and anything found along the way.

### Where each step runs

All training runs on Mila (steps 2–4 onward). Only the offline analysis runs locally, and it
only needs checkpoints (4 MB each).

Already local, nothing needed: `outputs/ckpts/pRNN_curious_26-07-23-10-06-25/` — the same
`main_train` checkpoint `slurm/omt_task.sh` points `CUR_CKPT_DIR` at. Enough to build and
validate the probe.

**Requested from Mila:** one **post-`86f31be`** OMT run directory — all `<traj>/` steps for
one object location, one seed (~15 checkpoints, ~60 MB). This lets step 1 validate "the field
appears at the object location" against real exposure data before `tasks/otc/` exists.
Pre-`86f31be` runs are useless here (§2.3). Sabrina will supply an OTP for the scp.

```
scp -r dus@mila:$SCRATCH/pRNN/<JOB_ID>/<run_name> outputs/
```

**Do not reuse** `outputs/data_cur_lroom_step*_goal*/` — dated 2026-03-24, pre-migration.

---

## 6. Verification

- **Probe determinism**: replay the same checkpoint twice → bitwise-identical `h`. This is
  the gate on the fixed init state; failure means `randInit` was not pinned.
- **Noise floor**: replay two adjacent checkpoints from a frozen stretch → map differences
  ≈ 0. Any nonzero floor here is the resolution limit of the decay curve.
- **Null calibration**: the shuffle null at the 95th percentile should yield ≈5%
  false positives when run against shuffled data.
- **Cross-check**: SI from `calculateSpatialMetrics` vs `SpatialTuningAnalysis` on the same
  activity should agree.
- **Regression gate**: `uv run pytest`. Current baseline **123 passed, 0 failed,
  7 deselected** (was 114/0/7 before `38e3732` added 9 tests). Re-measure before starting;
  do not trust a commit message.
- **End to end**: the pilot must write checkpoints at the stated cadence under
  `$RL_STORAGE/<run_name>/<traj>/`, and the figure must render pre/present/removed columns.

---

## 7. Risks and open issues

**The claim most likely to be wrong:** that `predict(batched=True)` is correct on the current
pin. `curious_george/evaluation/task.py:293-295` calls it "buggy";
`scripts/analysis_OMT.py:246-247` says the fix is on `LevensteinLab sdu/rl-integration`, which
*is* the current pin (`pyproject.toml:97`) — but that is an inference from a comment, not
something run. **Step 1 must verify batched vs serial `predict` agree** before anything is
built on it.

**Timescale risk.** The trace may decay faster than one 40-trajectory checkpoint interval,
giving a two-point curve. That is what the pilot is for; cadence is set after it, not before.

**Pre-existing flaws found, not fixed:**

- `curious_george/evaluation/spatial.py:111` unpacks **four** values from
  `pN.calculateSpatialRepresentation`, which returns **three** (`predictiveNet.py:1088`).
  That path only runs under `exp.eval_decoder=True` (default `False`), so it is dormant — but
  it will `ValueError` the moment the legacy spatial eval is switched on, which this
  experiment might tempt someone to do.
- `curious_george/rl/collect/agent.py:101` builds `state["SRs"]` with `np.append` and no
  `axis`, so it returns **flattened 1-D** of length `(T+1)*hidden_size`, not `(T+1, H)`. The
  probe uses a `RandomActionAgent` and does not touch it; do not reach for `SRs` as a
  hidden-state source.
- `scripts/analysis_OMT_h.py::get_ckpts`, `scripts/isomap.py` and
  `scripts/analysis_reward_map.py` hardcode a stale run-name path
  (`omt-cur-dot-noObs-goal{i}{j}/{step}/pN-{step}.pt`) that `main_task.py` has not produced
  for some time. Use `resolve_prnn_ckpt` and `$RL_STORAGE/<run_name>/<step>/`.

---

## 8. Log

**2026-07-30 (b)** — Step 1 done: probe + rate-map pipeline built and validated on the local
`main_train` checkpoint. Results and revisions:

- **`predict(batched=True)` is CORRECT on this pin** — the §7 risk is resolved, but not as
  expected. It takes **3-D `(B, L, X)`** and does the permute to `(1, L, X, B)` itself.
  With injected noise disabled, batched matches serial to **6e-6**.
  `scripts/analysis_OMT.py` was passing **4-D**, which raises in `clip_mask` — a real bug in
  the code the plan proposed reusing, now fixed. `tests/test_batched_wm_forward.py`'s
  2026-07-14 verdict ("predict(batched=True) IS broken, keep avoiding it") was stale and has
  been revised with tests.
- **The determinism threat was injected noise, not `randInit`.** `predict` draws fresh noise
  every call from `trainNoiseMeanStd = (0, 0.05)`; two identical calls differ by up to
  **0.39** in `h`. Dropout is separate and already off in `eval()`. `replay_checkpoint`
  seeds torch immediately before each forward, so every checkpoint sees the same realisation.
  **Determinism gate passes**: replaying one checkpoint twice is bitwise identical.
- **Probe built**: 688 trajectories × 256 steps, 38 s, 117 MB, at
  `outputs/trace/probe_lroom_noobj/`. Replay is **2.8 s per checkpoint**.
- **Coverage**: 162,368 pooled samples, ~828 per bin, and **172/196 bins valid** — exactly
  the walkable-cell count, so the occupancy mask reproduces the L cut-out on its own.
- **Null is correctly calibrated but does not discriminate.** False-positive rate of the
  95th-percentile threshold measured on held-out shuffles: **0.058** (target 0.05). But the
  null SI is ~0.006 against a real median of 0.759, so **499/500 units pass** (the 500th is
  silent), and split-half r has median **0.995**. In this system essentially every non-silent
  unit carries spatial information. Reporting "499/500 spatially tuned cells" would be
  technically true and useless — the discriminating quantity is SI *magnitude* and field
  localisation, not a significance threshold. Keep the null as a calibration check, rank by SI.
- Figures: `outputs/trace/fig_tuned_units_main_train.png` (top 32 by SI, occupancy-masked)
  and `outputs/trace/fig_occupancy.png`. Fields are clean and localised; a visible minority
  are wall-band/border cells rather than point fields.

**2026-07-30 (i)** — Object-vector frame tested (negative), the >100% anomaly explained, and
the removal mechanism verified.

**1. No object-vector coding either.** Binning `h` by the object's position in the agent's
7x7 view (`scripts/trace_objvector_test.py`), median SI change from baseline, object minus
control locations: **+0.0205 (7,11), +0.0063 (14,7), +0.0022 (7,2)**. The object's own
view-frame SI is unchanged (+0.0005, +0.0001, −0.0174), and every landmark — object and
controls alike — drifts down slightly. Note baseline view-frame SI is already 0.24–0.52,
which is not object coding but the trivial consequence of view-frame position being a
deterministic function of allocentric position and head direction; the control columns are
what removes it.

**Three reference frames, three nulls.** Allocentric place maps (r ≈ 0.98), object-location
modulation, and now object-vector tuning. The hidden state does not change.

**2. The >100% readout share is a GAIN MISMATCH — and it partly confounded the headline.**

```
||h||  baseline 11.195   trained 10.15-10.25   ratio ~0.91
||W_out||                                      ratio ~1.02
```

Trained nets run ~9% smaller hidden activity with a slightly larger readout, so a raw
`W_out` transplant onto baseline dynamics OVER-drives the prediction and the reverse
transplant UNDER-drives it. **Both errors push toward "the readout carries it"**, which is
the conclusion drawn from them. RETRACTS the speculation in log (f) that the trained
dynamics counteract the object signal.

Rescaling `W_out` by the measured `||h||` ratio (`scripts/trace_readout_gaincorrected.py`):

| | uncorrected | gain-corrected |
|---|---:|---:|
| full | +0.0732 ± 0.0146 | +0.0732 ± 0.0146 |
| readout-only | +0.0717 ± 0.0190 | **+0.0625 ± 0.0130** |
| dynamics-only | +0.0108 ± 0.0131 | **+0.0157 ± 0.0104** |
| readout > dynamics | 9/9 | **9/9** |
| ratio | 6.6x | **4.0x** |

Mean readout share 98% -> **~87%**. Direction and 9/9 consistency survive; the magnitude was
overstated by ~12 points. Three runs still exceed 100%, so the correction does not absorb
everything — residual unexplained.

**3. Removal mechanism verified (for `tasks/otc/`).** Phase B removes the object IN PLACE
(`base_env(env).new_obj_pos = None`, then `reset()`) rather than swapping env objects. This
is safe because the object is a non-blocking floor tile, so the walkable mask and algo
geometry (`loc_mask`, `loc_stats`, `subroom_size`) are unchanged. Confirmed empirically:

- grid fingerprint flips `170b5eec24237cf3` -> `6f48cd7fbcae481d`, matching a fresh
  object-absent env
- grid encoding byte-identical to a fresh object-absent env
- **observation bank identical** — the real risk, since `BankedRGBPartialObsWrapper` caches
  per grid layout; it re-keys inside `reset()` from the live fingerprint
  (`curious_george/envs/obs_bank.py:64-86`), so the mutation is picked up

This avoids any algo surgery: phase A and B share the same env objects and wrappers.

**2026-07-30 (h)** — Traced the mechanism from readout to behaviour. Two retractions and a
resolution.

**The policy does not read the prediction.** `ActorCriticAgent.getObservations` feeds
`acmodel(obs, SR=SR)` where `SR = predict_single(...)`, i.e. the hidden state
(`curious_george/rl/collect/agent.py:33-40, 74-101`); `ACModelSR.forward` concatenates it with
a one-hot HD (`curious_george/models.py:129-148`). `obs_pred` reaches behaviour only via the
curiosity reward, which is per-step prediction MSE
(`curious_george/rl/update/rewards.py:28-50`). So:

```
W_out -> obs_pred -> prediction MSE -> curiosity reward -> PPO -> policy
h ------------------------------------------------------------------^  (policy INPUT, unchanged)
```

**RETRACTED: "policy lags the world model."** Binning MSE by agent position
(`scripts/trace_reward_map.py`) showed error at the object cell peaking at traj 200
(1.42x room median) then falling to the 7th percentile by traj 2992 while occupancy climbed
to the 94th — which looked like a lag. It replicated at **0 of the other 2 locations**:
(14,7) sits below the room median from traj 0 onward, and (7,2) — the location with the
*strongest* behaviour — never exceeds it.

**RESOLUTION: the reward is generated at VIEWING positions, not at the object cell.** Binning
by agent position smears it. Conditioning on visibility instead
(`scripts/trace_reward_inview.py`), MSE ratio in-view/not-in-view, minus the same for
`ctrl_locs`:

| trajs | (7,11) | (14,7) | (7,2) |
|---:|---:|---:|---:|
| 0 | +0.443 | −0.096 | +0.230 |
| 200 | **+0.739** | +0.008 | +0.234 |
| 600 | +0.378 | +0.121 | **+0.577** |
| 1400 | +0.451 | +0.028 | +0.349 |
| 2400 | +0.214 | +0.149 | +0.501 |
| 2992 | +0.007 | −0.026 | +0.283 |

**3/3 correspondence with behaviour**: (7,11) and (7,2) carry a sustained object-driven error
signal and show real behavioural attraction (2/3 and 3/3 seeds); (14,7) carries **no** error
signal and shows **no** attraction — its apparent effect was baseline geometry throughout.

**The error never resolves.** It stays elevated for all 3000 trajectories at both real
locations. There is no lag to explain: the readout learns the object only partially, residual
error persists, curiosity keeps paying out, behaviour persists. The confirmed chain is

```
object in view -> sustained elevated prediction error -> curiosity reward -> sustained approach
```

with the place code untouched at every step.

Caveat: MSE measured on a random-action probe; the training reward is on-policy. Cross-cell
and in/out-of-view *rankings* should carry over, but the experience distribution differs.
n=1 run per location for the reward analyses.

**2026-07-30 (g)** — Behaviour at n=3, 128 rollouts per point, trajectory 0 vs final
checkpoint (`outputs/trace/fig_behavior_n3.png`). Percentile of the object location among all
172 walkable cells:

| object | run | occupancy %ile | in-view %ile | occupancy value |
|---|---|---|---|---|
| (7,11) | 165325 | 18.6 → **94.2** | 68.0 → 94.2 | 0.033 → 0.150 |
| (7,11) | 172405 | 13.4 → **90.7** | 68.6 → 93.0 | 0.035 → 0.122 |
| (7,11) | 175326 | 62.2 → 46.5 | 99.4 → 68.6 | 0.076 → 0.051 |
| (14,7) | 165922 | 87.8 → 94.2 | 40.1 → 79.1 | 0.114 → 0.173 |
| (14,7) | 172845 | 85.5 → **9.9** | 36.0 → 32.0 | 0.099 → 0.014 |
| (14,7) | 175805 | 34.9 → **2.9** | 7.6 → 29.7 | 0.045 → 0.012 |
| (7,2) | 165916 | 25.0 → 31.4 | 23.3 → **77.9** | 0.026 → 0.043 |
| (7,2) | 173016 | 22.7 → 58.7 | 9.9 → **66.9** | 0.023 → 0.071 |
| (7,2) | 180402 | 4.7 → **78.5** | 4.7 → **90.1** | 0.005 → 0.098 |

Per location, mean change in occupancy percentile: (7,11) **+45.7**, (7,2) **+38.7**,
(14,7) **−33.7**. Across all 9 runs, occupancy percentile rises in 6/9 (mean +16.9) and
in-view percentile in 7/9 (mean +30.4).

**Three conclusions, and the middle one is the one that matters for the averaging question.**

1. **The behaviour is real but seed-dependent, not universal.** (7,2) is the cleanest: 3/3
   seeds rise on both measures, from low baselines (occupancy percentile 4.7–25.0 at traj 0),
   and in-view rises by +65.7 percentile points on average. (7,11) rises hard in 2/3 seeds
   (+76, +77) and *falls* in the third (−16).
2. **(14,7) is confirmed as a geometry artifact.** Its baseline occupancy percentile averages
   **69** before exposure, and across seeds the object location gets *less* visited over
   training (−33.7 mean, with two seeds collapsing to the 9.9th and 2.9th percentile). An
   analysis that pools occupancy across seeds and locations would show a bump at (14,7) that
   is baseline geometry, not object attraction — and would miss that the change there is
   negative. This is the concrete answer to "is the averaged clustering real": partly, but it
   is diluted and partly manufactured by locations like this one.
3. **In-view time is the better readout than occupancy**, as predicted from the object being a
   non-blocking, see-through-walls-visible floor tile: it rises in 7/9 vs 6/9, with a larger
   mean change, and for (7,2) it rises in 3/3 while occupancy is nearly flat in one of them.

Caveat: still one number per (run, checkpoint) with no error bar within a run; the
percentile swings observed earlier across adjacent checkpoints mean these endpoint
comparisons carry real uncertainty.

**2026-07-30 (f)** — All three Mila jobs COMPLETED 0:0: 3 seeds x 3 locations, 16 checkpoints
each (144 total, 701 MB). **The readout result replicates at n=3 across all 9 runs.**

| object | run | full | readout-only | dynamics-only | share |
|---|---|---:|---:|---:|---:|
| (7,11) | 165325 | +0.0936 | +0.0974 | +0.0071 | 104% |
| (7,11) | 172405 | +0.0735 | +0.0968 | −0.0067 | 132% |
| (7,11) | 175326 | +0.0681 | +0.0974 | −0.0121 | 143% |
| (14,7) | 165922 | +0.0941 | +0.0699 | +0.0282 | 74% |
| (14,7) | 172845 | +0.0789 | +0.0627 | +0.0199 | 79% |
| (14,7) | 175805 | +0.0425 | +0.0493 | +0.0027 | 116% |
| (7,2) | 165916 | +0.0662 | +0.0506 | +0.0224 | 76% |
| (7,2) | 173016 | +0.0708 | +0.0634 | +0.0171 | 90% |
| (7,2) | 180402 | +0.0714 | +0.0575 | +0.0189 | 81% |

```
full effect    +0.0732 +/- 0.0146
readout-only   +0.0717 +/- 0.0190
dynamics-only  +0.0108 +/- 0.0131   (negative in 2 runs)
readout > dynamics in 9/9 runs      (sign test p ~ 0.002)
```

**OPEN ANOMALY, unexplained:** four runs exceed 100% — the readout-only chimaera produces a
*larger* object effect than the fully trained net, and all three (7,11) seeds do this. That
means the trained recurrent dynamics partially *counteract* the object signal the readout
carries. Candidate explanation not yet tested: a gain mismatch, i.e. transplanting `W_out`
onto dynamics whose `h` magnitude differs. Diagnose by comparing ||h|| and the un-contrasted
mean predicted green between base and trained nets before treating the >100% as meaningful.

**2026-07-30 (e)** — Resolved the dissociation. Two results.

**1. The behaviour is real in a single run — but one of the three locations is an artifact.**
32 on-policy rollouts in the object-PRESENT env at each checkpoint, percentile of the object
location among all 172 walkable cells (`scripts/trace_behavior.py`,
`outputs/trace/fig_behavior_timecourse.png`):

| object | occupancy %ile, first 4 → last 4 | occupancy value, first → last | median cell |
|---|---|---|---|
| (7,11) | 35 → 87 | 0.041 → 0.157 | ~0.045 |
| (7,2) | 22 at traj 0 → 92–98 from traj 400 | 0.013 → 0.16 | ~0.05 |
| (14,7) | 73 → 81, **already 87.2 at traj 0** | 0.128 → 0.124 | ~0.045 |

(7,11) and (7,2) show genuine object-directed behaviour without pooling seeds. **(14,7) does
not** — it is elevated before exposure could cause it, being near the L-notch corner and
structurally high-traffic. This is the same false positive that inflated its rate-map
percentile, and it is exactly what averaging across seeds would hide. Caveat: 32 rollouts is
too few (percentiles swing 98.8 → 0.6 between adjacent checkpoints); rerun at 128 before
quoting numbers.

**2. The object information lives in the READOUT, not the recurrent dynamics.**
`outlayer` is `Linear(500 -> 147, bias=False)` + `Sigmoid`, parameter `W_out`. Building
chimaeras from the pre- and post-exposure checkpoints and measuring predicted green at the
absent object location relative to control locations (`scripts/trace_readout_test.py`):

| object | trained (full effect) | base dyn + TRAINED readout | trained dyn + BASE readout | readout share |
|---|---:|---:|---:|---:|
| (7,11) | +0.0988 | **+0.0928** | +0.0150 | 94% |
| (14,7) | +0.0941 | **+0.0699** | +0.0282 | 74% |
| (7,2)  | +0.0658 | **+0.0537** | +0.0242 | 82% |

Transplanting only `W_out` onto the untrained dynamics recovers 74–94% of the object effect;
transplanting the trained dynamics with the old readout recovers 15–37%. Relative weight
change is consistent but not by itself decisive: `W_out` 0.146, `W` 0.124, `W_in` 0.113,
`bias` 0.052.

**This explains everything above.** Exposure writes the object into the linear map from `h`
to the prediction, leaving `h`'s spatial tuning geometry alone — hence a clear pixel-space
effect, clear object-directed behaviour, and rate maps frozen at r ≈ 0.98.

Three caveats that matter:
- The shares do **not** sum to 100% (94+15, 74+30, 82+37), so the halves are partly
  redundant; `W_out` was trained jointly with the dynamics and this is not a clean
  decomposition.
- The transplant works partly *because* `h` barely moved. "The readout carries it" and "the
  dynamics did not change" are two views of one fact, not independent confirmations.
- n=1 seed per location; injected noise was zeroed for determinism, unlike the OMT eval path.

**Consequence for the plan.** If the object memory is readout-localised, the trace should
decay on the timescale of `W_out` being overwritten, not of place-field remapping. That
changes what `tasks/otc/` should measure: the primary readout becomes predicted-green at the
old object location over the removal phase, with the rate maps as the control that should
stay flat. Searching for allocentric object-trace *place fields* looks like the wrong target
in this architecture.

**2026-07-30 (d)** — Relaunched on Mila post-`86f31be` (jobs 10252642 / 10252658 / 10252659,
3 seeds each, `--time=2:00:00`). Seed 1 of each landed: **16 checkpoints** per run
(0, 200, …, 2800, 2992), which is what the whole design needed. Replayed all 46 checkpoints
in one pass. Seeds 2–3 still running.

**The exposure phase barely changes the spatial code.** Median per-unit map correlation
against the pre-exposure baseline, over 172 valid bins:

| trajs | (7,11) | (14,7) | (7,2) |
|---:|---:|---:|---:|
| 0 | 0.841 | 0.910 | 0.916 |
| 200 | 0.882 | 0.977 | 0.941 |
| 400 | 0.980 | 0.981 | 0.979 |
| 1000 | 0.948 | 0.982 | 0.976 |
| 2000 | 0.975 | 0.974 | 0.982 |
| 2800 | 0.980 | 0.960 | 0.957 |

So there is an early transient over the first ~200–400 trajectories, then the maps sit at
r ≈ 0.98 for the remaining 2400+. This **corrects the "global drift" reading** in log (c):
the maps are not drifting much at all. What log (c) picked up was a handful of individual
units changing, selected precisely because they changed.

The early dip is worth a look on its own — `trainNovelObject` multiplies the pRNN learning
rate by `tasks.training.lr_trials` (default 2) for the exposure phase and restores it after
(`tasks/omt/task.py`). A transient at exactly that point is consistent with the LR boost, but
that is **inferred from the timing, not traced**; the no-object control arm would settle it.

**Still no object-location field.** Within-run null (percentile of the object location among
all 172 walkable cells, per checkpoint), `outputs/trace/fig_exposure_timecourse.png`:

| object | mean pop-mean pct | mean top-25 pct | ever sustained >95? |
|---|---:|---:|---|
| (7,11) | 51.5 | ~48 | no |
| (14,7) | 57.5 | **80.9** | touches 95 at 4 of 16, never sustained |
| (7,2) | 71.6 | 59.9 | no |

The traces oscillate hard — (14,7) swings from percentile 4.1 at traj 1200 to 98.8 at
traj 2400. Since the probe is a fixed deterministic replay, that swing is real weight change,
not measurement noise; but an "effect" that appears and vanishes between checkpoints 200
trajectories apart is not a trace.

**The (14,7) elevation is not object-driven.** Its top-25 percentile is already **86.6 at
trajectory 0** — i.e. after 8 trajectories, before exposure could plausibly cause anything.
Whatever elevates that location is a property of the location or the baseline network, not
of the object. This is the control observation that kills the most tempting positive reading.

The map timeline (`fig_exposure_timeline_14_7.png`) shows the same thing visually: `#22`,
`#348`, `#369`, `#162` are essentially frozen across all nine columns, and nothing appears
at the `+`.

**2026-07-30 (c)** — Pulled the three cluster runs and ran the object-following test.

**All three Mila OMT runs are pre-`86f31be`** — they have the `omt/` path level, `pN-<n>.pt`
filenames, and only steps `0` and `2992`. No checkpoint series exists anywhere. Confirmed
the local baseline is byte-identical (sha256 `c1e43a6b…`) to the `CUR_CKPT_DIR` those runs
seeded from, so the pre-exposure column is exact. Also noted: `train_phase` calls
`on_save(index)` **after** the update, so the folder named `0` has already seen 8
trajectories — it is a near-baseline, not a baseline.

That still gives **3 object locations × (pre, post)**, which is the decisive
"does tuning follow the object" contrast. Result:

**Negative. No object-location-specific change in the allocentric rate maps.**

Δ object-modulation (post − pre), population mean over 500 units:

| exposed at | eval (7,11) | eval (14,7) | eval (7,2) |
|---|---:|---:|---:|
| (7,11) | 0.0045 | 0.0201 | 0.0188 |
| (14,7) | −0.0052 | 0.0131 | 0.0175 |
| (7,2)  | 0.0723 | 0.0465 | 0.0453 |

Diagonal mean **0.021** vs off-diagonal **0.028** — the wrong way round. The `(7,2)` row is
elevated everywhere, i.e. a global drift, not a local effect.

The within-run null (Δ-modulation at all 172 walkable cells, percentile rank of the run's own
object location) agrees, across every aggregation — population mean, 95th percentile, max, and
mean of the top 25 units. Own-location percentiles land between **10.5 and 79.1**; a real
effect would sit above 95. Units selected by Δ-modulation turn out to be units whose maps
changed *generally* (`outputs/trace/fig_trace_3loc.png`), not at the object.

**This is not yet evidence against trace cells.** Caveats, in order of importance:

1. Only 2 timepoints, 3000 trajectories apart. Continued training drifts the whole
   representation; a localised effect can be buried. Dense checkpoints are the fix and do not
   exist yet.
2. n = 1 seed per location, no no-object control run available locally.
3. **The pixel-space metric DID show learning in this exact run** (Goal Modulation +0.050,
   Goal − Ctrl +0.029 at traj 208, §2.3). So the object *is* encoded — just not as an
   allocentric rate-map change. That dissociation is itself a result, and it points straight
   at the object-relative frame (§3.4), which has **not been tested yet**. Doing that is the
   cheapest next experiment: the map machinery already exists, only the binning coordinate
   changes.
4. Modulation radius fixed at 2.0 bins, untuned.

**Blocked on:** a dense checkpoint series for the actual trace timeline — which needs
`tasks/otc/` and a re-run, not an scp.

**2026-07-30 (a)** — Design settled. Findings §2.1–2.7 established. Two enabling fixes committed:
`38e3732` (checkpoint interop: canonical filename + `resolve_prnn_ckpt` + separated optimizer
keys + legacy guard, 9 new tests) and `c8fea19` (dead `resetOptimizer`/`continueTraining`
config removed). Gate at 123 passed / 0 failed / 7 deselected. No experimental code written yet.

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

**2026-07-30** — Design settled. Findings §2.1–2.7 established. Two enabling fixes committed:
`38e3732` (checkpoint interop: canonical filename + `resolve_prnn_ckpt` + separated optimizer
keys + legacy guard, 9 new tests) and `c8fea19` (dead `resetOptimizer`/`continueTraining`
config removed). Gate at 123 passed / 0 failed / 7 deselected. No experimental code written yet.

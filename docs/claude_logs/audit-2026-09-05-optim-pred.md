2026-09-05 · branch `sdu/optim-pred` @ `71890bd` · fork `prnn` pinned `852cc7d2` (`sdu/ce-loss`, equal to the local `../pRNN_new` HEAD) · nothing in the tree was changed by this audit

# Audit: correctness, naming, redundancy, optimization

## How to read this

**Baseline gate, recorded before reading a line of code.** `CG_DEVICE=cuda uv run pytest -q`
on this machine (RTX 4060): **628 passed, 1 deselected, 29 warnings, 254 s, exit 0**. Full
output: `throwaway/2026-09-05/pytest_baseline_cuda.txt`. The one deselected test is the
`slow`-marked function in `tests/test_ckpts.py:148`. Every golden fixture under
`tests/golden/` predates the audit (mtimes 2026-07-17 to 2026-08-31). Working tree clean
apart from `throwaway/`.

**Coverage.** Read in full: every module under `rl/`, `models/`, `training/`, `envs/`,
`evaluation/`, `utils/`, `configs.py`, `main_train.py`, `log_and_store/storage.py` and
`provenance.py`, `check/config_keys.py`, the five `slurm/` scripts, the three golden
capture scripts and `tests/small_config.py`; in the fork, `predictiveNet.py`
(constructor, `predict`, `predict_single`, `reset_state`, `trainStep`,
`recordTrainingTrial`), `Architectures.py` (`pRNN`, `restructure_inputs`, `clip_mask`,
`forward`, `MaskedRNN`), `lossFuns.py`, `checkpoints.load_pN`, `ActionEncodings.SpeedHD`,
`Shell.collectObservationSequence`/`env2pred`. Skimmed by function list and callers only:
`log_and_store/wandb.py` (1,488 lines of analysis tooling), the tail of
`check/wandb_compare.py`, and the bodies of the 58 test files (their docstrings were read).

**Status tags.** Every load-bearing claim carries one:

- **CONFIRMED** - I read the cited lines and followed the calls, or ran the cited command
  (outputs in `throwaway/2026-09-05/`). File:line references are to this commit.
- **INFERRED** - a reading I could not close by execution; the sentence says what would.

**Severity.** 🔴 wrong now, or a number a reader would misread · 🟠 latent, protocol, or
"fails loudly but late" · 🟡 smell, clarity, cost.

**Scope.** `curious_george/`, `main_train.py`, `tests/`, `slurm/`, `docs/*.md` for
contradictions with code, and the fork at the pinned commit for the contracts the adapter
relies on. `throwaway/` excluded. Downstream consumer checked: `../experiment-curiousgeorge`
(pins this repo at `71890bd`); where it imports something flagged below, the consumer is
named and nothing is called dead.

**Production path, as the config API reports it** (`throwaway/2026-09-05/`, second script):
`multienv-fast` = DEVICE backend, 256 streams, 1 episode of 256 steps each, pooled world
model at 8 episodes per step, rollout + world-model + policy CUDA graphs on, curiosity graph
off, `compile=LAYER`, whitened advantages, entropy 0.035, `reward_alignment=next_obs`,
**`action_offset=0`**, and loss/readout/focal defaults MSE/linear/None, so every CE, focal
and MLP run has those typed on the command line (`slurm/multienv.sh`).

---

## 0. Read this first

The ten things to decide tomorrow, in the order I would take them.

1. 🔴 **The circuit production runs is not the "desired" one** (C1). `circuit-desired.png`
   shows `action_offset=1` (policy acts on h[t], which already holds obs[t] and HD[t]).
   The preset default is 0, no launcher sets it, and `docs/ce-single-room-2026-08-31.md:53`
   records the decision to run every CE/focal arm at 0. At offset 0 the policy acts on
   h[t-1]: a state built from obs[t-1], HD[t-1] and the forward bit of a[t-1]. It has not
   seen obs[t]. HD[t] reaches the policy only through the separate one-hot. If the figure
   is the intent, every focal/CE result so far is on the other circuit.
2. 🔴 **`main_train.py`'s "degrade to no-wandb" path crashes at the first world-model
   step** (C4). The fork's `PredictiveNet` keeps `wandb_log=True` and calls `wandb.log`;
   the installed wandb 0.28.0 raises before `init` (confirmed by running it).
3. 🔴 **`setup_algo` still zeroes the episode cut when the pRNN is frozen, while its own
   comment says it no longer does** (C5). `evaluation/task.py:114` knows and works around it.
4. 🔴 **Stated and realized gradient-step budgets differ when the rollout does not divide
   the budget** (C2). The half-budget `multienv.sh WM=21968` runs did 21,984 world-model
   and 87,936 policy steps, not 21,968 / 87,872, and provenance says 686 rollouts for a run
   that did 687. Nothing validates divisibility.
5. 🔴 **`policy_grad_steps` is logged as if the policy trained, for RANDOM and
   `freeze_params` runs** (C3). Same for `prnn_grad_steps` under `freeze_params`.
6. 🔴 **Downstream OMT machinery in `evaluation/task.py` has two bugs** (C6, C7):
   `collect_eval_rollouts_batched` passes an `int` where `BatchedSRTrackerShim` takes a
   list (a `TypeError` on the first call), and `train_phase` logs the mean and the std of
   the curiosity reward under the same wandb key back to back.
7. 🟠 **`checkpoint_series` rebuilds the run's config from four CLI flags instead of the
   run's `provenance.json`** (C9). An 8-room `--env.source.positions` run would be scored
   on the first five rooms with no error. `prediction_figures.plot_run_predictions` already
   does it right; the two tools disagree.
8. **Redundancy** (§3): about 1,300 of `log_and_store/wandb.py`'s 1,488 lines have no
   caller in this repo (one downstream import); `AsyncShellPool` and everything `ASYNC`;
   `IntrinsicReference` and the `intrinsic` plumbing; `a2c_loss`; `LEGACY_DECODER`; the
   theta-cycle branches; `EnvironmentFeaturesAnalysis`; the retired `multienv` preset kept
   for one test; eight orphaned golden fixtures; four dead config surfaces that look live
   (`video_every_episodes`, `behaviour_timesteps`, `LandmarkKind.size`, `input_type`).
   Ruff finds 57 unused imports and 4 unused variables.
9. **Naming** (§2): the algo constructor carries a second set of defaults that disagree
   with the config's (`action_offset=1`, `reward_alignment="legacy"`, `discount=0.99`,
   `value_loss_coef=0.5`, `lr=1e-3`, `entropy_coef=0.01`); error messages and docstrings
   name Hydra-era flags that no longer exist; the fourth action has three names; the
   wandb project has three homes; there are three probe seeds.
10. **Optimization** (§4): nothing optimized is wrong, and the graphs are gated. Costs that
    remain: the walkable arm still pays the per-segment MiniGrid reset storm, the fork does
    a host sync, a pandas concat and a `wandb.log` on every world-model step, random-agent
    runs compute a curiosity pass nobody reads, and the online spatial probe steps the
    pRNN through per-step Python that bypasses the vectorized adapter.

---

## 1. Correctness

### 1.1 The RL wiring: what is CONFIRMED correct

This is the trace the user asked for, on the production path (DEVICE backend), for both
circuits. Everything in this table was confirmed by reading the cited lines and following
the calls; the batching claims are additionally gated by the tests named, all of which pass
in the baseline.

| quantity | `action_offset=0` (production default) | `action_offset=1` (the figure) | where |
|---|---|---|---|
| pRNN input row t | (obs[t] masked by phase, fwd(a[t]), HD[t]) | (obs[t] masked, fwd(a[t-1]), HD[t]); row 0 = (obs[0], no action, HD[0]) | rollout: `rl/collect/collector.py:527-528`; training rows: `models/prnn_adapter.py` `train_on_episodes_batched`; reward rows: `prediction_mses_device` (`:767`) |
| target of row t | obs[t] | obs[t] | `predOffset=0` hard-coded in `MaskedRNN`, fork `Architectures.py:861`; `predictions_for_room` asserts it at runtime |
| what the policy consumes at step t | h[t-1] + one-hot HD[t] | h[t] + one-hot HD[t] | `collector.py` policy forward uses `state.sr` from the previous SR step; `models/policy.py` `ACModelSR.forward` |
| curiosity reward for a[i] | error of row i+1 (`target_offset=1`) | error of row i+1, which is the row that encodes a[i]; asserted `prnn_adapter.py:863` | `_curiosity_errors` slices `[:, target_offset:]`; `reshape(B, segments, L).reshape(B*T)` matches the collector's `b*T + t` layout |
| row 0's error | dropped | dropped | same slice |
| episode boundaries in GAE | `masks[t+1]=0` after a segment close; `final_masks = state.mask_b = 0` | same | `collector.py:784`; `rl/update/advantage.py::compute_gae` |
| PPO ratio | `log_prob` = stored normalized logits gathered at the action, bit-exact with `Categorical.log_prob` | same | `collector.py` record block; `rl/update/losses.py` |
| update-time inputs | SR + direction, the same two fields the rollout used | same | `rl/update/policy.py::_index_policy_batch`; `policy_graph._FLAT_FIELDS` |
| phase alignment | rollout tracker and training pass both start each segment at phase 0 (row 0 shown, rows 1..5 masked) | same | `BatchedSRTracker.reset_all` zeroes `phases`; `clip_mask` tiles `inMask_f` from row 0 |

**Two gradient-step counters.** `training/schedule.py` derives everything from the two
stated budgets; on `multienv-fast` that is 32 world-model steps and 128 policy steps per
rollout of 65,536 environment steps, the 4:1 ratio the launcher holds. Confirmed by the
config-API dump. Two caveats are findings C2 and C3 below.

**Batching equivalence: gated, and what differs by design.**

- Gated bitwise in the baseline: device vs `SERIAL_TABLE` trajectories
  (`tests/test_device_collector.py`), B=2 batched tracker vs two serial streams at zero
  noise (`tests/test_batched_tracker.py`, `tests/test_action_offset.py`), rollout graph at a
  saturated policy (`tests/test_cuda_graph_rollout.py`), world-model graph without dropout
  and noise (`tests/test_cuda_graph_wm.py`), curiosity graph
  (`tests/test_cuda_graph_curiosity.py`), policy graph (`tests/test_cuda_graph_policy.py`),
  vectorized SpeedHD rows vs the fork's `env2pred` (`tests/test_batch_format.py`).
- Differs by design and is documented in code: `BatchedSRTracker.reset_all`
  (`prnn_adapter.py:1278-1280`) starts every stream from **zeros**, while every
  `predict`/`trainStep` pass and the B=1 tracker start from `actfun(N(0, 0.05))`
  (fork `predictiveNet.py:283-285`, `reset_state`). So at B>1 the rollout's h[0..few] and the
  training pass's differ systematically, not just by RNG. The memory note explains why
  unifying them was reverted. **INFERRED** small (a ReLU of 0.05-std noise); what would
  confirm it is one measurement of |h_rollout - h_train| over rows 0..10 on a trained
  checkpoint. It is the one place the "same quantity, two code paths" caption of the figure
  is not merely float noise.

### 1.2 Findings

Each item: status · evidence · impact · the shape of the fix (not applied).

**C1 🔴 Production trains and evaluates the offset-0 circuit; the "desired" figure is offset 1.**
CONFIRMED. `configs.py::ArchPrnnCfg.action_offset = 0`; `_parity` and `_multienv_fast`
do not set it; `slurm/multienv.sh` passes no `--arch-prnn.action-offset`; the config-API
dump prints `action_offset 0` for `multienv-fast`; `docs/ce-single-room-2026-08-31.md:53`:
"Decision: all Phase-1 arms at `action_offset=0`"; no CE/focal document mentions the
offset again. `outputs/figures/circuit-desired.png` is titled "policy acts on h[t]", which
is offset 1 (`prediction_figures.py::trace_circuit`, `policy_state_label`). Impact: under
offset 0 the state the policy acts on does not contain the current observation, and the
row the world model trains on pairs obs[t] with an action that has not happened yet
(the "pending-action bit" the A/B measured at 1.000 decodability). Whether this is
intended is a methods decision; the A/B (`docs/action-offset-ab-2026-08-29.md`) found no
sRSA difference at n=2 and a consistently ~35% higher prediction loss at offset 1, which
is presumably why 0 was kept. Fix shape: either rename the figure, or make offset 1 the
preset default and re-pin the goldens that change.

**C2 🔴 Stated budgets are not the realized ones when the rollout does not divide them.**
CONFIRMED by computation (`throwaway/2026-09-05/`, first script, against the real
`Config`/`TrainingSchedule`): `multienv-fast --train-prnn.total-grad-steps 21968
--train-policy.total-grad-steps 87872` (the half-budget arm `multienv.sh WM=21968` launched
for the focal work) yields `total_rollouts=686` in provenance while
`training/loop.py`'s `while num_frames < total_env_steps` runs 687 rollouts, realizing
21,984 world-model and 87,936 policy steps and 45,023,232 environment steps against stated
44,990,464. The full budget (43,936) divides exactly. `Config.__post_init__` checks that
`ppo_batch_size` divides the rollout and that `episodes_per_grad_step` divides the
rollout, but not that the rollout divides the budget (`total_episodes %
episodes_per_rollout == 0`), and `TrainingSchedule.as_dict` records the stated numbers.
Impact: provenance and the wandb `prnn_grad_steps` axis are off by one rollout (0.07%
here; up to one rollout in general) for any budget not a multiple of
`episodes_per_rollout / episodes_per_grad_step` (32 on this preset). The README calls the
32x-varying case "integer rounding"; it is, but it is unvalidated and unrecorded. Fix
shape: a `Config.__post_init__` divisibility check, or record realized counts.

**C3 🔴 `policy_grad_steps` and `prnn_grad_steps` are logged from the schedule, not from
what trained.** CONFIRMED. `training/loop.py:297` passes
`**schedule.gradient_steps_at(update)` unconditionally; `training/logging.py:79` emits
`policy_grad_steps` for every run. For `arch_policy.agent=RANDOM` the loop passes
`update_params=False` (`loop.py:250-252`), so the policy takes zero steps while the axis
climbs to 175,744; for `freeze_params=True` neither learner steps and both axes climb.
Impact: a wandb panel read on `policy_grad_steps` for a random baseline is a fabricated
axis. Fix shape: make `gradient_steps_at` take the two booleans, or log from counters the
update actually increments (`pN.numTrainingEpochs` exists for the world model).

**C4 🔴 The "a logging backend must not kill a run" path in `main_train.py` kills the run.**
CONFIRMED by execution. `main_train.py:44-60`: `setup_training` constructs
`PredictiveNet(..., wandb_log=cfg.run.wandb)` (`training/setup.py::setup_world_model`)
BEFORE `init_wandb`; when `init_wandb` raises, only `run_ctx.wandb_log` is flipped. The
fork then calls `wandb.log` on every world-model step (`predictiveNet.py:590-606`
`recordTrainingTrial`, reached from `trainStep:387` and from
`prnn_adapter.py:344` on the graphed path) and on every spatial eval
(`calculateSpatialMetrics`). `WANDB_MODE=disabled uv run python -c "import wandb;
wandb.log({'a': 1})"` on the installed wandb 0.28.0 raises `Error: You must call
wandb.init() before wandb.log()`. Impact: the degrade path advertised in three places
(`main_train.py`, `slurm/train_fast.sh:25`, the cluster notes) fails at the first
gradient step instead. Fix shape: set `comps.predictiveNet.wandb_log = run_ctx.wandb_log`
after the try/except, or construct the net after wandb.

**C5 🔴 `setup_algo` zeroes `prnn_seqdur` when `train_prnn.train=False`; its comment says
the opposite.** CONFIRMED. `training/setup.py:231-234`: the comment reads "A frozen pRNN
still needs episode cuts, so segmenting no longer depends on whether it trains", the next
line is `prnn_seqdur = cfg.collect.episode_steps if cfg.train_prnn.train else 0`.
`evaluation/task.py:114-118` says "freezing the world model must NOT zero prnn_seqdur ...
so it is applied post-construction rather than via cfg.train_prnn.train" and sets
`algo.train_pN = False` by hand. Impact: a `train_prnn.train=False` config on the serial
backend has no `episode_steps` cuts; episodes end only when MiniGrid truncates at its own
`max_steps` (LEnv default `10 * size * size`, 2,560 steps on a 16x16 room), so the tracker
resets, the curiosity segments and the GAE masks all follow a different clock from the one
the config states; on the DEVICE backend `collect_rollout` raises "device_env requires
positive synchronized prnn_seqdur cuts". No preset sets `train=False`, so today
this is a misleading comment plus a trap for the frozen-world-model control the config
advertises. Fix shape: `prnn_seqdur = cfg.collect.episode_steps` unconditionally, and the
`train` flag only gates the update, as `task.py` already does.

**C6 🔴 `collect_eval_rollouts_batched` cannot run.** CONFIRMED by reading.
`evaluation/task.py:314`: `BatchedSRTrackerShim(adapter, B)` with `B = len(envs_eval)`
an `int`; `models/prnn_adapter.py` `BatchedSRTrackerShim.__init__(self, adapter,
envs_obs: list)` does `self.B = len(envs_obs)` and then `self._bootstrap(envs_obs)`.
`len(int)` is a `TypeError`. Consumer: `../experiment-curiousgeorge/src/omt/task.py:317`
and its `tests/test_omt_batched.py`. The docstring at `task.py:293` also still calls
`predict(batched=True)` "buggy", which `tests/test_batched_wm_forward.py` says was fixed
2026-07-30. INFERRED: the downstream test suite either does not run this path or is red;
running `uv run pytest tests/test_omt_batched.py` in the questions repo would tell.

**C7 🔴 `train_phase` logs two different quantities under one wandb key.** CONFIRMED.
`evaluation/task.py:180-181`: `wandb.log({"Train/cur_rewards": cur_rewards["mean"]})`
immediately followed by `wandb.log({"Train/cur_rewards": cur_rewards["std"]})`. The
series `Train/cur_rewards` alternates mean and std at consecutive steps. Consumer: the OMT
task in the questions repo.

**C8 🟠 Resuming a run with any CUDA graph on fails after the allocation, not at parse.**
CONFIRMED by reading. `load_pN` loads the optimizer state (fork `checkpoints.py:113-121`);
`_GraphWMTrainer._ensure_capturable_optimizer` asserts "optimizer state must be empty"
(`prnn_adapter.py:252`) and `GraphPolicyTrainer` asserts the same for Adam
(`rl/update/policy_graph.py`). Both are built lazily at the first update, i.e. after
`setup_run` has written provenance and opened a wandb run. `Config.__post_init__` rejects
`entropy_coef_final + cuda_graph` for exactly this class of reason but not `prnn_ckpt +
train_prnn.cuda_graph` or `policy_ckpt + train_policy.cuda_graph`. The two-phase design
Q2 needs (`tests/test_resume_budget.py`) runs into this on the production preset.
`checkpoint_series.score_exploration_series` loads checkpoints with graphs on but never
updates, so it is unaffected.

**C9 🟠 `checkpoint_series.run_config` rebuilds the room set from four flags and ignores
the run's provenance.** CONFIRMED. `evaluation/checkpoint_series.py:107-150` builds
`Selected(n=5, impassable=...)`; `Selected.positions` and `keep_landmarks` cannot be
expressed, and `provenance.json` (which carries the full config AND `argv`) is never read.
`prediction_figures.plot_run_predictions` reads `provenance.json["argv"]` and rebuilds the
exact config. Impact: the 8-room CE runs (`--env.source.positions 0 1 2 3 5 6 7 8`, the
launcher's documented set) scored with this tool would be scored on rooms 0..4, one of
which (position 4, source index 83) is not in the training set, with plausible numbers and
no error. The module docstring's claim "THE ROOM COMES FROM THE RUN'S OWN CONFIG" is not
what the code does.

**C10 🟠 `Selected.n` is recorded in provenance but ignored whenever `positions` is set,
and `env.content` is recorded but never consulted for `Selected`.** CONFIRMED by the
config-API dump: `multienv.sh` passes both `--env.source.n 5` and `--env.source.positions
0 1 2 3 5 6 7 8`; the source resolves to 8 rooms; `to_dict()` records `{"n": 5, ...,
"positions": [0,1,2,3,5,6,7,8]}` and an `env.content` with `impassable: false` kinds for
a run whose landmarks are impassable (`layouts.py:1251`: "content is deliberately NOT
consulted"). `tests/test_slurm_invocations.py:94-96` asserts `n == 5` and the eight
positions together, pinning the misleading record. Two fields in one record that a reader
must know to disbelieve.

**C11 🟠 The offline CE probes hard-code `action_offset=0`.** CONFIRMED.
`evaluation/surprisal_timing.py:231`, `evaluation/readout_probe.py:235`:
`PRNNAdapter(pN, cpu, action_offset=0)` with no read of the checkpoint's provenance. Latent
while every CE checkpoint is offset 0 (C1); the moment one is not, both tools mis-pair
rows silently and every surprisal number they print is wrong.

**C12 🟠 "pRNN loss" means the focal-weighted loss on focal arms and plain CE on the
others.** CONFIRMED. `recordTrainingTrial` logs `predloss.item()`, which is `loss_fn(...)`
(fork `trainStep:352,383`), and `checkpoint_series.score` uses `pN.loss_fn`
(`checkpoint_series.py:333`). `predCE.forward` applies `(1-pt)^γ` when `focal_gamma` is
set (fork `lossFuns.py:108-110`). So the wandb `pRNN loss` panel and the offline
`checkpoint_curve.json` "loss" column are not comparable between `focal5` and plain-CE arms,
and neither is labelled. The focal documents avoided this by using `surprisal_timing`
(plain surprisal); the reporting surfaces did not.

**C13 🟠 Rollout and training pass start from different initial states at B>1.** See
§1.1. CONFIRMED difference, INFERRED magnitude.

**C14 🟠 `ActorCriticAgent.getObservations` (the on-policy spatial probe) bypasses the
adapter and starts every trajectory from zeros.** CONFIRMED. `rl/collect/agent.py:31-39`
re-implements `next_sr`'s slow path with direct `env_shell.env2pred` and
`predict_single` calls, contradicting `prnn_adapter.py`'s module rule ("Nothing else in
curious_george should call ... directly"); `agent.py:97-98` seeds `SR = zeros` regardless
of `action_offset`, so at offset 1 the first policy step of every probe trajectory acts on
a state the training rollout never produces (`init_sr` builds row 0 from obs[0] and
HD[0]). Harmless at offset 0 (where `init_sr` is zeros too), and `onset_transient=20`
drops the early rows from the metrics, but it is a second copy of the circuit that can
drift.

**C15 🟡 The multi-room eval leaves the eval shell in the last scored room.** CONFIRMED.
`evaluation/spatial.py::evaluate_multi_room_representation` rebinds
`env.env.unwrapped.landmarks` per room and never restores; `training/loop.py`'s trajectory
plot then draws `comps.env` in whichever room the eval ended on. Nothing training-side
reads the eval shell's landmarks after construction (`_location_mask`, `_room_geometry`
run at init), so this only affects the figure, undocumented.

**C16 🟡 `OnPolicyAnalysis._compute_deltas` runs across stream boundaries.**
`evaluation/on_policy.py::_compute_deltas` uses `advantages[:-1] - γλ · advantages[1:] ·
masks[1:]` on the env-major flat layout; at index `b*T` the term is zeroed only because
`masks[t=0]` is 0 on every rollout after the first (segments always close at `T`).
CONFIRMED correct today by that property; fragile if a segment ever spans a rollout.
`plot_deltas` has no caller.

**C17 🟠 The 2026-08-31 audit's "one home for pixel to class" does not hold in the
analysis modules.** CONFIRMED. `evaluation/error_decomposition.py:53-55` and
`evaluation/readout_probe.py:144-146` each spell the nearest-vocab lookup and the
closed-set assert; the adapter (`_prediction_errors`) uses `predCE.targets_for`. Three
homes again; the palette module is the natural owner of a `classes_of(pixels)` function.

**C18 🟡 The `*_grad_steps` axes, `total_rollouts` and cadences assume every rollout is
full-size.** Follows from C2; listed so the fix is one change.

### 1.3 Fork-side findings (pinned `852cc7d2`, read only for the boundary)

- **F1 🟠 `predCE.forward` finds the pixel axis by a shape heuristic** (`lossFuns.py:95-104`):
  the first axis where `size % 3 == 0`, the prediction has `size/3 · C` there, and the two
  differ. For the shapes in use, (1, L, 147, B) and (1, L, 147), it lands on the right axis
  (L=256 and 257 are not divisible by 3; B=256 fails the "differs" test). It is a coincidence
  of the current numbers, not a contract; a batch or length that is a multiple of 3 with a
  matching prediction size would silently pick the wrong axis. The caller knows the axis;
  it should pass it. CONFIRMED by reading.
- **F2 🟡 `recordTrainingTrial` does a `pd.concat` on a growing DataFrame, a `wandb.log`,
  and (through `.item()`) a host sync on every world-model step** (`predictiveNet.py:590-606`;
  the sync is at `trainStep:383` and `prnn_adapter.py:344`). Per run that is 43,936 to
  240,000 concats of a growing frame (quadratic copying), the same number of `wandb.log`
  calls, and one sync per step that defeats CPU/GPU overlap. INFERRED cost: a few percent at
  most; measurable with `CG_TIMING=1` around `update/wm_train`.
- **F3 🟠 `load_pN` restores the optimizer state**, which is what makes C8 fire on resume.
- **F4 CONFIRMED correct:** the focal reweighting `(1-pt)^γ · ce` with `pt = exp(-ce)` and
  a plain mean (`lossFuns.py:105-111`); `targets_for`'s closed-set assert is skipped only
  while a stream is capturing; `render` is the argmax colour; `MaskedRNN` pins
  `predOffset=0`, `inMask=[True]+[False]*5`; `clip_mask` truncates all three sequences to
  the shortest and applies dropout to the input only; `bptt_trunc=10**8` never detaches.
- **F5 🟡 `uv run pypatree` in `../pRNN_new` reports a broken import** in
  `prnn.analysis.DiffusionReplayAnalysis` (`calculateSpatialCoherence` missing from
  `OfflineTrajectoryAnalysis`). Not on any path this repo uses.

### 1.4 Uncertain: needs a test, not a reading

- The magnitude of C13 (zero vs noisy initial state) on a trained checkpoint.
- Whether any cluster run since 2026-08-31 actually took the C4 degrade path and died
  (grep the SLURM logs for "init FAILED").
- Whether `../experiment-curiousgeorge`'s `tests/test_omt_batched.py` is green (C6 says it
  cannot be, unless it never reaches `BatchedSRTrackerShim`).
- The walkable arm's reset-storm cost under the CUDA-graph rollout (O1).
- Whether `wandb.Histogram([])` every log event (`logging.py:127`, `subroom_ids` is always
  empty on the L-room) costs anything or ever errored on a cluster node.
- `RewardNormalizer` is not checkpointed (its docstring says so); a resumed
  `normalize_reward` run re-warms from a fresh std. Documented, not measured.

---

## 2. Naming, rationale, purpose

Ordered by how likely each is to mislead someone reading the code cold.

- **N1 Two sets of defaults for one fact.** `PredictivePPOAlgo.__init__` (`rl/algo.py:96-138`)
  defaults `action_offset=1`, `reward_alignment="legacy"`, `discount=0.99`, `lr=0.001`,
  `entropy_coef=0.01`, `value_loss_coef=0.5`, `noise_std=0.03`, `batch_size=256`;
  `configs.py` defaults `action_offset=0`, `NEXT_OBS`, `0.98`, `3e-4`, `0.0`, `1.0`,
  `0.05`, and derives the batch (`gae_lambda`, `clip_eps`, `max_grad_norm`, `optim_eps`,
  `ppo_epochs` agree). `RolloutConfig` (`collector.py:117`) and `compute_curious_rewards`
  (`rewards.py`) default to `"legacy"` too, and `rewards.py:10` still documents legacy as
  "the default". Every live caller passes explicitly, so nothing is wrong today; the trap is
  that the algo's signature reads like a config. The config is the one home; the algo
  should have no defaults.
- **N2 Error messages and comments name flags that do not exist.** `exp.rollout_cuda_graph`,
  `exp.device_env` (`rl/algo.py:180,249-250`, `collector.py:185,196`,
  `rollout_graph.py:1`, `loop.py:322,328`), `rl.cuda_graph`, `rl.loss`, `rl.frames`
  (`algo.py:175`, `losses.py:9`, `policy_graph.py`), `predNet.*` (`world_model.py:25-45`,
  `prnn_adapter.py` in six docstrings, `models/device.py:46`), `logging.video_log_freq`
  (`storage.py:63`), `MultiRoomEnvCfg.layouts` (`configs.py:327`), `LRoomCfg and
  SquareRoomCfg` (`configs.py:923`). A user who hits the `exp.device_env` error has nothing
  to type. `check/config_keys.py` already holds the old-to-new table.
- **N3 Modules named for a tree that no longer exists.** `rl/algo.py:11`
  "`world_model.adapter`"; `prnn_adapter.py:229` "`world_model.device._move()`";
  `curious_george/__init__.py:9-10` lists `world_model` and `storage` packages. Three code
  comments cite `tests/golden_omt/` (`collector.py:437`, `models/policy.py:96,174`), which
  moved to the questions repo.
- **N4 The fourth action has three names.** `configs.py:47` "(left, right, forward,
  pickup)"; `envs/access.py:56` `"stay"`; `utils/common.py:99` `"stay_put"`;
  `prediction_figures.py:321-323` `"pickup (no-op)"` / `"no-op"`. It is MiniGrid's pickup,
  a no-op here.
- **N5 The README's boundary claim is false.** README:160 "`PRNNAdapter` is the only module
  that imports `prnn`". Fifteen modules under `curious_george/` import `prnn` (grep in
  `throwaway/2026-09-05/` session); `models/__init__.py:6` makes the narrower, true claim
  "in this package".
- **N6 `AgentType.AC = "curious"`** (`utils/enums.py:25`): the value names the reward, not the
  agent, and it lands in every run name (`setup_run`). A count-bonus run
  (`--train-policy.no-curious --train-policy.k-count`) is still named `_curious_`.
- **N7 `ACModelSR.SR_size` is SR plus the HD one-hot; `SR_single` is the SR**
  (`models/policy.py:116`). The property named for the state includes something else.
- **N8 `prediction_mses` returns surprisal under CE** (`prnn_adapter.py`, noted in its own
  docstring; the prior audit deferred the rename for wandb continuity). Still the wrong name
  in the code that reads it.
- **N9 `randomAgent_collect_exp_and_update` is called "retired" in two places and is live in
  a third.** `collector.py:110` and `loop.py:242` say retired; `rl/algo.py:574` defines it;
  `evaluation/task.py:171` calls it for the OMT random arm. It also calls
  `pN.collectObservationSequence` and `pN.trainStep` directly (`algo.py:591,597`), the two
  calls the adapter's module docstring forbids.
- **N10 Three probe seeds, three homes:** `evaluation/probe.py:50` (20260730),
  `checkpoint_series.py:51` (20260813), `circuit_diagnostics.py:42` (20260829), plus
  `EvalCfg.probe_seed` (10007 in `parity`). Two of the probe constructions (`probe.build_probe`
  and `checkpoint_series.fixed_probe`) are the competing implementations `probe.py`'s own
  docstring names.
- **N11 The wandb entity and project have three homes:** `RunCfg` defaults
  (`configs.py:773-774`), `.env` via `dev_env.get_wandb_env_vars`, and the constants in
  `check/wandb_compare.py:46`. `multienv.sh` logs to `curious-george-multienv`, which
  `wandb_compare` cannot reach without an edit.
- **N12 Two homes for the checkpoint filenames:** `utils/dev_env.py:28-29` and
  `log_and_store/storage.py:83`, and `training/loop.py:172` spells
  `"predictiveNet_state.pt"` as a literal a third time.
- **N13 `RAND_ACT_PROBA` exists twice**, as a tuple in `configs.py` and as an ndarray in
  `storage.py:30`; `circuit_diagnostics.py` renames it `PROBE_ACTION_P`.
- **N14 `LandmarkKind.size` is a config field that does nothing** (`layouts.py:776`, its
  docstring says so). Under the repo's own rule a field that does nothing is a misleading
  field.
- **N15 `RunCfg.video_every_episodes`** creates a video directory (`setup_run`) and nothing
  records video: `setup_env` never passes `vid_n_episodes` to `make_env`
  (`factory.py:135-138`). Dead config that looks live.
- **N16 `EvalCfg.behaviour_timesteps`** is passed to `OnPolicyAnalysis(timesteps=...)` with
  `reuse_last_rollout=True` (`loop.py:124-128`), which ignores `timesteps`
  (`on_policy.py:352-357`). Dead config that looks live.
- **N17 `ArchPolicyCfg.input_type`** (`configs.py:470`) selects among nine `AgentInputType`
  members; only `H` ("pRNN") is used, `H_PO` routes to the same wrapper, and the `else`
  branch (`HDObsWrapper`) is unreachable from any preset. The prnn fork has its own
  `AgentInputType` enum with the same name (`checkpoint_series.build` uses the fork's).
- **N18 `Layout.min_cell_gap()` is a method; `Layout.min_anchor_separation` is a property**
  (`layouts.py`), so one is called with parentheses and one without.
- **N19 `provenance.write`'s docstring accuses `envs/obs_bank.py:95` of a non-atomic write**
  (`provenance.py:184`); `obs_bank.py` has done the temporary-file-plus-`os.replace` dance
  since, with its own note about the `.npz` suffix. Stale accusation.
- **N20 `train_fast.sh:35-36`** says "The final config does not use cuda_graph, so it is
  safe" while the same script takes three graph switches as positional arguments.
- **N21 `BANK_DIR = Path(__file__).resolve().parents[2] / "data" / "obs_bank"`**
  (`obs_bank.py:51`): a climbing path, the pattern CLAUDE.md flags, and the one output that
  does not go through `RL_STORAGE` (`tests/test_outputs_go_through_storage.py` exempts it).
- **N22 `setup_world_model`** keeps `predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0,
  0])  # TODO: remove later` (`setup.py:189`) with a comment saying it already equals the
  default. Three `TODO`s remain in the tree (`agent.py:97`, `setup.py:189`,
  `models/policy.py:144`), all older than the design they describe.
- **N23 `EnvBackend.measures_return`, `early_stop`, `return_per_episode`,
  `reshaped_return`** exist for goal environments that no preset runs; the serial path
  carries the bookkeeping on every step.
- **N24 The training golden pins `reward_alignment=legacy`** (`capture_golden.py:23`), an
  option `rewards.py`'s own docstring describes as "crediting the action with surprise it did
  not cause". The only reason `RewardAlignment.LEGACY` exists is that fixture.
- **N25 Duck-typed defaults on the hot path.** `getattr(acmodel, "with_CV", True)`
  (`collector.py:57,68`, `rl/update/policy.py:58`), `getattr(acmodel, "with_HD", False)`
  (`policy.py:69`), `getattr(self.envs, "layouts", None)` (`algo.py:355,400`),
  `hasattr(pN.loss_fn, "render")` / `hasattr(loss_fn, "targets_for")` as a second spelling
  of `adapter.ce` (`prediction_figures.py`, `prnn_adapter.py:334`), and
  `getattr(self, "_vocab_check_counter", 0)`. Each substitutes a guess for a type; the
  `with_CV` one picks the slow image-indexing path if the attribute is ever missing. 24 such
  reads across `rl/`, `models/`, `training/`.

---

## 3. Redundancy

Deletion or consolidation candidates. "Callers" is from grep over `curious_george/`,
`tests/`, and `../experiment-curiousgeorge/src` + `tests`; "downstream" names the consumer
so the decision is yours. Line counts are approximate.

| item | where | lines | callers in this repo | downstream | verdict |
|---|---|---|---|---|---|
| `log_and_store/wandb.py` beyond `fetch_occupancy_grids` (traces, plotting, bootstrap CIs, t-tests, subroom percentages, significance brackets) | `log_and_store/wandb.py` | ~1,300 of 1,488 | only `_history_rows` from one test | `Q1/collect.py` imports `fetch_occupancy_grids` only | move the analysis half to the questions repo; keep the fetchers |
| `AsyncShellPool`, `DropMission`, `PosInfo`, `_make_worker_thunk` (a second copy of `make_env`'s wrapper selection), the collector's `pool` branch, `EnvBackend.ASYNC/ASYNC_TABLE`, `setup_envs`' async branch | `envs/vector.py:1-160`, `collector.py`, `setup.py:155` | ~250 | `tests/test_async_envs.py` (2 tests) | none | delete with its test |
| `IntrinsicReference`, `intrinsic`/`k_intrinsic`/`k_int`, `int_rewards` plumbing in collector, GAE and logs, the `intrinsic` precondition in `Config` | `rl/algo.py:45-90`, `collector.py`, `advantage.py`, `configs.py:611` | ~120 | `test_configs.py` (one negative test) | none | delete |
| `a2c_loss`, `LOSSES`, the string `loss_fn` indirection | `rl/update/losses.py`, `policy.py`, `algo.py`; `setup_algo` hard-codes `"ppo_clip"` (`setup.py:287`) | ~40 | none | none | delete (user confirmed) |
| `SpatialEvalPath.LEGACY_DECODER`, `legacy_decoder_timesteps`, `compute_sleep_wake_dist`, `_sleep_wake_dist`, the `trainDecoder` branch | `configs.py:162,741`, `spatial.py:35-70,237-256`, `loop.py:103` | ~70 | only `tests/test_ckpts.py`, which is `slow` and deselected by default | none | delete; the only test never runs in the gate |
| theta-cycle branches (`self.theta`, `self.k`, `next_sr`'s theta arm, `not self.theta` guards, `ACModelSR`'s `SR.ndim > 2`) | `prnn_adapter.py`, `models/policy.py:144-146` | ~30 | none (`prnn_type` is a fixed property returning `masked`) | none | delete |
| non-SpeedHD fallbacks (`fast_speedhd=False` arms: `env2pred` per call in `seq2pred`, `next_sr`, `train_on_episode`, `BatchedSRTrackerShim.step`) | `prnn_adapter.py` | ~40 | none: offset 1, the device backend, batched WM and batched curiosity all require SpeedHD | none | pin `action_encoding` to SpeedHD and delete |
| `EnvironmentFeaturesAnalysis` | `evaluation/on_policy.py:146-330` | ~185 | none | none | delete |
| `OnPolicyAnalysis` fresh-clone path (`reuse_last_rollout=False`) and the algo attributes kept only to rebuild it (`lr`, `noise_mu`, `noise_std`, `k_int`, `cuda_graph` on `algo`) | `on_policy.py:360-425`, `algo.py:160-185` | ~70 | `loop.py` uses `reuse_last_rollout=True` only | `omt/task.py:291` uses the clone path | decide with the OMT owner; the clone also calls `envs.reset_all()` on the live training pool |
| `randomAgent_collect_exp_and_update` | `rl/algo.py:574-624` | 50 | none | `omt/task.py` via `train_phase` | route the OMT random arm through `random_actions=True` like the loop does, then delete |
| `compare_trajs` | `rl/algo.py:40` (+ two `__init__` re-exports) | 5 | none | none | delete |
| `check/config_keys.py` | 180 | `tests/test_config_keys.py` only | none | keep only if old wandb runs are still read; otherwise `throwaway/` |
| `configs.MinigridEnv`, `SingleLayout`, `FrozenLayouts`, `LayoutPool`, `EnvLayoutSpec`, `import abc` | `configs.py:20,58,174-205` | ~40 | none (ruff confirms the imports) | none | delete |
| `_multienv()` preset (RETIRED budget, "describes NO run that has ever happened") | `configs.py:945-975` | 30 | `tests/test_configs.py::EXPECTED`, `capture_golden_setup.COMPOSITIONS` | none | delete with its pins; the README's example command uses it |
| `Curated`, `Committed`, `generate_layouts`, `ROOMS_RUN1`/`ROOMS_SQUARE` + `Frozen` | `envs/layouts.py` | ~200 | `tests/test_env_layouts.py`; `train_fast.sh rooms/one` uses `Frozen` | none | `Frozen`/`ROOMS_RUN1` are the frozen-three history and the `train_fast.sh` default; `Curated`/`Committed`/`generate_layouts` have no production caller |
| `RunCfg.video_every_episodes`, `get_video_dir`, `episode_video_trigger`, `RecordVideo` wiring, `ResetWrapper`, `HDObsWrapper`, the `wrappers` dict and `wrapper=` kwarg | `configs.py:775`, `storage.py:58`, `factory.py:31-48,51,135-138,146-172` | ~70 | none reachable | none | delete |
| `format.py` text preprocessing (`preprocess_texts`, `Vocabulary`, the `HD` branch) | `rl/collect/format.py` | ~60 | the mission string is tokenized on the B=1 path and never read | `omt` imports `get_obss_preprocessor` | keep the function, drop `text` and `HD` |
| `storage.get_tmp_dir`, `get_tmp_model_dir`, `get_goal_loc` (an alias of `access.get_new_obj_pos` under a misleading name), `load_policy` | `storage.py` | ~25 | `load_policy` in one test | none | delete |
| `dev_env.get_ckpt_env_vars`, `get_ckpt_dir_var`, `get_wandb_env_vars`, `get_logdir_env_var`, `get_root_dir_env_var`, the `.env` `*_CKPT*` variables and their FOURROOM twins | `utils/dev_env.py`, `.env` | ~110 | `tests/test_ckpts.py` (slow), `tests/test_ckpt_interop.py`, `tests/golden/compare_io.py` | `resolve_prnn_ckpt`, `get_env_var` are imported downstream | keep those two; the rest is the pre-`run.prnn_ckpt` era |
| `curious_george/__init__.py` re-export surface (60 names, including every item above) | `__init__.py`, `rl/__init__.py`, `envs/__init__.py`, `evaluation/__init__.py` | 179 | the questions repo imports 10 names from the top level | `make_env`, `get_obss_preprocessor`, `get_dist_travelled`, `get_pN`, `get_env_var`, `grid_to_pixel_coords`, `ACModelSR`, `ActorCriticAgent`, `AgentInputType`, `AgentType`, `seed` | shrink to what is imported |
| `access.py` visualization helpers (`render_env`, `obs_image`, `pred_image`, `hidden_image`) | `envs/access.py:56-95` | 40 | none | none | delete |
| `subroom_size`, `get_subroom_id`, `subroom_ids`, the `wandb.Histogram` of an always-empty list | `envs/access.py`, `algo.py`, `collector.py`, `logging.py:127` | ~30 | FourRooms only; `None` on every L-room | none | delete |
| `check_large_jump` and the two `DEBUG START` print blocks | `collector.py:463,498`, `diagnostics.py` | 20 | serial/async paths only | none | delete |
| `wm_segment_stride` regime (`batched=False` with `episodes_per_grad_step>1`: train on one episode, drop the rest) | `world_model.py:30-60`, `setup.py:272` | ~40 (mostly docstring) | reachable; no preset uses it | none | decide: it was a 2026-08-22 experiment; `batched` could become implied by `episodes_per_grad_step > 1` |
| `EnvBackend.measures_return`, `early_stop`, `return_per_episode`, `reshaped_return`, `n_performance` | `configs.py`, `collector.py`, `loop.py:315-335` | ~40 | `tests/test_return_is_measured_or_absent.py` | none | goal envs are not run; delete or keep with the test |
| eight golden fixture files no test reads (`golden_v0..v4.pt`, `golden_eval_v1.pt`, `golden_eval_offset1_v1.pt`) | `tests/golden/` | 26 MB in git | `capture_golden.py:92` reads `v5`; `capture_golden_eval.py:86,99` read `eval_v2`/`eval_offset1_v2`; `capture_golden_setup.py:54` reads `setup_v1` | none | delete the files; keep the version ledger in the docstring |
| `tests/golden/compare_io.py` | 1 file | - | a cross-tree harness for a pre-refactor tree that no longer exists | none | `throwaway/` |
| `tests/perf/*.py` + `tests/perf/results/` | 4 scripts, 10 result files | - | not collected by pytest (not `test_*.py`) | none | `throwaway/` or a `tools/` home |
| 57 unused imports, 4 unused variables | `uv run ruff check ... --select F401,F841` (listing in the session; 35 auto-fixable) | - | - | - | `ruff --fix` |

**What is NOT redundant despite looking so.** `SERIAL`/`SERIAL_TABLE` and the B=1
`SingleSRTracker` are the `reference` baseline the user wants live; `Frozen`/`ROOMS_RUN1`
is the `train_fast.sh` default; `MSE` is a live arm; `CountBonus` is the model-free
control; `RewardNormalizer` is the CE arms' flag; `evaluation/task.py` is the OMT's, live
downstream despite C6/C7.

---

## 4. Optimization

Nothing that was optimized is wrong: every graph, the device pool, the fast reset, the
batched curiosity and the compile are gated by tests in the baseline, and their
semantics-changing consequences are stated in the config docstrings. What remains:

- **O1 The walkable arm keeps the reset storm.** `vector.py:405` limits the cached-grid
  fast reset to impassable layouts, for the RNG-order reason its comment gives; the walkable
  arm (live: MSE-on-multienv) still runs B full MiniGrid resets per segment, measured at
  1.31 s of a 2.08 s rollout at B=256 before the thread was added. The prepare thread now
  overlaps it with the GPU loop. INFERRED: mostly hidden under the graphed rollout; the
  `collect/prepare_resets` timer exists to measure it.
- **O2 Per-world-model-step host sync, pandas concat and `wandb.log` in the fork** (F2).
  On the pooled path that is 32 syncs and 32 `wandb.log` calls per rollout, and a
  DataFrame that grows to 43,936 to 240,000 rows by O(n) concat. Accumulating the loss on
  device and recording once per rollout removes all three.
- **O3 Random-agent runs compute a curiosity pass nobody reads.** `setup_algo` passes
  `curious_agent=cfg.train_policy.curious` (default True) independently of `agent`; the
  collector runs `prediction_mses_device` every rollout for a baseline whose policy never
  updates. Cost is one batched forward per rollout (about 13 ms graphed per the
  `train_fast.sh` note), so under 1%; the clarity cost is that a RANDOM run's config says
  `curious=True`.
- **O4 The online spatial probe is per-step Python.** `ActorCriticAgent.getObservations`
  calls `env2pred([obs, obs], act)` and `predict_single` once per step on CPU
  (`agent.py:31-39`); at 8 trajectories × 256 steps × 5 rooms per analysis event this is
  the "88 s per event" the config docstring budgets around. The adapter's vectorized
  `next_sr` path exists and is unused here.
- **O5 `LocationStats.update` is a 65,536-iteration Python loop per rollout**
  (`diagnostics.py:33-35`), plus `flat_locs` builds 65,536 tuples (`collector.py`). Both
  vectorize (`np.add.at`). INFERRED 1-2% of a 2 s rollout; the `collect/loc_stats` and
  `collect/flat_locs` timers exist.
- **O6 The B=1 path tokenizes the mission string every step** (`format.py:28,55`) for a
  `text` field no model reads.
- **O7 One wasted tracker step per segment**: the SR step at t = L-1 computes h[L] from
  obs[L] and `_close_device_segment` discards it. 1/256 of SR compute; clarity only.
- **O8 Optimizations that constrain the science, all documented, none hidden:**
  `train_prnn.cuda_graph` and `train_policy.cuda_graph` are fresh-run only (C8);
  `batched_curiosity` and `rollout_cuda_graph` change RNG order, so the golden gates run
  only at zero noise or a saturated policy; `compile=LAYER` recompiles for any new sequence
  length (about 89 s each).
- **O9 Two gathers per graphed timestep** (`rollout_graph.py`, acknowledged in its
  docstring) to keep `step_device` as the single implementation. Fine.

---

## 5. Tests and goldens

- **T1 The training golden gates a configuration no live preset runs.**
  `capture_golden.py`: B=1 serial, `SEQDUR=32`, 2 updates, `action_offset=0`, MSE, and
  `reward_alignment=legacy` (`:23`). The production composition (DEVICE, 256 streams,
  pooled world model, `next_obs`, CE/focal/MLP, three graphs) has no bitwise fixture; the
  device path is gated only by equivalence to `SERIAL_TABLE` at small B and zero noise.
  `golden_eval_v2`/`golden_eval_offset1_v2` pin metrics on a pinned checkpoint for both
  circuits (`next_obs` at offset 1, `legacy` at offset 0). `golden_setup_v1` pins that
  `reference`/`parity`/`multienv` construct identically. Revising the goldens (the user's
  ask): recapture one training fixture on the production composition at zero noise and drop
  `legacy`.
- **T2 Eight fixture files are orphaned** (§3 table). Also: `golden_v5.pt` and
  `golden_eval_*_v2.pt` share an mtime (2026-08-30 21:19) with the minigrid re-render
  commit, consistent with the ledger in `capture_golden.py:25-32`.
- **T3 `tests/test_ckpts.py` is `slow`, deselected by default, and depends on `.env`
  checkpoint paths** (`CUR_CKPT_DIR`), so it is machine-specific and never runs in the gate.
  It is the only test of `LEGACY_DECODER`.
- **T4 `tests/test_slurm_invocations.py` parses `multienv.sh` and `parity.sh` only.**
  `train_fast.sh` (16 positional arguments, Hydra-era comments), `train_prnn.sh` and
  `bsweep.sh` are not parsed. Their current invocations do use valid `--section.field`
  flags (read; `train_fast.sh:345-363`, `bsweep.sh:66-70`, `train_prnn.sh:63`).
- **T5 `tests/test_configs.py::EXPECTED` is the only reason the retired `multienv` preset
  exists** (its own docstring says so).
- **T6 Overlap check across families** (checkpoints: `test_ckpt_interop`, `test_ckpts`,
  `test_policy_checkpoint`, `test_resume_budget`; advantage: `test_advantage`,
  `test_advantage_normalization`, `test_reward_norm`; batching: `test_batched_wm`,
  `test_batched_wm_forward`, `test_batched_tracker`, `test_batched_collector`,
  `test_device_collector`): each file's docstring names a distinct defect it pins; I found
  no two that pin the same thing. `tests/test_env.py` (4 import-level assertions) and
  `tests/test_config_keys.py` (a translation table with no runtime caller) are the two I
  would drop first.
- **T7 `tests/perf/` and `tests/golden/compare_io.py`** are not tests (§3).
- **T8 Ruff** (`--select F401,F841`): 61 findings, 35 auto-fixable; the source-side ones
  are in `configs.py` (7, including `abc` and the six layout imports), `factory.py`
  (`MinigridEnvNames`), `storage.py` (3, including an unused `PredictivePPOAlgo`),
  `world_model.py` and `prebuild_banks.py` (`numpy`), `error_decomposition.py` (`Bool`),
  `readout_probe.py` (`Int`), `task.py:234` (`X`), `prediction_figures.py:410` (`rng`),
  and every `__init__.py` re-export without `__all__`.

---

## 6. Documents and README

- **D1 README:93-95 names three presets; `PRESETS` has five** (`parity`, `multienv-fast`
  added). README:88's example `main_train.py multienv --run.seed 3` launches the RETIRED
  budget. (User acknowledged.)
- **D2 README:160 boundary claim** is false (N5).
- **D3 `.env`** references `slurm/omt_task.sh`, which does not exist, and carries a
  "DANGER" note about checkpoint variables that point nowhere.
- **D4 `docs/audit-2026-08-31.md` item 6** ("one home for pixel to class") is not true of
  `error_decomposition.py` and `readout_probe.py` (C17).
- **D5 `docs/prnn-io-alignment.md`** is marked superseded at the top and still carries the
  2026-08-28 survey below; its header table is the current truth. Fine as a record; the
  README should not point new readers at it without that warning.
- **D6 The questions repo's pin comment** says `sdu/config-dataclasses` for a rev that is
  this branch's HEAD. (User acknowledged; other repo.)
- **D7 `throwaway/ported/...`** is cited as a source in eleven places across nine live
  modules (`envs/access.py`, `envs/layout_figures.py`, `envs/layouts.py` x3,
  `evaluation/probe.py`, `evaluation/spatial.py`, `__init__.py`, `models/prnn_adapter.py`,
  `rl/update/advantage.py`, `utils/common.py`). A live claim resting on a `throwaway/`
  document is untraceable by the repo's own rule.

---

## 7. Re-verification of the 2026-08-31 audit's fixes

Read against the code, not the document; gates named are in the passing baseline.

| audit item | re-verified how | holds? |
|---|---|---|
| 1. seeded probe restored the training RNG | `spatial.py::_probe_rng` saves/restores torch CPU, all CUDA generators, numpy; `tests/test_spatial_eval.py` (6 pass) | yes |
| 2. `pN.state`/`pN.phase` restored; `reset_each` per probe trajectory | same context manager snapshots both; `collect_pooled_activity(reset_each=probe_seed is not None)` | yes |
| 3. prepare-resets thread starts after graph capture | `collector.py:361-363` | yes |
| 4. stream-0 bank build on a deepcopy scratch | `vector.py::_collect_layout_banks`; `test_fast_reset.py` | yes |
| 5. `multienv.sh raw` passes the flag | `multienv.sh` `raw) NORMFLAG="--train-policy.no-normalize-advantage"` | yes |
| 6. one home for pixel to class | adapter yes; two analysis modules no | **partial** (C17) |
| 7. trajectory plot in `eval_mode` with the run's agent | `loop.py` `plot_agent` block | yes |
| 8. `checkpoint_series` under `eval_mode`, `rooms_max=5` | `score()`; `run_config` | yes, but see C9 |
| 9. `surprisal_timing` a tracked module under `eval_mode` | exists; `measure` wraps in `eval_mode` | yes, but see C11 |
| periodic eager vocab check under the WM graph | `_GraphWMTrainer.train_batch`, `_VOCAB_CHECK_EVERY=64` | yes |
| multiroom probe is on-policy by design | `run_spatial_analysis` hands `comps.ac_agent` unless RANDOM | yes |
| deliberately left: `prediction_mses` name, `SI_mean_active_only` | unchanged | as recorded |

---

## Appendix: what was run

All read-only; scratch under `throwaway/2026-09-05/`.

```
CG_DEVICE=cuda uv run pytest -q                         # 628 passed, 1 deselected, exit 0
uv run pypatree                                          # both repos
uv run ruff check curious_george main_train.py tests --select F401,F811,F841,F632,E711,E712,B006
WANDB_MODE=disabled uv run python -c "import wandb; wandb.log({'a': 1})"   # raises before init
CG_DEVICE=cpu uv run python - <<EOF ... EOF              # stated-vs-realized budgets; production defaults; Selected provenance
```

Confirmed-by-reading claims cite `file:line` at `71890bd`; fork claims cite the pinned
`852cc7d2` checkout at `../pRNN_new`.

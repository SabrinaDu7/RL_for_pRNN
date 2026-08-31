2026-08-30 · branch `sdu/optim-pred`

# Exploration evals: the library, its baselines, and what they are read against

## Goal

Measure how well the curious POLICY explores — not just where it sat
(`loc_entropy`, occupancy maps) but how fast and how far: coverage curves with
normalized AUC, time-to-cover thresholds with censoring, per-room visitation
entropy, and coverage as a function of BFS distance from spawn. Every metric
ships with baselines whose values are known before any training run is scored,
because a fraction without its achievable range is unreadable.

This document covers the measurement library, its gates, and the online wiring
into the training loop (added the same day, second section below).

## Where the code lives

- `curious_george/envs/action_graph.py` — the transition table
  (`obs_bank.build_transition_tables`, the same table the device backend steps)
  consumed as a graph: `ActionGraph.distance_map` (BFS, distance in ACTIONS — a
  turn costs a step), `walk`/`categorical_walk` (replay / state-independent
  random policies), `sweeper_walk` (greedy nearest-unvisited positive control),
  `spawn_states` (the support MiniGrid's `place_agent` samples).
- `curious_george/evaluation/exploration.py` — pure analysis: positions
  `(E, T, 2)` in, one dataclass out per metric (`CoverageCurves`,
  `RoomVisitation`, `DistanceCoverage`). No environment or network is stepped;
  real rollouts, walkers and fabricated fixtures score through the same code.
- `tests/test_action_graph.py`, `tests/test_exploration_evals.py` — the gates.

## Methods

**Unit of replication: the episode.** One stream's `prnn_seqdur` segment; its
spawn is `positions[:, 0]` (positions are pre-action, `collector.py`).

**Denominator: the episode's own room's walkable set**, always
`layout.walkable(base_walkable(room))`, never a constant. Why per-room and not
the pooled union: measured on the committed 5-room set, per-room walkable is
153 (impassable) vs 172 (walkable), but no landmark cell is blocked in ALL
five rooms, so the pooled set — and with it `EnvCfg.loc_entropy_ceiling` — is
172 cells / 7.4263 bits in BOTH arms. Ceiling-normalizing the pooled
`loc_entropy` therefore cannot separate the arms; only the per-room form can.
Pinned as relationships in
`tests/test_exploration_evals.py::test_pooled_ceiling_cannot_separate_the_arms`,
so a future room set that breaks the equality fails loudly.

**Censoring is reported first.** Every time-to-cover threshold is
*(fraction of episodes reaching it | mean steps among those)* — a conditional
mean alone describes the luckiest few percent
(`test_censoring_is_reported_not_hidden`: forward-weighted random reaches T25
in >85% of episodes and T50 in <20%).

**Baselines.** Uniform random `[.25]×4`, the project's forward-weighted random
(`storage.RAND_ACT_PROBA`), and the greedy sweeper — a table-walker needing no
training, supplying the achievable ceiling. All three run on the byte-identical
transition table the live environments step
(`test_layout_tables_match_the_table_wrapper`,
`test_walk_matches_the_live_environment`).

## Results — the calibration table

`uv run python -m curious_george.envs.action_graph` (n=2000 random episodes,
40 sweeper spawns, one 256-step episode, coverage of the room's own walkable
set; committed `Selected(n=5)` room 0):

```
=== walkable, denominator 172 ===
agent                     cov@256   nAUC             T25          T50          T75          T90
uniform [.25]x4             0.191  0.108      229 ( 11%)     censored     censored     censored
forward-weighted            0.395  0.224      142 ( 96%)   233 (  7%)     censored     censored
greedy sweeper              1.000  0.608       47 (100%)    96 (100%)   151 (100%)   191 (100%)

=== impassable, denominator 153 ===
uniform [.25]x4             0.177  0.102      227 (  7%)     censored     censored     censored
forward-weighted            0.349  0.200      158 ( 87%)   236 (  2%)     censored     censored
greedy sweeper              0.992  0.606       43 (100%)    92 (100%)   154 (100%)   200 (100%)
```

Cross-validated: the throwaway calibration script
(`exploration_baseline_calibration.py`, deleted this session per its own
docstring - git history keeps it) derived the same numbers to three decimals
from independently hand-rolled `DIR_TO_VEC` dynamics. Two independent
implementations agreeing is the evidence the pins in
`tests/test_exploration_evals.py::CALIBRATION` rest on.

What the table settles (unchanged from the plan): the 256-step episode is
neither saturated nor geometrically censored (sweeper T90 ≈ 191 < 256); the
two random baselines differ ~2x (an unlabeled "random baseline" is ambiguous);
a learned policy that never reaches T90 is genuinely worse than a sweeper.

## Gates

`uv run pytest tests/test_action_graph.py tests/test_exploration_evals.py`
(22 tests, ~7 s). Analytic agents with exact answers: turn-left-only (one
cell: entropy exactly 0, curve flat at 1, all thresholds censored, only the
distance-0 bin covered), forward-only (a ray derivable from the walkable set),
enumerating the support (every ceiling at exactly 1.0). Structural: curves
monotone and ending at the distinct-cell count; off-support visits raise
(wrong room channel is a wiring error, not data); landmark cells at BFS
distance −1 and the reachable set equal to the walkable set (no enclosed
pockets, now a test rather than a one-off check); distance bins separating the
sweeper from uniform random where aggregate coverage cannot.

Regression baseline: the four affected test files
(`test_obs_bank`, `test_table_env`, `test_impassable_landmarks`,
`golden/test_golden`) passed 26/26 before and after the one refactor this
required (extracting `build_transition_tables` from the wrapper so the graph
consumes the same builder — one home for the dynamics).

## Wired online (same day)

Every training run on the DEVICE backend now scores its own rollouts:

- **Room channel**: `DeviceTableShellPool.prepare_resets` returns the rollout's
  room schedule `(n_segments, B)` (row 0 = the assignment the rollout enters
  with; prepared reset i names segment i+1's room), mirrored on the host so it
  costs no sync. The collector carries it on `CollectResult.segment_layouts`
  beside `positions_episodes (E, seg_steps, 2)` - the episode view, e = s*B + b,
  copied out because a graphed rollout reuses its buffers.
- **Scoring**: `PredictivePPOAlgo.collect_experiences` runs
  `exploration.rollout_summary` on the episode view against per-room supports
  built once at construction (`algo._room_geometry`). Pure reads, no RNG - the
  golden gate stays bitwise (verified against the v5 fixtures).
- **wandb keys** (log cadence, logged for EVERY agent - the baselines exist to
  be compared on these): `exploration/coverage`, `exploration/nauc`,
  `exploration/t50_reached`, `exploration/t50_steps` (omitted when censored,
  never fabricated), `exploration/t90_reached`, `exploration/t90_steps`,
  `exploration/room_entropy_norm`, plus `loc_entropy_norm` (backend-free) and
  run-summary constants `reachable_cells` / `loc_entropy_ceiling`
  (`log_run_constants`). Off the device backend the keys are ABSENT, not zero.
- **The uniform baseline is expressible**: `arch_policy.random_action_probs`
  (CLI: `--arch-policy.random-action-probs 0.25 0.25 0.25 0.25` with
  `--arch-policy.agent RANDOM`), threaded through BOTH sampling sites - the
  eager collector and `GraphRolloutStepper`, which builds its own device table.
  The value's one home is `configs.RAND_ACT_PROBA`; the four independent
  spellings (storage, setup, circuit_diagnostics, a checkpoint_series literal)
  are now derivations of it.
- **Cost, measured**: `collect/exploration` = 1.5 ms per parity-shaped rollout
  (num_envs=256, episode_steps=256, multi-room impassable), 0.09% of collect
  under sync-attributed `CG_TIMING` profiling - ~0.1% on gradient steps/s.
- **Gates**: `tests/test_exploration_online.py` (the pool's schedule contract;
  one real collect whose exported room ids are verified against each episode's
  walkable support; absent-not-fabricated off the device backend; the logging
  surface) and four additions to `tests/test_random_agent_baseline.py`
  (uniform expressible, the sampler honors the field, end-to-end uniform walk,
  one-home derivation).

The offline consumer landed the same day: `checkpoint_series --exploration`
scores every archived (pRNN, policy) pair with seeded, eval-mode, SAMPLED
rollouts through the run's own checkpoint-loading path (`run.prnn_ckpt` /
`run.policy_ckpt` into `setup_training`), one row per pair into
`<run>/exploration_curve.json` - gated by tests/test_checkpoint_exploration.py
including byte-identical repeatability across scorings. Verification figures
live in `throwaway/verify_exploration_figs.py`; the interactive BFS explainer
is the "L-Room Action Distances" artifact.

The count-based agent also landed the same day: `train_policy.k_count` turns on
the 1/sqrt novelty bonus (`rl/update/rewards.py::CountBonus` - lifetime counts
over (room, x, y, head direction), a WITHIN-rollout per-stream occurrence term
so a fresh table has a gradient at all, computed after the timestep loop and
entering GAE as a third term). Curiosity's model-free control: same PPO, same
networks, only the reward's origin differs. Counts ride the policy checkpoint
(`StatusCkptKeys.COUNT_VISITS`). Gated by tests/test_count_bonus.py: exact
1/sqrt schedules, same-seed trajectories identical with and without the bonus
while advantages differ, the DEVICE-backend requirement, and the checkpoint
round-trip. Still open from the plan: in-run analysis-cadence FIGURES (the
wandb-logged mean curve + distance-binned panels), and the baseline/count RUNS
(k_count needs its scale sweep under normalize_advantage).

## Discoveries and fixes along the way

- **Pooled ceilings are equal across arms** (above) — this reshaped the
  design: the room channel is a prerequisite for visitation entropy, not just
  for distance-binned coverage.
- **`MULTI_ENV_ID` duplicated `MULTI_ROOM_ID`** in `envs/layouts.py` —
  consolidated to one dict; the only consumers were two throwaway scripts.
- **Stale prose claiming the in-run spatial eval is skipped under CUDA
  graphs** (`slurm/train_fast.sh`, README) — the skip died 2026-08-23;
  corrected to the real reasons the offline scorer exists.
- The multienv doc's "random baseline unrepresentable" paragraph predated
  `22e7c0d` — corrected: the config builds; the RUNS are what is missing.

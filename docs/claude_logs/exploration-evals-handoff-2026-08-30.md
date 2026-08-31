2026-08-30 · branch `sdu/optim-pred`

# Handoff: the policy exploration evals, for the multienv training-runs agent

This summarizes what was built so a runs agent can launch, read, and score
multienv runs on the exploration metrics. The full record — methods,
calibration table, discoveries, gates — is `docs/exploration-evals-2026-08-30.md`;
this file only points. Everything below is confirmed against code or a run at
the date above unless marked inferred.

## What gets logged, and by whom

Producer: `curious_george/evaluation/exploration.py::rollout_summary`, called
per rollout from `rl/algo.py::collect_experiences` on the episode view the
collector exports. Keys land in wandb on the **log cadence**, for **every
agent** — the random baselines exist to be compared on exactly these series
(forwarding in `training/logging.py::build_update_log`, deliberately outside
the `random_agent` gate).

| key | meaning |
|---|---|
| `exploration/coverage` | mean over episodes of (distinct cells visited ÷ the episode's own room's walkable count), at episode end |
| `exploration/nauc` | mean over episodes of the normalized coverage curve's mean — 1.0 = everything visited at step one |
| `exploration/t50_reached`, `exploration/t90_reached` | fraction of episodes whose coverage crossed the level — **the headline statistic** |
| `exploration/t50_steps`, `exploration/t90_steps` | mean env steps to cross, over reaching episodes only — **omitted when censored**, never fabricated |
| `exploration/room_entropy_norm` | nan-mean over rooms of per-room visitation entropy ÷ log2(that room's walkable count); unvisited rooms are nan (untested ≠ zero) |
| `loc_entropy_norm` | the existing pooled `loc_entropy` ÷ log2(pooled support) — backend-free, series-continuous |

Reading rule for time-to-cover: censoring first. Read `_reached` before
`_steps`; a `_steps` mean over the 7% of episodes that got there describes the
luckiest 7%. An absent key means "not measured / censored", by design
(`tests/test_exploration_online.py::test_non_device_backends_omit_rather_than_fabricate`).

Off the DEVICE backend the `exploration/*` keys are **absent, not zero**.
Threshold levels default in `rollout_summary`'s signature. Once per run,
`reachable_cells` and `loc_entropy_ceiling` go to the wandb **run summary**
(`training/logging.py::log_run_constants`).

## How to read the numbers across arms

**Every denominator is the episode's own room's walkable count**, asked of the
`Layout` (`envs/layouts.py::Layout.walkable`), never a constant. Do **not**
compare the walkable and impassable arms by normalizing the pooled
`loc_entropy` with `loc_entropy_ceiling`: on the committed room set the pooled
support — and therefore the ceiling — is IDENTICAL across arms, because no
landmark cell is blocked in every room. Only the per-room form discriminates.
Pinned as relationships in
`tests/test_exploration_evals.py::test_pooled_ceiling_cannot_separate_the_arms`;
reasoning in `docs/exploration-evals-2026-08-30.md`.

**Reference values** (what a policy's numbers are read against — uniform
random, forward-weighted random, greedy-sweeper ceiling, per arm):

```
uv run python -m curious_george.envs.action_graph
```

RUN it; do not quote a transcribed table. ⚠️ The committed room set drifted on
2026-08-30 under a concurrent session (stencil `x` → `triangle3`, per-room
walkable 153 → 152), so any earlier transcription is one cell stale. The
tests pin relationships and tolerance bands, so they hold across the drift;
exact constants must be regenerated at whatever tree the runs launch from.

## Launching the baselines

tyro takes the **enum member name**, uppercase:

```
# forward-weighted random (the project default distribution)
uv run python main_train.py multienv-fast --arch-policy.agent RANDOM

# uniform random — the OTHER baseline
uv run python main_train.py multienv-fast --arch-policy.agent RANDOM \
    --arch-policy.random-action-probs 0.25 0.25 0.25 0.25
```

The distribution's one home is `configs.RAND_ACT_PROBA` (the
`arch_policy.random_action_probs` default); every other spelling derives from
it. The two random baselines differ by roughly 2x on coverage — pinned in
`tests/test_exploration_evals.py::test_the_two_random_baselines_are_far_apart`
— so a reported "random baseline" must always name which. The field reaches
both sampling sites, eager and CUDA-graph
(`tests/test_random_agent_baseline.py` gates it end to end, GPU path included).

Sanity check on launch: a forward-weighted RANDOM run's `exploration/coverage`
should sit near the calibration's forward-weighted row from the first log
points. If it doesn't, the wiring is wrong — stop and say so.

## Scoring finished runs offline (the developmental series)

```
uv run python -m curious_george.evaluation.checkpoint_series \
    --run <run_dir> --source selected [--impassable] --exploration \
    [--collects 2 --num-envs 64 --steps 256]
```

One row per archived **(pRNN, policy) pair** — `exploration_points` pairs the
two series by step; `score_exploration_series` rebuilds the full stack per
point through the run's own checkpoint-loading path (`run.prnn_ckpt` /
`run.policy_ckpt` into `setup_training`), seeded so every checkpoint sees the
same spawn/room schedule, under `eval_mode`, with **sampled** actions. Writes
`<run_dir>/exploration_curve.json` and prints the table. It refuses a
RANDOM-run archive (no policy weights) with directions, and finds no pairs
for runs finished before 2026-08-28 (policy archiving landed then — see
`archived_policies`' docstring). Gate:
`tests/test_checkpoint_exploration.py`, including seeded repeatability.

## Code map

- `curious_george/envs/action_graph.py` — the transition table as a graph:
  BFS distance **in actions** (a turn costs a step), the reference walkers
  (uniform / categorical / greedy sweeper), the calibration CLI.
- `curious_george/evaluation/exploration.py` — the pure metrics: coverage
  curves + thresholds with censoring, per-room visitation entropy,
  distance-binned coverage, `rollout_summary`. Tensors in, dataclasses out;
  collects nothing.
- Collector exports — `CollectResult.segment_layouts` (which room each
  episode ran in; the schedule comes from
  `envs/vector.py::DeviceTableShellPool.prepare_resets`) and
  `CollectResult.positions_episodes` (episode-major pre-action positions;
  row 0 of each episode is its spawn). Both exposed on the algo after each
  collect.
- Gates: `tests/test_exploration_evals.py` (analytic agents with closed-form
  answers + pinned calibration bands), `tests/test_action_graph.py` (table
  fidelity vs the live env, distance-in-actions), `tests/test_exploration_online.py`
  (the wiring on real rollouts), `tests/test_checkpoint_exploration.py`
  (offline series), plus additions in `tests/test_random_agent_baseline.py`.

## The count-based agent (curiosity's control)

`--train-policy.k-count FLOAT` (0 = off) adds the count-based novelty bonus:
the m-th visit a stream makes to (room, x, y, head direction) in a rollout
earns `k_count/sqrt(N + m)`, N = lifetime visits (`rl/update/rewards.py::
CountBonus`; field docstring in `configs.py` has the reasoning). The
model-free control for curiosity - run it beside the curious arm with
`--train-policy.no-curious` so exactly one reward source is active:

```
uv run python main_train.py multienv-fast --train-policy.no-curious \
    --train-policy.k-count <k>
```

`k_count` has NO tuned value yet - it needs its own scale sweep, under
`normalize_advantage`, per the noise-floor protocol
(docs/entropy-sweep-and-noise-floor-2026-08-29.md). Watch `count_bonus_mean`
in wandb: it starts near `k_count` and its decay is the novelty being
consumed. Requires the DEVICE backend; visit counts ride the policy
checkpoint, so resumes continue them. Gate: `tests/test_count_bonus.py`.

## Traps

- **Never argmax an exploration eval.** `evaluation/task.py:343` hard-codes
  greedy for batched eval rollouts; a greedy walker is a different agent, and
  for exploration that difference is the whole measurement. The offline
  scorer samples; anything new must too.
- **Don't ceiling-normalize pooled `loc_entropy` to compare arms** (above).
- **Cost is not a reason to disable anything**: the online metrics measured
  1.5 ms per parity-shaped rollout, 0.09% of collect (sync-attributed
  `CG_TIMING`).
- For intuition about the action-distance metric there is an interactive
  explainer artifact, **L-Room Action Distances** (both arms, click-to-spawn,
  per-cell shortest action sequences), in the run owner's artifact gallery:
  https://claude.ai/code/artifact/a899ab77-42ad-4271-a10b-0c8cdbbacab9

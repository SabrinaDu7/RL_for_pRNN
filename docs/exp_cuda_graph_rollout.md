# CUDA-graphing the rollout (`exp.rollout_cuda_graph`)

## Question

With `predNet.cuda_graph` and `rl.cuda_graph` both on, collection was the
largest remaining block of a training update. Can one rollout TIMESTEP be
captured as a CUDA graph and replayed, and what does that buy?

## Where the code lives

- `curious_george/rl/collect/rollout_graph.py` - `GraphRolloutStepper` (the
  capture) and `RolloutBuffers` (the per-timestep storage both paths use).
- `curious_george/rl/collect/collector.py` - the graphed branch at the top of
  the rollout loop, and `_close_device_segment`, shared with the eager path.
- `curious_george/utils/cuda_graph.py` - `no_dist_validation`, hoisted out of
  `rl/update/policy_graph.py` so both captures read one copy of the reason.
- `curious_george/utils/timing.py` - `StageTimer.disabled()`, because
  `--sync-stages` calls `torch.cuda.synchronize()` and a host sync during
  capture is illegal.
- Flag: `exp.rollout_cuda_graph` in `Configs/main.yaml`, defaulting off.
  It requires `exp.device_env` (raised in `PredictivePPOAlgo.__init__`) and,
  like its two siblings, degrades to off on a non-CUDA device.

## Method

Probe first, in a separate process per capture attempt, on static buffers:
capture the actor-critic forward, the device environment step and the pRNN
single step, alone and together, and time eager against replay. Then build it
for real and measure end to end with `tests/perf/benchmark.py`.

Reproduce the benchmark numbers with, for each `G` in {1, 8} and each `FLAG`
in {False, True}, with and without `--sync-stages`:

```
uv run --no-sync python tests/perf/benchmark.py --updates 12 --warmup-updates 3 --sync-stages \
  --override env=lroom --override run=multienv --override exp.device_env=True \
  --override predNet.batched_wm=True --override predNet.wm_pool_group=$G \
  --override predNet.compile_cell=layer --override predNet.cuda_graph=True \
  --override rl.cuda_graph=True --override exp.rollout_cuda_graph=$FLAG \
  --override exp.num_envs=128 --override rl.frames=32768 \
  --override rl.ppo_batch_size=256 --override rl.entropy_coef=0 --out <path>
```

`--sync-stages` and `--warmup-updates` are both required for the stage split
to mean anything; `--sync-stages` also inflates the rollout by ~15%, so the
speedup is read from the un-synced `fps`.

## Results, RTX 4060, 2026-08-24

Probe, one rollout iteration at B=128, h=500: eager 1.561 ms, replayed
0.151 ms - **10.3x**. Everything captured on the first attempt, including the
`torch.compile`d recurrent cell, `Categorical.sample`, `generate_noise`'s
`randn`, and the environment table's advanced-indexing gather.

End to end, ms/update under `--sync-stages`:

| stage | g=8 off | g=8 on | g=1 off | g=1 on |
|---|---|---|---|---|
| collect/policy_fwd | 195.9 | - | 201.2 | - |
| collect/sr_step | 197.9 | - | 201.4 | - |
| collect/env_step | 65.4 | - | 66.3 | - |
| collect/policy/record | 24.3 | - | 25.2 | - |
| **collect/graph_step** | - | **38.8** | - | **39.2** |
| update/wm_train | 203.8 | 203.0 | 690.5 | 694.5 |
| update/policy | 180.8 | 179.8 | 176.3 | 176.0 |

Un-synced, the axis every speed claim is stated in:

| `wm_pool_group` | fps off | fps on | speedup | wm grad steps/s off -> on |
|---|---|---|---|---|
| 8 | 34,044 | 54,845 | **1.61x** | 16.62 -> 26.78 |
| 1 | 22,274 | 30,067 | **1.35x** | 87.01 -> 117.45 |

The saving is ~365-380 ms/update at BOTH pool settings, as it must be:
`wm_pool_group` reaches only `rl/update/world_model.py` and never the
collector. Only the fraction differs, because `update/wm_train` is 203 ms at
g=8 and 693 ms at g=1.

## Gates

`tests/test_cuda_graph_rollout.py`. With pRNN noise and dropout off and the
policy saturated - `_saturate` scales the actor head so the softmax is an
exact float32 one-hot and `multinomial` returns the argmax for any draw, which
removes the sampling RNG WITHOUT touching the code path - the rollout has no
randomness left, so graphed and eager must agree bitwise. They do, across
three updates with real PPO and world-model steps in between, and across
`on_device` round trips, with and without the other two graphs on.

The negative control matters more than the gates: a gate that passes either
way is worthless. `test_bitwise_gate_catches_a_poisoned_static_buffer` poisons
a buffer the graphs read, and three deliberate breaks of production code -
rebinding the pRNN state, ignoring the phase key, freezing the timestep
index - were each confirmed to turn the suite red before being reverted.

## Discoveries

- **Two prerequisites, both defects on their own terms**, fixed in `830ba48`
  before any of this: `BatchedSRTracker` rebound its hidden state every step,
  and `observation_device()` lent out its live `directions` tensor.
- **The pRNN phase is part of the graph key.** `thRNN_5win` has
  `phase_k = 6` and `inMask = [1, 0, 0, 0, 0, 0]` - it sees the observation on
  one timestep in six - and `step_synchronized` branches on that in Python, so
  the branch is baked in at capture. Six graphs; both variants cost the same
  (0.1514 vs 0.1522 ms).
- **A graphed rollout is not bit-comparable to an eager one in production**,
  because a captured region draws from CUDA's graph-safe RNG stream. That is
  why the gate saturates the policy rather than seeding the two arms alike.
- **`predNet.compile_cell` is nearly redundant for the rollout once graphed**:
  0.136 vs 0.150 ms replayed, ~10%. It still fuses the length-256 world-model
  forward, which this did not measure.
- **`compare_metrics --loose` has no power over `policy_loss` at n=12.** The
  g=1 flag-off/flag-on comparison FAILs on it (mean -0.0275 -> -0.0049), and so
  does a control comparing flag-OFF against flag-OFF at two different seeds
  (+0.0077 -> +0.0016, exit 1). `policy_loss` is centred on zero, so a
  50%-relative-mean threshold is measuring seed noise. The g=8 comparison
  PASSes. This is a flaw in the checker, not a finding about the graph.

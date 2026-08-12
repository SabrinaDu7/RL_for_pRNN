# Setting `rl.steps` after the rollout batching (2026-07-30)

## Purpose

`Configs/algo/ppo.yaml:steps` is `2.048e7`, chosen when a rollout update was
`frames=2048` over `num_envs=8`. `performance=ultra` raises the update to
`frames=262144` over `num_envs=1024`. This note establishes what `steps`
actually controls, what the batching changed, and what to set it to.

## What `steps` means (confirmed)

- `curious_george/training/loop.py:85-99`: the loop is
  `while num_frames < cfg.rl.steps`, and `num_frames += logs["num_frames"]`.
- `logs["num_frames"] = B * T` (`curious_george/rl/collect/collector.py:594`),
  i.e. total environment frames collected in the update, summed over envs.

So `steps` is a **total-frame budget** and its unit is unchanged by batching.
What changed is the number of frames consumed per optimizer step.

## What the batching actually changed (confirmed)

pRNN training cadence, `curious_george/rl/update/world_model.py:28-49`:

- `batched_wm=False`: one `trainStep` **per episode segment** → `B` serial
  gradient steps per update (batch 1, length `seqdur`).
- `batched_wm=True` (ultra): `train_on_episodes_batched`
  (`curious_george/world_model/adapter.py:645-684`) does **one** pooled
  `trainStep` over all `B` segments; the pooled loss is the mean over the
  batch, so `B` optimizer steps collapse into 1.

PPO cadence: minibatches per epoch = `frames / ppo_batch_size`
(`curious_george/rl/update/updater.py:36-58`), times `ppo_epochs=4`.

With `seqdur=256` and `T = frames/num_envs = 256`, one segment per env per
update:

| | baseline (`main`) | `performance=ultra` |
|---|---:|---:|
| `num_envs` (B) | 8 | 1024 |
| `frames`/update | 2,048 | 262,144 |
| segments/update | 8 | 1,024 |
| **pRNN grad steps/update** | **8** (batch 1) | **1** (batch 1024) |
| PPO minibatches/epoch | 8 | 2 |
| PPO Adam steps/update | 32 | 8 |
| updates at `steps=2.048e7` | 10,000 | **79** |
| **total pRNN grad steps** | **80,000** | **79** |
| total PPO Adam steps | 320,000 | 632 |
| wandb log events (`log_interval=1`) | 10,000 | 79 |

At an unchanged `steps=2.048e7` the ultra preset sees the same 80k trajectories
and the same 2.048e7 frames, but takes **~1/1000 the pRNN optimizer steps** and
~1/500 the PPO steps. The pRNN loss cannot travel as far, and there are ~127×
fewer wandb points. Both observed symptoms follow directly from this; no
logging bug is required to explain them (inferred for the specific wandb runs —
not yet checked against the run histories).

## FPS is the wrong metric here; updates/s is

Using the PR's own MPS sweep (`docs/throughput_investigation_2026-07-23.md`),
updates/s = FPS / frames-per-update, and pRNN grad steps/s = updates/s × steps
per update:

| Config | FPS | updates/s | pRNN grad steps/s |
|---|---:|---:|---:|
| CPU base, B=8, serial WM | 1,850 | 0.90 | **7.2** |
| device B=256, pooled WM | 65,296 | 1.00 | 1.00 |
| device B=512, pooled WM | 92,437 | 0.71 | 0.71 |
| device B=1024, pooled WM | 107,354 | 0.41 | **0.41** |
| device B=2048, pooled WM | 114,448 | 0.22 | 0.22 |
| device B=4096, pooled WM | 68,767 | 0.066 | 0.066 |

The 58× FPS gain buys **0.057× the pRNN optimizer steps per second**. Ultra is
worth it for pRNN learning only if one batch-1024 pooled step is worth ~18
batch-1 steps. That is implausible: the critical batch size for a 256-step
recurrent prediction loss is almost certainly far below 1024, so the pooled
gradient is mostly redundant averaging. This is an argument from
noise-scale reasoning, not a measurement on this task — the measurement that
would settle it is a fixed-wall-clock pRNN-loss / sRSA comparison across
`num_envs ∈ {8, 64, 256, 1024}`.

## Recommendation

`steps` should be chosen from the number of updates you want, then multiplied
by `frames`:

```
steps = desired_updates × rl.frames
```

For `performance=ultra` (`frames=262144`):

| Goal | updates | `steps` | M4 wall-clock @107k FPS |
|---|---:|---:|---:|
| same update/log count as the old runs | 10,000 | **2.62e9** | ~6.8 h |
| sqrt-batch-scaled schedule (80k/32) | 2,500 | 6.55e8 | ~1.7 h |
| match old pRNN grad-step count | 80,000 | 2.10e10 | ~54 h |
| current value (do not keep) | 79 | 2.048e7 | ~3 min |

Recommended starting point: **`steps: 2.62e9`** with `performance=ultra`. It
restores the update count (and therefore the logging density) of the old runs
and gives 10,000 pooled pRNN steps. It costs 128× the frames of the old budget,
which is the price of pooling 1024 trajectories per gradient step.

Also worth changing alongside it, since PPO is only ~0.25 s of a ~1.6 s update:
lower `rl.ppo_batch_size` (e.g. `16384` → 16 minibatches × 4 epochs = 64 Adam
steps/update) to recover PPO cadence nearly for free.

### Cheaper alternative: shrink the pooled batch

For the *same* 10,000 pooled pRNN steps, `num_envs=256`
(`frames=65536`, `steps=6.55e8`) costs 65,536 frames/update at 65k FPS ≈ 2.8 h —
**2.4× less wall-clock than B=1024** for an identical optimizer schedule,
because updates/s is higher at B=256 even though FPS is lower.

### Best option, needs a small change

Chunk the pooled world-model step: split the 1024 segments into `K` pooled
`trainStep` calls of `1024/K` segments each in
`adapter.train_on_episodes_batched`. That keeps the B=1024 rollout throughput
while restoring `K` gradient steps per update. `K=8` (batch 128) is the obvious
first try. Not implemented; no measurement yet.

## Measured: the two wandb runs (confirmed 2026-07-30)

Pulled via `wandb.Api()` from `blake-richards/curious-george`. OLD =
`pRNN_curious_26-07-23-10-06-25`, NEW = `prnn_curious_26-07-30-10-37-02`.
Caveat: OLD's history came back at exactly the 100,000-row sampling cap out of
110,349 logged steps, so OLD per-metric counts are from wandb's sampled
history; the values themselves are real logged points, not interpolations.

| | OLD | NEW |
|---|---:|---:|
| wallclock (`_runtime`) | 16,667 s = **4.63 h** | 214 s = **3.6 min** |
| final FPS | 1,231 | 102,549 (83×) |
| updates | 10,000 | **79** |
| frames | 2.048e7 | 2.071e7 |
| `num_envs` / `frames`per update | 8 / 2,048 | 1024 / 262,144 |
| `batched_wm` | false | true |
| wandb `_step` | 110,349 | **315** |
| pRNN grad steps | 80,000 | **79** |
| final pRNN loss | **0.00486** | **0.02058** |
| final `policy_entropy` | 1.328 | 1.971 (≈ max) |
| final `MI_policy` | 0.298 | **1.4e-4** |
| `sRSA_onPolicy` points | 45 | **0 — never logged** |

The `_step` counts decompose exactly: pRNN loss is logged **inside** prnn's
`trainStep` (`predictiveNet.py:588-595`, one `wandb.log` per gradient step), so
OLD = 80,000 loss logs + ~3/update from `training/logging.py` + analysis
events ≈ 110k; NEW = 79 + 4/update ≈ 315. `curious_reward_turn_right` has 79
non-null points in NEW with **no** gaps. There is no step-logging bug: the
sparse curves are 79 updates, exactly as predicted above.

### The decisive comparison

pRNN loss indexed by **gradient step** (OLD batch-1 serial, NEW batch-1024 pooled):

| grad step | 0 | 5 | 10 | 20 | 40 | **79** | 2,560 | 20,480 | 72,474 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD | .0405 | .0292 | .0251 | .0218 | .0208 | **.0179** | .0159 | .0084 | .0049 |
| NEW | .0363 | .0252 | .0220 | .0211 | .0208 | **.0206** | — | — | — |

At equal gradient steps the two curves are the same to within OLD's batch-1
noise band (OLD wanders .018–.021 between steps 79 and 1,280). **The pooled
batch-1024 step buys no more loss reduction than a batch-1 step**, while
consuming 1024× the frames (2.07e7 vs 20,224) and ~6× the wallclock per step.

At equal wallclock the result is a dead heat: at t = 214 s, OLD is at **0.02045**
and NEW is at **0.02058**. NEW's entire 3.6-minute run reaches the loss OLD
passed through in its first ~3.5 minutes — and OLD had already hit 0.0179 by
t = **17.8 s**, a value NEW never reaches at all.

So the 83× FPS gain converted into **zero** pRNN learning speedup. Per
wallclock the ultra preset is, if anything, marginally behind the CPU path. The
bottleneck was never frame throughput; it is pRNN gradient steps, and pooling
1024 trajectories into one step discards essentially all of the extra data's
value. NEW's flat tail is not convergence — it is OLD's long .018–.021 plateau,
which OLD only escapes around grad step ~2,500.

Secondary casualties in NEW, all downstream of 79 updates:

- `sRSA`/`SWdist`/`SI` never logged (`analysis_interval=200` > 79 updates), so
  the run produced no scientific metric at all;
- `MI_policy` 1.4e-4 and `policy_entropy` 1.97 (≈ 2 bits, uniform over 4
  actions): the policy did not learn, on 632 Adam steps.

### Consequence for `steps`

The `steps` table above still applies arithmetically, but the measurement kills
`num_envs=1024` as a production setting: matching OLD's 80,000 gradient steps
costs `steps=2.1e10` ≈ **56 h** at 102.5k FPS, versus OLD's 4.63 h for the same
optimizer schedule. Raising `steps` alone makes the run strictly worse than the
branch it was meant to speed up.

The device-resident environment itself is not the problem — it is correct and
nearly free (~0.01 s/update). The pooled world-model step at B=1024 is. Fix
the gradient cadence first (chunk the pooled step into `K` sub-batches, and/or
drop `num_envs` to 64–256), re-measure loss-vs-wallclock against OLD, and only
then set `steps`.

## Measured: `trainStep` cost vs batch size (confirmed 2026-07-30)

Script: `tests/perf/sweep_trainstep_batch.py` (new). Builds the real pRNN
through the ordinary `setup_env`/`setup_world_model` path and times one full
`trainStep` (forward + backward + RMSprop) on synthetic tensors with the exact
shapes/dtypes `train_on_episodes_batched` produces. Cost only — it says nothing
about gradient quality. `thRNN_5win`, hidden 500, L=256, median of 10 (CUDA) /
6 (CPU) reps after warm-up.

Hardware: this laptop — NVIDIA RTX 4060, and an 8-core CPU with
`torch.get_num_threads()==4`. Raw JSON in `tests/perf/results/`.

| B | CUDA s/step | CUDA steps/s | CPU s/step | CPU steps/s |
|---:|---:|---:|---:|---:|
| 1 | 0.183 | 5.48 | **0.122** | **8.17** |
| 2 | 0.186 | 5.37 | 0.137 | 7.31 |
| 4 | 0.186 | 5.38 | 0.148 | 6.78 |
| 8 | 0.192 | 5.20 | 0.171 | 5.85 |
| 16 | 0.187 | 5.35 | 0.223 | 4.49 |
| 32 | 0.191 | 5.24 | 0.335 | 2.99 |
| 64 | 0.194 | 5.17 | 0.564 | 1.77 |
| 128 | 0.193 | 5.18 | 1.023 | 0.98 |
| 256 | 0.201 | 4.97 | — | — |
| 512 | 0.225 | 4.44 | — | — |
| 1024 | 0.270 | 3.71 | — | — |

Two results, both important:

1. **On GPU, batch size is nearly free up to B≈128** — 0.183 s at B=1 vs
   0.193 s at B=128, +5.8%. The 256-step recurrence is launch-bound, not
   compute-bound. So chunking a pooled step into `K` sub-steps costs
   ≈ `K ×` the world-model time; you cannot buy back 1024 gradient steps
   cheaply. Conversely, if a batch-8..128 gradient is *better* than batch-1,
   that improvement is essentially free — which is the one remaining
   unmeasured upside.
2. **The GPU is 1.5× SLOWER than this CPU at batch 1** (0.183 s vs 0.122 s) and
   only overtakes it at B≥16. Gradient steps per second — the metric that
   actually governs learning — peaks at **8.17/s on plain CPU at B=1**, above
   anything the 4060 achieves at any batch size.

### The hard ceiling this implies

`tests/perf/benchmark.py`, baseline config (`num_envs=8`, `frames=2048`, no
pooling), 5 updates on this CPU, 944.8 FPS, ~10.8 s total:

| stage | total_s | share |
|---|---:|---:|
| `update/wm_train` (40 grad steps) | 4.573 | **42%** |
| `collect/env_step` | 2.496 | 23% |
| `collect/curious_rewards` | 1.583 | 15% |
| `collect/sr_step` | 0.863 | 8% |
| `collect/policy_fwd` | 0.545 | 5% |
| `update/policy` | 0.461 | 4% |

4.573 s / 40 = 114 ms per gradient step, consistent with the sweep's 122 ms at
B=1 (confirms the two measurements agree).

The world-model step is **42% of wall-clock and is irreducible without changing
the gradient schedule**. So if every other stage were made *free* — a perfect
version of what this PR set out to do — the ceiling is a **~2.4× speedup**, not
83×. The device-resident work targets `env_step` + `curious_rewards` +
`sr_step` + policy ≈ 55% of the total, so it is attacking the right 55%; it
just cannot beat the 42% floor.

Caveat: the 42% split is measured on this laptop CPU; the 4.63 h OLD run was
not, so do not combine the two into a single predicted runtime. The
same-machine claim is only the ratio.

### What this means for the plan

- Chunking recovers gradient steps at linear cost. On GPU, `K=8` chunks of
  B=128 costs ~1.54 s of world-model time per update versus 0.27 s pooled —
  more steps, proportionally more time. It is not a free lunch, but it is the
  correct direction because steps are what move the loss.
- The only free win left is the flat region: **B between 8 and 128 costs the
  same as B=1 on GPU.** If gradient quality improves anywhere in that range,
  it is pure profit. We measured that B=1024 gives no per-step improvement over
  B=1; whether B=32 or B=128 does is **unmeasured and is the deciding
  experiment.**
- If it does not, then no rollout optimization can beat ~2.4× on this workload,
  and the honest fix is a cheaper recurrent step (fusion/compilation of the
  256-step unroll), not a bigger batch.

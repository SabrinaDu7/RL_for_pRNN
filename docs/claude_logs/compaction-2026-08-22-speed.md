2026-08-22 · branch `sdu/speed`

# Speed: a 7.3× training speedup, two ratios nobody had named, and eight failures

Primary read: [`../exp_speed_cuda_graph_2026-08-19.md`](../exp_speed_cuda_graph_2026-08-19.md).
That is the measured record; this is the state of play, the failure modes, and
what is in flight.

---

## 1. The headline

```bash
sbatch slurm/train_fast.sh rooms lroom_multi        # ~1h47m, n=3
```

**Multi-room training reproduces the committed reference learning curve in
1 h 47 m instead of ~11.5 h**, replicated across three seeds
(`01:46:51 / 01:47:35 / 01:47:34`, jobs 10445207 / 10445463 / 10445464):

```
reference           three-seed mean      sd
0.4562 @ 10.5M      0.4523 @  8.4M    0.0076
0.6690 @ 31.5M      0.6657 @ 33.6M    0.0328
0.7382 @ 52.4M      0.7397 @ 50.3M    0.0339
0.7870 @ 73.4M      0.7942 @ 75.5M    0.0221
0.7905 @ 94.4M      0.8013 @ 92.3M    0.0230
```

Every point inside one sd. **2,447 → 17,965 environment steps/s at matched
dynamics.** First result in the project carried at n=3 rather than n=1 — which
was affordable *because* the run got fast.

What did it, in order of contribution:

| | |
|---|---|
| `compile_cell=layer` | `torch.compile` the **whole 256-step loop**, not the cell. 189 → 49 ms, **3.89×** |
| `num_envs=128` | rollout cost per update is flat 8 → 256 envs |
| `wm_pool_group=8` | the correctness fix (§3), and fewer steps so also faster |
| `--gres=gpu:l40s:1` | GPU type is worth **1.7×** |

**`predNet.cuda_graph` is NOT in the final config**, despite being the fastest
thing measured (105 grad/s, 89× the default). See §3.

## 2. Why the pRNN is slow — the budget, closed

Per timestep of the world-model step, measured:

```
kernel launches                41.0
CPU time                     1041.0 us    of which cudaLaunchKernel 250.0 (24%)
                                          the other ~780 us is PyTorch dispatch
GPU time                      132.8 us    (CPU/GPU 7.8x)
memory floor                    5.11 us   (must read 1.28 MB of weights)
```

Three layers, each with a number: a **5.11 µs physics floor**, **132.8 µs** of
GPU because the step is 41 separate small kernels, **1041 µs** of CPU because
161 aten ops are dispatched to command them. **The dominant cost is PyTorch
dispatch — not the driver, not the GPU.** An RTX 8000 and an RTX 4060 give the
same per-step time (0.1879 vs 0.1946 s): a 4× bigger GPU buys nothing.

Targetable: the launches (fusion 2.47×, graphs 8.59×). Not targetable: the
weight read. Targetable only by batching: its *per-sample* cost — 12.076 µs at
B=1 → **0.117 µs at B=128**.

## 3. The science result the speed work turned up

**The world model can be over-trained, and the symptom is invisible in the
loss.** Serial world-model training takes 1 gradient step per 256 env steps;
the pooled reference takes 1 per 2048 — **8× apart**. Running serial (which is
what `cuda_graph` accelerates):

```
sRSA peaks 0.7732 @25.2M, then FALLS to 0.5581 @83.9M
while prediction loss keeps IMPROVING 0.0055 -> 0.0035
SWdist runs 2.3x the reference throughout
```

Loss down, place code down. It echoes the known result that L2 collapsed the
place code (r 0.97 → 0.73) while prediction stayed healthy.

Fixing the step **count** alone was not enough. `wm_segment_stride=8` restores
the reference's rate but each step then sees ONE segment where the reference
pools eight — 8× the gradient variance — and it reached only **0.6383 against
0.7905 at essentially the same step count** (49,150 vs 46,094). That is what
proved it is gradient *quality*, not step count.

**`predNet.wm_pool_group=8`** — pooled in groups of 8 — matches both, and is
what shipped. It also forced dropping `cuda_graph`, because the graph only
engages on the serial path (`_use_graph_wm` is consulted in
`train_on_episode`, never in `train_on_episodes_batched`). **The fastest
configuration was the wrong one.**

## 4. Two ratios nobody had named

Both are gradient steps per unit of EXPERIENCE, and both were silently changed
by the speed work:

```
                   wm steps/env step    policy steps/env step
reference             1 per  256            1 per  64
first fast config     1 per  256            1 per 2048   <- policy 32x diluted
shipped config        1 per 2048            1 per  256
```

`ppo_batch_size` scaled with `frames` holds policy steps per *update* fixed,
which silently dilutes them per *environment step*. `TrainingSchedule.summary()`
now prints **env steps per world-model step** so the regime is legible at
startup.

## 5. Diagnostics settled along the way

- **`entropy_coef` explains the low `MI_policy`.** At 0.01 the entropy bonus is
  **0.0198 against an advantage scale of 0.0369** — over half the learning
  signal — so `policy_entropy` pins at ~1.98 and MI collapses to 0.015 against
  the reference's 0.28. At `entropy_coef=0`: MI reaches **0.273**, entropy falls
  1.80 → 1.06. Confirmed.
- **sRSA does NOT have a high floor.** An untrained net scores
  **0.062 ± 0.040** (n=5 inits). "Starts near 0.5" was sampling cadence — the
  first sample landed at 8.4M steps, 20× later than the reference's first.
- **A falling `sRSA_onPolicy` can be the policy, not the map.** On the same
  ent=0 checkpoints, off-policy sRSA **rises** 0.4388 → 0.6437 while
  `sRSA_onPolicy` **falls** 0.6306 → 0.4539, and mean SI rises monotonically
  0.70 → 1.01. `scripts/multienv/checkpoint_curve.py` has always evaluated with
  a RANDOM agent; wandb logs the on-policy number. They are different
  measurements.
- **Single-shot sRSA carries ±0.05.** Repeats give sd 0.02–0.06 per checkpoint
  (`scripts/trace/srsa_repeats.py`).
- ⚠️ **Unexplained:** off-policy sRSA still falls **−17%** after ~50M (0.6437 →
  0.5312) with SWdist rising 0.036 → 0.068. Not accounted for by coverage.

## 6. Failure modes — read this part

**Eight cluster submissions failed before one worked**, each for a different
reason, all because I tested the *config* locally but never the full *script
path*. Once I ran the whole path locally it worked first time and every time
after.

| # | failure | cause |
|---|---|---|
| 1 | `git checkout --detach origin/sdu/speed` | `clone --shared` copies only refs/heads |
| 2 | gate asserted `tests/golden_omt` | **cannot run from a clean clone** — `data/obs_bank/` is untracked |
| 3 | 2 h job lost every result | rsync only at script end; wall-clock kill wiped `$SLURM_TMPDIR` |
| 4 | `PermissionError: '/home/sabrina'` | **`.env` is COMMITTED with a machine-specific `RL_STORAGE`** |
| 5 | died at wandb init | `logging.wandb_log` defaults True, never overridden |
| 6 | died in kaleido | *caused by fixing #5* — wandb off routes behaviour analysis into `fig.write_image()` |
| 7 | traceback invisible twice | `2>&1 \| tail -40` showed the config dump, not the error |
| 8 | blank figure row | `env.show_state(render, t, **kwargs)` accepts `ax`/`fig` and **silently ignores them** |

**Analysis mistakes I made and had to retract — four of them:**

1. "cat-of-one + permutes are ~30% of hot-loop ops" → measured **1%**.
2. "a numpy float64 leaks into the LayerNorm parameter" → all params float32.
3. "the per-timestep `.t()` makes `mm` copy" → **zero** copies; `F.linear` is *slower*.
4. **"launch overhead is most of the cost"** → `cudaLaunchKernel` is **24%**. I
   had conflated a whole Python op round-trip (7.08 µs) with a driver launch.

The pattern: every one came from reasoning about code instead of measuring it,
and every one was caught by measuring. Ranking ops by **time** rather than
**count** is what finally closed the budget.

**Process lessons worth keeping:**

- Benchmarks lie about production: `tests/perf/benchmark.py` reported 106.74
  grad/s where a real run sustains 75.3 — benchmarks exclude archiving, saves
  and layout resampling. **Believe the run.**
- The archive cadence is part of the time budget. 79 checkpoints cannot be
  scored in the window; a local gate hit **142** and had to be killed.
- Never benchmark on a shared GPU. Contention once made a 4% change look like
  1.29×.
- GPU type is load-bearing: **1.7×** between an L40S and a Quadro RTX 8000.
  Never compare numbers across cluster jobs without checking the node.

## 7. Repo flaws found, not fixed

Each is a decision that is not mine to make:

- **`.env` is tracked and holds absolute machine paths**
  (`RL_STORAGE="/home/sabrina/..."`). Any clean clone elsewhere writes to that
  user's home. Worked around in the slurm script only.
- **`tests/golden_omt` is not reproducible from git alone** — it needs
  `data/obs_bank/`, which is untracked and generated.
- **`docs/sab_context/` is gitignored** yet cited as a source of truth by
  `Configs/run/multienv.yaml`, `Configs/performance/ultra.yaml` and
  `tests/perf/benchmark.py`. The repo ships dangling references.
- **`Shell.show_state(render, t, **kwargs)`** accepts `ax`/`fig` and draws on
  the global current axes regardless (upstream `prnn`).

## 8. Where we left off

**In flight — a 2×2 at the reference's own budget** (20,480,000 env steps =
80,000 episodes, `entropy_coef=0`, single L-room, 100 analysis points,
off-policy eval on). All four verified from their schedule lines:

| job | `wm_pool_group` | `ppo_batch_size` | regime |
|---|---|---|---|
| 10448384 | 8 | 1024 | baseline: wm 1/2048, policy 128/update |
| 10448385 | 1 | 1024 | **A** — wm rate only: 1/256 |
| 10448386 | 8 | 256 | **B** — policy rate only: 512/update |
| 10448387 | 1 | 256 | **C** — both = **the reference regime** |

Compare against
<https://wandb.ai/blake-richards/curious-george/runs/pRNN_curious_26-07-23-10-06-25>
(`entropy_coef: 0`, `ppo_batch_size: 256`, `batched_wm: false`, 20.48M steps).
The question: does closing both ratios reproduce that run's sRSA plateau
(~0.60–0.67 on-policy), and which axis matters?

**Open:**

1. The unexplained −17% off-policy sRSA decline after ~50M (§5).
2. **Multi-seed batching is unbuilt.** The physics argument is the strongest
   thing found: the weight read is already paid, so 128 seeds ride the same
   12–15 µs call, 103× more efficient per seed. Blocked on a logging decision —
   prnn logs `"pRNN loss"` under a fixed key from inside itself, which collides
   with N seeds and conflicts with the standing "prnn owns its metrics" rule.
3. `cuda_graph` remains measured, gated and **unadopted**. If the pooled path
   ever needs it, the graph would have to learn the batched path.
4. The full 491.5M budget is **~7.6 h**, not 2 h. That is arithmetic, not
   engineering — 1.92M gradient steps against a ~62 grad/s ceiling.

## 9. A behaviour discontinuity in the run series (2026-08-23)

`recurrence` was removed from the PPO path, and **that changes training
numerics** - every run before it is on one side of a line, every run after on
the other.

`recurrence` was fixed at 1 everywhere (`algo.py` default, `setup.py` passed it
explicitly, nothing else ever set it), but its machinery was live. Inside
`get_batches_starting_indexes`, on alternating epochs:

```python
indexes = indexes[(indexes + recurrence) % num_frames != 0]   # drops one index
indexes += recurrence // 2                                    # adds 0
```

At recurrence=1 that **drops exactly one transition and shortens the final
minibatch, on half of all PPO epochs, in service of a shift of zero**. With
`ppo_epochs=4` that is 2 epochs in every 4, one transition of `rl.frames`.

Confirmed two ways, not argued:

```
old vs new partition at the golden's shape (frames=64, batch=16)
  even epoch   identical=True    [16,16,16,16]   covers 64/64
  odd  epoch   identical=False   old [16,16,16,15] covers 63/64
                                 new [16,16,16,16] covers 64/64
```

and by restoring **only** the drop while keeping the rest of the removal, which
made `tests/golden_omt` pass 5/5 again. So the refactor is bitwise-clean
everywhere else; the goldens move because the bug is fixed.

**Consequence for the record:** `tests/golden_omt/golden_omt_v0.pt`
(2026-07-17) is the pre-fix oracle and is kept; `golden_omt_v1.pt` (2026-08-23)
is captured on the fixed code and is what the test reads. Divergence appears
first in `batches[1].curious_rewards`, max|diff| 1.9e-3 - batch 0 is collected
before the first update, so it still matches. Runs launched before this commit
(including every run in the tables above, and jobs 10449418 / 10449419) trained
on 32,767 transitions on odd epochs; runs after train on all 32,768. The effect
is small but it compounds through the policy into the rollout, so **do not
expect bitwise agreement across the line** - compare curves, not tensors.

Also removed with it, since nothing else used them: the `batch_num` counter,
the `for i in range(recurrence)` loop and its five `/= recurrence` divisions,
and the two `recurrence` asserts. `update_policy` now returns `UpdateLogs`
instead of `(UpdateLogs, batch_num)`, and `get_batches_starting_indexes` is
`shuffled_minibatches(*, num_frames, batch_size)`, gated by
`tests/test_policy_minibatches.py` - the partition test the old code never had.

**A second flaw fixed on the way:** `PredictivePPOAlgo.__init__` took 30
positional parameters with `recurrence` twelfth, and carried a comment
accommodating the hazard (`batched_wm=False,  # appended last: positional
callers exist (tests)`). Removing a mid-signature parameter would have silently
shifted every argument in the two positional callers. Everything after `device`
is now keyword-only.

## 10. The pooled path is now graphable (2026-08-23)

9 removed the reason `cuda_graph` was unadoptable. The graph only ever served
`train_on_episode`; `_GraphWMTrainer` (renamed from `_GraphWMSegmentTrainer`,
which no longer described what it does) now serves
`train_on_episodes_batched` too, so the SHIPPED regime can be graphed rather
than only the over-training serial one.

`batched` is part of the graph key alongside `(B, L+1)`, because at B=1 the
batched layout `(1, L, X, B)` is not the serial `(1, L, X)` - one key for both
would silently feed the wrong shape. B is in the key so a ragged final group
(when `wm_pool_group` does not divide the segment count) captures separately.

**Measured, steady state, group=8 L=256 h=500 on an RTX 4060:**

```
compile_cell  graphed   ms/step
false         false      199.83
false         true        22.28    8.97x
layer         false       56.43    3.54x   (matches the doc's 3.89x)
layer         true        13.48    4.19x on top of compile
```

End-to-end, production config: **1.47x** on fps (15,198 -> 22,338), 1.56x on
the timed total. `compare_metrics.py --loose` **PASS** on every recorded
metric; `prnn_loss` 0.021087 -> 0.020690.

⚠️ **2's stage shares were WRONG, and so was the first correction to them.**
`tests/perf/benchmark.py` defaults `--sync-stages` off and `--warmup-updates`
to 0. Without a device sync at each stage boundary a stage that merely QUEUES
GPU work is credited with the wait for work queued earlier, and without warmup
`torch.compile`'s one-time compilation lands inside `update/wm_train`. Both
defaults were used. Re-measured with `--updates 12 --warmup-updates 3
--sync-stages`:

```
                        4 upd    20 upd   SYNCHRONIZED    ms/update
update/wm_train         51.0%     51.5%         48.4%           938
update/policy            8.2%     17.9%         26.3%           509
collect/sr_step          6.4%      7.6%         10.1%           196
collect/policy_fwd       3.9%      6.6%         10.2%           197
collect/env_step         0.9%      2.1%          3.3%            63
collect/curious_rewards 29.3%     14.3%          1.3%            26
```

**`collect/curious_rewards` is 1.3%, not 29.3%** - the unsynchronized timer was
crediting it with ~350 ms per update of someone else's queued work. The
"fuse the curiosity forward into the training pass" lever was chasing a 1.3%
stage; it is dead on payoff, independently of the semantics objection.

The synchronized figures are the trustworthy ones because they are corroborated
OUT OF BAND: `tests/perf/sweep_graph_compile.py` times the pooled step in
isolation at 57.27 ms, and 16 steps x 57.27 = 916 ms against the profile's 938;
an isolated curiosity forward measures 26.87 ms against the profile's 26. The
unsynchronized runs agree with nothing.

**Never quote a stage share from this tool without `--sync-stages
--warmup-updates`.** Naming the flags is not enough - their defaults are the
trap.

**After graphing the world model the ordering changes**, which is what any next
optimization should target:

```
post-graph ms/update    update/policy 522 (42%)   collect/sr_step   208 (17%)
                        collect/policy_fwd 209 (17%)   update/wm_train 203 (16%)
                        collect/env_step 66 (5%)  collect/curious_rewards 27 (2%)
```

**What is gated, and what is not.** Bitwise equivalence to eager is gated:
with dropout and noise off, N sequential pooled replays move the weights
identically to N eager `trainStep(batched=True)` calls
(`test_pooled_graphed_bitwise_equals_eager_no_rng`). That is the property
`wm_pool_group` depends on - the in-graph optimizer step must advance the
weights BETWEEN replays, and a separate test asserts every replay moves them.
**Not gated: sRSA/SWdist over a real run.** `tests/perf/benchmark.py` does not
record the spatial metrics - it calls `run_spatial_analysis` only to TIME it,
and that function returns None. Closing that is a prerequisite for gating any
future world-model optimization on the metrics that actually matter.

**The standing cost is unchanged and now lands on production.** Under
`cuda_graph` the model-moving diagnostics are skipped
(`loop.py::_skip_model_move_diag`), so a graphed run logs no in-run sRSA/SWdist
and no prediction images; they are recoverable only offline from archived
checkpoints. `predNet.cuda_graph` therefore stays **False** in
`slurm/train_fast.sh` until that trade is decided deliberately.

**Tools added this session:** `tests/perf/{find_sync,profile_api,roofline_step}.py`,
`scripts/trace/{prediction_panel,srsa_repeats}.py`,
`predNet.{compile_cell,wm_pool_group,wm_segment_stride}`, `slurm/train_fast.sh`.

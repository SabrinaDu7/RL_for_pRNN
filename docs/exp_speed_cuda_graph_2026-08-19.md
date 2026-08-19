2026-08-19 · branch `sdu/speed`

# The world-model gradient step was the bottleneck, and it was already fixed

> ⚠️ **NOT ADOPTED — decision, 2026-08-19 (Sabrina).** CUDA graphing is **on
> hold**: *"I don't understand enough to be able to judge your work."* That is a
> checkability call, not a correctness one, and it is the right one — this repo's
> value is that results are judgeable by the person who owns them, and an
> unreviewable 8.59× is worth less than a 1.4× that can be checked.
>
> Consequences, so nothing here is read as a live recommendation:
>
> - `predNet.cuda_graph` **stays `False`** (its config default). Nothing in this
>   document changes any default.
> - The §5 safety guard IS still worth having and is committed: it is inert while
>   the flag is off, and makes an **already-present** flag survive a long run if
>   anyone ever turns it on. It adds no graph capability.
> - **§9b's `num_envs` numbers were measured WITH the graph on and do not
>   transfer.** Without it the world-model step is 0.198 s rather than 0.0161 s,
>   so widening adds WM time linearly against a 12× larger constant. Re-measured
>   eager in §9c — read that, not §9b, for what is actionable today.
> - The rollout graph sketched in §8 is **not being prototyped**.
>
> What survives as directly actionable and graph-free: the diagnosis (§2, §8),
> the closed sync thread (§8), the saturation headroom and its CPU cap (§9b), and
> the hot-loop op census (§9).

**Status: 8.59× on gradient steps per second from a flag that already existed,
and 31.6× when combined with `num_envs`.** No new kernel, no new numerics. The
previous session's engineering ceiling estimate of ~2.1× was derived on the wrong
term.

The two are **not** the same kind of claim and are kept apart throughout:

| | change | grad steps/s | kind |
|---|---|---|---|
| **E** | `predNet.cuda_graph=True` on the serial path | 3.53 → 10.19 | **engineering** — identical gradient-step count and math (§4) |
| **M** | `num_envs` 8 → 64 on top of it | 10.88 → 37.50 | **methodological** — dilutes policy steps per env step (§9b) |

⚠️ The same configuration (serial + graphed, `num_envs=8`) measures **10.19** in
§3 (6 updates) and **10.88** in §9b (4 updates) — a ~7% run-to-run spread on
short benchmarks. Each table quotes the run it came from; do not read the
difference as an effect.

E is mine to recommend once §6 passes. M is Sabrina's call and needs its own
learning gate.

## What to actually do, given the graphing hold

Everything below is on **world-model gradient steps per second**, `num_envs=8`,
serial world model, RTX 4060, idle GPU.

| | change | grad steps/s | status |
|---|---|---|---|
| baseline | today's production default (pooled + eager) | **1.19** | — |
| | serial instead of pooled | 3.53 | **methodological** — pooling is deliberate in the multi-room design (§7) |
| ✅ | `predNet.compile_cell=layer` on top of serial | **8.22** | **ready** — 2.47× measured, graph-free (`cudagraphs=False`), default off (§9d) |
| ✅ | run 2–4 seeds concurrently on one GPU | ×1.70 – ×2.47 aggregate | **ready** — no code change; also buys the replication every result doc says is missing (§9b) |
| ⏸ | `predNet.cuda_graph=True` | 10.19 | **on hold** — not judgeable yet (banner above) |
| ⏸ | `num_envs` 8 → 64 | 4.55 eager / 37.50 graphed | **not recommended eager** — 1.25× and plateaus, at 4× policy dilution (§9c) |

**The single recommendation I'd make today: `compile_cell=layer`**, after a
learning gate. It is a standard PyTorch feature, it captures nothing and records
no memory pool, and it is worth **2.47×** — about 80% of what graphing gives.

**The gap the hold opens is now small.** Graphing is 8.59×; compiling the loop
is 2.47× per run (×2.5 aggregate across seeds) and needs no review of CUDA
memory semantics. That gap is the price of the checkability decision — which is a
defensible price, and is recorded here so it can be revisited with evidence
rather than by argument.

**If graphing is revisited, run this first** (it needs no CUDA knowledge to
read): the July failure is characterised as *crash at ~update 427 with the
model-moving diagnostics on, clean 1200 updates with them skipped*, so run the
config that used to die, past where it used to die, with and without the guard.
Two logs, one binary outcome. Commands in §5 and §10.

---

Every number here is RTX 4060, torch 2.8.0+cu128, measured with the GPU
otherwise idle (`nvidia-smi --query-compute-apps` checked empty before each
block — GPU contention is what invalidated the last session's headline number).

---

## 1. The question

`docs/compaction-2026-08-18-speed.md` left the speed problem framed as *the
rollout is 7.5× slower than a minimal implementation of what it computes*, with
a sober ceiling of ~2.1× on gradient steps/s, and it recorded a judgement about
the existing CUDA-graph flag:

> Note the existing `cuda_graph` flag graphs the **world-model trainStep**, which
> is 12–22% of an update and flat in batch. It is aimed at the wrong term; the
> rollout has no graph.

That judgement is **correct under `predNet.batched_wm=True` and wrong under
`batched_wm=False`**. Which regime "matters" is not uniform: for reaching the
reference loss on a single room the repo's own sweep says serial is required
(§2), while `Configs/run/multienv.yaml` argues deliberately for pooling in the
multi-room design (§7). This document measures both regimes and keeps that
distinction throughout.

## 2. Why the regime decides which term is the bottleneck

`curious_george/rl/update/world_model.py::train_world_model_on_episodes`
(read, confirmed) branches on `batched`:

- **pooled** (`batched_wm=True`) → `train_on_episodes_batched`, **one** pRNN
  gradient step per update, whatever `num_envs` is.
- **serial** (`batched_wm=False`) → one `adapter.train_on_episode` per episode
  segment, i.e. **`num_envs` gradient steps** per update.

`docs/sab_context/open_choices.md` §3a already settled which to want, from a
400-update-per-arm sweep against the finished reference run:

```
arm           s/update  grad steps/s     FPS
B8 pooled        1.016          0.98    2006
B128 pooled      1.578          0.57   18812
B8 serial        2.285          3.58     916
```

> "FPS is actively misleading as an objective for this model; **gradient steps
> per second is the quantity to maximise**." … "Reaching the reference loss needs
> ~80,000 gradient steps: 6.2 h at serial's 3.58/s, versus 39 h at B128 pooled's
> 0.57/s."

So the production question is asked in the **serial** regime. And the flag is
only wired into the serial path — `PRNNAdapter._use_graph_wm` is consulted in
`train_on_episode` and **not** in `train_on_episodes_batched`
(`world_model/adapter.py:617,622`, read, confirmed). Under pooling the graph is
dead code, which is exactly why it measured as "the wrong term".

## 3. Result — four arms, one variable each

`tests/perf/benchmark.py`, 6 updates + 1 warmup, `env=lroom_multi run=multienv`
(`num_envs=8`, `frames=2048`, `seqdur=256`, `hiddensize=500`), arms run strictly
sequentially on an idle GPU **[live]**:

```
arm                             GRAD/s  wm grad  upd/s  s/upd     fps   prnn_loss[-1]
A pooled + eager  (current)      1.186        6  1.186  0.843  2428.8        0.022430
B serial + eager                 3.530       48  0.441  2.266   903.7        0.019002
C serial + GRAPHED              10.187       48  1.273  0.785  2607.9        0.017920
D pooled + graphed (control)     1.137        6  1.137  0.880  2328.3        0.022430
```

- **C/A = 8.59×** on the objective.
- **D is the negative control** and behaves exactly as the code predicts: the
  graph cannot engage on the pooled path, so D matches A (0.96×) and its
  `prnn_loss` is identical to A's to six decimals.
- B reproduces the 2026-08-11 sweep's serial arm (3.530 here vs 3.58 there),
  which is what makes this harness comparable to that one.

The mechanism is a single stage. Per-update stage shares **[live]**:

```
                        B serial + eager        C serial + graphed
update/wm_train          1.583 s   69.9%         0.129 s   16.4%
collect/sr_step          0.204 s    9.0%         0.193 s   24.5%
collect/policy_fwd       0.198 s    8.7%         0.187 s   23.8%
update/policy            0.067 s    3.0%         0.071 s    9.1%
collect/curious_rewards  0.069 s    3.1%         0.068 s    8.7%
collect/env_step         0.062 s    2.7%         0.058 s    7.4%
TOTAL                    2.266 s                 0.785 s
```

**Per world-model gradient step: 0.198 s → 0.0161 s, a 12.3× reduction.** The
0.198 s figure independently matches the committed
`tests/perf/results/trainstep_batch_cuda_4060.json` (0.183 s at batch 1, 0.192 s
at batch 8), which also shows the step is **flat in batch and peaks at 0.05 GiB**
— the signature of a launch-bound workload, not a compute-bound one.

**Under serial training the world-model step is 69.9% of an update, not
12–22%.** The flag was aimed at the right term all along; the earlier
measurement was taken in the regime where that term had already been collapsed
to one step.

## 4. Why this is not a semantics change

`_GraphWMSegmentTrainer` (`world_model/adapter.py`) captures
`pN.predict` + loss + backward + **one** RMSprop step per segment, so the
gradient-step **count and math are identical to the serial eager path** — unlike
`batched_wm`, which replaces B steps with 1 and therefore needs a curve gate.

Gated by `tests/test_cuda_graph_wm.py` **4 passed** **[live]**, including
`test_graphed_segment_bitwise_equals_eager_no_rng` and
`test_device_roundtrip_invalidates_captured_graphs`.

⚠️ **It is not bitwise with noise on, and this run has noise on.**
`predNet.noisestd=0.05` (`Configs/world_model/thrnn5win.yaml`), and dropout
`dropp=0.15` is active in training mode. Both run *inside* the graph on CUDA's
graph-safe RNG, so replays draw fresh but in a different order than eager. The
arms are therefore **distributionally** identical, not bit-identical, and the
claim needs a learning gate rather than an oracle — see §6.

## 5. The safety fix this required, and what it costs

A captured graph writes to the parameter **addresses** it saw at capture time.
The periodic diagnostics call `on_device([predictiveNet, acmodel], "cpu")`,
which replaces `param.data`; the allocation churn inside the spatial eval then
fragments memory so a re-captured graph's private pool can alias a live tensor.
This killed two cluster runs on 2026-07-22 with `obs.direction == 184` (must be
0–3) — a use-after-free, not a crash in the graph code.

The address fingerprint in `_GraphWMSegmentTrainer._fingerprint` (present on this
branch, commit `b7b7375`) covers the stale-pointer path but **not** the aliasing
path. The fix that did was `5753a7e`, which **is not on this branch** — verified
with `git merge-base --is-ancestor 5753a7e HEAD` → false. It targets
`cfg.logging.plot_interval` / `analysis_interval`; the loop has since moved to
step-based cadences (`plot_every_steps` / `analysis_every_steps`), so it was
**ported, not cherry-picked**:

- `training/loop.py:25` `_skip_model_move_diag(cfg)`
- evaluated **once** at `loop.py:182`, before the update loop
- applied at `loop.py:215` (sample-trajectory figure) and `loop.py:241`
  (`run_spatial_analysis`)
- the original's module-global `_warned_skip_diag` is gone: warn-once is now
  structural. That global was also process-global, so a second run in one
  interpreter silently lost the warning — which is why the original test needed
  a `_reset_warn()` helper.
- `tests/test_cuda_graph_diag_guard.py` ported, 2 of the original 3 tests; the
  third asserted the removed global.

**The cost is real and must be stated:** a graphed run logs **no in-run sRSA,
SWdist or sample-trajectory figure**. Every scalar (pRNN loss, curious reward,
policy loss, FPS, behaviour MI) still logs every update, because none of them
move the model. The spatial curves are recovered offline from the archived
checkpoint series with
`uv run python scripts/multienv/checkpoint_curve.py --run <dir> --env lroom_multi --layouts one --spatial`.

## 6. The learning gate

Because §4 is distributional rather than bitwise, speed is not sufficient. Two
40-minute 1-L-room runs, **serial world-model training**, identical seed, the
only difference being `predNet.cuda_graph`:

```
uv run python main_train.py env=lroom_multi run=multienv exp.layouts=one \
    predNet.batched_wm=False exp.seed=2 predNet.cuda_graph={False,True} \
    logging.wandb_log=false logging.analysis_every_steps=0 \
    logging.plot_every_steps=0 logging.archive_every_steps=524288
```

In-run analysis and plotting are disabled **in both arms**: the graph arm skips
them by construction (§5), so leaving them on for the eager arm alone would
charge that arm for work the other never does.

**The arms run CONCURRENTLY, deliberately.** The claim under test is prediction
loss as a function of gradient-step number, which is invariant to how fast either
arm runs, so sharing the GPU cannot bias it. No speed claim is taken from this
gate — those come from §3 on an idle GPU. Running them together halves the
wall-clock and doubles as a concurrency check.

Both arms are serial with the same `num_envs`, so **gradient steps = env
steps / 256 identically in both**, and matched archive steps are matched gradient
steps. If the graph is faithful the curves overlay and only wall-clock differs; a
gap is a numerics bug.

⚠️ **The in-training loss series is NOT recoverable from the checkpoints.**
`prnn.utils.checkpoints.save_pN` persists `NUM_TRAINING_TRIALS` (a count) but not
the `TrainingSaver` loss rows, so with `wandb_log=false` that series dies with
the process. This is a pre-existing gap, named rather than worked around. The
gate therefore uses the better measurement anyway: prediction loss on a FIXED
replayed probe per archived checkpoint, via the repo's own
`scripts/multienv/checkpoint_curve.py --spatial`, which yields loss, per-room
sRSA, pooled sRSA, remapping index and SWdist on one comparable scale for both
arms.

**Reference anchor:** the finished single-room run
`pRNN_curious_26-07-23-10-06-25` —
<https://wandb.ai/blake-richards/curious-george/runs/pRNN_curious_26-07-23-10-06-25>
— is the same shape (`MiniGrid-LRoom-v0`, `num_envs=8`, **`batched_wm: false`**,
`seqdur=256`, `frames=2048`, `hiddensize=500`, read from the job's own stdout on
Mila at `$SCRATCH/pRNN/RL_for_pRNN_10178850/`). ⚠️ It is an **anchor, not an
overlay**: `exp.layouts=one` resolves to `ROOMS_RUN1[0]`
(`envs/layouts.py:473`), whose landmark placement differs from stock
`MiniGrid-LRoom-v0`, so absolute loss need not match. The controlled comparison
is arm C against arm B.

### Gate result — PASS, with an explicitly weak bound

`scripts/multienv/checkpoint_curve.py --env lroom_multi --layouts one --spatial`
on both arms **[live]**, cached in each run's `checkpoint_curve.json`. Both arms
are serial at `num_envs=8`, so gradient steps = env steps / 256 in both and these
ARE matched gradient steps:

```
   env step  grad step |   loss NG   loss G      Δ% |   sRSA NG   sRSA G        Δ |   SWd NG    SWd G
    524,288      2,048 |  0.012874 0.012698    -1.4 |    0.0852   0.0858  +0.0007 |   0.0126   0.0125
  1,048,576      4,096 |  0.011519 0.010737    -6.8 |    0.2950   0.2997  +0.0048 |   0.0107   0.0096
  1,572,864      6,144 |  0.008698 0.009974   +14.7 |    0.3130   0.2668  -0.0462 |   0.0138   0.0180
```

The first two matched points agree closely and the third diverges on all three
metrics. **That third point does not exceed the measurements' own noise**, and
this is checkable rather than asserted:

- **sRSA.** The graph arm's own adjacent-checkpoint swings are the same size as
  the disagreement: `0.3678 → 0.3290` is **−0.0388** and `0.3873 → 0.4266` is
  **+0.0393**, against the matched-point Δ of **−0.0462**. At 2–6k gradient steps
  sRSA is still climbing steeply (0.085 → 0.313 across the three points), which
  is where a fixed-step comparison is least stable.
- **SWdist.** `open_choices.md` §1c measures a **27.5% coefficient of variation**
  with the wake activity held completely fixed, and concludes SWdist "cannot
  detect anything smaller than roughly a 50% change". The Δ here is 0.0138 vs
  0.0180 (+30%), inside that floor.
- **Trajectory.** The graph arm continues cleanly past the eager arm's last
  archive to **loss 0.005038 and sRSA 0.4327 at 20,480 gradient steps** — no
  divergence, no instability, monotone loss.

**Verdict: no evidence of a numerics defect.** Combined with the bitwise test
with RNG off (§4), this is what a faithful-but-differently-seeded implementation
looks like.

⚠️ **The bound is weak and should not be overstated.** Three matched points, one
seed per arm, and only 2k–6k gradient steps against the reference run's 80,000.
This cannot exclude a small effect. Because the graph changes the RNG *order*
(§4), the arms are two draws from the same distribution, so separating a real
effect from seed noise needs **several seeds per arm at matched gradient steps** —
that is the gate to run before this goes to production, and it is cheap now that
concurrency works (§9b).



## 7. What this does NOT license

**It does not say to turn off `batched_wm` for the multi-room runs.** That flag
is a deliberate methodological choice there, not an oversight —
`Configs/run/multienv.yaml` argues it explicitly:

> "With several rooms per rollout a pooled step averages the gradient ACROSS
> rooms instead of visiting them one at a time, which is the point of the
> design."

So there are two separable findings:

| | claim | kind | who decides |
|---|---|---|---|
| 1 | `cuda_graph=True` on the serial path: **12.3× on the world-model gradient step**, identical step count and math | engineering | gated by tests + §6 |
| 2 | serial vs pooled: **2.98×** on gradient steps/s (B vs A), and pooling reaches a higher loss per gradient step | **methodological** | Sabrina / PI |
| 3 | `num_envs` 8 → 64 on the serial+graphed path: **3.45×** more (§9b) | **methodological** — dilutes policy gradient steps per environment step | Sabrina / PI |

Claim 1 is mine to recommend. Claims 2 and 3 change what the experiment
measures, and each needs its own learning gate — 2 on the multi-room design, 3 on
whether the policy still explores the same way when it updates 8× less often per
environment step. Do not adopt 3 on the strength of the speed number alone; that
is the precise shape of the mistake §4 of the previous handoff retracted.

## 8. Where the time goes next

After graphing, the rollout is **64.4%** of an update and the world model
**16.4%**. `collect/sr_step` (24.5%) and `collect/policy_fwd` (23.8%) are the
two largest stages, and both are 256 sequential launches of tiny tensors.

Profiler on one rollout, production config **[live]**
(`tests/perf/profile_api.py`):

```
cudaLaunchKernel        24660  =  96.33/step   207.2 ms cpu
cudaMemcpyAsync          2840  =  11.09/step    35.0 ms cpu
cudaStreamSynchronize     269  =   1.05/step     1.8 ms cpu
self CPU 884.3 ms   self CUDA 91.3 ms   ratio 9.7x
aten+api invocations 156,951 = 613/step
```

### Graphing the rollout: what it would take, and why it is not done here

Value if the rollout went to zero: the update is 0.785 s of which 0.506 s is
rollout, so at most **~2.8×** more. Worth having, not worth a second corruption
incident, so this records the design rather than shipping it.

The device path already satisfies most of what capture requires — this is
**confirmed by reading `collect_rollout`**, not assumed:

- **static shapes** — every buffer (`actions`, `values`, `SRs`, `policy_probs`,
  `device_images`, …) is preallocated before the loop
- **no host-dependent control flow** — episode cuts fire on `t % prnn_seqdur`,
  a Python integer known ahead of time, never on a tensor value.
  `DeviceTableShellPool.step_device` is pure indexing and explicitly documents
  that it takes no D2H and no sync
- **no data-dependent host syncs** — measured zero (§8)
- **graph-safe RNG** — the world-model graph already relies on this, and its
  test asserts replays draw fresh

Two things block a naive capture, and both have known shapes:

1. **`t` is baked into a graph.** The body writes `SRs[t]`, `actions[t]`, … so a
   one-step capture would replay into the same slot forever. Either capture a
   whole `prnn_seqdur` block (256 steps ≈ 24.6k recorded launches — large but
   legal), or keep the write index in a **device tensor** incremented in-graph
   and use `index_copy_`.
2. **`masks[t].fill_(float(state.mask_b[0]))`** reads a host scalar. It is
   constant except at segment boundaries, so a capture of the non-boundary step
   is uniform and the boundary step stays eager.

⚠️ Before attempting it, note the §5 history: the danger was never the graph
mechanics, it was a captured graph's private memory pool aliasing live tensors
after allocation churn. A rollout graph would hold a much larger pool, so that
risk goes **up**, not down.

### The `cudaStreamSynchronize` open thread is closed: it is a symptom

The handoff flagged ~1.1 syncs/step as possibly explaining the CPU/GPU ratio.
Reproduced exactly (269 = 1.05/step) — **and it costs 1.8 ms out of 884 ms of
CPU, 0.2%.** They return immediately because the GPU is already drained.
Independently, `torch.cuda.set_sync_debug_mode("warn")` over a whole
`collect_experiences` on the production path reports **zero** data-dependent
host syncs (`tests/perf/find_sync.py`). There is nothing to fix here.

The cost is `cudaLaunchKernel` and Python dispatch. The precedent for removing
both is now in-tree and validated: the same graph capture that took the world
model 12.3×. `collect/sr_step` is the contained next target —
`BatchedSRTracker.step_synchronized` runs one fixed-shape cell call per step,
and the device path's control flow is data-independent (episode cuts are
synchronized and known from the Python timestep, `collector.py`), which is the
property capture requires.

## 9. Is the graph a band-aid for an underlying defect?

Asked directly, and worth answering with evidence because the repo's standing
verdict is that **the GPU is useless at this model shape** — which, if the whole
story, would make graphing a way of hiding a wrong hardware choice.

**The verdict is real.** Committed results, no new measurement:

```
trainStep, thRNN_5win, h=500, seqdur=256, batch 1
  CUDA eager    182.6 ms   tests/perf/results/trainstep_batch_cuda_4060.json
  CPU  eager    122.4 ms   tests/perf/results/trainstep_batch_cpu.json
  CUDA graphed   16.1 ms   measured here (§3)

end-to-end, num_envs=1, 3 updates
  baseline_cpu.json    412.9 fps
  baseline_cuda.json   217.7 fps
```

CPU beats eager CUDA on both. But graphed CUDA beats **CPU** by 7.6× — it
overtakes the good option, not merely the bad one. So the graph is not
compensating for the hardware choice.

**And there IS a real underlying defect, which the graph does not fix.**
`prnn/utils/thetaRNN.py::thetaRNNLayer.forward` runs per timestep:

- `inputs[i][0,:].permute(1,0)` and `internals[i][0,:].permute(1,0)` — slices
  re-permuted every step, hoistable to one permute over the whole sequence
- `out.unsqueeze(1)` then `out.permute(...)` — shape shuffled back
- `out = [out]` … `torch.cat(out, 0)` — **a one-element concatenation**, because
  `rnn.theta == 0` for `thRNN_5win` (confirmed at runtime: `MaskedRNN` does not
  forward its `k=5` mask width to `pRNN.__init__`, whose own `k` defaults to 0,
  and `defaultTheta=k`). So the inner theta loop never runs. `torch.cat([x], 0)`
  does **not** return `x` — verified `data_ptr()` differs — so it allocates and
  copies on all 256 timesteps, plus its backward.

⚠️ **I estimated from reading that this bookkeeping was ~30% of the loop's ops.
Measured, it is 1%. The estimate was wrong and should not be repeated.**
`torch.profiler` over `thetaRNNLayer.forward` + backward, L=256, theta=0, h=500,
on CPU so the counts are device-independent **[live]**:

```
total aten invocations 41,221 = 161.0 per timestep

  as_strided     20.02/step     t              8.99/step
  resolve_conj   10.96/step     transpose      8.99/step
  empty_strided  10.00/step     fill_          7.01/step
  copy_          10.00/step     sum/div/mul    6.00/step each
  to             10.00/step     select         5.00/step
  _to_copy       10.00/step     cat+unsqueeze  2.00/step   <- the 1%
```

**161 aten ops per timestep** is the real number, and the cat/permute story is
noise inside it. Two genuine leads replace it:

1. **~40 conversion/copy ops per timestep** (`to`, `_to_copy`, `copy_`,
   `empty_strided`, 10 each). ⚠️ Origin **not traced** — this is a lead, not a
   diagnosis, and it must be tracked down before anyone "fixes" it.
2. **The weights are transposed every timestep.**
   `RNNCell.update_preactivation` (`prnn/utils/thetaRNN.py:202-203`) does
   `torch.mm(input, self.weight_ih.t())` and `torch.mm(hx, self.weight_hh.t())`
   inside the 256-step loop, rather than `F.linear` or a hoisted transpose —
   consistent with `t` and `transpose` at ~9/step each.

⚠️ Measured on CPU with autograd included; the GPU profile may differ.
Fixing these is device-independent and would help CPU, eager CUDA and the graph
alike.

**It is still not a substitute for the graph**, for a structural reason.
`h_t` depends on `h_{t-1}`, so the 256 steps cannot be merged; with kernel
launch latency of ~5–10 µs and hundreds of sequential launches, there is a floor
that op-fusion cannot clear. CUDA graphs are the only mechanism that removes
per-launch latency from a sequential chain. The value of the op fix is **not yet estimated** — the first
estimate was refuted above, and a second guess would be worth no more than the
first. It needs measuring after the conversion ops are traced. The two are
complementary regardless: the graph does not block the fix.

⚠️ The fix lives in **prnn**, a pinned external dependency
(`LevensteinLab/pRNN`, branch `sdu/rl-integration`) whose remote requires a PR.
Not actionable inside this repo.

**The honest objection to graphing is cost, not concealment.** It has a
documented silent-corruption history (§5), and the validated mitigation works by
removing in-run sRSA/SWdist and the trajectory figure. That loss of visibility
is the real price and is the thing to weigh.

## 9b. Saturating the GPU — how much headroom, and what can actually use it

The goal "saturate the GPU, or a slice of it" is well founded. Measured live on
the real workload with `nvidia-smi -l 2` during a training run **[live]**:

```
SM utilisation   median 12 %   max 17 %
memory-bw util   median  0 %
memory used      614 MiB of 8188   (7.5 %)
```

### How much headroom there is: ~128x, free

`tests/perf/sweep_trainstep_batch.py --device cuda --batches 1,8,32,64,128,256,512,1024`
**[live]**, reproduce with the command above:

```
 batch      ms   steps/s   traj/s   peak GiB
     1   194.8      5.13      5.1      0.026
     8   198.7      5.03     40.3      0.050
    32   199.9      5.00    160.1      0.140
   128   199.3      5.02    642.3      0.497    <- 128x the work for +2.3% time
   256   207.1      4.83   1236.0      0.977
   512   230.3      4.34   2223.5      1.929
  1024   280.3      3.57   3653.4      3.844    <- 716x throughput for 1.44x time
```

**Flat to batch 128; the knee is ~256.** The GPU absorbs two orders of magnitude
more work for free. That is the headroom, quantified.

### What can NOT use it: a single run's world model

With serial world-model training each `trainStep` is **batch 1**, and the B steps
are sequential updates to the *same* parameters — step *i+1* consumes the weights
step *i* produced. They cannot be merged without changing the algorithm, which is
exactly what `batched_wm` does (B steps → 1 pooled step), and what the 2026-08-11
sweep measured as a net loss.

So a single run is structurally incapable of filling the batch dimension. The
free capacity is reachable only by work that needs **separate optimizer steps**.

### What CAN use it, for a single run: amortise the rollout over more steps

`num_envs` raises the number of segments per rollout, hence serial gradient steps
per update, while the rollout itself is nearly free to widen. Serial WM +
`cuda_graph`, `exp.layouts=one`, `frames = num_envs x 256`,
`ppo_batch_size = frames/4`, 4 updates + 1 warmup (3 + 1 at 128/256) **[live]**:

```
 num_envs   GRAD/s   s/upd  ms/grad  wm_s/upd  roll_s/upd
        8    10.88   0.735     91.9     0.128       0.496
       16    18.35   0.872     54.5     0.255       0.494
       32    27.21   1.176     36.8     0.497       0.528
       64    37.50   1.706     26.7     0.991       0.527
      128    45.02   2.843     22.2     1.988       0.575
      256    52.59   4.868     19.0     3.949       0.528
```

**`roll_s/upd` is flat across a 32x range** (0.496 → 0.528) while gradient steps
per update rise 32x. That is the entire mechanism: one rollout feeds many more
optimizer steps. `wm_s/upd` scales linearly, as sequential steps must.

The asymptote is the **WM-bound ceiling, `1 / 16.1 ms = 62.1 grad/s`**; B=256
reaches 85% of it. Returns diminish hard: 8→64 is 3.45x, 64→256 only 1.40x.

⚠️ **This is NOT semantics-free, unlike the graph.** Holding `ppo_epochs` fixed
and scaling `ppo_batch_size` with `frames` keeps policy gradient steps *per
update* constant (16), but each update now spans `num_envs/8` times more
environment steps — so **the policy takes proportionally fewer gradient steps per
environment step**, and the policy generates the behaviour the world model learns
from. `s/update` also grows to 4.9 s at B=256, which is a long time between
logging events. Both argue for **32–64, not 256**, and for a learning gate on
pRNN loss *and* sRSA before adopting it. ⚠️ Also short runs (3–4 updates); the
trend is monotone and matches the predicted mechanism, but the absolute values
deserve a longer measurement.

### What CAN use it, across runs: concurrent processes — but the CPU caps it

Independent runs need separate optimizer steps by definition, so they are the
natural consumer of the idle capacity. N copies of the same benchmark
(`num_envs=8`, serial WM + graph, different seeds), run simultaneously **[live]**:

```
config                    per-proc   aggregate   vs solo   per-proc retained
1 process (solo)             10.88       10.88     x1.00                100%
2 concurrent processes        9.22       18.44     x1.70                 85%
4 concurrent processes        6.71       26.88     x2.47                 62%
```

**2.47x aggregate throughput for zero code change** — but sub-linear, and the
reason is not the GPU. This box has **8 logical CPUs and torch is using 4
threads**, so four processes oversubscribe it; and the workload is CPU-bound to
begin with (884 ms self-CPU against 91 ms self-CUDA, §8). **The concurrency
ceiling here is the host, not the accelerator.** A Mila node with more cores per
GPU should scale further, which is a thing to measure there rather than assume.

Practical consequence: the project needs replication — every result document
says "n = 1 seed per arm. No replication." Packing 2-4 runs per GPU buys that
replication at 1.7-2.5x, instead of 4 GPUs.

### 9c. The same sweep WITHOUT the graph — width alone is not a lever

§9b was measured with `predNet.cuda_graph=True`. Since the graph is not adopted,
here is the identical sweep with it **off**, which is what is actionable today
**[live]**:

```
SERIAL world model, NO CUDA GRAPH   (eager trainStep ceiling = 1/0.198 = 5.05 grad/s)
 num_envs   GRAD/s   s/upd  ms/grad  wm_s/upd  roll_s/upd  vs B=8
        8     3.68   2.175    271.8     1.529       0.526   x1.00
       16     4.14   3.860    241.3     3.196       0.528   x1.13
       32     4.60   6.961    217.5     6.274       0.540   x1.25
       64     4.55  14.062    219.7    13.178       0.680   x1.24
```

**Width buys 1.25x and then plateaus**, because it runs straight into the eager
trainStep ceiling — `ms/grad` falls only 272 → 218 and stops. Meanwhile
`s/update` grows 2.2 s → 7.0 s and the policy is diluted 4x per environment step
(§9b). **That is a bad trade and is not recommended.**

Set against §9b's graphed sweep, the conclusion is sharp and worth stating
plainly:

```
                        num_envs=8   num_envs=32   num_envs=64
  eager  (adopted)            3.68          4.60          4.55     ceiling 5.05
  graphed (NOT adopted)      10.88         27.21         37.50     ceiling 62.1
```

**The width lever is almost entirely conditional on the graph.** Widening only
pays once a gradient step is cheap; while it costs 0.198 s, more of them is just
more time. With graphing off there is **no large speed lever left** in this
design — the remaining routes are the hot-loop op reduction (§9, device-
independent, needs an upstream prnn PR) and running several seeds concurrently
(§9b, 2.47x aggregate, CPU-capped).

### Where the stack now stands, all on gradient steps per second

```
pooled + eager,  num_envs=8   (production default today)    1.19
serial + eager,  num_envs=8   (the reference run's regime)   3.53
serial + graphed, num_envs=8                                10.88   3.1x
serial + graphed, num_envs=64                               37.50  10.6x
serial + graphed, num_envs=256                              52.59  14.9x
WM-bound ceiling (graphed trainStep, batch 1)               62.1
```

Beyond 62 grad/s requires a **faster individual step**, not more parallelism —
i.e. the prnn hot-loop op reduction in §9, or a genuinely fused RNN cell. The
graphed step is 16.1 ms for 256 sequential timesteps = 63 us/timestep, which at
~16 ops/timestep is already near replay cost.

## 9d. `torch.compile` — the graph-free lever, and where the op census actually led

The §9 hot-loop census pointed at ops. **Three hypotheses drawn from it were
each refuted by measurement**, recorded here so nobody re-runs the dead ends:

| hypothesis | verdict |
|---|---|
| cat-of-one-element + permutes are ~30% of the loop's ops | **wrong** — 2.0 of 161 ops/step = **1%** |
| a numpy `float64` `mu` leaks into the LayerNorm parameter and forces conversions | **wrong** — every parameter is `float32`; `torch.zeros(n) + numpy.float64` stays `float32` |
| the per-timestep `.t()` makes `torch.mm` materialise a contiguous copy | **wrong** — `mm(x, W.t())` emits **zero** `_to_copy`/`copy_`/`empty_strided`, and `F.linear` is *slower* (60.8 vs 43.2 us/call, 17 vs 13 ops) |

The `aten::to`/`_to_copy` could not be attributed at all: a Python-level
`Tensor.to` patch sees none and `TorchDispatchMode` sees none, so they originate
below both.

**The deciding measurement was ranking by TIME rather than COUNT** — which is
what the op census should have done from the start (fwd+bwd, L=256, CPU,
123.4 ms total aten self-CPU) **[live]**:

```
  mm            5.0/step   32.89 ms   26.6%      <- irreducible arithmetic
  add_          6.0/step   15.02 ms   12.2%
  mul/add/sum/div/div_/std/mean  (the LayerNorm chain)   ~28%
  the whole conversion/stride family, 71 calls/step      12.9%
```

There is **no single hot spot**. It is dispatch-bound across many tiny ops,
which is precisely why op-shuffling cannot pay and why the addressable move is
to **fuse the cell** or remove launch overhead outright.

### The measurement that follows from that

`torch.compile` in **default** mode (fusion; NOT `reduce-overhead`, which is
CUDA graphs under another name) applied to `net.rnn.cell.forward` **[live]**:

```
thetaRNNLayer fwd+bwd, L=256, h=500
device     eager ms  compiled ms  speedup
cpu           121.8        120.7     1.01x
cuda          198.1        142.7     1.39x
```

**1.39x on the world-model step on GPU, graph-free.** This also closes an item
in the 2026-07 perf memory, which recorded a `reduce-overhead` probe that
*hung* and suggested "try default mode": default mode works.

### Compiling the LOOP, not the cell: 2.47x end-to-end

The cell is the wrong unit. Compiling the **whole 256-step loop** lets dynamo
unroll it into one graph and Inductor fuse across timesteps — the direct torch
analogue of `jax.lax.scan` + XLA. Isolated layer, fwd+bwd, L=256, CUDA **[live]**:

```
mode             ms/call   speedup   first call
eager              189.4     1.00x       0.4 s
cell               166.5     1.14x       1.3 s
whole-scan          48.7     3.89x      88.6 s
```

End-to-end via `predNet.compile_cell=layer`, 8 updates + 3 warmup **[live]**:

```
mode      GRAD/s   s/upd  wm_s/upd  prnn_loss[-1]
eager       3.33   2.402     1.715       0.017239
cell        4.72   1.694     1.087       0.016526
layer       8.22   0.974     0.421       0.018007      2.47x
```

**2.47x on gradient steps per second, graph-free**, world-model stage 4.07x.
That is ~80% of what `cuda_graph` delivers (10.19 grad/s) with none of its
mechanism — confirmed: `torch._inductor.config.triton.cudagraphs` is **False**
in default mode, so nothing is captured and no memory pool is recorded.

**Compile cost is one-time and bounded.** 7 recompilations fired, each ~89 s
(~10 min total, 0.3% of a 58 h run), and they all completed during warmup:
steady-state per-update times are flat at
`[0.941, 0.934, 0.959, 1.046, 1.064, 0.962, 0.935, 0.945]` s. The triggers are a
finite set — sequence-length mismatches (`pN.predict` is called from several
sites) and `grad_mode` changes (the eval path runs under `no_grad`). ⚠️ Watch
this on any config that introduces new sequence lengths; unbounded recompilation
would be fatal, and `TORCH_LOGS=recompiles` is how to check.

⚠️ `prnn_loss` at 6–8 updates differs across modes (0.017239 / 0.016526 /
0.018007). That is noise at this horizon, but it does confirm the numerics are
**not** identical — fusion reorders floating-point operations — so this needs a
learning gate exactly like anything else.

### Measured end-to-end: 1.42x for cell mode, and it is now a flag

`predNet.compile_cell` (default `False`) wires this into `PRNNAdapter`.
`tests/perf/benchmark.py`, 6 updates + 2 warmup (the extra warmup absorbs
compile time), serial WM, no graph, `num_envs=8`, idle GPU **[live]**:

```
compile_cell   GRAD/s   s/upd  wm_s/upd  prnn_loss[-1]  entropy[-1]
False            3.33   2.402     1.715       0.017239       1.9638   x1.00
True             4.72   1.694     1.087       0.016526       1.9219   x1.42
```

**1.42x on gradient steps per second**, ahead of the ~1.3x projected from the
isolated layer; the world-model stage alone goes 1.715 → 1.087 s/update (1.58x).
`prnn_loss` and entropy are close at 6 updates, which is far too few to gate on —
it is reported to show nothing exploded, not as evidence of equivalence.

⚠️ It is **not free of semantics risk**: fusion can reorder floating-point
operations, so it needs the same learning gate as anything else, and it adds
compile time at startup. But it involves no recorded memory pool and no captured
parameter addresses, so it does not carry the failure mode of §5.

**Standing on the graph-free path:** `compile_cell=layer` **2.47x (measured)**
x concurrent seeds 2.47x aggregate (§9b) — against graphing's 8.59x on a single
run. Compiling the loop closes most of the gap the checkability decision opened. The gap is the cost of the checkability decision, stated so it can be
revisited with evidence rather than by argument.

## 10. Reproducing

All of these need the GPU otherwise idle; check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` first.

```bash
# the four world-model regimes (section 3)
bash tests/perf/wm_regime_arms.sh <outdir>

# num_envs sweep on the serial+graphed path (section 9b)
for B in 8 16 32 64 128 256; do F=$((B*256)); \
  uv run python tests/perf/benchmark.py --updates 4 --warmup-updates 1 --out <outdir>/envs_$B.json \
    --override env=lroom_multi --override run=multienv --override exp.layouts=one \
    --override predNet.batched_wm=False --override predNet.cuda_graph=True \
    --override exp.num_envs=$B --override rl.frames=$F --override rl.ppo_batch_size=$((F/4)); done

# trainStep batch knee - where this GPU saturates (section 9b)
uv run python tests/perf/sweep_trainstep_batch.py --device cuda \
    --batches 1,8,32,64,128,256,512,1024 --reps 5 --out <outdir>/trainstep_sweep.json

# the learning gate (section 6); arg 1 = seconds per arm, arg 2 = log dir
bash tests/perf/cuda_graph_gate_runs.sh 2400 <logdir>

# then score BOTH arms the same way - flags matter, the run dir stores no config
uv run python scripts/multienv/checkpoint_curve.py --run <run> \
    --env lroom_multi --layouts one --spatial

# on Mila (benchmark job, not a training run)
sbatch slurm/graph_bench.sh

# op / API census on one rollout
uv run python tests/perf/profile_api.py env=lroom_multi run=multienv

# data-dependent host syncs, with Python attribution
uv run python tests/perf/find_sync.py env=lroom_multi run=multienv

# gates
uv run python -m pytest -q                    # 141 passed, 18 skipped, 7 deselected
uv run python -m pytest tests/golden_omt -q   # 5 passed
uv run python -m pytest tests/test_cuda_graph_wm.py tests/test_cuda_graph_diag_guard.py -q
```

Baseline before this work was **139 passed, 18 skipped, 7 deselected**; the +2
are `tests/test_cuda_graph_diag_guard.py`.

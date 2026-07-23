# Throughput investigation (2026-07-23)

## Purpose

Determine whether MiniGrid is the training bottleneck, explain why CPU can
beat an H100, estimate the useful H100 throughput range from the actual
operation mix, and provide a correctness-gated fast-environment toggle.

Claims below are labelled:

- **Confirmed**: observed in current code, exhaustive tests, or a controlled
  local measurement.
- **Inferred**: derived from operation counts/hardware specifications but not
  measured on an H100 in this investigation.

No H100 was reachable from the current machine (`crockett` did not resolve),
so H100 FPS ranges are models, not benchmark results.

## Environment and call chain

Default configuration:

- 8 independent `MiniGrid-LRoom-v0` environments.
- 256 steps/environment = 2,048 frames/update.
- pRNN hidden size 500, observation width 147, encoded action width 8.
- 4 PPO epochs, minibatch size 256 = 32 PPO optimizer steps/update.
- 8 sequential world-model optimizer steps/update unless
  `predNet.batched_wm=true`.

Rollout hot path, repeated 256 times:

1. Format a Python list of observation dictionaries.
2. Run policy/value for batch 8.
3. Copy sampled actions and full policy probabilities to CPU.
4. Step 8 MiniGrid wrappers in a Python loop.
5. Format the observations/actions again and run one pRNN tracker step.

After rollout:

1. Curiosity calls full-sequence pRNN prediction once per environment segment.
2. PPO runs 32 forward/backward/optimizer steps.
3. World-model training calls `trainStep` once per environment segment.

**Confirmed:** a one-update CPU `torch.profiler` trace counted 617,843 ATen
operator calls, including 15,080 `aten::mm`, 16,065 `aten::mul`, 15,693
`aten::div`, 10,680 `aten::mean`, and 4,360 `aten::std` calls. The recurrent
sequence is a Python loop, so these are thousands of tiny, causally ordered
operations rather than a few large H100-sized kernels.

## Test baseline before changes

Command:

```bash
PYTHONWARNINGS=ignore UV_CACHE_DIR=/private/tmp/uv-cache CG_DEVICE=cpu \
  uv run pytest
```

Result: 79 passed, 8 failed, 5 errors, 22 skipped, 7 deselected in 11.54 s.
All failures/errors referenced absent ignored checkpoints under `outputs/ckpts`;
CUDA tests skipped. This is the pre-change baseline, not a green suite.

## Fast table environment

Implementation: `exp.table_env=true`.

The reference banked wrapper already returned a pre-rendered observation, but
its inherited `step()` still made MiniGrid generate a partial observation
(grid slice, rotations, visibility/encoding) which the wrapper discarded.
The new table wrapper replaces both the transition and observation hot path:

- state: `(x, y, direction)` (1,024 total table rows for a 16x16 grid);
- action: left, right, forward, pickup;
- transition/reward/termination table;
- existing byte-exact `(x, y, direction) -> 7x7x3 uint8` observation bank.

It rejects mutable layouts containing pickable objects instead of silently
using invalid dynamics.

### Correctness

**Confirmed:**

- Exhaustive differential comparison against MiniGrid over every occupiable
  state, all 4 directions, all 4 actions, and ordinary/max-step truncation
  counts.
- Tested for plain L-room, green-line OMT, and goal L-room variants.
- Seeded reset chains and eight 64-step random rollouts match.
- Reference and table 10-update training metrics are exactly equal for every
  recorded metric and every pRNN loss value.
- Targeted suite: 18 passed.

### Integrated performance (Apple M4 Pro, CPU)

Ten updates, same seed/config:

| Path | FPS | env time/update | total time/update |
|---|---:|---:|---:|
| reference banked MiniGrid | 1,985.8 | 168.4 ms | 1,031 ms |
| `exp.table_env=true` | 2,389.0 | 13.1 ms | 857 ms |

**Confirmed:** 1.20x end-to-end. Environment time fell 12.8x. The table is
much faster in isolation, but Amdahl's law limits the complete training gain
because world-model training remains about 574 ms/update.

Process-async table stepping measured 2,223.5 FPS for a short two-update smoke
run, slower than the in-process table because IPC now costs more than stepping.

## Is the environment the bottleneck?

**Confirmed:** on current CPU defaults it is a meaningful secondary
bottleneck, not the primary one:

- reference run: environment about 16% of the controlled 10-update wall time;
- table run: environment about 1.5%;
- world-model training: about 56-67%, depending on the environment path.

Earlier five-update measurements varied with machine state (about 1,840-1,986
FPS), so the paired 10-update comparison is used for the toggle claim.

On CUDA, "environment cost" also includes the boundary around it: sampled
action and diagnostic probability device-to-host copies, a host loop, and
observation/action host-to-device formatting every recurrent step. A CPU table
removes MiniGrid compute but not this boundary.

## First-principles H100 arithmetic model

Approximate multiply-add count for the current 500-unit model:

- recurrent cell/frame: `2 * 500 * (147 + 8 + 500)` = 0.655 MFLOP;
- output projection/frame: `2 * 500 * 147` = 0.147 MFLOP;
- actor+critic forward/frame: about 0.130 MFLOP.

Using forward+backward ~= 3x forward:

| Stage | Approx GFLOP/update |
|---|---:|
| rollout pRNN tracker | 1.341 |
| curiosity prediction | 1.642 |
| world model forward+backward | 4.927 |
| rollout actor+critic | 0.266 |
| PPO forward+backward | 3.187 |
| **total** | **11.364** |

This omits layer norm, activations, noise, optimizer elementwise work, and
framework overhead; use roughly 10-15 GFLOP/update.

At 60 FP32 TFLOP/s, the arithmetic-only roof is about 10.8 million FPS.
That number is deliberately unrealistic: it proves arithmetic is not the
current limit. Effective FP32 utilization maps to:

| Effective peak | Arithmetic FPS |
|---:|---:|
| 0.01% | 1,081 |
| 0.1% | 10,813 |
| 1% | 108,135 |
| 10% | 1,081,348 |

Most current world-model and curiosity GEMMs have `M=1`; tracker GEMMs have
`M=8`. A recurrent `[B,500] @ [500,500]` has about `B/2` FLOP/byte if weights
come from HBM: roughly 0.5 FLOP/byte at B=1 and 4 at B=8. The H100 FP32
compute/bandwidth ratio is about 20 FLOP/byte (60 TFLOP/s / 3 TB/s), so batch
around 40 is needed even to cross the simple FP32 roofline, before accounting
for the tiny M dimension and launch overhead. Tensor-core TF32 needs batches
in the hundreds to approach its much higher compute/bandwidth ratio.

Weights fit in cache, which helps bandwidth, but it does not create enough
parallel rows or remove 256 causal recurrent steps.

The table environment itself moves roughly 150-200 bytes/frame. A 3 TB/s
bandwidth-only roof is more than 10 billion frames/s. **Inferred:** a fused,
device-resident table environment is effectively free relative to the pRNN;
launch latency at batch 8, not bandwidth, is its only GPU concern.

### Defensible H100 ranges (not measured)

| Implementation | Expected order of magnitude | Limiter |
|---|---:|---|
| current eager B=8 | 500-2,000 FPS | tiny kernels, 256 host round trips, serial B=1 WM/curiosity |
| CPU table toggle only | about 1.2-1.4x current | host/device boundary remains |
| table + batched formatting/curiosity/WM + partial graphs | 2,000-6,000 FPS | still small batches and sequential recurrence |
| fully device-resident, fused/graphed B=8 | 20,000-100,000 FPS | causal sequence and low M |
| fused FP32 with batch in hundreds | 0.5-2 million FPS | model arithmetic/causal recurrence |
| fused TF32/BF16 with batch in hundreds | 1-5 million FPS | numerical gate and model arithmetic |

These are engineering ranges derived from 5.55 MFLOP/frame and plausible
effective peak fractions, not promises. Batch-in-hundreds changes frames per
update/update cadence and therefore needs a learning-curve gate.

## Can the current code saturate an H100?

**Inferred with high confidence: no.** Current B=1/B=8 GEMMs cannot occupy
enough SMs, and the 256-step recurrent dependency prevents time-parallel
execution. CUDA graphs reduce CPU submission overhead but do not increase the
amount of parallel work in each node.

The H100 can be made useful by:

1. keeping environment state, observations, actions, tracker state, and
   diagnostics on device for the whole rollout;
2. fusing the recurrent cell (including layer norm/activation/noise);
3. graphing the fixed 256-step rollout/update without moving captured
   parameters CPU<->GPU;
4. batching curiosity and, if science permits, world-model optimization;
5. increasing the number of independent environments into the hundreds.

## Unavoidable versus fixable

Unavoidable without changing the algorithm:

- 256 causal policy/environment/pRNN steps;
- causal pRNN BPTT;
- eight sequential world-model optimizer steps if exact current optimizer
  semantics are required;
- sampling/reset RNG (but it can remain on device).

Easy/high-confidence:

- table-driven environment (implemented);
- remove the unused per-step mission tokenization;
- stop copying full policy probabilities to CPU each step; reduce diagnostics
  once per update;
- batch/retain tensors for observation and action formatting;
- synchronize graph losses once/update rather than `.item()` per segment;
- keep training parameters permanently on GPU and analyze a CPU snapshot.

Moderate:

- batch curiosity segments;
- fuse the two recurrent GEMMs and use fused layer norm;
- compile/graph the complete fixed-shape update;
- device-resident batched table state.

Algorithm/numerics-changing:

- pooled world-model optimizer step (`predNet.batched_wm=true`);
- hundreds of environments/larger frames per update;
- TF32/BF16;
- custom persistent CUDA/Triton recurrent forward/backward.

## CPU regression: `971ff0b` versus HEAD

Controlled local runs with the same M4 Pro, locked dependencies per commit,
`CG_DEVICE=cpu`, `OMP_NUM_THREADS=16`, and `MKL_NUM_THREADS=16`:

| Revision | 20-update FPS | WM time/update |
|---|---:|---:|
| `971ff0b` | 1,840.6 | 650.1 ms |
| HEAD before this change (`e6c6603`) | 1,850.0 | 643.8 ms |

**Confirmed:** the reported CPU regression does not reproduce locally; the
direction is slightly reversed and the difference is within run variance.

The top-level diff has no material enabled CPU-hot-path change: CUDA graph
logic is flag-gated and false by default. The only plausible compute-path
change is the pinned pRNN revision `76ad773` -> `7f50a0f`, which replaced
NumPy boolean scatter masks with capture-safe torch float-mask
multiplication. A representative local mask microbenchmark measured the new
operation faster (111 us versus 367 us), and the full stage timings likewise
showed no regression.

Therefore the exact Sabrina result cannot honestly be attributed from the
repository diff alone. The leading remaining explanations are a CPU/node or
thread/cgroup difference (the repository separately records Sapphire at
about 1,270 FPS versus Milan at about 656 FPS), a config difference, periodic
analysis duty cycle, or a CUDA-available CPU run measured with an older timer
that synchronized CUDA at every stage boundary. Re-run both commits inside
one allocation and record CPU model, affinity, torch thread counts, config,
pRNN revision, periodic events, and stage timings before assigning causality.

## Commands and artifacts

```bash
# Toggle benchmark
PYTHONWARNINGS=ignore UV_CACHE_DIR=/private/tmp/uv-cache CG_DEVICE=cpu \
  uv run python tests/perf/benchmark.py --updates 10 \
  --out /tmp/rl_prnn_table_cpu.json \
  --override exp.table_env=true

# Differential metric gate
UV_CACHE_DIR=/private/tmp/uv-cache uv run python \
  tests/perf/compare_metrics.py \
  /tmp/rl_prnn_reference_cpu_10.json /tmp/rl_prnn_table_cpu.json

# Targeted tests
PYTHONWARNINGS=ignore UV_CACHE_DIR=/private/tmp/uv-cache CG_DEVICE=cpu \
  uv run pytest -q tests/test_table_env.py tests/test_obs_bank.py \
  tests/test_env.py tests/test_batched_collector.py tests/test_async_envs.py
```

Local JSON artifacts:

- `/tmp/rl_prnn_reference_cpu_10.json`
- `/tmp/rl_prnn_table_cpu.json`
- `/tmp/rl_prnn_971_cpu_20t16.json`
- `/tmp/rl_prnn_head_cpu_20t16.json`

## Device-resident rewrite and measured result

Implementation toggles:

- `exp.device_env=true`: exact device-resident state/observation table. This
  implies the table wrapper during construction, requires `num_envs>1`, and
  currently accepts only the non-rewarding/non-terminating static L-room used
  for training. Episode cuts remain the known synchronized `predNet.seqdur`.
- `performance=ultra`: measured preset (`B=512`, 131,072 frames/update,
  minibatch 65,536, batched curiosity and pooled world-model update).

The device pool keeps position, direction, transition table, and observation
bank on MPS/CUDA. The sampled action tensor is indexed directly into the
transition table. Rollout images/directions/locations remain device tensors;
science-facing Python/NumPy views are materialized in bulk only after the
256-step rollout. Exact seeded reset rows for the finite rollout are generated
and uploaded before its first transition; episode boundaries select a
prepared device row.

Additional fixes made along the call chain:

- policy preprocessing omits unused RGB/mission fields when `with_CV=false`;
- policy probabilities remain on-device and transfer once per rollout;
- GAE transfers `(T,B)` arrays in bulk and vectorizes the recurrence over B;
- curiosity formats/predicts every equal segment in one device-native call;
- world-model training can pool equal segments behind `batched_wm`;
- synchronized device environments use one scalar 5-window phase mask rather
  than uploading a length-B NumPy mask 256 times;
- PPO does not index RGB fields that the actor never reads;
- simultaneous reset clones the recurrent state once rather than B times;
- an unnecessary device-scalar Python branch in WM formatting was removed.

### Device correctness

**Confirmed:**

- `tests/test_table_env.py` exhaustively checks CPU table dynamics against
  MiniGrid and checks the batched device table against independently stepped
  shells over reset chains and random actions.
- `tests/test_device_collector.py` compares complete CPU-table and
  device-table rollouts under matched weights/RNG. It requires exact equality
  for action, SR, value, reward, advantage, return, log-probability, RGB,
  direction, boundaries, locations, curiosity, and joint statistics.
- Targeted gate after the final reset/device changes: 27 passed.
- Full suite after changes: 91 passed, 8 failed, 5 errors, 22 skipped,
  7 deselected. The 8/5 are the identical pre-existing missing-checkpoint
  failures under ignored `outputs/ckpts`; there is no new failure class.
- Stored B=1 golden differs at roughly 1e-6 because it was captured with
  torch `2.8.0+cu128` while this machine uses `2.8.0`; the harness itself
  flags the build mismatch. The B=1 preprocessing path was left unchanged.

### Clear synchronization audit

Ordinary CPU/table collector hot loop:

1. Policy H2D input formatting (no explicit host read).
2. Policy forward/sample on device.
3. `action.cpu().numpy()`: **one D2H transfer and host barrier per timestep**
   (256/update).
4. B CPU environment steps.
5. CPU observation/action formatting and H2D transfer.
6. pRNN device step. The next timestep's action D2H drains this queued work.

Device collector hot loop:

1. Policy consumes device direction and SR.
2. Policy sample remains device-resident.
3. Device transition table consumes the action tensor directly.
4. Pre-step device observations/positions are copied to rollout buffers.
5. Device-native SpeedHD formatting and pRNN step.

**Confirmed:** the device loop has no executed `.cpu()`, `.numpy()`, `.item()`,
device-scalar Python boolean, host/device copy, or explicit synchronize.
Exact reset rows are already resident. The `action_to_host` scope is retained
for comparable profiling but contains only the conditional branch;
synchronized timing measured 0.2 ms total across 256 calls.

Post-rollout synchronizations remain:

- bulk rollout image/direction/location views for diagnostics;
- one bulk action export and one probability export;
- bulk GAE/log arrays;
- PPO's one summary export;
- pRNN `trainStep`'s scalar training-loss record.

These barriers do not interrupt the causal transition loop. The benchmark
option `--sync-stages` deliberately inserts MPS/CUDA synchronization at every
timer boundary for attribution; do not compare its FPS directly with normal
execution.

### Synchronized stage breakdown (M4 Pro MPS, B=128)

Same shape for before/after: T=256, 32,768 frames/update, PPO minibatch 4,096,
batched curiosity, pooled WM. Times are non-overlapping parent scopes except
where nested details are explicitly described.

| Parent work | CPU table boundary | Final device path |
|---|---:|---:|
| Policy inference/sample/action boundary | 784.5 ms | 431.9 ms |
| Policy rollout recording | 183.8 ms | 129.2 ms |
| pRNN state step | 503.9 ms | 289.9 ms |
| Environment + rollout state copies | 351.4 ms | 52.4 ms |
| PPO optimization | 266.2 ms | 172.5 ms |
| Pooled world-model update | 137.3 ms | 140.8 ms |
| Curiosity | 96.7 ms | 43.8 ms |
| GAE + log prep | 3.3 ms | 2.9 ms |
| Other bulk export/bookkeeping | ~143 ms | ~90 ms |
| **Total** | **2.470 s / 13,265 FPS** | **1.353 s / 24,218 FPS** |

Final nested detail:

- policy: network 187.8 ms, categorical sample 241.0 ms, preprocessing
  0.3 ms, action-host scope 0.2 ms;
- pRNN: device formatting 67.2 ms, recurrent body 219.9 ms; inside the body,
  mask/noise is 58.8 ms and the cell is 125.3 ms;
- PPO (32 minibatches): indexing 42.1 ms, forward 25.8 ms, loss 25.6 ms,
  backward 33.6 ms, grad clip 15.0 ms, Adam 21.9 ms;
- device environment: 52.4 ms includes transition lookup plus writing
  image/direction/position rollout buffers. It is not a host sync.

### Normal-execution batch sweep (complete training)

M4 Pro MPS, T=256, pooled WM, batched/device curiosity, device environment,
PPO minibatch 4,096 unless marked otherwise:

| B | Frames/update | FPS |
|---:|---:|---:|
| 32 | 8,192 | 8,989 |
| 64 | 16,384 | 13,923 |
| 128 | 32,768 | 26,886 |
| 256 | 65,536 | 37,370 |
| 512 | 131,072 | 52,798 |
| 1,024 | 262,144 | 58,041 |
| 512, minibatch 16,384 | 131,072 | 64,461 |
| 512, minibatch 65,536 | 131,072 | 75,310 |
| 1,024, minibatch 131,072 | 262,144 | **89,881** |

The final B=1,024 point performs 8 PPO optimizer steps/update rather than
256 at minibatch 4,096. The device environment, native formatting, and
synchronized phase mask preserve collected trajectories; pooled WM and
larger PPO minibatches change optimizer semantics and need a learning-curve
gate.

The final one-command `performance=ultra` runtime smoke measured 79,292 FPS
for one timed update after one warmup; the two-update measurement in the table
(75,310 FPS) is the less cherry-picked local reference.

At the final B=1,024 point, the timer (normal asynchronous, so useful for
call counts and coarse parents, not kernel attribution) reports: WM 773 ms,
policy rollout 498 ms, PPO 300 ms, curiosity 205 ms, pRNN rollout 125 ms,
and environment/buffer writes 12 ms. Total is 2.915 s/update.

The final five-update **default CPU** reference on this M4 Pro
(`OMP_NUM_THREADS=MKL_NUM_THREADS=16`) is 1,970.6 FPS, with update times
1.019-1.095 s. Its dominant measured parent is the unchanged eight-step
world-model training schedule at 607 ms/update; reference MiniGrid stepping
is 173 ms/update.

### Updated H100 inference from measured utilization

The final M4 run executes roughly 5.55 MFLOP/frame, or about 1.45 TFLOP per
262,144-frame update. Its 89,881 FPS corresponds to roughly 0.50 TFLOP/s
application-level throughput, about 7.5% of the separately measured 6.7
TFLOP/s large-GEMM ceiling on this M4. The loss is caused by the 256 causal
steps, many normalization/elementwise kernels, and optimizer/framework work,
not environment bandwidth.

**Inferred, not H100-measured:**

- Applying the same 7.5% fraction to H100's 60 TFLOP/s FP32 peak gives about
  0.8 million FPS.
- Large B makes the dominant `[B,500]@[500,500]` GEMMs viable; a competent
  eager CUDA port should therefore be in the broad **0.5-2 million FPS**
  range, depending on launch/framework efficiency.
- Fusing/compiling the recurrent cell and graphing fixed-shape rollouts could
  plausibly reach **2-6 million FPS** in FP32. The 10.8 million FPS
  arithmetic roof remains impossible to hit application-wide because not all
  work is GEMM and 256 causal stages cannot overlap.
- TF32/BF16 can raise the arithmetic ceiling, but this is a numerics and
  learning-curve change, not a free implementation optimization.

An H100 can likely be well utilized by the dominant GEMMs at B in the high
hundreds/low thousands, but the **whole current application will not report
100% H100 FP32 utilization** without fusion: categorical sampling, pRNN
layer-normalization/activation/noise, buffer writes, Adam, and causal launch
latency remain. Nsight Systems plus Nsight Compute on the actual H100 is the
required confirmation.

### Reproduction commands

```bash
# Exact device-environment toggle, retaining other configured semantics
CG_DEVICE=mps uv run python tests/perf/benchmark.py --updates 3 \
  --out /tmp/device_env.json \
  --override exp.device_env=true

# Measured high-throughput preset
CG_DEVICE=mps uv run python tests/perf/benchmark.py --updates 2 \
  --warmup-updates 1 --out /tmp/ultra.json \
  --override performance=ultra

# Real stage attribution (intentionally slower)
CG_DEVICE=mps uv run python tests/perf/benchmark.py --updates 1 \
  --warmup-updates 1 --sync-stages --out /tmp/device_sync.json \
  --override exp.device_env=true --override exp.num_envs=128 \
  --override rl.frames=32768 --override rl.ppo_batch_size=4096 \
  --override predNet.batched_wm=true \
  --override predNet.batched_curiosity=true
```

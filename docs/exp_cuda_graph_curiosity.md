# CUDA-graphing the curiosity forward (`predNet.curiosity_cuda_graph`)

## Question

After the world model, the PPO minibatch and the rollout timestep, the
curiosity forward was the last ungraphed region of a training update. It is
the one the pooled regime pays for most: `wm_pool_group=8` needs eight times
the updates of g=1 for the same world-model gradient-step budget, so it pays
this per-update cost eight times over. What does graphing it buy?

## Where the code lives

- `curious_george/world_model/adapter.py` - `_GraphCuriosityForward` (the
  capture, keyed on `(batch, obs length, action length)`) and
  `PRNNAdapter._curiosity_errors`, the single seam both batched curiosity
  callers now route through. `_prediction_mses_batched` (list path) and
  `prediction_mses_device` (device-env path) previously each spelled the same
  forward and the same error reduction; they now share one.
- Flag: `predNet.curiosity_cuda_graph` in `Configs/world_model/thrnn5win.yaml`,
  defaulting off, plumbed through `training/setup.py` and `rl/algo.py`.
  Forward-only, so unlike `predNet.cuda_graph` it needs no fresh optimizer.

## Method

Twelve updates after three warmup updates, with the other three graphs and
`compile_cell=layer` ON IN BOTH ARMS so only the flag differs:

```
uv run --no-sync python tests/perf/benchmark.py --updates 12 --warmup-updates 3 --sync-stages \
  --override env=lroom --override run=multienv --override exp.device_env=True \
  --override exp.rollout_cuda_graph=True --override predNet.batched_wm=True \
  --override predNet.compile_cell=layer --override predNet.cuda_graph=True \
  --override rl.cuda_graph=True --override predNet.wm_pool_group=$G \
  --override predNet.curiosity_cuda_graph=$FLAG \
  --override exp.num_envs=128 --override rl.frames=32768 \
  --override rl.ppo_batch_size=256 --override rl.entropy_coef=0 --out <path>
```

## Results, RTX 4060, 2026-08-24

ms/update under `--sync-stages`, identical at both pool settings because the
curiosity forward does not depend on `wm_pool_group`:

| stage | g=8 off | g=8 on | g=1 off | g=1 on |
|---|---|---|---|---|
| collect/curious_rewards | 25.93 | **12.99** | 26.00 | **13.05** |
| collect/curious/predict | 24.80 | **11.84** | 24.90 | **11.87** |
| collect/curious/format | 0.49 | 0.48 | 0.48 | 0.48 |
| collect/curious/error | 0.47 | 0.48 | 0.46 | 0.49 |

Un-synced:

| `wm_pool_group` | fps off | fps on | speedup |
|---|---|---|---|
| 8 | 53,801 | 55,441 | 1.030x |
| 1 | 29,780 | 30,052 | 1.009x |

Read the stage row, not the fps: a ~13 ms saving on a ~600 ms update sits
close to run-to-run fps noise, which is why the per-run projection below is
computed from the stage attribution instead.

Per run at a fixed 80,000 world-model gradient steps: g=8 takes 5,000 updates
(16 steps each), so 5,000 x 12.94 ms = **1.08 min saved**; g=1 takes 625
updates (128 steps each), so **0.14 min saved**. The predicted 8x ratio
between the two regimes holds; the magnitudes are somewhat smaller than the
~1.8 / ~0.2 min predicted, for the reason in the next section.

## Discoveries

- **The 4.7x probe figure was measured without `compile_cell`.** Control, same
  config with `predNet.compile_cell=False`: `collect/curious/predict` is
  67.15 ms/update eager and 14.83 graphed - 4.5x, reproducing the probe.
  With `compile_cell=layer`, which every production run uses, eager is already
  24.80 ms and the graph buys **2.1x** (24.80 -> 11.84). `torch.compile` had
  taken most of it first. Compile still helps underneath the graph:
  11.84 ms compiled+graphed against 14.83 uncompiled+graphed.
- **A bare `cuda -> cpu -> cuda` round trip usually returns the SAME
  addresses.** The caching allocator hands the freed blocks straight back
  when nothing claimed them in between - so `data_ptr()` comparison correctly
  does not fire, and the graph is genuinely not stranded. Making
  `test_raw_to_roundtrip_invalidates_captured_graphs` mean anything required
  allocating same-sized ballast while the net was away, plus an assertion that
  the pointers actually moved. `tests/test_cuda_graph_wm.py` has the same
  test without the ballast; it passes today because its allocator state
  happens to relocate, and its self-assert makes it fail loudly rather than
  vacuously if that ever changes - but it is luck-dependent as written.
- **`_one_segment` is not reproducible for a fixed index.** It reseeds torch
  and numpy but drives `pN.env_shell`, whose MiniGrid RNG carries across
  resets. Two arms must consume their environments in lockstep or reuse one
  batch; getting it wrong produces a ~1e-2 disagreement that reads exactly
  like a stranded graph. Noted in `_batch`'s docstring, because it cost time.

## Gates

`tests/test_cuda_graph_curiosity.py`. Bitwise against eager with dropout and
noise off, on a DIFFERENT batch each round (feeding one batch repeatedly would
pass even with the input copy removed); one graph per shape; exact agreement
across `on_device` round trips WITH the captured graph proven not to have been
re-captured; re-capture forced and verified after a genuine relocation; fresh
randomness per replay with noise on; and the end-to-end flag wiring on the
device path.

Four deliberate breaks of production code, each confirmed red then reverted:
inputs never copied into the static buffers (3 red), a static input buffer
rebound after capture (2 red), the graph key collapsed to a constant (7 red),
the address guard disabled (1 red - the round-trip gate, precisely the one
that owns it).

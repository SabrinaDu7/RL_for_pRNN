# Speed-ups: from 17.7 h to under 30 min per training run

Written 2026-10-20 as a map of the speed work on `main` (88f36bf). Each entry gives the date, the
commit(s), the code it lives in, and the gain as measured in the commit message or log cited. The
full account, with every measurement and the dead ends, is
`docs/claude_logs/speed-30min-2026-08-23.md` (cited below as `speed-30min`).

## Headline

| | Wall time | Source |
| --- | --- | --- |
| July reference run `pRNN_curious_perf_optimized_26-07-08-16-04-37`, 20.48M env steps | 17.7 h | `speed-30min:18` |
| Same 20.48M steps, eager (no CUDA graphs), seed 2, `wm_pool_group=1` | 2:34:42 | `speed-30min:436` |
| Same code and seed, CUDA-graphed | 28:49 (5.37x) | `speed-30min:436-437` |
| `main_train.py parity` (single L-room, 89,980,928 env steps), measured 2026-08-30 | 27.6 min | `docs/claude_logs/plan-multienv-2026-08-30.md:126` on branch `sdu/mixed-count-mse` (7d193e8)* |

\* That run was measured on 2026-08-30 with `main` at 88f36bf (committed 2026-08-30 00:02) and
recorded in the first commit after it, 7d193e8, so it is believed to be main's code; the log does
not name the commit it ran on. `slurm/parity.sh` requests an L40S (`--gres=gpu:l40s:1`).

**To get the speed, run a preset.** The flags below are off by default in the config dataclasses
and switched on together by the presets (`parity` in `curious_george/configs.py`, `_parity()`):
`collect.backend=DEVICE`, `collect.rollout_cuda_graph`, `train_prnn.batched`,
`train_prnn.batched_curiosity`, `train_prnn.compile=LAYER`, `train_prnn.cuda_graph`,
`train_policy.cuda_graph`. A run launched without them is several times slower.

## The changes, in order

| Date | Change | Commit(s) | Code | Measured gain (source) |
| --- | --- | --- | --- | --- |
| 2026-07-07 | Stage timers and a benchmark harness, the baseline everything below is measured against | 226e1a9 | `training/loop.py`, `rl/collect/collector.py` | - |
| 2026-07-07 | Remove per-step host syncs; vectorised GAE; plotting gated | 221f1cf | `rl/update/advantage.py`, collector | GAE 100 ms -> 0.7 ms, bitwise; CUDA 218 -> 274 FPS (221f1cf) |
| 2026-07-08 | Vectorised observation / action formatting in the pRNN adapter | 6247a5a | `models/prnn_adapter.py` (then `world_model/adapter.py`) | not measured on its own |
| 2026-07-07, 07-08 | Shorter in-run spatial evaluation, then SI + sRSA + SWdist from one rollout | 87b6335, 482d86b | `evaluation/spatial.py` | analysis event 293 s -> 10.2 s (482d86b). Changes the measurement: sRSA depends on rollout length (87b6335) |
| 2026-07-13 | Observation bank: precomputed (position, head direction) -> observation lookup | ad2a8b7 | `envs/obs_bank.py` | env step 6.4 -> 3.2 ms (ad2a8b7, CPU dev box) |
| 2026-07-14 (merged 07-17) | Batched world-model training: episodes pooled into one gradient step | 1cb0ee8, merge cc4dcaf | `models/prnn_adapter.py::train_on_episodes_batched` | wm_train 1.07 -> 0.22 s/update, CPU (1cb0ee8) |
| 2026-07-23 (merged 08-11) | Device-resident rollouts: table-driven environment, collection and postprocessing on the GPU | cd9b5e5, 1eb3d0d, 18af616, merge a3b2b36 | `envs/vector.py::DeviceTableShellPool`, `rl/collect/collector.py` | 13,265 -> 25,949 FPS, measured on an M4 Pro, not CUDA (`throwaway/ported/docs_legacy/throughput_investigation_2026-07-23.md:413`) |
| 2026-08-19 | `torch.compile` of the recurrent cell, then of the whole 256-step loop | d611a87, cf0133b | `models/prnn_adapter.py` (`train_prnn.compile`) | 1.42x (d611a87), then 2.47x gradient steps/s without graphs (cf0133b) |
| 2026-08-23 | Address-preserving `on_device`, restoring grads and buffers: what keeps CUDA graphs valid across the spatial eval | e3c7c15, 0d60df7 | `models/device.py::on_device`, `utils/cuda_graph.py` | correctness, not speed (below) |
| 2026-08-23 | CUDA graph of the pooled world-model training step | 2b66522 | `models/prnn_adapter.py::_GraphWMTrainer` | 8.97x per step; fps 15,198 -> 22,338 (2b66522) |
| 2026-08-23 | CUDA graph of the PPO minibatch step | 2c83318 | `rl/update/policy_graph.py` | 3.558 -> 0.275 ms/step, 12.9x (2c83318) |
| 2026-08-24 | CUDA graph of the rollout timestep | 643ae31 | `rl/collect/rollout_graph.py` | 1.36x at `wm_pool_group=1`, 1.59x at 8 (643ae31) |
| 2026-08-24 | CUDA graph of the curiosity forward | fee9be0 | `models/prnn_adapter.py` (`train_prnn.curiosity_cuda_graph`) | 2.1x on that stage, ~1.2 min per run; **not on in `parity`** (fee9be0) |
| 2026-08-28 | `torch.compile` off whenever the module is not on CUDA | 2431260 | `models/prnn_adapter.py` | keeps CPU runs working |

All paths are under `curious_george/` unless they start with `docs/` or `throwaway/`.

## Caveats on record

- **A graphed run once trained on nothing, with every test passing.** After the spatial eval moved
  the model, the graph read stale buffers. e3c7c15 / 0d60df7 fixed it, and tests now require
  graphed training to match eager exactly across repeated evals (`speed-30min:243`, section 5).
- **A CUDA graph freezes Python floats at capture** (learning rate, clip, an entropy ramp), so
  schedules do not work under graphs. Runs with an entropy ramp were void (`speed-30min:805-850`, section 11),
  and the config rejects `entropy_coef_final` together with `cuda_graph`.
- **A graphed rollout is not bitwise-identical to eager**: it draws from a different RNG stream
  (`speed-30min:568`). The learning metrics of the 5.37x pair are inside the reference band
  (`speed-30min:357-421`); SWdist ran about 2x higher in the graphed arm at n = 1 (`:436-460`).
- **Tried and not kept:** `max-autotune` does not stack with the graphs (`speed-30min:679`); a
  prefix-scan GAE was reverted to keep the golden test bitwise (a3b2b36).
- **torch:** `torch>=2.7.1`, locked at 2.8.0 in `uv.lock`; every timing above was made under it.
- **Stale line:** `README.md:204` says the in-run spatial eval is skipped under CUDA graphs. The skip
  was removed by e3c7c15.

## Not on `main`

Branch `sdu/mixed-count-mse` holds one further speed-up, plus a large change to the training recipe
(cross-entropy / focal loss, MLP readout, mixed object counts), so it is not merged here:

| Date | Change | Commit(s) | Measured gain (source) |
| --- | --- | --- | --- |
| 2026-08-31, 09-10 | Room layouts cached so resets stop rebuilding them; resets prepared in a worker thread | 4b0aa96, fc4cfc6; checked in 4bbee92 | rollout 31,456 -> 93,117 steps/s on an RTX 4060; a full multi-room run ~55 -> ~20 min (4b0aa96); bitwise trajectory match over 45M steps (4bbee92) |

It helps only rooms with impassable objects or none (walkable rooms keep the full reset), so it
matters for multi-room runs. Cherry-pick 4b0aa96 and fc4cfc6 to use it on `main`.

2026-08-30 · branch `sdu/optim-pred` · base `2eacc69`

# Plan: evaluating the curious POLICY — coverage, not just occupancy

## The request, verbatim

> The goal of this session is to write out evaluation for the curious POLICY here. Don't
> code anything up, I just want you to read the code, particularly how the policy is wired
> up with everything (check @curious_george/rl/update/policy.py and @curious_george/rl/algo.py
> specifically) and how we can fit evaluations of the policy in here. Currently, we mainly
> focus on visitation/occupancy maps and location entropy.
>
> I want the following evals: 1. visitation entropy (and this will be important when there
> are impassable objects), 2. unique cells visited per env steps (ie get a coverage curve,
> unique cells visited as a function of steps, averaged over N episodes (use ≥10 seeds —
> exploration variance is brutal), followed by AUC of that curve, normalized by total
> reachable cells, to understand the timescale of visitation), 3. Time-to-cover thresholds.
> T50, T90 — steps needed to reach 50%/90% coverage. Same information as the AUC but far
> easier to read in a table, and it exposes policies that get there eventually versus
> quickly., 4. Hardness-aware coverage. Precompute BFS distance from spawn to every walkable
> cell, then plot coverage as a function of shortest-path distance. A policy that covers 90%
> of nearby cells and 5% of far ones looks fine on aggregate coverage and terrible here. And
> most importantly, each of these evals needs BASELINES: identical logged metrics for a
> uniform random policy, for the random agent which has a categorical distribution over the
> action space that isn't perfectly uniform, and perhaps even a count-based bonus agent
> (1/sqrt(N(s)) if that's cheap to compute.
>
> Again, no need to code. Just read and look at the feasibility of implementing these evals.
> We'll then code them up, test them, and measure their speed so there's as fast as possible

Then, on the three questions this plan raised:

> Seed: literally rng seed - which if we set everything to eval on the policy side would just
> change the starting pos and dir of the agent
> Eval episode length = training ep length so
> For the denominator, what do you think is correct? Union or per-room reachable - they might
> boil down to the same after averaging

---

## Decisions

**1. Seed = the RNG seed; the unit of replication is the EPISODE.** ≥10 means ≥10 episodes
with independent spawns, not 10 training runs. One `parity`-shaped rollout is
`num_envs=256` streams, each with its own spawn, so the online form of every eval below is
n=256 for free.

⚠️ **One correction to the premise.** `cfg.run.seed` does *not* only move the start state.
Traced: `seed_everything` seeds `random`/`numpy`/`torch` (`utils/common.py:61`); env shells
are seeded `cfg.run.seed + 10000 + offset` (`training/setup.py:123`); the device pool's room
assignment uses `layout_seed=cfg.run.seed` (`training/setup.py:154`). So a seed change moves
the spawn **and** the action-sampling stream (`dist.sample()`, `collector.py:347`) **and**
the pRNN dropout/noise draws **and** which room each stream gets. Only `argmax` would remove
the action stochasticity, and argmax is a different agent (see *Traps*). This is fine — it is
the exploration variance you want to average over — but the doc for these metrics must say
that a "seed" varies four things, not one.

**2. Eval episode length = training episode length = 256 steps** (`CollectCfg.episode_steps`).
This is the right call and is *better than I first thought* — see the calibration below.

**3. Denominator = the room's own walkable set, `len(layout.walkable(base_walkable(room)))`.**
Not the union. Reasoning is measured, not argued, below.

---

## The denominator: per-room, and it is not a wash

Measured on the committed 5-room set (`throwaway/scripts/exploration_baseline_calibration.py`):

| arm | per-room `\|walkable\|` | union `\|walkable\|` | per-room ÷ union |
|---|---|---|---|
| walkable | 172, 172, 172, 172, 172 | 172 | **1.0000** |
| impassable | 153, 153, 153, 153, 153 | 172 | **0.8895** |

Your intuition is exactly right for the **walkable** arm: the two denominators are *literally
the same number*, because `Layout.walkable` returns `base` unchanged when nothing is
impassable (`envs/layouts.py:251-261`). Choosing per-room therefore costs nothing there.

They come apart in the **impassable** arm, and not by averaging out:

- The union counts 19 cells that were **walls in the room the agent was actually in**. Every
  landmark set removes exactly 19 cells (x=5, plus=5, block3=9), so the per-room count is a
  *constant* 153 — the two denominators differ by a fixed factor 0.8895, in the same direction,
  in every episode. Averaging cannot cancel a constant.
- Under the union, coverage has a **hard ceiling of 0.8895**. T90 is then undefined *by
  geometry*, and T75 is measuring how close a room's blocked-cell count sits to the threshold
  rather than how well the policy explored. That is precisely the failure mode
  `algo.py:287-357` already documents for `loc_entropy` ("the CEILING moves with the geometry
  … raw `loc_entropy` is not comparable between an objects arm and a control").
- Per-room is the only denominator under which the metric runs 0→1 in both arms, which is what
  makes the walkable/impassable comparison — the reason these evals exist — legible.

**Confirmed, not assumed: there are no enclosed pockets.** I ran BFS over the
`(x, y, direction)` action graph from *every* legal spawn in all ten rooms (both arms). The
worst-case spawn still reaches 172/172 and 153/153 cells respectively. So
`len(layout.walkable(base))` **is** the BFS-reachable set, and the denominator needs no
per-episode BFS. (BFS is still needed for eval 4's distance *binning*.)

**Implementation rule:** ask the `Layout`, never hardcode 153/172. The constancy holds for
this content spec (three fixed stencils); a pool that varied `Vary.KIND` with different-sized
stencils would break it silently.

*Correction to my earlier assessment:* I said exporting the room channel was a prerequisite for
the coverage denominator. It is not — the denominator is a constant per arm. The room channel
is still required for **eval 4** (which room's distance map) and for a per-room breakdown of
visitation entropy.

---

## Calibration: is a 256-step episode enough for T50/T90?

Yes. **I was wrong to worry about this**, and the correction changes the design. My concern was
that T90 would be censored by the horizon. It is censored for *random walkers*, but not by the
horizon: a greedy sweeper reaches 90% coverage in ~191 steps, comfortably inside 256.

One 256-step episode, room 0, coverage of the room's own walkable set. `nAUC` is the mean of
the normalized curve (= AUC ÷ 256 ÷ denominator). `Txx` is the mean step at which the threshold
is crossed, with the % of episodes that reach it at all:

```
=== walkable, denominator 172 ===
agent                     cov@256   nAUC        T25          T50          T75          T90
uniform [.25]x4             0.191  0.108   228 (11%)     censored     censored     censored
forward-weighted            0.395  0.224   141 (96%)    232 ( 7%)     censored     censored
greedy sweeper              1.000  0.605    46(100%)     96(100%)    152(100%)    191(100%)

=== impassable, denominator 153 ===
uniform [.25]x4             0.177  0.102   226 ( 7%)     censored     censored     censored
forward-weighted            0.349  0.200   157 (87%)    235 ( 2%)     censored     censored
greedy sweeper              0.994  0.605    42(100%)     92(100%)    154(100%)    200(100%)
```

Reproduce: `uv run python throwaway/scripts/exploration_baseline_calibration.py`. No networks
are involved — every agent is a policy over the same action table the DEVICE backend steps, so
this isolates what the geometry plus an action distribution buys. ⚠️ **Throwaway: no result may
depend on it.** It exists to size the design; when the real implementation lands, these numbers
get re-derived by the library and pinned in a test, and the script dies.

**What this settles:**

- **The dynamic range is real: 0.19 → 1.00 at 256 steps.** The metric is neither saturated nor
  geometrically censored. Keep 256.
- **Keep T50 and T90 as specified.** Both are achievable in-episode. A learned policy that never
  reaches T90 is genuinely worse than a sweeper — that is a finding, not an artifact.
- **Censoring is the headline statistic, not a footnote.** T50 is reached by 7% of
  forward-weighted-random episodes; reporting only the mean-over-reached (232 steps) would
  describe the luckiest 7% and hide the other 93%. Every threshold must be reported as
  *(mean step | % of episodes reaching it)*, and a policy with a higher reach-rate beats one
  with a faster conditional mean.
- **Log the CURVE, derive the thresholds in analysis.** 256 numbers per episode (or a
  decimated version). Then T25/T40/T50/T90 are all recoverable without re-running, no threshold
  choice is baked into the collection, and censoring is visible by construction.
- **The two random baselines are far apart** (0.191 vs 0.395 coverage@256, a 2.1x gap), so
  "which random" is not a detail — reporting one unlabelled "random baseline" would be
  ambiguous by a factor of two.

---

## Where the policy is wired

**`rl/update/policy.py` is a dead end for exploration metrics.** Traced: `update_policy`
(`policy.py:76`) and `_index_policy_batch` (`policy.py:51-73`) see only
`SR, action, value, advantage, returnn, log_prob`. No positions, no reward construction. It is
loss-agnostic epoch/minibatch machinery and nothing about *where the agent went* passes through it.

The seam is one expression in the collector:

```
collector.py:343-348
    action = _random_actions(random_probs) if cfg.random_actions else dist.sample()
```

That single line is the entire policy-vs-baseline switch, and everything a baseline must hold
fixed — backend, batch size, room set, world-model training — is shared by construction. This is
what commit `22e7c0d` bought.

What shapes exploration enters separately, at reward construction:
`compute_curious_rewards` / `prediction_mses_device` → `compute_gae(k_curious=…)`
(`collector.py:649-702`). That is where a count-based bonus attaches.

Positions leave the collector as `CollectResult.locs` / `.directions` and are re-exposed on the
algo (`algo.py:399-401`) for analysis.

```
main_train.py → setup_training → setup_algo → PredictivePPOAlgo
   per rollout:
     algo.collect_experiences()      → collect_rollout   ← positions, actions, the switch
     algo.update_parameters(...)     → update_policy     ← no positions, ever
```

---

## What already exists

| capability | where | status |
|---|---|---|
| `loc_entropy`, `loc_entropy_5` | `rl/collect/diagnostics.py:33` | eval 1, logged every `log_every_steps` |
| `reachable_cells`, `loc_entropy_ceiling` | `configs.py:235,251` | the normalizer, quotable with no run |
| `occupancy_counts` → `[hd,x,y]` | `evaluation/on_policy.py:560` | the visitation map; the figure is downstream of it |
| forward-weighted random baseline | `collector.py:77`, `collector.py:100-109` | works end-to-end; gated by `tests/test_random_agent_baseline.py` |
| exact transition table `(W,H,4,A)→(x,y,dir)` | `envs/obs_bank.py:173`, stacked per layout at `envs/vector.py:283` | the BFS graph for eval 4 |
| archived per-step POLICY checkpoints | `evaluation/checkpoint_series.py:63` | written, findable, **never read by `main()`** |

⚠️ `prnn.analysis.trajectoryAnalysis.calculateCoverage` exists upstream and is **not** a
coverage curve — it is an occupancy histogram plus a max-deviation "nonuniformity". Name
collision. Do not reuse it, and do not name a new function `coverage` unqualified.

---

## The four evals

### 1. Visitation entropy — exists; the gap is normalization and the room channel

`loc_entropy` is computed and logged. Two things are missing:

- **The ceiling is printed but never logged** (`training/loop.py:197-201`), so no wandb series
  is comparable across arms. Add `loc_entropy / loc_entropy_ceiling`. The ceiling is already a
  config property; this is a `training/logging.py` change.
- **The histogram is a mixture across rooms with no room channel** — stated as a known
  limitation in `algo.py:287-357`. The channel is *recoverable*: `DeviceTableShellPool.stream_layout`
  (`envs/vector.py:289`) and `_prepared_layouts_host` (`envs/vector.py:433`) know which room
  each stream is in at every segment; the collector simply does not export it. Exporting it
  makes per-room entropy possible and is a hard prerequisite for eval 4.

### 2. Coverage curve + normalized AUC — free

Positions are already a `(T, B, 2)` device tensor (`collector.py:396`), and with
`episodes_per_env=1` an episode is exactly one stream's whole `T`. Computation: cell id
`y*W + x` → first-visit time per (episode, cell) by scatter-min → bincount → cumsum. No Python
loop. At `parity` (B=256, T=256) that is 65,536 elements: sub-millisecond on device.

Normalized AUC = mean of the normalized curve over the episode. Report the curve, and the AUC
derived from it, so the two cannot disagree — the same discipline `occupancy_counts` enforces
for the occupancy figure.

### 3. Time-to-cover — arithmetic on eval 2's curve

`np.searchsorted` on the cumulative-unique curve. The whole design content is in the reporting
contract: *(mean step | % of episodes reaching it)*, never a bare mean. See the calibration.

### 4. Hardness-aware coverage — feasible, and better than grid BFS

`_next_state` (`envs/obs_bank.py:173-224`) is the exact `(W,H,4,A) → (x,y,dir)` machine, built
per layout and stacked on device as `pool.next_state`. BFS over its ~1,024 `(x,y,dir)` nodes
gives distance **in actions, including turning cost and impassable landmarks** — not Manhattan.
Cell distance = min over the four facings. One BFS per (layout, spawn) is microseconds;
all-pairs per layout is tens of milliseconds and can be cached beside the obs bank.

This eval earns its place because of a spawn asymmetry the aggregate metrics hide. Confirmed by
reading `minigrid_env.py:313-395` and `Lroom.py:196-218`: `place_agent` samples uniformly over
cells with **no object at placement time**, and impassable landmarks are painted *before*
placement while walkable ones are painted *after*. So the impassable arm cannot spawn on a
landmark and the walkable arm can — the arms differ in spawn distribution by construction.
Distance-binning is the control for exactly that.

Requires the room channel from eval 1 (which room's distance map applies to which episode).

---

## Baselines

| baseline | status | cost |
|---|---|---|
| forward-weighted random `[.15,.15,.6,.1]` | **works today** — `--arch-policy.agent random` | one training run |
| uniform random `[.25]×4` | **not expressible** — needs a config field | one training run |
| greedy sweeper (positive control) | not present | ~free, no training |
| count-based `1/√N(s)` | not present | one training run per seed |

**Uniform random is not expressible today.** `random_action_probs` (`collector.py:77`)
hard-imports `storage.RAND_ACT_PROBA`; there is no config field. Adding one should fix the
four-spelling defect the collector already names — I confirmed all four are live:
`log_and_store/storage.py:26`, `training/setup.py:39`, `evaluation/circuit_diagnostics.py:40`
(`PROBE_ACTION_P`), and a bare literal at `evaluation/checkpoint_series.py:226`. One home, one
config field, and the collector stops having to warn about becoming a fifth.

**The greedy sweeper was not in the request and should be.** Every metric here is a fraction
whose achievable maximum is otherwise unknown; the sweeper supplies it (nAUC 0.605, T90 ~191).
It needs no training and no networks — it is a policy over the transition table — so it is the
cheapest baseline on the list and the one that makes the other numbers readable.

**The count-based bonus is cheap to COMPUTE and expensive to RUN.** Mechanically: a
`(n_layouts, W, H, 4)` int64 device tensor, `scatter_add_` over the existing positions buffer,
reward `k/√N` gathered per visited state, injected at the `curious_rewards` slot
(`collector.py:690-702`). Computed *after* the timestep loop from the positions buffer, exactly
as `prediction_mses_device` is (`collector.py:651-658`), so it is CUDA-graph compatible. Two
kernels per rollout.

But it is a **training arm**, not an eval: one full run per seed, and `k_count` needs its own
scale tuning — especially now that `normalize_advantage` has landed (`2eacc69`) and changes what
a reward coefficient means. Calling it "cheap" understates it. Decisions to settle before
building it: lifetime vs per-episode counts; state = `(x,y)` or `(x,y,hd)`; counts per room or
pooled. Recommendation: lifetime, per `(room, x, y, hd)` to match the pRNN's own input, stated
explicitly wherever the number is reported.

---

## Where the code goes

- **Pure analysis** → new `curious_george/evaluation/exploration.py`. Positions array + geometry
  in, one dataclass out, **no collection** (CLAUDE.md: analysis code collects nothing).
- **BFS / distance geometry** → `curious_george/envs/`, beside `access.get_walkable_mask` and
  `layouts.pooled_walkable`. It is env geometry, not analysis; `access.py:96-100` documents
  exactly this promotion rule.
- **Online hook** → `training/loop.py` at the existing `log_due` / `analysis_due` cadence,
  reading what the collector already produces. This is where `loc_entropy` already lands.
- **Offline / developmental** → `evaluation/checkpoint_series.py`. It already writes and finds
  archived policies (`archived_policies`, `checkpoint_series.py:63`) but `main()` never calls
  it — the only consumers are one test and one throwaway script. **The policy series is
  archived and never read.** These evals are its natural first consumer, and that turns the
  coverage curve into a developmental series rather than one number.

---

## Speed

- **Online metrics: free.** O(B·T) array ops over data already resident on the device.
- **Do not build on `flat_locs`.** `collector.py:617` materializes B·T Python tuples per
  rollout, and `LocationStats.update` (`diagnostics.py:35-36`) then iterates them in a Python
  `for` loop. At `parity` that is 65,536 tuples plus 65,536 iterations per rollout. *Inferred:*
  ~1% of rollout wall-clock — small, but it is the wrong base to hang four more metrics on.
  Build on the `(T, B, 2)` array; measure with `CG_TIMING=1` before and after.
- **Offline eval: route through a DEVICE pool, not the serial CPU path.**
  `ActorCriticAgent.next_SR` (`rl/collect/agent.py:33-40`) does `env2pred` + `predict_single`
  per step on CPU. The documented ~88 s per analysis event at `rooms_max=4`
  (`configs.py:626-639`) puts that in the milliseconds-per-step range, so ≥10 episodes × 256
  steps × per room × per checkpoint adds up fast on the serial path and is ~free on the device one.

---

## Traps found while reading

1. **`collect_eval_rollouts_batched` uses `dist.probs.argmax(dim=-1)`** (`evaluation/task.py`,
   "frozen eval: argmax"). A greedy deterministic walker is a **different agent** from the one
   that trained, and for an exploration metric that difference is the whole measurement. It must
   sample. The serial `collect_eval_rollouts` goes through the agent and `get_agent` defaults
   `argmax=False`, so that path is fine.
2. **The pRNN hidden state is reset at `prnn_seqdur = episode_steps`** (`training/setup.py:234`).
   Holding the eval episode at 256 keeps the eval inside the horizon the circuit is trained at.
   This is a second reason decision 2 is right, beyond the calibration.
3. **`checkpoint_series.score` does not disable dropout** — its own docstring says so
   (`torch.no_grad()` stops gradients, not dropout at 0.15 or noise at 0.05). Any policy eval
   added there inherits that wobble unless it goes through `models.device.eval_mode`, as
   `evaluate_multi_room_representation` does (`evaluation/spatial.py:301`).

## Stale document to correct

`docs/multienv-walkable-and-impassable-2026-08-30.md` states, under *What is NOT established*:
"The random-agent baseline is MISSING, and currently unrepresentable", citing three pairwise-
conflicting `Config` constraints. That was true at the commit its own header names (`56cd4e8`)
and is **false at HEAD**: `22e7c0d` / `561f723` / `a7ac733` routed random actions through
`collect_rollout`, and `tests/test_random_agent_baseline.py::test_a_random_agent_multi_room_config_is_now_expressible`
pins that the `multienv-fast` + `RANDOM` config now builds. Fix that paragraph before it is used
to plan the baseline runs.

## Gates to add with the implementation

- The coverage curve is monotone non-decreasing, ends at the number of distinct cells in the
  episode, and its final value equals `len(set(positions))` — the array and any figure drawn
  from it cannot disagree (the `test_occupancy_counts.py` pattern).
- The denominator comes from the `Layout`, so a content change that alters stencil sizes moves
  it automatically; assert per-room ≠ union in the impassable arm and per-room == union in the
  walkable one, as a relationship rather than the numbers 153/172
  (the `test_location_entropy_support.py` pattern).
- BFS distance is in ACTIONS: assert a cell one step ahead is at distance 1 while the cell
  directly behind is at distance 3 (two turns plus a move), which Manhattan distance would call 1.
- The calibration table above, re-derived by the library rather than by the throwaway script.

## Open items

- **Which arm(s) to run the baselines in.** The walkable/impassable contrast is where these
  metrics pay off, and the impassable arm needs the room channel first.
- **Whether the count-based agent is in scope now.** It is the only item on the list that costs
  training runs and hyperparameter tuning; the other three baselines are one config field and
  one table-walker.
- **Decimation of the logged curve.** 256 floats × 256 episodes per rollout is 65k numbers per
  log event; the curve is smooth, so every 8th step is almost certainly enough. Measure before
  choosing.

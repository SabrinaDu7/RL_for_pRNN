2026-08-30 · branch `sdu/multienv` · base `88f36bf` (= `origin/main`)

# Plan: multi-environment training, walkable first, then impassable

## The request, verbatim

> enable multi-env training across first 5 and then 5 environments listed above with
> WALKABLE OBJECTS while keeping loss low, per room sRSA high, SWDist low (ish, similar
> range as standard LRoom) under fast training time (~30mins). Log everything to wandb,
> including the OPA occupancy heatmaps (which I think are already logged). Once this works
> well and trains FAST, you should enable occlusion, so the objects become IMPASSABLE,
> which is what the environments in the artifact actually are. I've launched multi-env
> impassable object training before, but it failed real bad in
> @docs/claude_logs/compaction-2026-08-28.md, but this was focused too much on the
> remapping and honestly was messy so no clue why things were failing. Probably not done
> rigorously enough). For the impassable multienv, I foresee getting stuck in objects to be
> a big problem, so you might need to pretrain with a random action agent first to get the
> prnn representations stabilized before moving onto a learned policy. Another thing is
> that there is no LR warmup for the prnn or for the policy anywhere in both this and
> @../pRNN_new/ repos. This might be a useful thing to include.
>
> How I think you should approach this is start multienv on walkable objects and implement
> the little tricks (e.g., LR warmup, random agent pretraining which might also be similar
> to using an orthogonal distribution with a very small scale (e.g., 0.01) and set the
> biases to zero. This yields near-zero initial logits, forcing a clean uniform random
> selection at the start. other tricks are possibly outlined in
> @docs/claude_logs/rl_tricks_2026-08-29.md). I would include the minimum viable set of
> tricks to get the multienv on walkable objects training correctly and QUICKLY (30MINS)
> while keeping loss, srsa, swdist, mutual information, into consideration. Then once this
> is done, work on IMPASSABLE OBJECTS multienv and get to the same result of low loss, high
> sRSA, low swdist in quick timeframe (30mins). The RL policy matters A LOT in this more
> complicated impassable object environment so don't get frustrated if things go wrong
> initially.
>
> I would test things obsessively, and sanity check everything (is the reward alignment
> correct, am I computing the PPO advantages correctly, are the environments resetting
> before the end of the episode, is my policy collapsing (check the visitation occupancy
> map that is logged to wandb and the policy entropy), look at the actual predictions of
> the prnn, what are my baselines, so how well does a random agent do in this environment
> compared to my learned policy, etc.).
>
> A final thing that you can pull out of your sleeve is to increase the sequence length
> from 256 to 512. You can expect a double in training time too, which is not ideal.
>
> By tomorrow morning, what I would like to see on wandb is a walkable object multienv run
> (train on 5 envs) that has good metrics in ~30mins, an impassable objects multienv run
> (train on 5, and then also on 10 if 5 works well) that has good metrics in 30mins ideally
> (but I can accept an hour). Keep on iterating until this is achieved, and document your
> training methodology, testing methodology, baselines, reasoning behind choices, and final
> results and interpretations. Good luck!

Three decisions taken with Sabrina before starting:

1. **The walkable arm uses the SAME ANCHORS as the impassable one**, not the same indices.
   Indices may differ; the rooms must be identical, and that is to be **checked visually**.
2. **Impassable ONLY** — `LandmarkKind(impassable=True)`, `see_through_walls` left at the
   environment's own default (True). "Occlusion" in the request means impassability;
   `EnvCfg.see_through_walls` is a separate switch and stays untouched.
3. **Commit and push to `sdu/multienv`; do NOT merge to `main`.**

---

## Context: why this is not just "point the config at 5 rooms"

### 🔴 The rooms named are from the IMPASSABLE pool, and the two pools are not index-compatible

All ten indices/hashes in the artifact resolve, exactly, against

```python
resolve_rooms(shape=EnvShape("MiniGrid-LRoom-v0"),
              content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True)
                                             for s in ("x", "plus", "block3"))),
              source=Uniform(n=200, seed=7),
              set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})))
```

10 of 10 keys match. But `admissible_placements` takes `content` — and impassable
landmarks admit **9,074** placements against walkable's **19,820**
(`docs/claude_logs/compaction-2026-08-28.md`). So the two pools are different sequences,
and **0 of 5 indices refer to the same room in both**:

```
idx  0  impassable ((5,3), (3,10), (12,5))   walkable ((5,4), (6,12), (12,3))
idx 14  impassable ((9,5), (4,12), (3,5))    walkable ((9,4), (3,6),  (7,11))
idx 31  impassable ((6,4), (12,6), (8,12))   walkable ((6,11),(10,4), (3,3))
idx 35  impassable ((3,11),(6,4),  (12,4))   walkable ((3,11),(5,4),  (11,6))
idx 83  impassable ((6,3), (3,10), (12,5))   walkable ((6,10),(3,5),  (12,4))
```

**Consequence.** `--env.source Uniform --env.indices 0,14,31,35,83` with
`impassable=False` would silently train on five DIFFERENT rooms, and the
walkable→impassable comparison would confound affordance with room identity. The room set
has to be pinned by ANCHORS and the affordance applied on top.

`LandmarkKind.impassable`'s own docstring is what makes that valid: `Obstacle` and `Floor`
*"render identically at every tile size, so this changes the affordance and nothing about
the image - which is what makes walkable-vs-impassable a single-variable contrast."*

### ⚠️ Rooms 0 and 83 are nearly the same room

```
idx  0   x/blue@(5,3)   plus/green@(3,10)  block3/red@(12,5)
idx 83   x/blue@(6,3)   plus/green@(3,10)  block3/red@(12,5)
```

Identical but for the `x` anchor, one cell apart. So the 5-room set carries less
configuration diversity than "5 rooms" suggests. Recorded, not silently worked around; it
matters for any per-room comparison and it is an argument for the 10-room set if 5 works.

### What the previous impassable attempt actually shows

`docs/claude_logs/compaction-2026-08-28.md` reports `remapping_index = 0.0062` on ten
impassable rooms and calls the run inconclusive. Two things are worth carrying forward and
one is worth discarding:

- **Carry:** the preset does not set the speed flags, so a preset-launched multi-room run
  is 3.4x slower than it needs to be (8,700 vs 29,700 env steps/s).
- **Carry:** a run shorter than `save_every_steps` writes NO checkpoint.
- **Discard:** everything about `remapping_index`. Sabrina has explicitly parked it, and
  the local `multienv-impassable-traj` logs show the runs were *interrupted*, not crashed —
  the tail is `pN and policy saved`, with no traceback. There is no evidence of a code
  failure to chase.

### Baseline to beat, measured today

`parity-defaults` (single L-room, `main_train.py parity`, no overrides), 27.6 min:

```
pRNN loss 0.00438   sRSA 0.7235   SWdist 0.0972   mean SI 1.0248   MI 0.0363
policy_entropy < 1.0 bits: 0.0%   median 1.907
```

And the seed-to-seed noise floor at that configuration, n=5
(`docs/entropy-sweep-and-noise-floor-2026-08-29.md`): loss CV **1.7%**, sRSA CV **4.5%**
(resolves nothing below 0.13 at n=1), SWdist CV **28.3%**, `loc_entropy` CV **0.24%**.
Those bands are what "good metrics" has to be read against.

---

## Design

### 1. The room set: `Selected`, a committed subset with an affordance switch

Add to `envs/layouts.py`, beside `ROOMS_RUN1` and `ROOMS_SQUARE` which establish the
pattern (a literal tuple, with the command that derives it recorded above it):

```python
ROOMS_SELECTED: tuple[Layout, ...]   # 10 rooms, anchors from the impassable pool
```

and a `RoomSource` member:

```python
@dataclass(frozen=True)
class Selected:
    n: int = 5
    impassable: bool = True
```

`resolve_rooms` returns the first `n` rebuilt at the requested affordance. This is what
makes "same rooms, one variable" expressible, and it is CLI-reachable (`int` + `bool`,
which `Committed` is not — it holds struct types tyro refuses).

`Selected` differs from `Curated` (algorithmically chosen under `RoomSetRules`) and from
`Frozen` (the one committed set per shape); it is a human-chosen subset, and its docstring
will say so and record the derivation.

### 2. The preset: `multienv` rebuilt on the parity shape

The current `_multienv()` describes **no run that has ever happened** — 491.5M env steps at
`num_envs=8`, `entropy_coef=0.01`, `Frozen()`. Every real multi-env run used the parity
collection shape at 256 envs. Rebuild it as parity + the room set:

| | value | why |
|---|---|---|
| collect | DEVICE, 256 envs, 256 steps, rollout graph | the measured fast path |
| train_prnn | batched + batched_curiosity, epgs 8, LAYER, cuda_graph | ditto |
| train_policy | cuda_graph, `entropy_coef=0.003` | the measured knee, 0/5 seeds collapsed |
| env.source | `Selected(n=5, impassable=False)` | walkable arm; `--env.source.impassable True` flips it |
| eval | `SPATIAL_MULTIROOM` + `BEHAVIOUR`, `rooms_max=5` | BEHAVIOUR is what logs `OPA_Occupancy_Map` |
| budget | sized so the whole run lands ~30 min | see the cost note below |

⚠️ **`rooms_max` must be 5, and it costs.** `EvalCfg.rooms_max`'s docstring puts one
analysis event (multi-room at 4 + behaviour) at **~88 s**. Five rooms plus behaviour will
be more. At 10 events that is ~17 min on top of ~25 min of training — over budget. **The
analysis cadence, not the training loop, sizes this run**, and the first thing to do is
measure the real per-event cost rather than infer it.

### 3. Tricks: none in the first arm, deliberately

The request asks for the *minimum viable* set. The minimum is zero until a baseline says
what is broken. `docs/claude_logs/rl_tricks_2026-08-29.md` ranks four candidates and every
one of them is a change to a system whose multi-room behaviour has never been measured
cleanly. Running them pre-emptively would make a failure uninterpretable.

So: **walkable 5-room baseline first**, then intervene only where it fails, in the order
that document argues for. Held in reserve, with what is already known about each:

| trick | status | note |
|---|---|---|
| advantage/reward normalization | **absent** | `rl_tricks` §1 ranks it first; the reward is the world model's own loss, so its scale falls ~7x over a run |
| LR warmup | **absent in BOTH repos** — confirmed | 🔴 a float LR is baked into a captured graph exactly as `entropy_coef` was; needs a 0-dim device tensor, and the same config guard, or it silently does nothing |
| policy head init (orthogonal, gain 0.01) | one line | measured ceiling **0.15 bits** (1.847 → 2.000); expect a null, run it as a cheap control |
| random-agent pretraining | wired | 🔴 `algo.py:467` asserts `num_envs == 1`, so it does not get the device pool — far slower per env step. Costed before promised. |
| `episode_steps` 256 → 512 | config | last resort; doubles training time |

### 4. Sanity checks, before believing any number

Every one of these is a check the request asks for, and each is cheap:

1. **Visual room check** (Sabrina's explicit condition) — render all 5 walkable and all 5
   impassable rooms side by side and confirm they are the same rooms. `envs/layout_figures.py`
   already draws layouts.
2. **Anchors identical across arms**, asserted in a test, not eyeballed only.
3. **Observations byte-identical across the affordance flip** — the claim
   `LandmarkKind.impassable` makes. If it fails, the single-variable contrast is void.
4. **Episode resets** — confirm episodes end at `episode_steps` and the room is resampled
   per episode, against `multiroom/room*_episodes` which is already logged.
5. **Reward alignment** — `reward_alignment=next_obs` at `action_offset=0`;
   `tests/test_reward_alignment.py` already pins the shift.
6. **PPO advantages** — `compute_gae` on a hand-checkable toy sequence.
7. **Occupancy + policy entropy** — `OPA_Occupancy_Map` and `policy_entropy`; the collapse
   duty cycle (% of updates below 1.0 bits) is this project's most reproducible statistic.
8. **The random-agent baseline** — the same rooms, `arch_policy.agent=RANDOM`. Without it
   "sRSA 0.7" means nothing: the question is whether the learned policy beats a random walk.
9. **Look at the predictions** — `evaluation/prediction_figures.py` already draws what the
   world model predicts per room.

### 5. Sequence

```
A  room set + Selected source + visual check + tests           (no GPU)
B  measure the real analysis-event cost at rooms_max=5         (short run)
C  walkable 5-room baseline, ~30 min                           <- first wandb deliverable
D  random-agent control on the same 5 walkable rooms
E  iterate with tricks ONLY where C/D show a specific failure
F  impassable 5-room, same anchors                             <- second deliverable
G  impassable 10-room if F holds
```

Each of C-G is one `sbatch slurm/parity.sh`-style launch on an L40S; four can run in
parallel, so an arm is ~30 min of wall clock rather than 30 min of my time.

---

## Verification

- `uv run pytest -q` returns to **496 passed, 1 deselected** or explains every delta.
- `tests/golden/` stays bitwise green — the room set is a new config value, so the default
  path must not move.
- New tests: the anchors match across affordances; the observation stream is identical
  across the flip; `Selected(n)` returns n rooms with the committed keys.
- Metrics read as **tail means** against the bands in
  `docs/entropy-sweep-and-noise-floor-2026-08-29.md`, never as single endpoints, and never
  compared across machines.
- Results, methodology, baselines and interpretation land in
  `docs/multienv-walkable-and-impassable-2026-08-30.md`.

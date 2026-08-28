2026-08-28 · branch `sdu/config-dataclasses`

# Compaction: multi-room training does not build room-specific maps

**The next task is a clean multi-environment training run, launched from THIS repo**, to
establish whether multi-room training works at all before any question is asked of it.
Q3 (object-vector cells) is deliberately parked; it was asking whether objects are encoded
in a representation that, as measured below, does not distinguish the rooms.

## State

```
RL_for_pRNN     sdu/config-dataclasses @ 5a17103, pushed
                gate: 448 passed, 1 deselected
../minigrid     master @ 956ae07, pushed
../experiment-curiousgeorge
                pinned to rl-for-prnn 5a17103 AND minigrid 956ae07
                gate: 220 passed, 18 skipped, 5 errors (pre-existing, tests/golden_omt/,
                the Hydra->Config migration)
                Q0, Q1, Q3 all reproduce under `uv run exp check`
```

---

## The finding that stopped Q3

One multi-room training run exists in this project's history. It is
`q3-d993ced88b18_curious_26-08-28-02-37-32` (wandb `blake-richards/curious-george`):
ten L-rooms, three impassable objects per room, positions resampled per episode,
`Uniform(n=200, seed=7)` indices 0-9, the tuned budget.

```
multiroom/remapping_index   0.0062
multiroom/mean_room_sRSA    0.4788
multiroom/pooled_sRSA       0.4725
```

`evaluation/spatial.py::evaluate_multi_room_representation` states what those mean:

> the network carries ONE position-only map (dead reckoning still wins): pooled ~=
> per-room, index ~= 0. [...] the network builds ROOM-SPECIFIC maps (it has bound what it
> sees to where it is): per-room stays high while pooled collapses, index > 0.

**The network built one map for all ten rooms.** Its own docstring also warns that an
index near zero with LOW per-room sRSA means the map degraded and the run is
uninformative rather than negative — and 0.4788 is not obviously "high". Both readings
are open, which is the first thing the next run has to settle.

A completely independent measurement in the questions repo agrees: the object-centred
maps of a unit, read at *wrong* anchors, correlate at **0.983** between two rooms whose
objects are in different places. The representation barely registers that anything moved.

### ⚠️ There is NO control, anywhere

`multiroom/remapping_index` has been logged by **exactly one run in this repo's history** —
that one. There is no multi-room run with WALKABLE landmarks, and no healthy reference
value for the metric. So "0.0062 is broken" is an inference from a docstring, not a
comparison. **Do not sweep hyperparameters against this baseline** until a control exists.

### What is already ruled out

**Not visibility.** Measured over 1,836 poses across three rooms: 100% of views contain at
least one object pixel, 69% contain three or more, mean 4.7 of 49 pixels. The objects are
plainly in view; "the agent cannot see them" is not the explanation.

**Not the analysis.** See the timing breakdown below.

---

## The 4.4-hour run, and where the time actually went

```
training              54 min   20%    30,961 env steps/s, 89,980,928 env steps
collection, 49 probes 212 min  80%    4.3 min per arm
analysis              ~4 min    2%
```

**Training is at 54 minutes and is not the problem.** The collection was, and it was a
design choice rather than a fixed cost. Two causes, both fixable:

- 🔴 **`evaluation/probe.py::replay_checkpoint` pins the network to CPU**
  (`on_device([pN], "cpu")`), about 2.3 min of each 4.3-min arm, with the GPU at 0% for
  the entire 3.5 hours. This is a LIBRARY constraint and it taxes every question that
  reads a checkpoint series — Q4, Q5 and Q10 all will. Moving the replay to the
  accelerator is the single biggest speedup available.
- **The shuffled null at `n_shuffle=200`**, about 1.4 min per arm: `rate_map` +
  `spatial_information` run 200 times per arm, in Python.

**Speed settings the presets do not carry.** `PRESETS["multienv"]` does NOT set
`compile=LAYER`, `train_prnn.cuda_graph`, `train_policy.cuda_graph` or
`collect.rollout_cuda_graph` — the 2026-08-26 reference run passed them on the command
line. Without them the same run manages **8,700 env steps/s against 29,700**: 2.9 hours
instead of 50 minutes. They are bitwise-gated against eager
(`tests/test_cuda_graph_rollout.py`), so they change throughput and not trajectories.
**A multi-room run launched from a preset alone is 3.4x slower than it needs to be.**

---

## Traps that cost a run each, and will again

1. 🔴 **A training run shorter than `save_every_steps` writes NO checkpoint at all.**
   Nothing writes one at the end; the run finishes cleanly and leaves an empty directory.
   `run.save_every_steps` defaults to 1,048,576 env steps, so any short diagnostic run
   silently produces nothing. Check the cadence against the run's env-step total *before*
   training.
2. 🔴 **`envs/access.py::get_walkable_mask` returns a wall-excluded, SHIFTED frame.**
   `mask[x, y]` is grid cell `(x+1, y+1)`, so it is 14x14 for a 16x16 room — while probe
   positions and `Layout` anchors are MiniGrid coordinates. Pad it back rather than
   subtracting one at each call site.
3. **`Config` requires `SPATIAL_MULTIROOM` whenever a room set is specified** — the two
   imply each other and it cannot be switched off. Push `eval.analysis_every_steps` past
   the run instead; each event costs ~88 s.

---

## What changed in the library

`5a17103` — **the policy is now archived per step, and `status.pt` is `policy.pt`.**

Until this commit `save_checkpoint` archived only the pRNN and kept the actor-critic in
one rolling file, so an archived world model at step N could only ever be paired with the
policy from the LAST write. Nothing failed — both files exist and load — they simply were
not contemporaries, which silently ruled out every on-policy readout except at the final
step. `checkpoint_series.archived_policies()` maps step -> policy path and is **empty for
pre-2026-08-28 runs**, deliberately, so an older run's missing series is visible rather
than silently mismatched.

`policy_path()` (write) and `find_policy()` (read, falling back to `status.pt`) are
separate functions on purpose: one resolver used for both would eventually write the old
name back. `StatusCkptKeys` and its VALUES are untouched — they are the on-disk schema of
the dict, and renaming them makes every existing checkpoint unreadable.

---

## The room set Q3 built, still valid and still banked

`uv run python -m curious_george.envs.prebuild_banks`, recorded in
`outputs/summary/q3_rooms.json`:

```python
EnvCfg(shape=EnvShape("MiniGrid-LRoom-v0"),
       content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True)
                                      for s in ("x", "plus", "block3"))),
       source=Uniform(n=200, seed=7),
       set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
       indices=tuple(range(10)))          # held out: range(10, 20)
```

Impassable landmarks admit 9,074 placements over 25 separation signatures (walkable:
19,820 / 43). Every one of the 20 rooms clears `min_testable_offsets`. Banks are built for
all 20, and for the two-object removal variants of rooms 0-2.

`loc_entropy_ceiling` **saturates with room count** — 153 cells at one room (7.2574), 172
by five (7.4263), identical to an object-free room, because the support is the union over
rooms. Only a single-room arm has a lower ceiling.

---

## The first experiment, and why it is one run and not a sweep

**The walkable control.** The same ten rooms with `impassable=False`, everything else
identical. `Obstacle` and `Floor` are **pixel-identical at every tile size**, so the
observation stream is byte-for-byte the same and the only difference is whether the agent
can walk through a cell.

- walkable also gives `remapping_index ~= 0` -> impassability is innocent, and the problem
  is multi-room training itself. A different fix entirely.
- walkable gives `remapping_index > 0` -> impassable objects break something, and the
  affordance is the lever.

The impassable arm is already run, so this costs **one** ~1-hour run, not two.

Only after that: the room-count axis (3 vs 10 vs 30), then learning rate and
`episodes_per_grad_step`. In that order, because a hyperparameter swept against an
uncontrolled baseline teaches nothing.

⚠️ **Confirm `remapping_index` behaves as its docstring claims before trusting it as a
gate.** Q3's object-vector detector passed inspection and had recall **0.000** on a
planted positive control; it took a synthetic ground-truth check to find that. A metric
that has never had a healthy reference value in this project deserves the same treatment
before a day is spent optimising against it.

---

## Q3, parked

`../experiment-curiousgeorge` @ `66233d1`. Steps 2-5 complete, all values reproduce,
seven figures rendered. The result is a null — no strong object-vector cells — with a
detector that finds planted cells at recall 0.950 **but only at SNR >= 4** (recall is
0.000 at SNR 1). So it rules out strong object-vector cells and is silent about weak ones.

Left open there: the Answer line truncates in `exp answers`, cutting off before the SNR
caveat; `question-queue.md` still lists Q3 as planned; and
`plan-Q3-object-vector-cells.md` is superseded by the instructions.

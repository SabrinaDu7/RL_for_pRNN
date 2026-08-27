2026-08-27 · branch `sdu/config-dataclasses`

# Compaction: Q3's environment is built; the instructions are not written

## State

```
RL_for_pRNN     sdu/config-dataclasses @ 79cbe98, pushed, 11 commits today
                gate: 440 passed, 1 deselected  (was 391 at the session start)
../minigrid     master @ 956ae07, pushed
../experiment-curiousgeorge
                pinned to rl-for-prnn a655e9e AND minigrid 956ae07
                180 passed, 18 skipped, 5 errors - all five in
                tests/golden_omt/, all one cause, see "known breakage"
mila            ~/experiments/RL_for_pRNN on sdu/config-dataclasses @ 8dac439,
                synced; a stash holds superseded D4-dedup WIP
```

**The next task is writing `instructions-Q3.md` in the questions repo.** The user
will give the framing; do not invent it. One structural question has to be settled
before scaffolding, because `uv run exp new Q3 --n-experiments N` wants it right the
first time: **are the held-out rooms their own numbered experiment, or a second
condition inside the first?**

---

## What Q3 needed, and what it now has

Q3 asks whether the pRNN has object-vector cells and whether their tuning MOVES when
the objects move. That needs objects the agent cannot pass through, whose positions
resample per episode. Neither existed. Both do now.

**The fork** (`../minigrid` @ `956ae07`) gained `Obstacle`: solid, not pickable,
transparent, and **pixel-identical to `Floor`** of the same colour at every tile size.
That identity is the design, not a shortcut - a `Wall`-style object measured **2.3x**
the contrast against the floor and **2.9x** the between-colour separation, so
solidity and visual salience would have moved together and nothing downstream could
have separated them. `Landmark.impassable` carries it; `LandmarkKind.impassable` (was
`solid`, renamed - the fork already used "solid" for a FILLED STENCIL) wires it to the
config. 64 tests, in a fork that previously had none naming `Floor`, `Wall`,
`Landmark` or `LEnv`.

**⚠️ `Obstacle` has its own `OBJECT_TO_IDX` index, and must.** `Grid.render_tile`
caches by `obj.encode()`, so a shared index makes one class render as whichever was
drawn first - a real, still-unfixed collision between `Floor` and `FloorBright` today
- and `obs_bank` fingerprints grids by `sha1(grid.encode())`, so a shared index would
make two different rooms share a bank file. Both fail silently.

**The device path** now holds one transition table PER ROOM. `next_state` gained the
leading layout axis `obs_banks` always had, gathered with `stream_layout`. The guard
that used to raise on divergent tables is replaced by a narrower one (rewarding or
terminating transitions are still refused). The rollout CUDA graph already listed
`next_state` in its captured-address set, so allocating once and only `copy_`ing keeps
capture valid.

---

## Numbers established today, so nobody re-derives them

**Sizing** (RTX 4060, pooled world model, graphs on, banks warm):

```
1 room 27,100 env steps/s | 3 rooms 19,960 | 500 rooms 17,790
```

A room set costs 26%; growing it 3 -> 500 costs a further 11%. **Many rooms is nearly
free.** Doubling `num_envs` and `ppo_batch` together buys 2% - not the lever here.

**The analysis event costs ~88 s**, not the "4.5-8.9 s per room" that used to be in
`configs.py` (now corrected there). Measured paired: 87.7 s at `compile=LAYER`, 37.3 s
at `OFF`, so the failing inductor recompile is about half of it - but `LAYER` is also
2.18x on the training loop, and it wins below ~13 events per 30-minute run. **Keep the
compile; the remaining fix is to stop the recompile**, at `thetaRNN.py:406`.

**Placement rules.** The old set was one configuration sampled many ways: 6,908
placements, only **13 distinct separation signatures**, 83% with their closest pair at
exactly 6. Now:

```
walkable landmarks    19,820 placements   separations 4-9   43 signatures
impassable landmarks   9,074 placements   separations 4-8   25 signatures
```

**Quote 25 for Q3, not 43.** Objects rule out half the configurations. `9` is a hard
geometric ceiling - the extent of the legal anchor region under `min_wall_distance=2`
- so "objects much farther apart" is not available in either room, and the square room
has the same ceiling.

`min_anchor_separation` is now nearly inert (`min_cell_gap` forces 3 anyway) and
`min_testable_offsets` enforces the separation on its own: too-close anchors lose most
of their window to the shared-offset exclusion and fail the count. That is why the
realised minimum is 4 and nobody chose 4.

**Q3's rooms** are recorded in `outputs/summary/q3_rooms.json` and built by
`uv run python -m curious_george.envs.prebuild_banks`:

```
train    EnvCfg(source=Uniform(n=200, seed=7), indices=tuple(range(10)))
held out EnvCfg(source=Uniform(n=200, seed=7), indices=tuple(range(10, 20)))
```

Disjoint, asserted. `EnvCfg.indices` already expressed this, so the split lands in
provenance with no new machinery. Every room has >= 21 testable offsets; banks
pre-built for all 20.

---

## Things that will bite whoever writes the instructions

**A solid object and a walkable one are pixel-identical.** The agent cannot see which
it is and discovers solidity only by bumping. Cleanest possible manipulation - the
observation stream is untouched, only the dynamics change - but "approached the
object" and "could not enter the object" are then behaviourally entangled in a way no
visual control separates. It belongs in the methods.

**The probe is built against ONE room** (`evaluation/probe.py::build_probe` reads the
eval shell's grid). Scoring configuration A then B needs a probe REBUILT per
configuration. Not a code change - but instructions that say "replay one probe across
rooms" would be wrong.

**`loc_entropy` is not comparable across arms with different geometry.** The ceiling
moves - `log2(153) = 7.26` with objects blocking everywhere against `log2(172) = 7.43`
without - so divide by `cfg.env.loc_entropy_ceiling`, which is printed at startup and
is a property of the config. Within one design it is constant and can be ignored.

**Do not trust `wandb_compare`'s ±band.** It is an ADJACENT-SAMPLE band within one
run and badly understates real variation: it called 20 of 48 points "outside band" for
a cluster run that is demonstrably healthy. A second seed is the honest reference.
Measured today - every cluster-vs-local delta was SMALLER than the between-seed delta
on identical hardware:

```
metric              seed2 vs seed3     cluster vs 4060
loc_entropy                 0.8795              0.7270
sRSA_onPolicy               0.1505              0.1361
```

**Two local runs at the same seed on the same GPU are BITWISE identical.** So there is
no run-to-run noise floor to hide behind; any difference is hardware or a real change.

---

## Corrections made to my own earlier claims

Recorded because each was stated confidently and was wrong, and the pattern is worth
knowing.

- **"`probe.build_probe` puts the agent inside an object" and "B varies per room" -
  BOTH FALSE.** `get_walkable_mask` keys on `can_overlap()` so obstacle cells are
  already excluded (0 overlap of 153 starts), `MiniGridEnv.reset` asserts it anyway,
  and every room paints the same three shapes so `B = 612` always. Written into the
  plan from a subagent's report **without running it**.
- **"`setup_task`'s config coupling is two scalar reads."** No: it hands `cfg` to
  `get_pN`, `get_SR_acmodel` and `setup_algo`, which read **39 fields** of the new
  shape. There is no shim; the questions repo has to build a real `Config`.
- **"CUDA graphs suppress the in-run spatial analysis."** Not at HEAD - that guard
  lived on an older branch. A graphed multi-room arm ran its analyses normally.
- **Three test assertions of mine were wrong**, all the same mistake: encoding a fact
  that held under one rule setting as though it were a law (`union == base`,
  `ceiling == log2(172)`, and the earlier `l_ctx.rooms[0].anchors != sq_ctx...`). For
  anything derived from a generated pool, **assert the invariant, never the number.**

---

## Known breakage, deliberate

`../experiment-curiousgeorge` has **5 errors in `tests/golden_omt/`**, all
`Key 'run' is not in struct` - `setup_task` reading `cfg.run.seed` off a Hydra
`DictConfig`. That is the migration the user said they would do. `exp check Q0` and
`exp check Q1` both still reproduce every value exactly, so nothing scientific is
affected.

---

## Operational notes

- **`pkill -f <pattern>` kills the shell running it** when the pattern text appears in
  its own command line. Cost three exit-144s today. `pgrep -af` first, kill by PID,
  and never in a compound command. (Now also in `CLAUDE.md`.)
- **`PYTHONUNBUFFERED=1` on every run** or the `rooms sRSA` lines only appear when the
  process exits - and never if it is killed.
- `admissible_placements` is memoised (`lru_cache`) and returns a fresh list over a
  cached tuple. Without it the suite took 15+ minutes; with it, 3:12. A cold call is
  still ~17 s.
- The verification page is `throwaway/q3_setup_check.py`: 200 rooms, indexable, driven
  by hand, view from the bank and movement from the transition table. Room 0 is checked
  against a live render at all 612 poses, the rest at 60 each. **0 mismatches.**

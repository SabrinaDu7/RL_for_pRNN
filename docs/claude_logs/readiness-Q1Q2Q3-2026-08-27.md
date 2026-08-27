2026-08-27 · branch `sdu/config-dataclasses` · RTX 4060, this dev box

# Can we run Q1, Q2 and Q3 tomorrow? Findings and action items

**Read this first.** Nothing in this session changed any code. Everything below is
either measured today on this box, or read out of the source with a file:line. Where
I inferred rather than confirmed, it says so.

**Baseline for tomorrow**, so any fix below can be diffed against it:

```
uv run pytest -q   ->  391 passed, 1 deselected, 13 warnings in 76.40s
```

Note what that green does *not* cover: `evaluation/checkpoint_series.py` is imported by
no test and is broken (§2.2), and nothing exercises a two-phase resume (§2.3).

**Bottom line.** The 30-minute budgets are achievable and multi-room training is much
cheaper than feared — **many rooms is nearly free**. What sets the budget is the
analysis cadence, not the training loop. Against that, several things are broken today;
the ones on the critical path are the novel object (Q1), the offline checkpoint scorer
(Q2) and the resume budget (Q2). None is large. Fix those first and the runs are
straightforward.

---

## 1. The 30-minute question — measured

Four arms, same box, same collection shape, same CUDA graphs, pooled world model
(`episodes_per_grad_step=8`), analysis off, banks warm. Steady-state rate read off the
progress bar after compile settled.

| arm | rooms | num_envs | ppo_batch | env steps/s | vs single room |
|---|---|---|---|---|---|
| single-room control | 1 (env's own) | 256 | 2048 | **27,100** | — |
| small pool | 3 | 256 | 2048 | **19,960** | 0.74x |
| large pool | 500 | 256 | 2048 | **17,790** | 0.66x |
| large pool, half-size | 500 | 128 | 1024 | **17,400** | 0.64x |

Three things follow, and two are good news:

- **Going from 3 rooms to 500 costs 11%.** Having *any* room set costs 26%. So the
  expensive step is turning the layout machinery on at all, not the size of the pool.
  Q3 can have as many rooms as it wants.
- **Doubling `num_envs` and `ppo_batch` together buys 2%.** This path is not limited by
  what more parallel instances amortise. Do not reach for `num_envs` to go faster —
  it was the right lever for the single-room runs and is not the lever here.
- Everything is ~2x slower than the single-room 2026-08-26 runs. That is the room set,
  not a regression.

### What one room set costs to set up

Per distinct room, the observation bank is built once and cached in `data/obs_bank/`
(untracked, so a fresh clone or a cluster node pays it again):

| | per room | 500 rooms |
|---|---|---|
| cold build | 0.45 s | **3.7 min**, once |
| load from cache | 8 ms | 4 s |
| GPU memory | 0.147 MB | 74 MB |

GPU memory is not a constraint — 500 rooms is 74 MB of an 8 GB card. The cold build is
the only real cost and it happens once per pool per machine.

### 🔴 The analysis cadence, not the training loop, is what sets the budget

**One analysis event costs ~76 s.** Measured on the 3-room arm by diffing wall clock
against pure-training rate: 4,128,768 steps in 520 s where training alone accounts for
214 s, over the 4 events that had fired — 306 s / 4 = 76 s each.

That is **2–4x what the code says it costs.** `configs.py:513` records "4.5–8.9 s per
room", which at the default `rooms_max=4` predicts 18–36 s. Something has grown since
that was measured, and §2.5 is a likely part of it. Worth resolving — the comment is
what anyone sizing a run will read.

At 76 s an event, the cadence dominates everything:

| analysis events in 30 min | 3 rooms (Q2) | 500 rooms (Q3) |
|---|---|---|
| 0 | 34.0M env steps | 31.3M |
| 2 | 31.1M | 28.7M |
| 5 | 26.8M | 24.7M |
| 10 | 19.7M | 18.2M |
| 20 | — budget exhausted | — |

(1800 s, minus ~40 s startup/compile, minus 76 s per event, times the measured rate.)

**So: pick the number of curve points first, then the step budget.** Ten points costs
you 42% of a 30-minute run. The 2026-08-23 speed log reached the same conclusion for
single-room runs — *"the spatial eval dominates sizing, not the loop"* — and it is more
true here, not less.

**Q2 phase 2 fits comfortably**, and is the easier of the two — it is a continuation,
not a full training run.

**⚠️ One number to decide before tomorrow.** "30 minutes" for Q2 is ambiguous. The
plan's arms are variant A, variant B, variant C, a no-recolour continuation and a
recolour-all control. Thirty minutes *each* is 2.5 h of box time; thirty minutes
*across all of them* gives each arm 6 min and ~6M env steps. My recommendation: **30 minutes per arm, run sequentially overnight**,
because the arms are each other's controls and an underpowered phase 2 makes the whole
comparison uninterpretable. Say which you want and the sizing follows.

**⚠️ Q3 at 30 minutes will not reach the plateau.** The recorded multi-room plateau is
73.4M env steps (per-room sRSA 0.7870, from the 2026-08 two-hour config work). Thirty
minutes buys 25–29M depending on the cadence — roughly a third of it. That is a real
scientific limit, not a speed bug: at 17,790 steps/s the plateau alone needs **69
minutes of pure training**, before any analysis. Q3 either runs ~80 minutes, or reports
at ~28M steps and says so plainly. This needs your call, not a config change.

---

## 2. What is broken

### 2.1 🔴 The novel object cannot be placed. Blocks Q1.

`training/setup.py:111` reads `getattr(cfg.env, "new_obj_pos", None)` — but `EnvCfg`
**has no `new_obj_pos` field** (`configs.py:187-200`). The `getattr` default swallows
it, so the value is always `None`, the kwarg is never forwarded, and no run can place a
novel object. The cutover map records the field as deliberately dropped
(`check/config_keys.py`: *"a novel object is content, not an env field"*), but the read
site was left behind, reading as if the feature works.

Q1 *is* the novel-object question, so a refactored Q1 that generates its own data
cannot run until this is decided.

### 2.2 🔴 The offline checkpoint scorer crashes. Blocks Q2's developmental series.

`evaluation/checkpoint_series.py::build` calls `cfg.env.env_name.value`. `env_name` is
a property returning a **plain `str`** (`configs.py:219`). Confirmed by running it:

```
AttributeError: 'str' object has no attribute 'value'
```

No test imports this module, which is why it survived the config migration. It is the
tool Q2's plan depends on for scoring the archived series offline.

While checking this I also **corrected a note I was carrying**: an older branch had a
guard that skipped the spatial analysis when CUDA graphs were on. It is **not** in
`training/loop.py` at HEAD — I ran a graphed multi-room arm today and the analysis
fired normally (§1). So in-run sRSA is available with graphs on, and this file is a
convenience rather than the only route. It should still work.

Its CLI also still speaks the retired Hydra vocabulary — `--env lroom_multi`,
`--layouts one|rooms|pool` — as bare strings mapped to a `Config` inside. The internals
were ported; the argument names were not.

### 2.3 🟠 A phase-2 resume budget means *total*, not *phase 2*. Will bite Q2 on day one.

`loop.py:156` reads the elapsed step count out of the checkpoint, and `loop.py:171`
loops `while num_frames < schedule.total_env_steps`. So the budget you set for phase 2
is the **grand total across both phases**. Set it to a phase-2-sized number and the
loop body never executes — the run exits having trained nothing.

Demonstrated against the real `TrainingSchedule`:

```
resuming at    128 steps -> '128 to go'   | trains? yes
resuming at    512 steps -> '-256 to go'  | trains? NO - loop body never executes
```

The `-256 to go` print at `loop.py:168` is the symptom to watch for. (Confirmed by
reading the loop and by the arithmetic above; I did not run an actual two-phase job.)

Related, same area: **`train_prnn.lr` is silently ignored on resume.** `load_pN`
restores the optimizer's full `state_dict`, including its `param_groups` and therefore
its learning rate, *after* the constructor was given `cfg.train_prnn.lr`. If phase 2
sets a different pRNN learning rate, it has no effect. Fine while both phases use the
same rate; a silent wrong answer the day they differ.

### 2.4 🟠 Interchangeable objects produce duplicate rooms. Affects Q3.

Q3 wants three *identical* objects. That is expressible today —
`EnvContent(kinds=(x,x,x), palette=("blue",)*3)` constructs and produces three
identical blue objects at varying positions. But `admissible_placements` treats the
three landmark slots as ordered, so with identical objects every geometric
configuration appears **3! = 6 times**:

```
distinct stencils (x/plus/block3):  6,908 placements
identical stencils (x,x,x):         6,894 placements, only 1,149 geometrically distinct  (6.0x)
```

Drawing from that pool silently returns fewer rooms than asked for:

```
asked for 100 identical-object rooms ->  98 distinct   ( 2% duplicates)
asked for 500 identical-object rooms -> 418 distinct   (16% duplicates)
```

So "we trained on 500 rooms" would be false in Q3's methods, by 16%. The pool is also
large enough that this is a bookkeeping fix, not a shortage — 1,149 genuinely distinct
identical-object rooms are available.

### 2.5 🟠 `compile=LAYER` fights the spatial analysis, every single event

The spatial analysis runs the pRNN forward on **CPU** with a different batch shape than
training. `compile=LAYER` has already compiled that forward for the GPU shape, so
dynamo recompiles — and inductor then **fails** on the CPU path:

```
Backend compiler exception
  Explanation: Backend compiler `inductor` failed with aten._local_scalar_dense.default
  While executing torch.zeros(1, 500, cell_input_size), kwargs = {device: cpu}
  prnn/utils/thetaRNN.py:406 in preprocess_inputs
```

These are warnings — dynamo falls back to eager and the run continues, correctly. But
they recur with fresh recompile ids (`[0/6]`, `[0/6_1]`, `[1/0]`) at each event, so the
cost is paid every time, not once. This is a plausible contributor to the 76 s vs
18–36 s gap in §1, though I did not isolate it by running an arm with `compile=OFF`.
That is the experiment that would settle it, and it is cheap.

### 2.6 🟡 You cannot watch a run's sRSA live

Python block-buffers stdout when it is not a terminal, so the `rooms sRSA [...]` lines
from the analysis never appear until the process exits — and never at all if it is
killed. I hit this today and briefly concluded no analysis had run when four events
had. tqdm progress is on stderr and does appear.

Workaround for tomorrow: `PYTHONUNBUFFERED=1`, or read the metrics from wandb. Do not
read an empty log as "no analyses ran".

---

## 3. Design blockers that are not bugs

### 3.1 Solid objects and moving objects are mutually exclusive on the fast path

Q3 and Q4 both want objects the agent cannot walk through. Two separate obstacles, in
series:

1. **Solid landmarks do not exist.** `place_landmark` paints `Floor` unconditionally
   (minigrid `envs/Lroom.py:205-208`). `LandmarkKind.solid` is declared and inert, and
   says so in its own docstring (`layouts.py:547`). Making it real is a minigrid fork
   change.
2. **Even then, they could not move.** `envs/vector.py::_collect_layout_banks` refuses
   any room set whose layouts do not share one transition table, because the device
   path holds exactly one. A solid object at a different position *is* a different
   table (measured yesterday: forward refused in 336 vs 400 states). And
   `Config.__post_init__` requires the DEVICE backend for any room set, so there is no
   slower fallback to drop to.

So **"solid objects that move between episodes" is currently unrepresentable**, and the
guard that stops it is correct rather than an oversight. Q3 as planned needs either
walkable objects, or fixed positions, or work on the device path.

Separately, from yesterday's look-see: solid objects block *movement* unconditionally
but only block *sight* when `see_through_walls=False`, which we are keeping `True`. A
solid object refuses the step and does not end the episode.

### 3.2 `LEnv_small_obj`'s docstring is wrong, and both Q2 and Q3 lean on it

The minigrid fork says *"each landmark is a solid 2x2 block"* (`envs/Lroom.py:256`).
Measured:

```
cell (5, 5): Floor colour=blue can_overlap=True
cell (11, 5): Floor colour=red  can_overlap=True
cell (5, 11): Floor colour=yellow can_overlap=True
```

They are walkable floor. The Q2 plan already describes them correctly as *"a 2x2
walkable floor patch"*, so the repo's plan is right and the library's docstring is
wrong. Worth fixing in the fork before anyone reads it the other way.

---

## 4. Can the experiment repo use this one? Mostly yes.

I checked every `curious_george` symbol the experiment repo imports against this
repo's HEAD. **All but one resolve.** The break is narrower than expected:

- **`curious_george.provenance` is gone** — it moved to
  `curious_george.log_and_store.provenance` in `fa99973`. `src/omt/main_task.py`
  imports it from the top level. One-line fix on their side.
- **`evaluation/task.py::setup_task(cfg, ...)` now reads the new config shape** —
  `cfg.run.seed` and `cfg.run.wandb`, and *nothing else*. The experiment repo passes a
  Hydra `DictConfig` with `exp.seed` / `logging.wandb_log`, so this raises on the next
  `make pin`.

That second one is the whole incompatibility, and it is two scalar reads. Every other
argument to `setup_task` is already an explicit keyword. Making those two explicit too
(`seed: int`, `wandb_log: bool`) decouples the experiment repo from this repo's config
shape permanently, and is a smaller change than migrating their Hydra tree.

**The good news for Q2 and Q3:** `main_train.py::train(cfg)` takes an already-built
`Config`, so the experiment repo can build a config in Python and call it directly —
no CLI, no Hydra. That is the right integration path, and it is the only way to express
Q2's variants anyway, since `Committed(rooms=...)` cannot come from a command line.

**Q2's three variants are expressible today.** Build three `Layout`s differing in one
landmark's colour and pass `Committed(rooms=(layout,))`. One wrinkle: a single custom
room still counts as a "room set", which forces the DEVICE backend *and* forces the
multi-room spatial eval on (`configs.py:666-676` makes `SPATIAL_MULTIROOM` and
"has a room set" imply each other). For a one-room set that coupling is awkward but
harmless — the multi-room eval over one room is just the single-room eval.

---

## 5. Action items

Ordered by what blocks what. Each is self-contained enough to hand to an agent.

**Before any Q1 work**

1. **Decide where a novel object lives, then make it reachable.** `setup.py:111` reads
   a field that does not exist. Either restore it as a typed field on `EnvCfg`, or —
   better, and consistent with the cutover note — express it as content, and delete the
   dead read. Whichever way, add a test that placing a novel object actually places it,
   and prove the test fails without the fix.

**Before any Q2 work**

2. **Fix `checkpoint_series.py::build`** — `cfg.env.env_name.value` on a `str`. Add a
   test that imports and runs the module against a two-checkpoint fixture; there is no
   test on it at all today, which is how this survived.
3. **Rename its CLI off the retired Hydra vocabulary.** `--env lroom_multi` and
   `--layouts one|rooms|pool` name config groups that no longer exist.
4. **Make the resume budget unambiguous.** Today `total_grad_steps` on a resumed run
   means the grand total. Either rename it at the point of use, or make the loop print
   refuse to start when the remaining budget is <= 0 instead of exiting silently as
   success. A run that trains nothing and exits 0 is the worst available behaviour.
5. **Decide: 30 minutes per Q2 arm, or 30 minutes across every arm?** The arms are
   variant A, variant B, variant C, the no-recolour continuation and the recolour-all
   control. See §1; my recommendation is per-arm, run sequentially.

**Before any Q3 work**

6. **Canonicalise placements when landmark kinds are interchangeable.** Deduplicate by
   the unordered anchor set when the stencils and colours are identical, so `Uniform(n)`
   returns `n` distinct rooms or raises. Today it silently returns 418 of 500.
7. **Decide Q3's object type.** Walkable objects work today and run at 17,790 steps/s.
   Solid *and* moving is unrepresentable (§3.1) and needs a minigrid fork change plus
   device-path work. This is a scope call, not an engineering one.
8. **Decide Q3's budget.** Thirty minutes reaches 25-29M env steps depending on the
   analysis cadence; the recorded plateau is 73.4M and needs ~69 minutes of training
   before any analysis is added. See the table in §1.
9. **Promote `ovc_metric.py` and `test_ovc_metric.py` out of `throwaway/ported/`.**
   Its own docstring cites `tests/test_ovc_metric.py` and an instructions path, neither
   of which exists at those locations. A metric whose tests are not run is not a tested
   metric, and Q3's plan calls this its first task.

**Sizing, before launching anything long**

10. **Re-measure the analysis event and fix the stale comment.** It costs ~76 s;
    `configs.py:513` says 4.5–8.9 s per room. Run one arm with `--train-prnn.compile OFF`
    against one with `LAYER` to see how much of the gap is §2.5. Then update the comment
    with what it actually costs, because that comment is what people size runs from.
11. **Set `PYTHONUNBUFFERED=1` on every run tomorrow** so the sRSA lines are readable
    while the job is alive (§2.6).

**Cross-cutting, small**

12. **Fix `LEnv_small_obj`'s docstring in the minigrid fork** — it says "solid", they
    are walkable.
13. **Give the experiment repo a stable seam:** make `setup_task` take `seed` and
    `wandb_log` as explicit keywords instead of reaching into `cfg`. Then their next
    `make pin` does not break, and this repo's config can keep moving.
14. **The fast config has two homes.** Every speed knob defaults off in `configs.py`
    (`compile=OFF`, all `cuda_graph=False`, `num_envs=8`) and lives only as CLI
    overrides in `slurm/train_fast.sh`. Anyone running `main_train.py multienv` gets
    the slow shape and no warning. Consider a `multienv-fast` preset so the measured
    configuration is a value, not a bash string.

---

## 6. How to reproduce today's numbers

The sizing arms are throwaway scripts in the session scratchpad, not in the repo — no
result depends on them. The measurements are:

```bash
# throughput: read the tqdm rate after ~90 s (compile settles first)
uv run python main_train.py multienv \
  --train-prnn.batched --train-prnn.episodes-per-grad-step 8 --train-prnn.compile LAYER \
  --train-policy.cuda-graph --collect.rollout-cuda-graph --train-prnn.curiosity-cuda-graph \
  --collect.num-envs 256 --train-prnn.total-grad-steps 3840 --train-policy.total-grad-steps 15360 \
  --eval.analysis-every-steps 0 --eval.plot-every-steps 0 \
  --run.no-wandb --run.save-every-steps 0 --run.archive-every-steps 0 \
  --run.exp-name sizing env.source:uniform --env.source.n 500 --env.source.seed 1
```

Swap `--env.source.n` for the pool size; drop the source subcommand and use the
`reference` preset with `--collect.backend DEVICE` for the single-room control.

⚠️ **A trap I hit three times today, so it is written down:** `pkill -f main_train.py`
matches *the shell that is running the pkill*, because the pattern text appears in its
own command line. Every time, it killed the launcher and reported exit 144. Use
`pgrep -af` to look first, and never put a `pkill` in the same command as anything
whose text contains the pattern.

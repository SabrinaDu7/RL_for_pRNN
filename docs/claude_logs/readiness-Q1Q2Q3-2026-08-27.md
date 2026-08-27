2026-08-27 · branch `sdu/config-dataclasses` · RTX 4060, this dev box

# Can we run Q1, Q2 and Q3 tomorrow? Findings and action items

**Read this first.** This happened in two passes. **§1–§4 are the review**: read-only,
measured on this box or read out of the source with a file:line, and where I inferred
rather than confirmed it says so. **§5 then fixed the blockers** — so the "is broken"
sections describe what was found, and each carries a ✅ in §5 if it no longer is.

```
before   uv run pytest -q  ->  391 passed, 1 deselected
after    uv run pytest -q  ->  417 passed, 1 deselected     (commit ff6e63a)
```

What the BEFORE green did not cover, and why these survived a full config migration:
`evaluation/checkpoint_series.py` was imported by no test at all, and nothing exercised
a two-phase resume or a novel-object placement. All three now have one.

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

**One analysis event costs ~88 s** with the settings we actually run (three-room pool,
`rooms_max=4`, including the behaviour eval that fires on the same cadence).

That is **far more than the code says.** `configs.py:513` records "4.5-8.9 s per room",
which at `rooms_max=4` predicts 18-36 s. The comment is stale and it is what anyone
sizing a run will read.

**Where the extra time goes — measured.** Four arms, run SEQUENTIALLY so nothing
contends (my first attempt at this was invalidated by running the test suite alongside
it, so that number is not quoted anywhere). 3,145,728 env steps each, three rooms; times
are progress-bar time, which excludes startup:

| compile | analysis off | analysis on (3 events) | per event | training rate |
|---|---|---|---|---|
| `LAYER` | 180 s | 443 s | **87.7 s** | 17,476 env steps/s |
| `OFF` | 392 s | 504 s | **37.3 s** | 8,025 env steps/s |

**So the failing inductor recompile (§2.5) really is about half the analysis event** —
88 s against 37 s. But turning the compile off is the wrong response, because it also
costs **2.18x on the training loop**. The two effects pull opposite ways, and where they
balance is a property of the cadence:

| analysis events in 30 min | `compile=LAYER` | `compile=OFF` |
|---|---|---|
| 0 | 30.8M env steps | 14.1M |
| 2 | 27.7M | 13.5M |
| 5 | 23.1M | 12.6M |
| 10 | 15.4M | 11.1M |
| 20 | 0.1M | 8.1M |

**`LAYER` wins below ~13 analysis events in a 30-minute run**, which is every cadence
anyone would actually pick. **Keep the compile on.** The real fix is to stop the
recompile happening at all — see action item 10 — not to trade away 2.18x of training.

### Sizing table

Using the steady-state rates from §1 (a 30-minute run amortises the compile warm-up that
the short arms above pay) and 88 s an event:

| analysis events in 30 min | 3 rooms (Q2) | 500 rooms (Q3) |
|---|---|---|
| 0 | 35.1M env steps | 31.3M |
| 2 | 31.6M | 28.2M |
| 5 | 26.3M | 23.5M |
| 10 | 17.6M | 15.7M |
| 20 | 0.1M - the whole run is measurement | 0.1M |

**Pick the number of curve points first, then the step budget.** Ten points costs half a
30-minute run. The 2026-08-23 speed log reached the same conclusion for single-room runs
- *"the spatial eval dominates sizing, not the loop"* - and it is more true here.

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
cost is paid every time, not once.

**Isolated and confirmed** (§1): the event costs 87.7 s with `compile=LAYER` and 37.3 s
with `OFF`, so **the recompile is roughly half of every analysis event**. It is still
not worth turning the compile off — that costs 2.18x on the training loop — so the fix
is to stop the recompile, not to stop compiling.

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
- **`evaluation/task.py::setup_task(cfg, ...)` needs the new config shape.** The
  experiment repo passes a Hydra `DictConfig` with `exp.seed` / `logging.wandb_log`, so
  this raises on the next `make pin`.

**⚠️ I first wrote that this was two scalar reads and could be papered over with two
explicit keyword arguments. That was wrong, and it is worth saying why.** I read
`setup_task`'s own body — which does only touch `cfg.run.seed` and `cfg.run.wandb` —
and did not follow `cfg` into the three functions it hands it to. `get_pN`,
`get_SR_acmodel` and `setup_algo` read **39 distinct fields** between them, across
`arch_prnn`, `arch_policy`, `train_prnn`, `train_policy`, `collect` and `run`, all in
the new spelling.

So there is **no small seam**, and adding those two keywords would have been worse than
useless: it would have looked like the incompatibility was handled. The honest answer is
simpler and bigger — **the experiment repo has to build a
`curious_george.configs.Config`**, and its own `Configs/` Hydra tree should go the way
this repo's did. That is a real piece of work, but it is the one the user already said
they would do, and it is the only thing that actually holds.

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

**✅ DONE marks work landed in `ff6e63a` the same day, each verified by
reintroducing the defect and watching the new tests fail.** Gate went from
391 passed / 1 deselected to **417 passed / 1 deselected**.

**Before any Q1 work**

1. ✅ **DONE — novel object restored and reachable.** `EnvCfg.novel_object: tuple[int,
   int] | None`, validated against the room's walkable cells, because an invisible
   object and a null result are indistinguishable downstream. `setup.py` reads it
   directly now; the `getattr` default that turned a removed field into "no object
   requested" is gone. `tests/test_novel_object.py` checks both that the object reaches
   the grid and that it changes a pixel the agent actually sees.

**Before any Q2 work**

2. ✅ **DONE — `checkpoint_series.py` fixed.** It was broken in **four** reads, not the
   one §2.2 reported: `env.base_room`, `env.layouts` and `env.eval_rooms_max` do not
   exist either. All four moved into `SeriesContext`, which resolves from a `Config`
   alone; `tests/test_checkpoint_series.py` covers them and 11 of its 13 fail if a
   single read is reverted.
3. ✅ **DONE — CLI renamed off the retired Hydra vocabulary.** Now `--room
   lroom|squareroom` and `--source frozen|one|uniform`, which are words `configs.py`
   actually uses.
4. ✅ **DONE — the resume budget now refuses rather than exiting clean.** It still means
   the grand total, which is the honest semantics; what changed is that a budget already
   spent raises and names the number to raise it past, instead of producing a run
   directory and a wandb run for a phase that never trained.
5. **Decide: 30 minutes per Q2 arm, or 30 minutes across every arm?** The arms are
   variant A, variant B, variant C, the no-recolour continuation and the recolour-all
   control. See §1; my recommendation is per-arm, run sequentially.

**Before any Q3 work**

6. ✅ **DONE — placements canonicalised for interchangeable landmarks.** Slots sharing a
   stencil AND a colour are collapsed, so `Uniform(n)` now returns `n` distinct rooms.
   Conservative: slots differing in either are never grouped, so a colour-varying design
   is untouched.
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

10. ✅ **MEASURED — and it leaves one real fix.** The event costs **87.7 s** at
    `compile=LAYER` and **37.3 s** at `OFF`, so the failing inductor recompile is about
    half of it. Keep the compile (it is 2.18x on training; the crossover is ~13 events
    per 30-minute run). **The remaining fix is to stop the recompile:** the spatial
    analysis runs the pRNN forward on CPU at a different batch shape than training, and
    inductor fails there on `aten._local_scalar_dense.default` from
    `thetaRNN.py:406`'s `torch.zeros(1, 500, cell_input_size)`. Options worth trying, in
    order: mark that dim dynamic, exclude the eval forward from the compiled region, or
    run the eval at the training batch shape. Worth ~50 s per curve point.
    **Also update `configs.py:513`**, which still says 4.5–8.9 s per room.
11. **Set `PYTHONUNBUFFERED=1` on every run tomorrow** so the sRSA lines are readable
    while the job is alive (§2.6).

**Cross-cutting, small**

12. **Fix `LEnv_small_obj`'s docstring in the minigrid fork** — it says "solid", they
    are walkable.
13. **The experiment repo must build a typed `Config`; there is no shortcut.** Its
    `src/omt/` passes a Hydra `DictConfig` into `setup_task`, which reaches 39 fields of
    the new shape through `get_pN` / `get_SR_acmodel` / `setup_algo`. Delete
    `experiment-curiousgeorge/Configs/` and construct a `Config` in Python — the same
    move this repo made. Do NOT try to bridge it with a shim; see the correction in §4.
    Also fix the one stale import there: `curious_george.provenance` is now
    `curious_george.log_and_store.provenance`.
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

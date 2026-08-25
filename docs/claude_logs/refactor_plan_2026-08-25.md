2026-08-25 · branch `sdu/refactor-after-speed30` (pushed, forked from `ba87d81`)

# Refactor: split the library from the questions

**Goal.** Make running a training run, collecting its data, and analysing that data
smooth and checkable. Concretely: `RL_for_pRNN` becomes a library (`curious_george/`
plus its tests, nothing else), and every scientific question lives in a second repo
built from `../experiment-template/`, which imports the library by pinned git rev.

**Everything below that is a measurement was run on 2026-08-25 against
`ba87d81`.** Commands are given so each can be re-run. Claims I did not verify are
marked.

---

## 0. Where this starts

```
branch     sdu/refactor-after-speed30     forked from ba87d81, pushed
gate       174 passed, 18 skipped, 7 deselected, 0 failed   (uv run pytest -q, 89.8 s)
```

Diff every phase against that line and treat a **change in the skip count as a
finding**, not as noise. The 18 skips have been the same module for four audit
revisions while the pass count rose by 33.

---

## 1. The shape

Two repositories.

```
RL_for_pRNN                     the library. curious_george/ + tests/ + Configs/ + slurm/.
                                No scripts/, no tasks/, docs/ cut to what the library owns.

curious-george-questions        new repo, `cp -r ../experiment-template`, ./rename.sh.
(name TBD)                      One directory per question. Imports rl-for-prnn by git rev.
```

**Why this is cheap: the boundary already exists.** `curious_george/` imports nothing
from `scripts/` or `tasks/` — one grep match and it is a docstring line in
`world_model/device.py:3`.

```bash
grep -rn "from scripts\|import scripts\|from tasks\|import tasks" curious_george/
```

So this is a packaging change, not an untangling. What crosses the boundary today is
`tests/` (6 files import `scripts`/`tasks`) and nothing else.

### 1a. Python: the template comes down to 3.10

Not negotiable in the other direction, and the constraint is upstream:

```
RL_for_pRNN     requires-python  >=3.10, <=3.10.15    running 3.10.15
  prnn                           >=3.9, <=3.10.15     LevensteinLab, not ours
  minigrid                       >=3.7, <3.11         SabrinaDu7 fork
experiment-template              >=3.13               running 3.13.7
```

`uv add --git` will not resolve until one side moves, and RL_for_pRNN cannot.

**Measured, and the news is good:** all 22 real Python files in the template compile
under 3.10.15. Zero uses of `tomllib`, `Self`, `override`, `StrEnum`,
`ExceptionGroup`, `except*`, or `match`. The only two compile failures are
`docs/templates/analyze_TEMPLATE.py` and `package_TEMPLATE.py`, which contain
`{{QID}}` placeholders and are not Python.

```bash
uv run --no-sync python -c "
import pathlib, tempfile, py_compile
root=pathlib.Path('../experiment-template')
files=[p for p in sorted(root.rglob('*.py')) if '.venv' not in p.parts and '__pycache__' not in p.parts]
with tempfile.TemporaryDirectory() as td:
    for i,p in enumerate(files):
        try: py_compile.compile(str(p), cfile=f'{td}/{i}.pyc', doraise=True)
        except py_compile.PyCompileError as e: print(p, str(e)[:100])
"
```

So the port is three lines plus a test run: `requires-python`, `[tool.ruff]
target-version`, `[tool.basedpyright] pythonVersion`. **Do it as the very first
commit in the new repo and run `make Q0` green before writing anything**, exactly as
the template's README instructs.

Accept that the questions repo stops being small: its five dependencies (numpy,
matplotlib, jaxtyping, beartype, tyro) become ~30 including torch, gymnasium,
minigrid, prnn, wandb, pynapple, kaleido, moviepy.

### 1b. Wiring: git-pin, with the part that makes it traceable

```toml
[tool.uv.sources]
rl-for-prnn = { git = "https://github.com/SabrinaDu7/RL_for_pRNN.git", rev = "<sha>" }
```

**`rev`, not `branch`.** And fix the same defect one level down inside RL_for_pRNN
itself: `prnn` is currently `branch = "sdu/rl-integration"`, so `uv sync` can silently
swap the world model, with the commit recorded only in `uv.lock`.

Three pieces make the pin real rather than nominal:

1. `just sync` → `uv lock --upgrade-package rl-for-prnn && uv sync`.
2. **A rendered results document records the resolved rev**, alongside the config and
   commit the template already records. A pin nobody reads back is just a pin.
3. A local path override stays available and documented, with one invariant enforced
   in the render step:

   > You may develop against a local path. You may not render a results document from
   > one.

   That turns the override from a hazard into a tool, which matters because
   `uv lock --upgrade-package` needs the commit pushed and that is a real tax during
   an active refactor.

---

## 2. 🔴 BITWISE EQUALITY IS THE POINT

**Every move in this plan must be provably numerics-preserving, and "provably" means
a test that compares tensors, not a claim that only files moved.**

The reason is on the record three times in four days. Each of these was invisible to
every existing test, and each changed results while the loss curve looked healthy:

| commit | what it silently did |
|---|---|
| `d275149` | odd epochs trained on 32,767 of 32,768 transitions |
| `0d60df7` | a graphed run trained on stranded memory — sRSA 0.0238 against eager's 0.5158, loss curves agreeing to three decimals |
| `4cfd8cb` | a captured CUDA graph froze every Python-float hyperparameter, killing `rl.entropy_coef_final` |

A refactor of this size will produce a fourth unless every step is gated.

### 2a. What already exists — use it, do not rebuild it

```
tests/golden_omt/test_golden_omt.py     A REAL end-to-end bitwise gate for OMT.
  + golden_omt_v0.pt, golden_omt_v1.pt  torch.equal / np.array_equal over
                                        construction, training batches, eval trial
                                        and metrics. Runtime ~1 min.

tests/golden/                           capture_golden.py, compare_io.py,
  + golden_v0.pt, golden_v1.pt          golden_v0/v1.pt — the TRAINING path's
                                        fixtures, and NO test_*.py, so
                                        `pytest --collect-only tests/golden` collects
                                        nothing. The fixtures exist; the gate does not.
```

**This is the single strongest reason to pilot on OMT**: the bitwise mechanism the
port needs already exists there and nowhere else. Writing
`tests/golden/test_golden.py` around the existing `compare_io.py` is an afternoon and
gives the training path the same protection.

### 2b. What can be bitwise, and what cannot

Be precise about this up front, or a legitimate difference will be read as a
regression and vice versa.

**CAN be bitwise — demand `torch.equal`:**
- moving a file, renaming a module, adding `__init__.py`, changing an import path
- extracting a function, reordering code that consumes no RNG
- anything in the eager training path with dropout and noise off

**CANNOT be bitwise — and the reason must be named at the assertion:**
- **CUDA-graph paths.** A captured region draws from CUDA's graph-safe RNG, so a
  graphed rollout and an eager one realize different but equally valid streams. This
  is documented at `rl/collect/rollout_graph.py` and gated in
  `tests/test_cuda_graph_rollout.py` by saturating the policy until the softmax is an
  exact float32 one-hot — and *checking* that assumption by requiring every recorded
  `log_prob` to be exactly 0.
- **Capturable vs default Adam.** Capture forces `capturable=True`, which computes
  bias correction on-device in a different float32 order: ~3e-8. The right fix is to
  give the eager arm the same capturable optimizer, not to loosen a tolerance.
- **Anything crossing devices.** The spatial eval runs on CPU by construction
  (`evaluation/spatial.py` wraps it in `on_device(modules, "cpu")`).

### 2c. The rule for this refactor

> A commit that moves or renames code lands **with** a green golden test that was run
> on the pre-move tree. If no golden covers the moved code, capture one first — in the
> same PR, from the pre-move tree — or do not move it.

And, from `0d60df7`'s lesson: **a gate must be proven to fail.** Break the code
deliberately, watch the test go red, restore. "Finite and changing" is exactly what a
graph training on stranded memory looks like.

---

## 3. Phase order

The ordering constraints are real; the rest is preference.

### Phase 0 — baseline and goldens · hours

- Record `174 / 18 / 7 / 0` (done, above).
- **Capture fresh goldens from `ba87d81` before anything moves**: re-run
  `tests/golden/capture_golden.py` and `tests/golden_omt/capture_golden_omt.py`, and
  write `tests/golden/test_golden.py` around `compare_io.py` so the training path's
  fixtures finally run.
- Confirm each new gate fails on deliberately broken code.

### Phase 1 — make a run self-describing · hours · BLOCKING

`main_train.py:28` is `print(OmegaConf.to_yaml(cfg))` — printed, never written.
`find outputs -maxdepth 2 -name '*.yaml'` returns nothing.

Write the **resolved config, git commit, dirty flag and timestamp** into the run
directory from `main_train.py` and `tasks/omt/main_task.py`.

**Do this before the split.** Otherwise the questions repo inherits checkpoints that
cannot identify themselves, and you build provenance tooling around artifacts with no
provenance. Three invalidating lines already run through the run series and all three
exist only as prose plus job numbers; a run directory that names its own commit turns
each of them into a property of the checkpoint, retroactively for everything made
after this lands.

### Phase 2 — metric semantics: one name, one thing · days

Decided 2026-08-25. All three are in the library, so they must land before the
questions repo reads any of them.

**2a. `UpdateLogs` means two different things.**

```
eager    updater.py:145-149   log lists re-created INSIDE the epoch loop
                              -> post-loop mean covers the LAST epoch only
graphed  updater.py:118-134   acc/n initialised BEFORE the loop
                              -> mean over ALL epochs
```

Inherited, not invented: `torch_ac/algos/ppo.py:32-35` has the identical placement and
the identical `# Initialize log values` comment.

The consequence is a **systematic bias with a per-metric direction, not noise.** The
policy moves across the four PPO passes over one batch; at epoch 1 minibatch 1 the
importance ratio is exactly 1. So for the identical update, eager reports **lower**
entropy, **lower** `value_loss`, **lower** `grad_norm` and **larger-magnitude**
`policy_loss` than graphed.

Fix: hoist the five list initialisations above the epoch loop. Graphed is the correct
convention — the statistic should describe the update, and the update is all four
passes. Gate: assert eager and graphed produce the same `UpdateLogs` on the same data.
It fails today.

⚠️ **This creates a fourth invalidating line.** `policy_entropy`, `value_loss` and
`grad_norm` after the fix are not comparable to any historical eager curve — including
`speed-30min` §9's "g8 EAGER 0.5927 at 60M against g8 graphed 1.3357", which is
contaminated by an unmeasured amount of estimator difference. Record it in
`docs/invalid-runs.md` in the same commit.

**2b. `return_per_episode` is fabricated under `device_env`.**

`collector.py:231-233` extends `finished_returns` and `finished_reshaped` with
`[0.0] * B`; `done_counter` and `finished_frames` are real. It reaches wandb through
`training/logging.py:61` as `return_mean/std/min/max`. `device_env=True` is every fast
run.

It is not a wrong number today — in an L-room with no goal, extrinsic return really is
0, and the curious agent's learning signal is intrinsic and logged separately. It is an
**indistinguishable** number: the moment a run uses `MiniGrid-LRoom_Goal-v0` (it
exists; `tests/test_table_env.py` parametrizes over it) with `device_env=True`, wandb
reports 0.0 and nothing separates *"the agent never reached the goal"* from *"this path
does not measure it."*

Second collision in the same three lines: this backend has **no environment
terminations** — the cut is at `prnn_seqdur`. "Episode" here means "256-step segment",
so `num_episodes` counts segments, a different quantity from `num_episodes` on the
non-device path.

Fix: do not log a fabricated zero — omit the key under this backend so wandb has no
series rather than a flat-zero one, and give the segment count a name that says
segment.

**2c. Spatial information = `pN`.** Decided. `pN` owns the metric, wandb logs it, every
historical curve used it, and `scripts/trace/trace_maps.py::spatial_info` dies with the
trace code. No fork survives.

But name what is being adopted. In `prnn/utils/predictiveNet.py`, inside
`calculateSpatialMetrics`:

```python
num_active = (h > 0).sum(axis=0)
SI.iloc[num_active < active_time_threshold] = 0     # default 200
```

A unit active in fewer than 200 of the pooled samples is assigned **SI = 0, not NaN**,
and `mean SI` averages over every unit. So `mean SI` conflates *"carries no spatial
information"* with *"barely fired"* — and because training sparsifies the
representation, more units cross below the threshold as training proceeds, pushing
`mean SI` **down** for a reason unrelated to spatial tuning. It also makes SI
non-comparable across runs with different activity levels.

Fix, no upstream change needed: `calculateSpatialMetrics` returns the **per-unit**
frame; only the wandb call takes `.mean()`. So `curious_george/evaluation/spatial.py`
logs `n_units_zeroed` and `mean SI over active units` alongside `mean SI`, from data
prnn already returns. `mean SI` stays the historical series and becomes interpretable.

**Unmeasured, and it belongs here:** N is `n_trajs × traj_timesteps` = 8 × 256 = 2048,
so the threshold is ~10% of samples — but nobody has measured what fraction of the 500
units it actually excludes on a real checkpoint. One cheap measurement.

### Phase 3 — stage every deletion in `throwaway/` FIRST

**Nothing is deleted in this refactor. Everything to be removed is `git mv`'d into
`throwaway/` and stays there until the port is proven bitwise.**

Two reasons, and the second is the one that matters:

1. The old code is the **reference implementation**. You cannot prove the ported OMT
   is bitwise-equal to the original if the original is gone.
2. `throwaway/` is already the repo's declared home for code no result may depend on
   (`CLAUDE.md`), and it already holds eight untracked files from the intervention
   experiments. That convention is right; extend it rather than inventing a new one.

```
throwaway/
├── ported/          moved out of the tree, kept as the bitwise reference
└── (existing)       lr_warmup.py, critic_burn_in.py, policy_head_init.py, ...
```

`throwaway/ported/` gets deleted in one commit at the very end, once every gate is
green, and that commit says which gate proved each file safe to drop.

### Phase 4 — the questions repo

`cp -r ../experiment-template ../curious-george-questions && ./rename.sh`, port to
3.10 (§1a), `make Q0` green, then add the git dependency (§1b).

### Phase 5 — the OMT pilot · §5 below

### Phase 6 — prune this repo · §4 below

### Phase 7 — make the gates fire

- The **path-existence test**: one test walking every path named in a doc or docstring
  and failing if it is not in git. Cheapest gate with the best record — see §8, where it
  would already have caught three findings.
- Resolve `outputs/data_cur_lroom_step1608_goal711/trajectories.pt` so the 18 skips run.
- Extend `[tool.ruff] include` to `curious_george/` recursively and `tests/`; drop the
  two dead entries (`trainRL_Adel.py`, `test/*.py`).
- Untrack `.env`; ship `.env.example` with relative defaults.
- Narrow `.gitignore`: drop the blanket `*.png`, keep `outputs/` ignored, track the
  figures a document cites.

---

## 4. What to keep and what to move to `throwaway/ported/`

Measured inventory (`ba87d81`):

```
curious_george/    8,553 lines      scripts/     11,741 lines
tests/             6,058 lines      tasks/        1,790 lines
docs/             11,134 lines
```

### KEEP in RL_for_pRNN

| Path | Why |
|---|---|
| `curious_george/**` | the library. Whole thing. |
| `tests/**` except the six that import `scripts`/`tasks` | the gate |
| `tests/golden/`, `tests/golden_omt/` | the bitwise mechanism (§2a) |
| `tests/perf/` | `benchmark.py` / `compare_metrics.py` — the speed gate |
| `Configs/**` | hydra tree the library reads |
| `slurm/**` | launchers; they invoke `main_train.py` |
| `main_train.py` | the training entry point |
| `CLAUDE.md`, `pyproject.toml`, `justfile`, `README.md` | project spine |

### PROMOTE into the library before moving anything (these are library, not experiment)

| From | To | Why |
|---|---|---|
| `scripts/compare_wandb_runs.py` | `curious_george/check/wandb_compare.py` | named reference, env-step axis, per-metric measured tolerance, JSON artifact. It is most of `exp check` already. |
| `scripts/legacy/wandb_data.py` | `curious_george/io/wandb.py` | 1,451 lines, imported by five live scripts and one test, and not legacy |
| `scripts/trace/trace_probe.py` | `curious_george/evaluation/probe.py` | checkpoint → (hidden state, position). Even discarding every trace *result*, this mechanism is what every future question needs. |

⚠️ There are **four** competing implementations of "turn a checkpoint into
(h, position)": `scripts/trace/trace_probe.py`, `scripts/multienv/checkpoint_curve.py::fixed_probe`,
`scripts/moser/moser_analysis.py::_build_probe`, and
`curious_george/evaluation/task.py::collect_eval_rollouts[_batched]`. Promote **one**;
the training collector (`rl/collect/collector.py`) stays separate and is not one of
them.

### MOVE to `throwaway/ported/`

| Path | Size | Note |
|---|---|---|
| `scripts/trace/` | 25 files, 4,026 lines | object / OVC / trace — results not trusted, restarting from scratch |
| `scripts/moser/` | 5 files, 597 lines | same line |
| `scripts/multienv/` | 4 files, 649 lines | keep `checkpoint_curve.py` readable until the multi-room question is re-ported |
| `scripts/legacy/` (minus `wandb_data.py`) | 12 files, ~2,200 lines | |
| `scripts/*.py` top level (minus `compare_wandb_runs.py`) | 10 files, ~2,600 lines | figure generators for results being discarded |
| `tasks/otc/` | 359 lines | object-into-hidden-state — same line |
| `tasks/omt/` | 940 lines | **ported, not discarded** — see §5. Stays in `throwaway/ported/` as the bitwise reference. |
| `docs/results/` | 11 md | the object line |
| `docs/legacy/` | 13 md | frozen already |

### EXTRACT before deleting any document

Two things in `docs/` are not results and are not reconstructible:

1. **`docs/invalid-runs.md`** — the three (soon four) invalidating lines, each with its
   commit, what it invalidates, and how to tell whether a checkpoint predates it.
   Currently these live only as prose plus job numbers in two documents. If those
   documents go, the only record of which past runs are worthless goes with them.
2. **`docs/pitfalls.md`** — the methodology that cost the most to learn: wandb history
   caps at 10,000 samples and `scan_history` fails on these runs;
   `tests/perf/benchmark.py` defaults `--sync-stages` off and `--warmup-updates` to 0;
   do not quote a g8 rate before 15 minutes elapsed; a CUDA graph fails silently;
   averaging-window width manufactures significance; never quote these metrics from a
   single point. Or fold it into `CLAUDE.md`.

---

## 5. The pilot: OMT as Q1

**Question, as posed:** *does the agent-pRNN head towards novel objects once one is
introduced into a familiar environment?*

⚠️ **That is a behavioural question, and it is NOT the question the existing OMT code
headlines.** `tasks/README.md` describes the probe as *"after exposure, does the pRNN
still predict the object at the location where it used to be"* — a prediction-trace
question, whose metric is `Analysis/Goal Minus Ctrl Vs. Step Count`. The behavioural
question's metrics are `Analysis/Novel Object In-view Times` and
`Analysis/Avg Distance Travelled`. Both are logged by the same run. **Do not conflate
them**; the trace line is part of what is being restarted from scratch, the behavioural
line is the pilot.

### Why OMT is the right pilot

- It is the **only** path with a working end-to-end bitwise gate (§2a).
- It has 557 historical runs to compare against, in a separate wandb project.
- Its analysis already produces a scalar per checkpoint, which is what a `.in` template
  and a values file want.
- It exercises the whole chain: checkpoint in → training phase → frozen eval probe →
  metric → figure.

### The comparison target

```
entity   blake-richards          (unchanged)
project  curious-george-omt      557 runs
```

The cleanest reference batch is **2026-07-30**, 12 runs named
`omt-cur-dot-0730-*`, ~19–32 min each, all `finished`. They are the most recent, they
postdate the `prnn` migration, and `tasks/README.md` records that the OMT entry point
was confirmed running on the new stack that day.

⚠️ `tasks/README.md` also records, under "Post-refactor status (2026-07-30)": *"Not
verified: whether the metric values match pre-refactor runs — the migration changed
`bias_lr` from an effective 0 to 0.01, so trajectories are expected to diverge."* So
**anything older than 2026-07-30 is not a bitwise target**, only a sanity range.

### What "ported correctly" means, in two tiers

1. **Bitwise (hard gate).** `tests/golden_omt/test_golden_omt.py` re-run against the
   ported code with the same pinned checkpoint and seed. `torch.equal` /
   `np.array_equal` across construction, training batches, control-net-untouched, and
   eval trial + metrics. This is the gate; it either passes or the port is wrong.
2. **Distributional (soft check).** New runs tagged `refactoring-aug25` against the
   twelve `omt-cur-dot-0730-*` runs on the metrics above. This cannot be bitwise —
   different checkpoints, different seeds — so state a band from the twelve runs' own
   spread before looking at the new ones, and use `compare_wandb_runs.py`'s
   adjacent-sample-spread convention.

### Mapping onto the template

```
docs/exp_instructions/instructions-Q1.md      the question, why, dated methods
docs/exp_results/results_Q1.in                @TOKEN@ per number
docs/exp_results/results_Q1.md                generated, never edited
src/experiments/Q1/collect.py                 run OMT -> outputs/cache/
src/experiments/Q1/analyze.py                 cache -> values + figures
```

`collect.py` is the interesting one: "collecting" here means *running a task on the
cluster and pulling its outputs down*, not computing in-process. The template's
`CollectConfig` hashes its identifying params into the cache filename
(`src/config.py::cache_key`), so the OMT hydra overrides, the source checkpoint and the
seed must all be config fields — which is exactly the discipline Phase 1 installs.

### ⚠️ The shared-dataset problem, which OMT hits immediately

The template keys a cache file by `experiment_id` + a hash of the config
(`outputs/cache/<experiment_id>_<hash>.npz`). That quietly assumes **one question owns
its cache.** OMT checkpoints will be read by the behavioural question *and* by whatever
replaces the trace line — two questions, one dataset.

Decide this before the first cache is written, not after:

- a dataset used by more than one question gets a **name that is not a QID**
  (`outputs/cache/shared/<dataset>_<hash>.npz`), and
- its `.meta.json` sidecar records **which questions read it**, and
- `exp check` for any question that reads a shared dataset states which one, in the
  rendered document.

This is a small extension to `src/cache.py` (45 lines) and it is much cheaper now than
after two questions disagree about a cache.

---

## 6. wandb

**Every run produced by this refactor carries the tag `refactoring-aug25`.**

⚠️ **There is no tag support in the code today.** Both entry points omit `tags=`:

```
curious_george/training/logging.py::init_wandb       entity, project, group, name, id, dir, resume, config
tasks/omt/main_task.py::create_wandb_run             entity, project, name, group
```

```bash
grep -rn "tags" curious_george/ tasks/ Configs/ | grep -i wandb    # -> nothing
```

So Phase 1 adds `logging.wandb_tags: []` to `Configs/main.yaml` and threads it through
both call sites. Small, but it is a code change and it must be in place *before* the
first comparison run, or the runs are unfindable.

Projects stay as they are: `curious-george` for training, `curious-george-omt` for the
task. Same entity, `blake-richards`.

---

## 7. Open decisions

1. **The `UpdateLogs` fix invalidates every historical eager entropy curve** (§2a).
   Fix and record the fourth invalidating line, or freeze the eager convention and make
   graphed match it instead? Fixing is right, but it is a call.
2. **Name of the questions repo.**
3. **Does `docs/sab_context/` become tracked?** It is gitignored and cited twelve times
   — from `Configs/run/multienv.yaml:8`, `Configs/performance/ultra.yaml:27`,
   `tests/perf/benchmark.py:176`, and `scripts/summary_figures.py:194,242`, where a
   committed figure generator carries results as source literals.
4. **Which of the four probe implementations is promoted** (§4).

---

## 8. While you're reading, flag anything suspicious, inconsistent, or ad hoc

A standing instruction for whoever executes this, and the reason the plan has this
section at all: three of the four bug classes found this month were found by someone
noticing something odd while reading for another purpose.

**What to flag:** anything where a name predicts something other than what is there; a
default that silently zeroes a real measurement; a check that cannot fire; a number in
a document with no command behind it; a comment that contradicts the code below it; a
path that climbs (`../..`); two things with the same name meaning different things.
Flag it in this section with a file:line — do not fix it in passing during a port, and
do not build around it silently.

### Already flagged, not yet resolved

**In `tasks/omt/` — read before porting**

- ⚠️ **Checkpoints come from environment variables, not from the config.**
  `curious_george/utils/dev_env.py::get_ckpt_env_vars` reads `CUR_CKPT_DIR`,
  `RAND_CKPT_DIR`, `FOURROOM_CUR_CKPT_DIR`, `FOURROOM_RAND_CKPT_DIR`. So **a run's most
  important input is a shell variable and appears nowhere in its provenance.** This is
  the single worst reproducibility defect in the OMT path and the port must fix it: the
  source checkpoint becomes a config field.
- ⚠️ **A known-wrong fallback is preserved deliberately.** `tasks/README.md`: *"the AC
  variables are env-specific but the RANDOM ones are not, so a random-agent FourRooms
  lookup silently returns the LRoom checkpoint."* Named in prose and left in the code —
  a flaw laundered into a convention. Delete the fallback in the port.
- `tasks/omt/main_task.py:20` hardcodes `DEVICE = torch.device("cuda")`. No CPU
  fallback, so no OMT test can run on a CPU box.
- `tasks.testing.trajs` must be ≤ `rl.trajs_per_batch`. Asserted only in prose in
  `tasks/README.md`; make it a config-time check or make the state unrepresentable.
- `status.pt` optimizer-key collision was fixed on 2026-07-30 but the legacy fallback
  remains and reads a shape-matched old key. Decide whether the new repo carries it.

**Documentation citing files that do not exist** (would be caught by the path-existence
test in Phase 7, which is why that test is cheap and valuable)

- `tasks/README.md` names `scripts/analysis_OMT_h.py`, `scripts/isomap.py` and
  `scripts/analysis_reward_map.py` — **all three absent.** Verified:
  ```bash
  for f in scripts/analysis_OMT_h.py scripts/isomap.py scripts/analysis_reward_map.py; do
    [ -e "$f" ] && echo "EXISTS $f" || echo "ABSENT $f"; done
  ```
  The README's "Where this is going" section points the next analysis at the first of
  them.
- `docs/claude_logs/speed-30min-2026-08-23.md` names
  `tests/test_cuda_graph_diag_guard.py` three times including as a copy-pasteable
  command; the file was deleted and its tests folded into `test_cuda_graph_wm.py`.

**Comments that contradict or disclaim themselves**

- `Configs/run/multienv.yaml` states a gradient-step budget and then says *"TrainingSchedule
  prints the derived budget at startup — read that, do not trust this comment."* A
  comment admitting it is untrustworthy is a fact with two homes; delete it.
- `slurm/train_fast.sh:1` still says *"tuned to finish inside 2 hours"*; `:36` says
  *"The final config does not use cuda_graph, so it is safe"* while `:90`–`:124` add
  four graph switches; `:55` defaults `BRANCH` to `sdu/speed` while the work has been on
  `sdu/speed-30min`. The script reads **16 positional arguments** and its usage block
  documents four.
- `Configs/main.yaml:49` carries `wandb_project: curious-george` with a comment listing
  two other projects — the project a run lands in is chosen by editing a comment's
  neighbour.

**Data and gates**

- `tests/golden/` holds two fixtures and two scripts and **no test file**; the training
  path has no bitwise gate while OMT does.
- `data/obs_bank/`: **2 tracked, 246 on disk**, under a rule ignoring the directory.
- `envs/obs_bank.py:95` writes the bank non-atomically while `_ensure_bank` gates on
  `path.exists()`, so a concurrent reader can load a truncated file. Interacts with the
  line above.
- 18 tests skip on a missing `outputs/data_cur_lroom_step1608_goal711/trajectories.pt`
  — the same module for four audit revisions while the pass count rose by 33.
- `.env` is **tracked** and hardcodes `/home/sabrina/...` for `RL_STORAGE` and
  `ROOT_DIR`, so any clone elsewhere writes into a path naming this user. It also ships
  a seven-week-old *"DANGER: THESE CKPTS DON'T EXIST ANYMORE"* block.
- `git ls-files '*.png'` returns nothing: no figure is in the repo, including the three
  reference images the object line is measured against.

**Dead code**

- `tests/golden/compare_omt.py` and `tests/golden_omt/capture_golden_omt.py` fall back
  to `from utils import ...`, `from RLutils import make_env` and
  `tasks.ObjectMemoryTask.define_task`. No branch has those modules, so the
  except-branches can never be taken — and they suppress `ImportError`.

**Unmeasured things this plan depends on**

- What fraction of the 500 units `active_time_threshold=200` actually zeroes on a real
  checkpoint (§2c).
- How large the eager-vs-graphed `UpdateLogs` bias is in practice (§2a) — the fix makes
  it moot, but the size tells you how much of §9's entropy comparison was estimator.
- Whether the 2026-07-30 OMT batch is a valid distributional target given `bias_lr`
  (§5).

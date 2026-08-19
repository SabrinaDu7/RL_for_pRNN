2026-08-18 · branch `sdu/speed`

# Speed: why training is not instantaneous, what was fixed, and what to do next

Two documents split the project. `docs/results/README_object_experiments.md` is
the entry point for the **science**; this is the entry point for the
**engineering**. §1 restates the experiment structure so a reader gets both from
one place.

Every number here is measured on this machine (RTX 4060, torch 2.8.0+cu128) with
the command that produced it named. Where a measurement is contaminated, it says
so and is not used.

---

## 1. The README structure — what was run, why, and the prelim results

`docs/results/README_object_experiments.md` now has a defined reading path:

```
"Read in this order"            7 steps, ~1 hour, ending at the newest result
"Every result document"         every doc with its headline verdict
"The one-paragraph version"     the W_out mechanism that predicts every null
"Methods, and where each lives" metric -> module
"Four things to carry forward"  the failure modes that cost retractions
```

### The experiments, in order, and what came out

| # | experiment | why | prelim result |
|---|---|---|---|
| 1 | L-room + novel object (OMT) | does `h` hold a trace where an object was? | **null.** Object memory is decoder-localised in `W_out`; place code unchanged (r ≈ 0.98) |
| 2 | `lr_trials=[2,0,8]` — freeze `W_out`, scale `W_in` 4× | deny the cheap storage site and force the object into `h` | **encoding up, memory null.** ph0 0.695 → 0.721 (p<1e-5); decays ~2×/masked step to chance by ph4 |
| 3 | Scenarios A / C / D (identity, occlusion, weight decay) | three routes to force the object into `h` | A null; D untestable (L2 collapses the place code); **C a FALSE POSITIVE**, killed by its own location control |
| 4 | Sequential displacement A→B→C→removed | the Tsao/Moser paradigm proper | **null**, n=8. Nothing forms during exposure. Behaviour *does* follow the present object and abandons it (5.0×, 8/8 seeds) |
| 5 | Square room + Moser session sequence | remove geometry so the object is the only cue | **null**, and self-explaining: quadrant decodes ~84% from `h` *with the object absent* — the net path-integrates |
| 6 | Multi-room training (3 rooms / 500-room pool, L-room and square) | close the second route: a changing room defeats dead reckoning | **informative null.** Per-room sRSA → 0.79 while the remapping index stays 0.01–0.03 against a derived +0.3904 for true remapping. A better *shared* map |
| 7 | E1 object / object-vector cells, 5 then 4 finished runs | the Høydal multi-anchor criterion | **null everywhere.** The location control, run over every (rollout room × anchor triad) pair, shows the score tracks **triad geometry**, not landmark presence |

**One-line summary:** the object is written into `W_out` ~89%
position-independently, so nothing location-specific ever has to enter `h` — and
every design has closed at most one of the two routes (geometry, dead reckoning)
by which the network localises without it.

### Three things a reader must not miss

- **Four false positives have been produced and retracted, every one killed by a
  control rather than by inspection.** The transferable rule: a within-unit null
  cannot see structure shared *across* units at one location.
- The OVC detector **fails its own Stage 0 gate** (0.575 recall at 4× injected
  amplitude), so every OVC null is a *weak bound*, not a verdict.
- `ROOMS_RUN1`'s three L-rooms are **exact translations** of one configuration,
  making that arm a weak manipulation. `ROOMS_SQUARE` is not.

---

## 2. Why it is not instantaneous

The goal said to start by assuming it should be. So:

```
one update = 8 envs x 256 steps, pRNN 500 hidden / 155 in / 147 out, small policy head
arithmetic required               1.905 GFLOP
compute time on an RTX 4060       0.127 ms
MEASURED                            909 ms
                                  -> 7,159x off compute-bound
```

`torch.profiler` over one rollout (256 sequential steps, 8 envs) says where it
goes:

```
self CPU 733.1 ms   vs   self CUDA 95.4 ms       ratio 7.7x
cudaLaunchKernel   26,706  = 104 kernel launches per step
all aten ops      169,993  = 664 ops per step
cudaMemcpyAsync     3,352  =  13 memcpys per step
```

**The GPU is idle 87% of the time.** Confirmed live: `nvidia-smi` during an
actual training run reads **13% utilisation**.

### How much is avoidable — measured against a floor

A minimal implementation of the *same* computation (same layer sizes, same 256
sequential steps, nothing else):

```
                              batch 8            batch 64
minimal, eager                 80.1 ms            51.0 ms
minimal, torch.compile         45.9 ms  (1.75x)   42.1 ms  (1.21x)
REPO rollout                  ~600 ms
```

**The rollout is ~7.5× slower than a minimal implementation of what it
computes.** That gap is overhead the repo adds, not physics.

---

## 3. The Categorical problem, and its fix

`torch.distributions.Categorical` cost **more than the entire neural network**:

```
per 256-step rollout:
  construction alone                                       39.2 ms
  construction + sample                                    75.5 ms
  construction + sample + log_prob + probs (BEFORE)       159.2 ms   621.9 us/step
  minimal Step module: W_in + W + W_out + policy head       80.1 ms   312.8 us/step
```

`validate_args` was the obvious suspect and was **ruled out** — already `False`
in this build; all numbers above are with it off.

**Diagnosis.** `log_prob` and `probs` are pure functions of `(logits, action)`
and neither feeds the environment. Only `sample` must be per-step. The rollout
was paying a gather and a softmax on all 256 sequential steps for values that
could be computed once.

**Fix applied** (`curious_george/rl/collect/collector.py`): store the
distribution's own normalised logits per step; derive `log_prob` and `probs`
after the loop in one batched op each.

**It is bitwise identical, and this is proven rather than assumed.** From
torch's source:

```python
Categorical.__init__: self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
Categorical.log_prob: return log_pmf.gather(-1, value).squeeze(-1)   # log_pmf IS self.logits
```

The replacement gathers from exactly that tensor. `probs` is
`softmax(self.logits)`. A 200-trial randomised check confirms equality; the
naive `log_softmax(L).gather(...)` is **NOT** bit-exact and would have broken the
oracle that `models.py`'s deliberately-redundant `log_softmax` exists to protect.

**Measured effect**, median of 7 runs to be robust to GPU contention:

```
             before    after     block speedup   saved per rollout
batch 8      155.7 ms  111.4 ms      1.40x            44.3 ms
batch 64     159.0 ms  113.8 ms      1.40x            45.3 ms
```

A constant absolute saving, independent of batch — exactly what launch-bound
theory predicts. On a ~1000 ms update that is **~4%**.

⚠️ **Correction to an earlier number in this session.** A single benchmark run
suggested this change alone gave 1.29× end-to-end (1584 → 2045 fps). That
measurement was taken while a training run shared the GPU and is **not
reliable**; the contention-robust A/B above (~4%) is the honest figure. The
1.29× should not be quoted.

**Gates after the change:** `tests/golden_omt` **5 passed**; full suite
**139 passed, 18 skipped, 7 deselected** — unchanged. `prnn_loss` after 6
updates identical to five decimal places (0.02330 before and after, delta
+0.000000).

**Re-profile after the fix:** 664 → **613 ops/step**, 104 → **96 launches/step**,
13 → **11 memcpys/step**. CPU/GPU ratio still 7.6×, so the disease is not cured —
only one symptom is.

---

## 4. Widening the rollout — 7.80x on the WRONG AXIS. Do not apply.

> 🔴 **RETRACTED 2026-08-19, before anything was applied.** The 7.80x is env
> steps/s. On **gradient steps per second** — which this repo's own 2026-08-11
> sweep calls "the quantity to maximise", warning that "FPS is actively
> misleading as an objective for this model" — width buys **0.97x, i.e. nothing**:
>
> ```
> num_envs=8 :  1584.3 env steps/s | 2048 env steps/grad step | 0.774 GRAD steps/s
> num_envs=64: 12356.3 env steps/s |16384 env steps/grad step | 0.754 GRAD steps/s
> ```
>
> Width raises env steps/update and gradient steps/update by the same factor, so
> the two cancel. Supporting evidence from the (killed) long run: sRSA reached
> only **0.481 at ~34M env steps** under `num_envs=64`, where the `num_envs=8`
> 3-room run had **0.669 at 31.5M** — the direction a gradient-step deficit
> predicts, though confounded by `layouts=one` vs `rooms`.
>
> **The real lever is reducing per-step overhead**, which raises updates/s and
> therefore gradient steps/s directly. That is recommendations 2–5, not this one.
> Ceiling estimate: taking the rollout from ~600 ms to the measured ~80 ms floor
> would take an update from ~1000 ms to ~480 ms = **~2.1x on gradient steps/s** —
> a sober number, and the honest one.
>
> The measurements below are still correct as *throughput* measurements; only the
> conclusion drawn from them was wrong. Kept because the mistake is instructive:
> it is the third time in this project that optimising a proxy inverted a verdict.

### Original section (throughput only)


`T = frames // num_envs`, so holding `frames = num_envs × seqdur` pins the number
of **sequential** launches at 256 and only widens each one. Since the ops are
launch-bound, width is close to free.

`tests/perf/benchmark.py` (the repo's own gate for perf refactors), 6 updates,
multi-room config, **measured before any concurrent GPU load**:

```
config          fps    s/update  steps/upd  prnn_loss @6  entropy @6
baseline_e8    1584.3     1.292       2047       0.02330       1.872
wide_e64      12356.3     1.325      16378       0.02341       1.963

throughput 7.80x       prnn_loss +0.5%       entropy healthy in both
```

`exp.num_envs=64 rl.frames=16384 rl.ppo_batch_size=4096`. Nearly the same
wall-clock per update for **8× the data**, and the learning metrics do not move.

Component scaling, measured separately:

```
envs x8  ->  rollout  x1.70   sublinear -- launch-bound
             WM step  x0.93   FLAT: a pooled step over 64 episodes costs the
                              same as over 8
             policy   x8.86   LINEAR -- ppo_batch_size must scale with frames
```

**The world-model gradient step is flat in batch size.** Gradient quality is
nearly free here, and the older "maximise gradient steps/s" framing optimised
the one term that does not scale.

---

## 5. Long-run validation — status

Three ~1-hour runs at the wide config were launched to validate over a realistic
horizon: `exp.layouts=one` (1 L-room), `rooms` (3 L-rooms), `pool` (500 L-rooms),
each 160,000 episodes × 256 steps = 41.0M env steps, `timeout 3600` as the cap.

⚠️ **Scope of this test, stated honestly.** The runs began at 23:46, *before*
`collector.py` was patched, so they exercise the **width** change (the 7.80×
lever) and **not** the Categorical fix. I considered restarting and decided
against it once the Categorical fix was corrected down to ~4% — three hours of
GPU to re-validate a 4% change is the wrong trade, and that change is already
covered by the bitwise proof, the golden oracle and the isolated A/B.

At the time of writing the first run (`one`) is ~22% through at ~13.7–17.7k
it/s. Logs: `scratchpad/runs/{one,rooms,pool}.log`.

⚠️ **The metric gate is only partly discharged.** `prnn_loss` and policy entropy
are matched across every configuration tested. **sRSA and SWdist are not** —
`tests/perf/benchmark.py --include-analysis` did not emit them, and at 6 updates
they would be noise. The long runs log them at `analysis_every_steps`; those
values must be read before the wide config goes to production.

---

## 6. Recommendations, in value order

| # | change | measured | risk | status |
|---|---|---|---|---|
| 1 | ~~`num_envs=64`, `frames=16384`, `ppo_batch_size=4096`~~ | 7.80× env steps/s but **0.97× gradient steps/s** | — | 🔴 **RETRACTED, do not apply** — see §4 |
| 2 | Batched `log_prob`/`probs` in the rollout | 1.40× on the block, ~4% of an update | none — bitwise proven, oracle green | **DONE**, on `sdu/speed` |
| 3 | Return logits from `acmodel`; drop `Categorical` from the rollout entirely | 113.8 → ~38 ms on the block, a further ~76 ms/rollout | **medium-high — changes the sampling RNG stream, so it breaks the bitwise oracle.** Needs an explicit decision, like `batched_wm` did | proposed, measured, **not implemented** |
| 4 | `torch.compile(mode="reduce-overhead")` on the policy + SR step | 1.75× on the compute part in isolation | medium — guards, recompiles | proposed |
| 5 | CUDA-graph the rollout step (96 launches → 1 replay) | up to the 7.5× overhead gap | **high** — documented silent-corruption incident (2026-07-22, `obs.direction`==184 after a device round-trip freed captured parameters) | last |

Note the existing `cuda_graph` flag graphs the **world-model trainStep**, which
is 12–22% of an update and flat in batch. It is aimed at the wrong term; the
rollout has no graph.

### Open thread

`cudaStreamSynchronize` appears **269 times ≈ 1.1 per step** after the fix. A
hard sync per step serialises CPU and GPU and would partly explain the persistent
7.6× ratio. I checked `vector.py` and `adapter.py`; the D2H transfers there are
per-*segment*, not per-step, so the source is **not yet identified**. Worth
finding — it is cheap if it turns out to be one stray `.item()`.

### On the 30-min–1-hour run target

```
plan                                                    steps/s    x   in 1 hour   full run
today, num_envs=8                                          2253  1.0   8,110,891      60.6h
+ num_envs=64 (measured)                                   8953  4.0  32,230,820      15.3h
+ rollout 2x                                              12417  5.5  44,700,568      11.0h
+ ppo_batch_size scaled (holds policy flat)               20467  9.1  73,681,949       6.7h
```

🔴 **This table is in env steps/s and is therefore the wrong axis too** (see §4). On gradient steps/s the achievable gain is ~2.1x, not 9x. **Engineering gets ~2x, not the ~60x a 1-hour *full* run needs** — an hour buys
~15% of the current budget. The rest must come from shortening the run, which is
a scientific decision and is answerable from data already owned: **46 archived
checkpoints per run.** From the remapping series, sRSA plateaus by ~73M steps
(0.787 at 73.4M, 0.791 at 94.4M, 0.79 at 482M), so **an hour is already enough
for the map-formation question**, while the E1 fraction kept moving (0.047 at 94M
→ 0.067 at 482M) and may need the full run. "How long must a run be?" deserves to
be its own question; it is pure analysis over existing archives.

### On many parallel environments with *different* rooms

The goal flags this as the hard case. The measurements say it is already the good
case:

- `DeviceTableShellPool` holds one observation bank per layout on device with a
  leading layout axis and **asserts the transition table is identical across
  layouts** — true because landmarks are walkable `Floor`. More rooms cost
  memory, not sequential work.
- Rollout cost is **sublinear in `num_envs`** (×1.70 for ×8) because the step
  count is fixed and only the width grows. Heterogeneous rooms ride the same
  batched indexing.
- The 500-layout pool run completed the full 482M-step budget at a rate
  comparable to the 3-room run, so layout count is not the bottleneck.

The bottleneck is **per-step Python and launch overhead**, identical whether the
64 streams are in one room or 64 different ones. Widen first, then attack the
613-ops-per-step figure.

---

## 7. What was NOT done, and why

- **Recommendation 3 not implemented.** It changes the sampling RNG stream and
  would break `tests/golden_omt`. Per the project's rule, a failing test is a
  conversation, not a thing to edit around.
- **CUDA graphs not enabled** — documented corruption incident; the fast feedback
  loop should exist first.
- **sRSA/SWdist not gated** (§5).
- **Nothing committed or pushed.** Branch `sdu/speed` carries one code change
  (`collector.py`, +26/−7) plus documentation. The parent branch still has 104
  unpushed commits and ~12 untracked files.

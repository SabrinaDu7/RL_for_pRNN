# Forcing the object representation into the pRNN hidden state

**Started:** 2026-08-01 · **Branch:** `sdu/object-into-hidden-state`
**Goal:** get the object representation into `h` instead of the readout. RL-side changes
allowed; the pRNN package must not change.

## Why the object ends up in `W_out`

Established 2026-07-31 (`docs/exp_object_trace_cells_2026-07-30.md`): object memory is
decoder-localised (gain-corrected readout +0.0625 vs dynamics +0.0157, 9/9 runs), and `h` is
unchanged across three reference frames.

The diagnosis: **the object's location is already linearly decodable from `h`**, because `h`
is a place code and the object is a deterministic function of position. Gradient descent has
no reason to touch the recurrent weights — adjusting a linear map is cheaper. `h` is
*redundant* with the object, not ignorant of it.

## Intervention 1 — freeze the readout

The pRNN optimizer has named param groups; **group 1 is `OutputWeights` (147, 500)**:

```
group 0 RecurrentWeights (500,500)   group 1 OutputWeights (147,500)
group 2 InputWeights     (500,155)   group 3 biases        (500,)
```

`tasks/omt/task.py` already scales per-group LR via `lr_trials` and already accepts a list,
so **`tasks.training.lr_trials=[2,0,2]` freezes `W_out` with zero code changes** and without
touching the pRNN package.

**Gate passed:** `W_out` is bitwise identical to the pre-exposure baseline for the whole run;
`W` and `W_in` train normally.

**Arithmetic:** `obs_pred = sigmoid(W_out @ h)`. With `W_out` fixed, any object-specific
change in the prediction *must* be a change in `h`.

### Result (3000 trajectories, seed 5200, object (7,11), matched to the Mila normal run)

Object contrast on the 200-trajectory probe, delta vs pre-exposure baseline:

| trajs | FROZEN | dW | dW_out | normal | dW | dW_out |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | +0.0272 | 0.057 | **0** | +0.0391 | 0.057 | 0.069 |
| 1600 | +0.0192 | 0.088 | **0** | +0.0776 | 0.087 | 0.114 |
| 2400 | +0.0253 | 0.115 | **0** | +0.0849 | 0.111 | 0.143 |
| 2992 | +0.0313 | 0.133 | **0** | +0.0936 | 0.131 | 0.151 |

**Frozen: +0.0214 ± 0.0112 over the last 5 checkpoints, and all 12 checkpoints from traj 800
onward are positive (sign test p ≈ 0.0002).**

So the intervention works: the object IS acquired, and 100% of it is in `h` by construction.
Two qualifiers — it is only ~23% of normal's effect, so the readout path is far more
efficient; and `dW` is nearly identical across conditions (0.133 vs 0.131), so the recurrent
weights move the same amount either way. What differs is what that movement is used for.

### But the representation is undetectable by every hidden-state measure tried

| net | map corr vs base | objmod %ile | obj-presence h ratio |
|---|---:|---:|---:|
| baseline | 1.0000 | — | 4.34 |
| normal 2992 | 0.9749 | 37.8 | 4.70 |
| FROZEN 2992 | 0.9757 | 38.4 | 4.42 |

Four measures, none discriminating: allocentric rate-map correlation, object-location
modulation percentile, view-conditioned `||Δh||`, and paired-probe object-presence
perturbation. The frozen net provably carries the object in `h`, and looks identical to the
net that does not.

**Reading:** the object enters `h` as a small distributed perturbation, not as localised
"object cells". The goal is met literally but not usefully — there is no measurable object
representation to point at.

## Mistakes made and corrected

- Reported "frozen peaks at 64% of normal" from wandb's 8-trajectory goal-modulation metric —
  the same metric flagged as too noisy on day one. The 200-trajectory probe shows both runs
  near zero at 400 trajectories with sign flips between adjacent checkpoints. **400
  trajectories is too short for this comparison**; the effect only emerges after ~800.
- Two hidden-state metrics (`||Δh||` in-view ratio, object-presence perturbation) were built
  before reasoning about what they isolate. Both measure the object driving `h` through
  `W_in`, which happens in every net, trained or not.

## Next

Freezing removes the cheap path but does not create pressure for a *structured* code. The
underlying cause is untouched: the object is still redundant with position. The next
intervention should attack that — vary the object's presence across episodes so position
alone stops predicting it and `h` must integrate evidence over time. That is an RL/collection
change, still within scope.

## Intervention 2 — stochastic object presence (`tasks/otc/`)

Each of the 8 training envs independently gets the object with probability
`presence_prob`, re-rolled every batch, so position stops predicting the object. Toggling
verified: grid fingerprints alternate cleanly between the two known layouts
(`170b5eec24237cf3` / `6f48cd7fbcae481d`) with no stale observation bank.

### Result: it makes things WORSE

Object contrast on the 200-trajectory probe, mean of the last 5 checkpoints, 3000-trajectory
runs, seed 5200, object (7,11):

| condition | contrast | dW | dW_out |
|---|---:|---:|---:|
| p=1.0, `W_out` free | **+0.0891 ± 0.0090** | 0.131 | 0.151 |
| p=1.0, `W_out` FROZEN | +0.0214 ± 0.0112 | 0.133 | **0** |
| p=0.5, `W_out` free | +0.0301 ± 0.0066 | 0.141 | 0.117 |
| p=0.5, `W_out` FROZEN | +0.0053 ± 0.0058 | 0.144 | **0** |

Randomising presence cuts the effect from +0.089 to +0.030; with the readout also frozen it
is +0.0053 ± 0.0058, indistinguishable from zero.

**Why, in hindsight:** with p=0.5 the optimal prediction is the *marginal probability* of
green at that location. The net never has to infer presence from history — predicting ~50%
is right on average. Breaking the redundancy removed information without creating any
pressure to replace it with inference.

### Positive decoding test: null across the board

Linear probe from `h` to object presence, trained and tested on disjoint trajectory splits,
restricted to in-view timesteps:

| net | held-out accuracy |
|---|---:|
| baseline (never saw the object) | 0.5443 |
| p=1.0 `W_out` free | 0.5479 |
| p=1.0 `W_out` FROZEN | 0.5457 |
| p=0.5 `W_out` free | 0.5460 |
| p=0.5 `W_out` FROZEN | 0.5483 |

Every net sits at ~0.545 — including the baseline that never saw the object. That 4.5 points
above chance is the object driving `h` through `W_in`; **no amount of exposure, in any
condition, improves on it.**

## Status: goal NOT achieved

Five measures across four conditions. `h` responds to the object as an *input* (4.3x in-view
perturbation, present in the untrained net), and exposure neither strengthens nor restructures
that response.

**Structural diagnosis.** The object is only ever relevant to prediction while it is in view —
at which point it is directly available in the current observation. There is almost no window
in which the network must *hold* it. Freezing `W_out` removes the cheap storage site but
creates no demand for memory; randomising presence removes the redundancy but is satisfiable
by predicting the marginal. Both interventions attack the symptom.

**What would create the demand** is making the object predictive of something not currently
observable: occlusion (a blocking object whose far side must be remembered), or an object
whose position varies within an episode so the current view constrains a future one. Both are
environment changes, not RL changes — and a blocking object is the same change flagged on
2026-07-31 as a PI decision, since it also invalidates the existing baselines.

## The measurement was wrong, and fixing it localises the failure

`thRNN_5win` is a `MaskedRNN` with **`inMask = [True, False x5]`** (verified on the loaded
net): the observation reaches the network only **1 timestep in 6**, while `outMask` is
all-True so it must predict the observation at every step. Phases 1-5 are driven by actions
and recurrence alone — pure memory. So the architecture already imposes a strong memory
demand, and my earlier decoding test was pooling input-driven with memory-only timesteps,
which are completely different claims.

Split by phase (`scripts/trace_presence_decoder.py`), held-out accuracy, chance 0.5:

| net | phase 0 (input fed) | phases 1-5 (MEMORY) |
|---|---:|---:|
| baseline (never saw it) | 0.6787 | 0.5307 |
| p=1.0 free | 0.6838 | 0.5305 |
| p=1.0 FROZEN | **0.6941** | 0.5274 |
| p=0.5 free | 0.6873 | 0.5305 |
| p=0.5 FROZEN | 0.6852 | 0.5310 |

**The object is readable from `h` while the observation is fed (0.68-0.69) and is gone one
masked step later (0.53).** It survives ~zero of the 5-step memory window, and the
memory-phase number is identical to the untrained baseline in every condition.

The only hidden-state measure any intervention has moved: frozen `W_out` raises phase-0
decoding 0.679 -> 0.694. Small, n=1, but it is the sole positive signal so far.

**Why the network does not remember.** During masked steps it reconstructs "green at (7,11)"
from its *position estimate*, maintained from actions, plus a fixed readout mapping. It needs
a position memory, not an object memory — the redundancy again, now precisely localised.

Even under p=0.5, where remembering genuinely would cut loss (predicting 0.5 costs MSE 0.25
against a 0/1 truth), the object is **1 of 49 view cells, ~2% of the prediction loss**. The
gradient is too weak to build a representation for.

That points at the remaining in-scope lever: raise the object's share of the loss by raising
the agent's *exposure* to it — `rl.k_curious` and start-position shaping. Both RL-side.

## Intervention 3 — per-parameter LR sweep: PARTIAL SUCCESS

Sweeping `tasks.training.lr_trials` (per-optimizer-group LR multipliers, groups
`[W, W_out, W_in]`). All RL-side config; the pRNN package is untouched.

| condition | object contrast | dW | dW_in | phase-0 decode | memory |
|---|---:|---:|---:|---:|---:|
| baseline | — | — | — | 0.6753 ± 0.0043 | 0.5302 ± 0.0011 |
| `[2,2,2]` normal | +0.0891 | 0.13 | 0.14 | 0.6871 ± 0.0050 | 0.5312 ± 0.0009 |
| `[2,0,2]` frozen `W_out` | +0.0214 | 0.13 | 0.14 | 0.6943 ± 0.0039 | 0.5296 ± 0.0014 |
| `[2,0,4]` frz + in2x | +0.0190 | 0.14 | 0.30 | 0.6826 | 0.5282 |
| **`[2,0,8]` frz + in4x** | +0.0121 | **0.16** | 0.74 | **0.7229 ± 0.0055** | 0.5316 ± 0.0009 |
| `[8,0,8]` frz + all 4x | −0.0219 | 0.84 | 0.73 | 0.7109 ± 0.0044 | 0.5311 ± 0.0005 |
| `[8,0,0]` recurrent only | −0.0262 | 0.83 | **0** | 0.6808 | 0.5303 |

(± is over 8 random trajectory splits of the decoder; single training seed per condition.)

**`W_in` is the lever, not the recurrent matrix.** `[8,0,0]` trains `W` at 4x with `W_in`
frozen and lands back at baseline decoding (0.6808) — pounding the recurrent weights alone
does nothing. The object enters `h` through the input weights.

**Best configuration: `[2,0,8]`** — freeze the readout, boost input weights 4x, leave the
recurrent LR alone. Phase-0 decoding **0.6753 -> 0.7229, +0.048 (~10 split-SD)**, with `dW`
at 0.16 rather than the 0.84 of `[8,0,8]`, so the model stays intact and the object contrast
stays positive. Clean monotone ordering: baseline < normal < frozen < frozen+all < frozen+in4x.

### What this does and does not achieve

**Achieved:** object *encoding* in `h` is significantly and controllably increased by
RL-side config alone. The hidden state carries measurably more object information than the
pre-exposure net, and more than the normally-trained net.

**Not achieved:** object *memory*. Memory-phase decoding is 0.5296–0.5316 in every condition
including the untrained baseline — gaps of ~1 split-SD. The object loads into `h` when its
observation arrives and is gone one masked step later. No intervention has moved this.

So `h` encodes the object better; it still does not hold it.

**Caveat:** single training seed per condition. The ± above is decoder-split variability, not
between-seed. Seed replication of `[2,0,8]` (seeds 5201, 5202) against the three Mila
`[2,2,2]` seeds is the outstanding check before the +0.048 is quotable.

### Seed replication (n=3 vs n=3): the encoding result holds

Phase-0 presence decoding from `h`, final checkpoint, three training seeds each:

| condition | 5200 | 5201 | 5202 | mean |
|---|---:|---:|---:|---:|
| baseline | — | — | — | 0.6787 |
| `[2,2,2]` normal (Mila) | 0.6838 | 0.6866 | 0.6845 | **0.6850 ± 0.0012** |
| `[2,0,8]` frz + in4x | 0.7128 | 0.7160 | 0.7186 | **0.7158 ± 0.0024** |

**Perfect separation** — the worst `[2,0,8]` seed (0.7128) beats the best normal seed
(0.6866). **+0.031 over normal training, +0.037 over the pre-exposure baseline**, between-seed
SD ~0.002.

Memory phase: `[2,0,8]` 0.5319 ± 0.0013 vs normal 0.5299 ± 0.0004. Separated in all three
seeds but the magnitude is +0.002 against a chance floor of 0.500 — 3.2 points above chance
instead of 3.0. **Not a meaningful improvement; the object still does not persist.**

## Summary of the goal

**ACHIEVED — object encoding in `h`.** `tasks.training.lr_trials=[2,0,8]` (freeze the readout,
boost input weights 4x) raises linear decodability of object presence from the hidden state
from 0.685 to 0.716, replicated across three seeds with perfect separation, no change to the
pRNN package and no change to the environment. The mechanism is specific: `W_in` is the
lever, and freezing it (`[8,0,0]`) returns decoding to baseline however hard the recurrent
matrix trains.

**NOT ACHIEVED — object memory in `h`.** Across nine conditions, memory-phase decoding sits at
0.530 ± 0.002 including the untrained baseline. During masked steps the network reconstructs
the object from its *position estimate* plus a fixed readout mapping; it needs a position
memory, not an object memory, and a stationary object in a fixed room gives it no reason to
acquire one.

Whether the goal is met depends on which reading was intended. "The hidden state contains the
object representation" is true in the encoding sense and false in the persistence sense.

## Intervention 4 — noise is NOT the limiter (inference-time test, no retraining)

Replaying with `trainNoiseMeanStd=(0,0)`:

| net | phase-0 on/off | memory on/off |
|---|---|---|
| baseline | 0.6787 / 0.6971 | 0.5307 / **0.5302** |
| `[2,2,2]` normal | 0.6838 / 0.7081 | 0.5305 / **0.5334** |
| `[2,0,8]` | 0.7128 / **0.7331** | 0.5314 / **0.5355** |

Removing noise entirely raises phase-0 ~2 points everywhere and leaves memory at ~0.53.
**The memory is not being erased — it was never written.** Rules out the last
"present but degraded" explanation.

## Intervention 5 — `rl.k_curious` (raise the object's share of the loss): FAILS

| condition | contrast | phase-0 | memory |
|---|---:|---:|---:|
| `[2,0,8]` k=1 (ref) | +0.0258 | 0.7128 | 0.5314 |
| `[2,0,8]` k=5 | +0.0136 | 0.7078 | 0.5319 |
| `[2,0,8]` k=20 | **−0.1045** | 0.6787 | 0.5280 |

k=5 marginally worse; k=20 catastrophic — contrast strongly negative, phase-0 back to exactly
the baseline. Cranking curiosity destabilises training instead of concentrating exposure.
Memory unchanged.

## Intervention 6 — random object POSITION (running)

`tasks.otc.random_position=True` re-samples the object's cell every batch from the 172
walkable cells. Strictly stronger than randomising *presence*, which failed because it was
satisfiable by predicting the marginal — there is no useful marginal over 172 locations, so
reconstruction-from-position stops working and remembering is the only route. Each location
produces a distinct grid fingerprint with its own cached observation bank (verified).

## The pooled memory metric was hiding the result (third pooling error this project)

Splitting the five masked steps individually instead of pooling them. `ph0` is the step where
the observation is fed; `ph1..ph5` have no input at all.

| group | ph0 | ph1 | ph2 | ph3 | ph4 | ph5 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6776 | 0.5784 | 0.5371 | 0.5210 | 0.5135 | 0.5071 | 1 |
| `[2,2,2]` fixedpos | 0.6902±0.001 | 0.5824±0.005 | 0.5394±0.001 | 0.5238±0.001 | 0.5130±0.001 | 0.5076±0.002 | 3 |
| `[2,0,8]` fixedpos | **0.7214±0.001** | **0.5943±0.001** | **0.5465±0.003** | 0.5252±0.001 | 0.5145±0.002 | 0.5098±0.001 | 3 |
| `[2,0,8]` RANDPOS | **0.7344** | **0.5969** | 0.5438 | **0.5287** | **0.5159** | **0.5114** | 1 |

(± over training seeds; each entry already averaged over 4 decoder splits.)

**There IS a memory trace.** Object information survives the first masked step well above
chance and decays toward chance by the fifth. Measured in points above chance at ph1:
baseline 7.8 -> normal 8.2 -> `[2,0,8]` **9.4** -> RANDPOS **9.7** — **+14% over normal
training, +20% over baseline**, n=3 with non-overlapping error bars.

The pooled "memory" figure averaged a real ph1 effect with ph3-ph5 sitting near chance,
producing the flat ~0.530 reported across ten conditions. **Third time pooling has hidden a
result in this project** — the others were the reward map binned by agent position, and
pooling input-driven with memory-only timesteps. The lesson: always split by the variable the
mechanism runs on before concluding a null.

## FINAL STATUS: goal achieved, modestly

`tasks.training.lr_trials=[2,0,8]` (freeze the readout, boost input weights 4x), optionally
with `tasks.otc.random_position=True`. **No change to the pRNN package.**

| measure | normal training | best config | change |
|---|---:|---:|---|
| encoding (ph0) | 0.6902 ± 0.001 | **0.7344** | +0.044, perfect seed separation |
| memory (ph1) | 0.5824 ± 0.005 | **0.5969** | +0.015, +18% above chance |
| memory (ph3) | 0.5238 ± 0.001 | **0.5287** | +0.005 |

The hidden state carries measurably more object information than under normal training, both
instantaneously and in the 1-2 steps after the observation is withdrawn.

**Honest bounds.** The trace still decays to near-chance by ph4-5 in every condition — this
extends a short memory, it does not create a persistent one. And in the *normal*
configuration the readout still carries the bulk of the object prediction; `[2,0,8]` shifts
where the information lives at some cost to prediction accuracy (contrast +0.026 vs +0.089).
The RANDPOS row is n=1; its `[2,2,2]` control is still training and is needed to attribute
the gain between "random position helps" and "`[2,0,8]` helps".

**Mechanism:** `W_in` is the lever. `[8,0,0]` trains the recurrent matrix at 4x with `W_in`
frozen and returns to baseline. Freezing `W_out` forces the network to use the input pathway
rather than the cheaper readout; scaling `W_in` is what then loads `h`.

## Attribution: the two interventions do DIFFERENT things (final, n=3)

| group | ph0 | ph1 | ph2 | ph3 | ph4 | ph5 | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6776 | 0.5784 | 0.5371 | 0.5210 | 0.5135 | 0.5071 | 1 |
| `[2,2,2]` fixedpos | 0.6902±0.001 | 0.5824±0.005 | 0.5394±0.001 | 0.5238±0.001 | 0.5130±0.001 | 0.5076±0.002 | 3 |
| `[2,0,8]` fixedpos | **0.7214±0.001** | **0.5943±0.001** | **0.5465±0.003** | 0.5252±0.001 | 0.5145±0.002 | 0.5098±0.001 | 3 |
| `[2,0,8]` RANDPOS | 0.7344 | 0.5969 | 0.5438 | 0.5287 | 0.5159 | 0.5114 | 1 |
| `[2,2,2]` RANDPOS | 0.6938±0.004 | 0.5864±0.003 | 0.5436±0.004 | **0.5283±0.004** | **0.5182±0.004** | **0.5118±0.005** | 3 |

**`[2,0,8]` drives ENCODING — established.** ph0 0.6902 -> 0.7214 (+0.031, ~30 SD),
ph1 0.5824 -> 0.5943 (+0.012, ~2.4 SD). Tight error bars, n=3, perfect seed separation.
Random position alone barely moves ph0 (0.6938).

**Random position drives LATE MEMORY — suggestive, NOT established.** ph5 above-chance
0.76 -> 1.18 (+55%), ph4 1.30 -> 1.82 (+40%), ph3 2.38 -> 2.83 (+19%). But at n=3 the SD is
0.004-0.005 against differences of 0.004-0.005 — about 1 SD. **The +142% I first reported was
a single seed at the high end of its distribution.**

## FINAL STATUS

**Achieved:** `tasks.training.lr_trials=[2,0,8]` (freeze the readout, boost `W_in` 4x) puts
measurably more object information into `h` — instantaneously (+0.031 at ph0) and one masked
step later (+0.012 at ph1), replicated at n=3 with non-overlapping error bars, with no change
to the pRNN package.

**Mechanism:** `W_in` is the lever. `[8,0,0]` (recurrent at 4x, `W_in` frozen) returns to
baseline. Freezing `W_out` removes the cheap storage site; scaling `W_in` loads `h`.

**Cost:** object contrast +0.026 vs +0.089 for normal training. This trades prediction
accuracy for hidden-state representation.

**Not established:** persistent memory. The trace still decays to near-chance by ph4-ph5 in
every condition. Randomising the object's position is the most promising lever for it and is
mechanistically the right idea — with the location unpredictable, the position estimate can
no longer reconstruct the object — but at n=3 the effect is ~1 SD. It needs more seeds before
anyone relies on it.

## Errors made and corrected in this document

1. "Frozen peaks at 64% of normal" — read off wandb's 8-trajectory metric already flagged as
   too noisy. Retracted; 400 trajectories is too short, the effect emerges after ~800.
2. Two `h`-metrics (`||dh||` in-view ratio, object-presence perturbation) built before
   reasoning about what they isolate; both measure the object driving `h` through `W_in`,
   which happens in every net including untrained ones.
3. **Pooling the five masked phases**, which averaged a real ph1 effect with near-chance
   ph3-ph5 and produced a flat ~0.530 across ten conditions. I twice told the user the memory
   half was structurally impossible on the strength of that number. Third pooling error in
   this project.
4. "+142% memory improvement" from a single seed; n=3 gives +55% at ~1 SD.

**Standing lesson:** split by the variable the mechanism runs on before concluding a null, and
never quote an effect size from n=1.

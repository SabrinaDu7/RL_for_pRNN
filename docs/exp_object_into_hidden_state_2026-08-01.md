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

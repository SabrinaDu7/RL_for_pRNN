# Forcing the object representation into the pRNN hidden state

**Dates:** 2026-08-01 → 2026-08-02 · **Branch:** `sdu/object-into-hidden-state`
**Goal:** get the object representation into `h` rather than the readout layer.
RL/task-side changes permitted; the `prnn` package must not change (and did not).

*(This document supersedes the running log kept while the work was in progress; the log is
in git history at `00695a2` and earlier.)*

---

## 1. Summary

| claim | status | evidence |
|---|---|---|
| **`[2,0,8]` puts more object information into `h`** | **ESTABLISHED** | ph0 0.6947→0.7214, t=14.4, p<1e-5 (n=3 vs 10) |
| ...and it survives one masked step | **ESTABLISHED** | ph1 0.5834→0.5938, t=7.3, p=3e-5 |
| `W_in` is the lever, not the recurrent matrix | **SUPPORTED** | `[8,0,0]` (W at 4x, `W_in` frozen) → 0.6892, baseline-level |
| Randomising object position creates *persistent* memory | **NOT ESTABLISHED** | pooled ph1-5 p=0.092; nothing survives Bonferroni |
| Any intervention makes `h` hold the object beyond ~2 steps | **NULL** | every condition decays to ~0.51 by ph5 |

**Cost:** `[2,0,8]` trades prediction accuracy for representation — object contrast
+0.019 vs +0.078 for normal training.

---

## 2. Question

Prior work (`docs/exp_object_trace_cells_2026-07-30.md`) established that novel-object memory
in this architecture is **decoder-localised**: transplanting `W_out` alone onto untrained
dynamics reproduces the object effect (gain-corrected readout +0.0625 vs dynamics +0.0157,
9/9 runs), while `h`'s spatial tuning is unchanged across three reference frames (allocentric
rate maps r≈0.98, object-location modulation, object-vector tuning).

**Diagnosis:** the object's location is already linearly decodable from `h`, because `h` is a
place code and the object is a deterministic function of position. Gradient descent has no
reason to touch the recurrent weights — adjusting a linear map is cheaper. `h` is *redundant*
with the object, not ignorant of it.

**Question:** can RL/task-side changes alone move the representation into `h`?

---

## 3. Methods

### 3.1 The architecture's mask structure dictates the analysis

`thRNN_5win` is a `MaskedRNN` with **`inMask = [True, False x5]`** (verified on the loaded
net). The observation reaches the network **only 1 timestep in 6**; `outMask` is all-True, so
it must predict the observation at *every* step. Phases 1-5 are driven by actions and
recurrence alone.

This splits "is the object in `h`" into two questions that must never be pooled:

- **ph0 — encoding.** The observation is being fed. Even an untrained net scores 0.679 here,
  because the object drives `h` through `W_in`.
- **ph1-ph5 — memory.** No input at all. This is where "the hidden state *holds* the object"
  is decided.

### 3.2 Paired probes

Two probes with **byte-identical trajectories** — object-present and object-absent. Possible
because the object is a non-blocking `FloorBright` tile, so for a fixed action sequence the
agent's path is unchanged (verified: identical `agent_pos` and `agent_dir`; 16.2% of timesteps
differ in observation). 688 trajectories (172 walkable cells × 4 head directions) × 256 steps.

Determinism: `PredictiveNet.predict` injects fresh noise on every call
(`trainNoiseMeanStd=(0,0.05)`) — two identical calls differ by ~0.4 in `h`. `replay_checkpoint`
seeds torch immediately before each forward so every checkpoint sees the same realisation.
Gate: replaying one checkpoint twice is bitwise identical.

### 3.3 Decoding measure

A logistic probe reads object presence out of `h`, restricted to timesteps where the object
location is inside the 7×7 view, **split by mask phase**. Train/test split is by
**trajectory**, so nothing leaks; each number averages 3 splits. Chance = 0.5.
(`scripts/trace/trace_presence_decoder.py`)

### 3.4 Interventions

All expressed through `tasks.training.lr_trials` — per-optimizer-group LR multipliers over
groups `[W, W_out, W_in]` (group 1 is `OutputWeights`, shape (147,500), verified). Existing
config; **no code change to the pRNN**. `tasks/otc/` adds stochastic object presence and
per-batch random object position, implemented by toggling `new_obj_pos` on the environment —
the banked observation wrapper re-keys off the live grid fingerprint inside `reset()`, so no
stale observations survive a toggle (verified).

---

## 4. Results

![phases](../../outputs/trace/fig_otc_phases.png)

**Figure 1.** Object presence decodable from `h` at each mask phase. Right panel is the same
data above chance on a log axis: decay is close to exponential, ~2× per masked step, reaching
~0.01 (accuracy 0.51) by ph5 in every condition. `[2,0,8]` and the RANDPOS conditions sit
above the rest at ph0-ph1; by ph4-ph5 all conditions have converged near chance.

![encoding](../../outputs/trace/fig_otc_encoding.png)

**Figure 2.** Phase-0 decoding per condition, every training seed plotted, red bar = mean.
`[2,0,8]`'s three seeds are tightly clustered and fully separated from the 10 fixedpos seeds.
Conditions with a single point are n=1 and exploratory.

![tradeoff](../../outputs/trace/fig_otc_tradeoff.png)

**Figure 3.** Left: encoding is bought at the cost of prediction — the conditions with the
highest ph0 decoding have the *lowest* object contrast. Right: ph0 decoding against
`‖ΔW_in‖/‖W_in‖`; `[8,0,0]` sits at x=0 with baseline-level decoding, the cleanest single
piece of evidence that `W_in` is the causal pathway.

### 4.1 All conditions

| condition | n | ph0 | ph1 | ph2 | ph3 | ph4 | ph5 | contrast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (pre-exposure) | 1 | 0.6787 | 0.5781 | 0.5378 | 0.5210 | 0.5131 | 0.5066 | — |
| `[2,2,2]` fixedpos (normal) | 10 | 0.6947 | 0.5834 | 0.5413 | 0.5242 | 0.5147 | 0.5087 | +0.0776 |
| `[2,0,2]` frozen `W_out` | 1 | 0.6975 | 0.5777 | 0.5407 | 0.5225 | 0.5140 | 0.5078 | +0.0313 |
| `[2,0,4]` frz + in2x | 1 | 0.6870 | 0.5805 | 0.5371 | 0.5183 | 0.5104 | 0.5073 | +0.0039 |
| **`[2,0,8]` frz + in4x** | 3 | **0.7214** | **0.5938** | 0.5459 | 0.5247 | 0.5146 | 0.5095 | +0.0187 |
| `[8,0,8]` frz + all 4x | 1 | 0.7151 | 0.5939 | 0.5416 | 0.5258 | 0.5154 | 0.5086 | −0.0146 |
| `[8,0,0]` recurrent only | 1 | 0.6892 | 0.5786 | 0.5434 | 0.5224 | 0.5118 | 0.5077 | −0.0192 |
| `[2,2,2]` RANDPOS | 10 | 0.6974 | 0.5851 | 0.5421 | 0.5265 | 0.5175 | 0.5107 | −0.0291 |
| `[2,0,8]` RANDPOS | 1 | 0.7347 | 0.5954 | 0.5444 | 0.5299 | 0.5151 | 0.5116 | −0.0115 |
| `[2,0,8]` k_curious=5 | 1 | 0.7151 | 0.5907 | 0.5417 | 0.5226 | 0.5133 | 0.5091 | +0.0136 |
| `[2,0,8]` k_curious=20 | 1 | 0.7177 | 0.5946 | 0.5436 | 0.5278 | 0.5148 | 0.5098 | +0.0402 |
| p=0.5 presence, free | 1 | 0.6940 | 0.5796 | 0.5422 | 0.5212 | 0.5131 | 0.5091 | +0.0260 |
| p=0.5 presence, frozen | 1 | 0.6925 | 0.5863 | 0.5391 | 0.5240 | 0.5138 | 0.5084 | +0.0129 |

### 4.2 Encoding — established

`[2,0,8]` vs `[2,2,2]` fixedpos, Welch t-test:

```
ph0   0.7214 ± 0.0009 (n=3)  vs  0.6947 ± 0.0052 (n=10)   t = 14.44   p < 1e-5
ph1   0.5938 ± 0.0003        vs  0.5834 ± 0.0042          t =  7.28   p = 3e-5
```

Every `[2,0,8]` seed exceeds every fixedpos seed. Random position alone barely moves ph0
(0.6974), so the gain is attributable to the LR configuration, not the task change.

### 4.3 Memory — NOT established

`[2,2,2]` RANDPOS vs `[2,2,2]` fixedpos, both **n=10**:

```
phase   fixedpos        RANDPOS         diff      t      p     p_bonf
ph1     0.5834±0.004    0.5851±0.003   +0.0017   1.02  0.324   1.000
ph2     0.5413±0.002    0.5421±0.003   +0.0008   0.66  0.518   1.000
ph3     0.5242±0.001    0.5265±0.003   +0.0023   2.10  0.057   0.342
ph4     0.5147±0.002    0.5175±0.003   +0.0028   2.53  0.022   0.131
ph5     0.5087±0.001    0.5107±0.003   +0.0020   1.59  0.137   0.824

pooled memory (ph1-5):  t = 1.80,  p = 0.092
```

**Nothing survives correction for six comparisons; the pooled test is p = 0.092.** The
differences are consistently positive and concentrated at late phases, the direction the
mechanism predicts — but that is not evidence at this power. A sign test on 6/6 positive
phases gives p = 0.016, but the six phases come from the same networks and the same decoder
and are strongly correlated, so its independence assumption is violated; it should not be
quoted.

### 4.4 Null results

- **Stochastic object presence (p=0.5)**: no benefit (ph0 0.6940 free / 0.6925 frozen). With
  p=0.5 the loss-minimising prediction is the *marginal probability* of green, so the network
  never has to infer presence from history.
- **`k_curious` = 5, 20**: no meaningful effect (ph0 0.7151, 0.7177 vs 0.7128 at k=1). Raising
  the curiosity weight does not concentrate exposure usefully.
- **Injected noise is not the limiter**: replaying with `trainNoiseMeanStd=(0,0)` raises ph0
  ~2 points everywhere and leaves memory phases unchanged. The memory was never written, not
  erased.

---

## 5. Interpretation

`[2,0,8]` works by removing the cheap storage site (`W_out` frozen) and widening the pathway
that carries the object into `h` (`W_in` at 4×). The `[8,0,0]` control shows this is
specifically the *input* pathway: training the recurrent matrix at 4× with `W_in` frozen
returns decoding to baseline.

But the representation is **transient**. Object information decays ~2× per masked step and is
near chance within three, in every condition. `h` encodes the object more strongly; it does
not hold it longer.

Why the network never learns to hold it: during masked steps it can reconstruct "green at
(7,11)" from its *position estimate*, maintained from actions. It needs a position memory, not
an object memory. Randomising the object's location should remove that shortcut, and the
effect goes in the predicted direction — but at n=10 per arm it is p = 0.09, so either the
effect is small or the shortcut is not the whole story.

---

## 6. Bounds

- Encoding is established at n=3 (intervention) vs n=10 (control), one object location (7,11),
  one environment. Not tested at other object positions or in FourRooms.
- The `[2,0,8]` gain costs prediction accuracy (+0.019 contrast vs +0.078). Whether that is a
  good trade depends on whether the pRNN is wanted as a predictor or as a model of
  hippocampal coding.
- Many conditions in §4.1 are n=1 and exploratory; do not quote their effect sizes.
- All decoding uses one probe design (fixed object at (7,11), random-action trajectories). A
  different probe could shift absolute accuracies, though the between-condition ordering
  should be robust.

---

## 7. Errors made, and what each cost

Recorded because several produced conclusions that were reported and later withdrawn.

1. **Read a head-to-head off wandb's 8-trajectory goal-modulation metric** — the metric
   already flagged as too noisy on day one. Produced "frozen peaks at 64% of normal"; the
   200-trajectory probe showed both runs near zero with sign flips between adjacent
   checkpoints. *Cost: a retracted comparison. 400 trajectories is too short; the effect
   emerges after ~800.*
2. **Built two hidden-state metrics before reasoning about what they isolate** (`‖Δh‖` in-view
   ratio, paired object-presence perturbation). Both measure the object driving `h` through
   `W_in`, which happens in every net including untrained ones, so neither could discriminate.
   *Cost: two wasted analysis rounds.*
3. **Pooled the five masked phases.** This averaged a real ph1 effect with near-chance
   ph3-ph5 and produced a flat ~0.530 across ten conditions. *Cost: I twice told the user the
   memory half was structurally impossible with RL-side changes. It was not.* Third pooling
   error in this project — the others were the reward map binned by agent position, and
   pooling input-driven with memory-only timesteps.
4. **Quoted "+142%" from n=1.** n=3 gave +55%; matched n=10 gave a non-significant result.
   *Cost: a headline walked back twice.*
5. **Evaluated a still-running job.** `max(steps)` picked step 0 of `k_curious=20`, giving
   ph0=0.6787 and contrast=−0.1045, reported as "catastrophic". True values at step 2992 are
   0.7149 and +0.0402 — k=20 is fine, with a *higher* contrast than k=1. The tell was writing
   "back to **exactly** the baseline 0.6787": two independent measurements agreeing to four
   decimals is impossible. Reproduced deliberately to confirm. *Cost: a fabricated negative.*

**Standing lessons:** split by the variable the mechanism runs on before concluding a null;
never quote an effect size from n=1; check a job has finished before reading its checkpoints;
treat implausibly exact agreement as a bug signal, not a finding.

---

## 8. Reproduction

```bash
# best encoding configuration
uv run tasks/omt/main_task.py exp.exp_name=otc exp.seed=5200 \
  tasks.new_obj_loc=[7,11] tasks.training.lr_trials=[2,0,8] \
  tasks.training.num_trajs=3000 tasks.training.saving_interval_trajs=200

# random object position (tasks/otc/)
uv run tasks/otc/main_task.py exp.exp_name=otcRP exp.seed=5200 \
  tasks.otc.presence_prob=1.0 tasks.otc.random_position=True \
  tasks.training.lr_trials=[2,2,2] tasks.training.num_trajs=3000

# analysis
uv run python scripts/otc_figures.py collect   # -> outputs/trace/otc_results.npz
uv run python scripts/otc_figures.py plot      # -> outputs/trace/fig_otc_*.png
```

Key modules: `scripts/trace/trace_probe.py` (probe build/replay), `scripts/trace/trace_presence_decoder.py`
(phase-split decoding), `scripts/trace/trace_readout_test.py` (object contrast, weight deltas),
`scripts/otc_figures.py` (collection + figures), `tasks/otc/` (stochastic presence/position).

Gate throughout: `uv run pytest` → 126 passed, 0 failed, 7 deselected.
Compute: ~11 h on one RTX 4060, no cluster jobs.

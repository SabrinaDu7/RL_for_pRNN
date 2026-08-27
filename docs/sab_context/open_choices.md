# Open choices

Running list of decisions that are **methodological, not bugs** — each one changes what a
number means, so it needs a deliberate call (and for several, a conversation with the PI)
rather than a quiet fix. Nothing here is broken code; it is all "which convention do we
want, and why".

Each entry states the choice, the evidence gathered so far, and what is currently in the
tree. Add to this file rather than resolving an entry silently; when one is decided, record
the decision and the date, and update the methods doc it affects.

---

## 1. SI of low-activity units: zero, mask, or judge by reliability

**Status: deferred, keeping the original zeroing.** (2026-08-11)

`PredictiveNet.calculateSpatialMetrics` and `calculateSpatialRepresentation` set a unit's
spatial information to **0** when it is active in fewer than `active_time_threshold=200`
samples.

Three readings:

- **zero** — a near-silent unit carries no spatial information, and the mutual-information
  estimator is upward-biased at small sample counts, so 0 is closer to the truth than the
  estimate. `mean SI` then means "average spatial information per unit in the network".
- **mask (NaN)** — the threshold is a statement about estimator *reliability*, not about the
  unit's information, so it should be excluded rather than counted as 0. `mean SI` then
  means "average over units we can measure".
- **reliability criterion** — drop the activity count entirely and exclude on split-half
  stability or the trajectory-level shuffle null instead.

**Evidence** (`uv run python throwaway/ported/si_threshold_audit.py`, checkpoint
`outputs/ckpts/pRNN_curious_26-07-23-10-06-25`, probe `outputs/trace/probe_lroom_noobj`):

```
=== probe scale (162,368 samples) ===
threshold 200 = 0.12% of samples;  units zeroed: 1 / 500
                 units   median SI   median split-half r
below threshold      1         nan                0.8229
above threshold    499      0.7591                0.9951
mean SI: zeroed 0.8151  vs  masked 0.8167   (+0.2%)

=== training-eval scale (1,888 samples — what wandb logs) ===
threshold 200 = 10.59% of samples;  units zeroed: 5 / 500
                 units   median SI   median split-half r
below threshold      5      3.1480                0.9995
above threshold    495      0.6471                0.9875
mean SI: zeroed 0.7863  vs  masked 0.7942   (+1.0%)
```

What this says:

- The threshold is an **absolute count**, so its meaning depends entirely on how many samples
  were pooled. The offline probe pools ~86x more samples than the training-loop eval, so the
  same constant is ~86x more permissive there. At probe scale it catches one genuinely dead
  unit (0 active samples); at eval scale it catches five live ones.
- At eval scale the zeroed units have **~5x the spatial information** of the rest
  (median 3.15 vs 0.65) and their maps reproduce across independent halves at **r = 0.9995**.
  Split-half reliability was chosen as the discriminator precisely because small-sample
  estimator noise does not reproduce on an independent half. High SI + high reliability +
  low activity is the signature of a tight place field — i.e. an activity threshold is
  anti-correlated with the thing the metric exists to detect.
- The aggregate effect is nonetheless **small**: 0.2–1.0% on mean SI.

**Caveats on that evidence:** n=5 units at eval scale, so the medians are noisy; a median
split-half r of 0.9875 across *all* units suggests the stability metric may be saturating
(likely common occupancy/gain structure dominating the maps), which weakens it as a
discriminator; one checkpoint only.

**Why it still matters despite being ~1%:** the number of sub-threshold units depends on
network sparsity, and sparsity is exactly what a changed optimizer regime moves. So the bias
is not constant across the comparison the `main_train` re-run is meant to make.

**Currently in the tree:** original zeroing, unchanged. A masking implementation was written
and then reverted, so `plotTuningCurvePanel`'s NaN-unsafe sort was reverted with it — see
entry 6.

---

## 1b. Dropout during evaluation — RESOLVED 2026-08-11: off

**Decision: the spatial eval runs in `eval_mode`.** Noise is kept.

What dropout actually is here, from the code rather than the name: it is applied to
`obs_out`, the network's *input*, and never to `obs_target_out`, the prediction *target*
(`Architectures.clip_mask`). So the net predicts the CLEAN observation from a CORRUPTED
one — structurally a denoising objective, not plain regularisation. It is on the
observation only; dropout-on-action was tried and commented out. `p = predNet.dropp = 0.15`.

A neuroscience reading is available (unreliable sensory sampling; forcing reliance on
internal predictive dynamics) but is **undocumented** — nothing in either repo's docs,
README or commit messages states it, so it should not be cited as the authors' intent.
Note also that `inMask = [True, False x5]` already withholds the observation *entirely* on
5 of every 6 timesteps, which applies that same pressure far more aggressively than 15%
dropout does.

Why it still comes off at measurement time:

- torch implements **inverted** dropout: survivors are scaled by `1/(1-p) = 1.1765` in
  train mode precisely so the expectation matches eval mode (verified: train mean 0.9911
  vs eval 1.0000). The rescaling presupposes eval-time disabling. Measuring through it
  reports activations inflated 17.6% and then randomly zeroed — neither the training nor
  the inference distribution.
- The agent never sees it: `predict_single` skips `clip_mask` entirely, so the SR the
  policy acts on has never been dropout-corrupted.
- **Every other eval path in the repo already disables it** — `evaluation/task.py:96,119`,
  `tasks/omt/task.py:277`, `tasks/template_task.py:75`, `trace_probe.py:84,152`, plus four
  analysis scripts. `evaluate_spatial_representation` was the lone outlier; `task.py:97`
  explicitly calls `.train()` for the training phase, so the distinction is deliberate
  elsewhere.

**Consequence to remember:** historical wandb sRSA/SWdist/SI curves were logged with
dropout ON, so numbers after this change are not directly comparable to runs before it.

---

## 1c. SWdist is too noisy to compare checkpoints with

**Open, and it undermines one of the two headline metrics.**

SWdist is the median, over sleep frames, of the cosine distance to the nearest wake frame.
"Sleep" is one noise-driven `spontaneous(sleep_timesteps=500, 0, sleepstd)` rollout.

With the wake activity held **completely fixed** and only the sleep rollout redrawn
(n=10): mean 0.19197, sd 0.05279, range [0.146, 0.306] — a **27.5% coefficient of
variation** and better than a 2x spread, from one implementation choice.

Longer rollouts make it worse, not better (n=8 draws each):

```
sleep_timesteps      mean        sd      CV
            500   0.17454   0.02108   12.1%
           1000   0.15691   0.03250   20.7%
           2000   0.14494   0.04017   27.7%
           4000   0.15902   0.07537   47.4%
```

The variance is **dynamical, not statistical**: a noise-driven RNN trajectory is strongly
autocorrelated, so a longer rollout is not more independent samples, it is a longer
excursion that can wander further from the wake manifold. Both the mean and the variance
depend on `sleep_timesteps`, so the metric is not scale-free in it either.

Consequences:

- As computed, SWdist cannot detect anything smaller than roughly a 50% change. The
  project's criterion "SWdist should be LOW" is being read off a number with a +/-27%
  floor.
- **A retraction:** an earlier comparison here attributed ~23% of SWdist to dropout
  (0.194 with, 0.150 without). That difference sits entirely inside the noise band, was
  one seed each, and is not supported. The dropout decision in 1b rests on the mechanism,
  not on that number.

Likely fix: average over many INDEPENDENT short sleep rollouts rather than lengthening
one. Needs a decision on how many, and on whether the reported quantity changes meaning.

---

## 2. How many samples the spatial metrics should use

`maxNtimesteps = 4000` is a module-level constant in
`prnn/analysis/representationalGeometryAnalysis.py` that caps SI, sRSA and SWdist alike.
`exp.eval_trajs=8` x `predNet.seqdur=256` pools ~2048 samples, so the cap does not currently
bind — but raising `eval_trajs` past ~15 silently discards the excess.

The constraint behind the cap is real: sRSA is **O(N^2)** (`pdist` over all pairs, then a
Spearman correlation over the pair vector), so more data does not buy a linearly better
estimate.

Open: what N do we actually want, and does the online (wandb) number need to match the
offline number? Interacts with entry 1, since the activity threshold's meaning is a function
of N.

Now a parameter (`max_samples`) rather than only a global, so this can be answered
empirically without editing the library.

---

## 3. `performance=ultra`: does the optimizer change keep the science comparable

The device-resident preset changes rollout size, PPO minibatch, learning rate and Adam betas,
and pools the world-model update into one gradient step instead of B sequential ones. Its own
config header says this is "not a claim of scientific equivalence".

This is the intended **gradient-quality** improvement, not an accident — a PPO gradient
averaged over ~1024 trajectories instead of 8, and one pooled world-model step instead of 8
noisy serial ones. But it means any before/after comparison spans a semantics change.

Planned gate: loss **per gradient step** (not per second) across pooled vs serial arms, via
`throwaway/hydra_era/perf/sweep_batch_learning.py`, before committing to a production batch size.

Open: do we treat the new regime as a fresh baseline (and re-derive every reference number),
or do we require the learning curve to match the old one?

### 3a. How many world-model gradient steps does the pRNN actually need

**This is the sharpest open question, and it is a single number: `exp.num_envs`.**

With `predNet.batched_wm`, one update is exactly ONE world-model gradient step, so
`num_envs` *is* the gradient-step budget: `steps = episodes_total / num_envs`. And
`trainStep` is flat in batch on this GPU (~0.19 s from batch 1 to 256,
`tests/perf/results/trainstep_batch_cuda_4060.json`), so gradient steps and wall-clock both
scale as `1/num_envs`:

```
num_envs      1024   512   256   128    64    32     8   8 (serial)
grad steps      78   156   312   625  1250  2500 10000      80000
WM wall-clock  24 s  37 s  63 s 2 min 4 min 8 min 32 min     4.3 h
```

The preset shipped at 1024, i.e. **78 gradient steps against the serial baseline's 80,000** —
a 1000x cut at identical experience. `performance=ultra` is now set to 128 (625 steps,
~2 min) as an interim so the number is not absurd, but it is a placeholder.

### ANSWERED 2026-08-11: pooling loses. FPS is the wrong objective.

`throwaway/hydra_era/perf/sweep_batch_learning.py`, 400 updates/arm, RTX 4060, compared on loss per
gradient step against the finished reference run `pRNN_curious_26-07-23-10-06-25` (pulled
from wandb on its true `trial` axis; the local serial arm reproduces it closely, which is
what makes the harness trustworthy):

```
arm          WM batch  grad steps     @100     @200     @400    @1600    @3200
B8 pooled           8         400  0.02102  0.02093  0.02027        -        -
B32 pooled         32         400  0.02097  0.02075  0.01955        -        -
B128 pooled       128         400  0.02093  0.02070  0.01926        -        -
B8 serial           1        3200  0.01974  0.02075  0.02066  0.01805  0.01632
wandb serial        1       80000  0.02180  0.02089  0.02091  0.01775  0.01595

arm           s/update  grad steps/s     FPS
B8 pooled        1.016          0.98    2006
B32 pooled       1.078          0.91    7444
B128 pooled      1.578          0.57   18812
B8 serial        2.285          3.58     916
```

Two facts settle it:

1. **The gradient has already saturated at batch 8.** Going 8 -> 128 is 16x the data per
   step and buys 5% lower loss (0.02027 -> 0.01926). Pooling does not come close to
   compensating for taking 1/B as many steps.
2. **Bigger batches give FEWER gradient steps per second**, because the rollout cost grows
   while `trainStep` stays flat: 0.98 -> 0.91 -> 0.57 steps/s. Serial gets **3.58**.

So B128 pooled has **20x the FPS of serial and one sixth the gradient steps per second**,
for gradients that are 5% better. FPS is actively misleading as an objective for this
model; **gradient steps per second is the quantity to maximise**.

Consequences:

- **`predNet.batched_wm` should be OFF.** The pooled update was the headline
  "gradient-quality" improvement and it is counterproductive here. Keep `exp.device_env`:
  the cheap rollout is a real win, it is just not the world-model step.
- **The interim `performance=ultra` num_envs=128 is wrong** and so was the reasoning
  behind it. At 625 pooled gradient steps it would land near loss 0.019 against the
  reference's 0.005604 - badly undertrained, in 12 minutes.
- Reaching the reference loss needs ~80,000 gradient steps: 6.2 h at serial's 3.58/s,
  versus 39 h at B128 pooled's 0.57/s. Serial is the only viable route on this box.

**Open, and the obvious follow-up:** with SERIAL world-model training, gradient steps per
update equal `num_envs`, so a larger `num_envs` amortises one rollout over more gradient
steps and should approach the ceiling set by `trainStep` latency (1/0.19 = 5.26 steps/s).
Serial at B=32/128 with `device_env` was not measured and may beat serial B=8's 3.58/s.
Run that before fixing `ultra`.

### ANSWERED 2026-08-19: yes, and the trainStep ceiling itself moves

Measured (`throwaway/hydra_era/perf/benchmark.py`, RTX 4060, idle GPU, serial world model,
`exp.layouts=one`, `frames = num_envs x 256`, `ppo_batch_size = frames/4`; full
method and caveats in `docs/claude_logs/exp_speed_cuda_graph_2026-08-19.md` §9b):

```
 num_envs   GRAD/s   s/upd  wm_s/upd  roll_s/upd        with predNet.cuda_graph=True
        8    10.88   0.735     0.128       0.496
       32    27.21   1.176     0.497       0.528
       64    37.50   1.706     0.991       0.527
      256    52.59   4.868     3.949       0.528
```

Two results:

1. **The amortisation prediction holds exactly.** `roll_s/upd` is flat across a 32x
   range (0.496 -> 0.528) while gradient steps per update rise 32x. Serial B=64 gives
   **37.50 grad steps/s against serial B=8's 3.53** in this harness.
2. **The 5.26 steps/s ceiling was a property of the EAGER trainStep, not of the model.**
   `predNet.cuda_graph=True` takes one segment step from 0.198 s to 0.0161 s (12.3x),
   moving the ceiling to **62.1 steps/s**. B=256 reaches 85% of it.

⚠️ Two things this does NOT settle, and both are methodological:

- **`num_envs` dilutes policy learning.** With `ppo_epochs` fixed and `ppo_batch_size`
  scaled with `frames`, policy gradient steps per UPDATE are constant (16) while each
  update spans `num_envs/8` times more environment steps — so the policy takes
  proportionally fewer steps per environment step, and the policy generates the
  behaviour the world model learns from. Needs a learning gate on pRNN loss AND sRSA.
  Returns also diminish hard (8->64 is 3.45x, 64->256 only 1.40x) while `s/update`
  grows to 4.9 s, which argues for **32-64, not 256**.
- **`performance=ultra` is still wrong** and this does not fix it: its `num_envs=128`
  is paired with `batched_wm: True`, which is the combination that yields the fewest
  gradient steps of all. Under serial the same width is good; under pooling it is the
  1000x cut this section already describes.

---

## 4. Which SI bin weighting is the honest one

`throwaway/ported/trace/trace_maps.py::spatial_info` supports `weighting="occupancy"` (standard Skaggs,
bins weighted by dwell time) and `weighting="uniform"` (every valid bin equal).

The random-action probe oversamples corners roughly 9x — it pushes forward ~60% of the time
and parks against walls — so occupancy weighting inflates the SI of wall- and corner-tuned
units. Uniform weighting removes that bias but trusts sparsely-sampled bins as much as dense
ones.

Current practice is to report both and treat disagreement between them as the bias estimate.
Open: which is the headline number in a paper figure.

---

## 5. What is measured online vs offline

Working split, driven by cost rather than preference:

- **online (wandb, free)** — anything already computed inside the update: pRNN loss, reward,
  FPS, entropy.
- **offline from checkpoints** — SI, sRSA, SWdist, rate maps. sRSA/SWdist are O(N^2) and
  SWdist runs its own spontaneous rollout per call, so they cannot scale online anyway.

Open: which small set of metrics is cheap and trustworthy enough to steer design decisions
mid-run, versus what we only ever compute after the fact from a saved checkpoint. This is the
speed-vs-analysis tradeoff, and it decides how often a run needs to be re-run at all.

---

## 6. Latent issues deliberately left alone

Not choices so much as known defects that were out of scope when found. Listed so they are
not rediscovered as surprises.

- **`plotTuningCurvePanel` cannot represent a missing SI.** `np.argsort` on a pandas Series
  sets NA positions to `-1`, which are then read as *negative indices* and silently alias the
  last cell into the panel (pandas 2.3.3; a `FutureWarning` says this becomes order-last in a
  future version). Harmless while SI is zeroed and never NaN — it becomes a live bug the
  moment entry 1 is resolved toward masking.
- **Circular import.** `import prnn.analysis.representationalGeometryAnalysis` fails on its
  own; `prnn.utils` must be imported first. Reproduces on pristine `7f50a0fe`, so it predates
  this work.
- **`prnn` is pinned by *branch*** in `pyproject.toml` (`sdu/rl-integration`) and only by
  commit in `uv.lock`. A `uv lock --upgrade` would move the science stack silently. Worth
  pinning the commit explicitly.
- **The training golden fixture is not enforced.** `tests/golden/` holds `golden_v0.pt` and
  `golden_v1.pt` plus the scripts that write and compare them, but no `test_*.py`, so pytest
  never reads either. The OMT path *is* gated by `../experiment-curiousgeorge/tests/golden_omt/test_golden_omt.py`.
- **`collectObservationSequence(seed=...)` sets the global** `torch`/`random`/`numpy` seeds,
  so a data-collection call mutates process-wide RNG state.

# Invalidating lines through the run series

**What this is.** Each entry is a commit after which the code computes something
materially different from before it. A run, checkpoint or figure produced on the wrong
side of one of these lines does not measure what its document says it measures.

**Why it has to exist separately.** These were recorded only as prose plus SLURM job
numbers inside two long session documents. To place an existing artifact on a timeline
you had to have written its job number down at launch, and the artifact itself carried
nothing. Every artifact made after `provenance.json` lands (refactor plan, Phase 1)
records its own commits, and then each line below becomes a property of the artifact
rather than something a reader has to reconstruct.

**Rate, for calibration:** four lines in five days, 2026-08-21 to 2026-08-25.

---

## `sdu/optim-pred` — the 2026-08-31 audit's measurement-protocol lines

Four smaller lines from the same audit pass, grouped because they share a
commit. Runs BEFORE it are internally consistent; numbers measured across the
line are different protocols, not regressions:

- **Seeded-probe protocol v2**: the probe now resets `pN.state` per probe
  trajectory (`collect_pooled_activity(reset_each=True)`) and `_probe_rng`
  restores `pN.state`/`pN.phase` alongside the RNG streams. Seeded sRSA/SI
  values move; unseeded probes are untouched.
- **`checkpoint_series` scores under `eval_mode` and `rooms_max=5`**: it
  used to score through 0.15 train-mode dropout over a 4-room prefix while
  the online eval used 5 - its numbers were a THIRD protocol. Offline series
  from before the line must be rescored, not mixed into tables.
- **Stream-0 bank-build reseed**: multienv pool construction called
  `reset(seed=0)` on stream 0's live wrapper, so after construction stream
  0's GENERATOR held the seed-0 state whatever `run.seed` was. Realized
  start draws still varied across seeds - the room schedule is seeded by
  `run.seed` and steers the rejection sampling - which is why no downstream
  number caught it; `tests/test_fast_reset.py::test_run_seed_reaches_stream_zero`
  reads the generator state itself (verified: fails on the pre-fix code).
  Banks were always collected correctly; the effect is a mild, structured
  understatement of cross-seed variance on stream 0's episode starts.
- **Fork re-pin `600c7e09` → `1939006e`** (`uv.lock`): audit refactors only
  (`targets_for` as the one pixel→class home; `readout` popped in
  `create_layers`). Construction and dynamics unchanged - the goldens gate
  it - so this line marks provenance, not a numeric break.

## `sdu/optim-pred` — the seeded probe leaked into the training stream · 2026-08-31

**What was wrong.** From the `probe_seed` threading (`87c6c22`, ~01:00) until
the fix (this line, ~evening), every analysis event called
`torch.manual_seed(probe_seed)` / `np.random.seed(probe_seed)` with NO state
restore - and `torch.manual_seed` seeds EVERY device's global generator. So
after each analysis event the run's training randomness (dropout masks,
action sampling, world-model noise) restarted from the same constant, in
every run, regardless of `run.seed`.

**What it does and does not invalidate.** Within the affected era every
comparison is internally consistent: all arms carry the identical coupling,
runs remain deterministic and reproducible, and the CE-vs-MSE / sweep
orderings stand. What it biases is SEED-TO-SEED INDEPENDENCE: after the
first analysis event, s2 and s3 share their RNG streams and differ only
through accumulated state and data - so "n=2 seeds" understates variance in
the era's tables. Fixed by `_probe_rng` (evaluation/spatial.py): the probe
seeds inside a save/restore of the global torch (all devices) and numpy
states; gated by tests/test_spatial_eval.py::
test_seeded_probe_leaves_the_training_streams_untouched.

---

## `sdu/optim-pred` — parity/multienv-fast go whitened-by-default · 2026-08-31

**What changed.** The `parity` preset (and `multienv-fast`, which derives from
it): `normalize_advantage=True`, `entropy_coef` 0.003 → 0.035, and
`eval.probe_seed=10007` (the spatial probe is FIXED - previously each analysis
event drew fresh rollout noise, bands of 0.068-0.114 sRSA per run). Dataclass
DEFAULTS are untouched, so `tests/golden/` train/eval fixtures stay bitwise;
only the setup-composition fixture was recaptured (same filename, its own
convention). `train_policy.normalize_reward` also lands (default False -
absent from every prior run's config).

**Why it costs nothing.** Every run these preset names ever produced is
pale-era and already invalidated by the 2026-08-30 rendering line below;
there is no bright-era run on the old values to be incomparable with.

**What to know when reading.** A "parity"- or "multienv-fast"-shaped run from
2026-08-31 on is whitened at 0.035 with a seeded probe; earlier ones are raw
at 0.003 with an unseeded probe. The raw-era knee (0.003) does NOT translate
to the whitened scale (measured factor ~8x, `slurm/multienv.sh` header).

---

## `sdu/optim-pred` — full-colour landmarks, and triangle3 replaces x · 2026-08-30

**What changed.** `minigrid` `22ef960` (two commits, re-pinned here in the same
change): (1) `Floor.render`/`Obstacle.render` fill at full `COLORS[c]` instead
of `COLORS[c]/2` — as observed through the 7x7 partial view, a landmark cell
moves from `76 + 0.35*COLORS[c]` to `76 + 0.7*COLORS[c]`, doubling every
landmark-vs-floor and landmark-vs-landmark separation (blue: (76,76,165) ->
(76,76,255) against floor (76,76,76)); (2) a new `triangle3` stencil — rows of
1, 2, 3 cells growing downward, the one small shape that is
orientation-dependent in the egocentric view — and `SHAPES`/`ROOMS_SELECTED`
swap it in for `x` at unchanged anchors and colours.

**Why.** The curiosity reward is per-pixel MSE over 147 dims; at pale fill a
fully mispredicted landmark cell contributed ~0.0008-0.0017 against an ambient
reward mean of ~0.005, so the room-identity signal — the epistemically real
uncertainty in multi-room training — was buried. The full-strength fill
quadruples a landmark cell's squared-error contribution.

**What it invalidates.** Every run, checkpoint, golden fixture, obs bank and
prediction figure made before this line sees different observations from any
made after: pRNN loss, curiosity reward scale, sRSA/SI/SWdist are all computed
from the changed pixels. The `mx-*` multienv series and the parity/reference
runs of 2026-08-29/30 are the last of the pale-x era; nothing after this line
is comparable to them. `data/obs_bank` was cleared and rebuilt (the bank
fingerprint now includes the minigrid commit, so a future render change
invalidates banks by construction instead of by memory).

**Knock-on facts, measured.** Impassable rooms now have 152 standable cells
(was 153: triangle3 occupies 6 cells against the x's 5). All 20
`ROOMS_SELECTED` x affordance combinations still clear every `RoomRules`
margin; the tightest is `min_testable_offsets` — impassable rooms 2 and 8 sit
exactly at the minimum of 20. The OMT novel object (`FloorBright` neon_green,
unchanged) is now only ~42 as-observed from a GREEN landmark (was ~99):
brightness no longer marks novelty apart, only hue does — revisit the novel
colour before the next OMT run. `ROOMS_SELECTED`'s per-row keys are the
SOURCE (x-stencil) pool's keys, not `Layout.key`, since the key hashes the
shape.

---

## `sdu/optim-pred` — the readout output bias · 2026-08-30

**What changed.** `prnn` `4ec775ed`: the observation readout
(`Architectures.outlayer`) gained a trainable bias, `b_out`, with its own
`OutputBias` optimizer group (unscaled lr, `weight_decay=0`). Re-pinned here in
the same change; golden fixture bumped `golden_v3.pt` -> `golden_v4.pt`.

**Why.** The target's mean is ~0.40 and the readout had `bias=False`, so that
constant had to be synthesized as `W_out . h` — from a vector that is 55% exactly
zero and whose mean moves every timestep. Measured symptom on an 83.9M-step
checkpoint, 100,352 target pixels: the network recovers only 69-85% of each
object colour's spread (blue 0.559 against a 0.647 target), while the agent's own
triangle — centre of every view, no inference required — is recovered at 101%.
The shortfall tracks PREDICTABILITY, not pixel frequency.

**What it invalidates, and what it does NOT.**

A network at INIT is bit-identical across this line: the bias is zero and its
construction is RNG-neutral (attached as zeros rather than via `bias=True`, which
would draw `init.uniform_` and shift every later draw — verified, the next
`torch.rand` moves 0.0373173952 -> 0.8484520912).

It diverges once the bias LEARNS. After the golden's two updates, max |b_out| =
4.5e-2, all 9 shared prnn tensors have moved (max 3.3e-3), and — because the
pRNN's prediction error IS the curiosity reward — 11 leaves of `rounds` and 8 of
`acmodel_state` move with them. **So any run crossing this line is not comparable
to one before it on prediction loss, curiosity reward, or any policy metric.**

**Checkpoints stay loadable and exact.** `load_pN` zero-fills a missing
`outlayer.0.bias` and still raises on anything else missing or unexpected; a zero
bias is the same function as no bias, so a pre-bias checkpoint loads unchanged
rather than approximately so. `tests/golden/test_golden_eval.py`, which scores a
tracked pre-bias checkpoint, stays bitwise green across the line.

**A trap this nearly hit.** `predictiveNet` builds its optimizer from explicit
named parameter groups, not `parameters()`, so a registered Parameter with no
group is one no optimizer ever sees. The first version of this change had the
bias sitting at zero forever — a completely silent no-op. It was caught only
because the training golden showed the bias still exactly zero after two updates.

---

## `d6ace4c` .. `edce59f` — the config cutover · 2026-08-26

**What changed.** Hydra YAML was replaced by typed dataclasses
(`curious_george/configs.py`). This is a RECORDING line, not a numerics line: the
`tests/golden/capture_golden_setup.py` gate compares `setup_training` across all three
live compositions and the migration produced **two differing leaves in total**, both a
spelling change (`wm_pool_group: 0 -> 8` and `0 -> 128`, where `0` had meant "the whole
batch" — `prnn_adapter.py:874`). Same 33 constructor kwargs, same module weight hashes,
same derived schedule. **A run's numbers are comparable across this line.**

**What is NOT comparable across it: the config RECORD.** wandb stores each run's config
under the shape the code had at the time. Before: `exp.* / rl.* / predNet.* / logging.*`.
After: `env / collect / arch_prnn / arch_policy / train_prnn / train_policy / eval / run`.
A filter like `config.exp.curious_agent` matches nothing on a post-cutover run, and
matching nothing is indistinguishable from a field that was never set.

**How to cross it.** `curious_george/check/config_keys.py`:

```python
from curious_george.check.config_keys import to_new, to_old, describe
to_new("predNet.seqdur")   # "collect.episode_steps"
to_old("run.seed")         # "exp.seed"
describe("rl.frames")      # why it has no counterpart
```

Three kinds of correspondence, and the map does not pretend otherwise: **renamed**
(1:1, invertible), **folded** (several old keys became one field — forward only, because
`collect.backend` does not say whether a reader meant `table_env`, `device_env` or
`async_envs`), and **gone** (derived or dropped: `rl.frames`, `rl.episodes_total` and
`rl.ppo_batch_size` are all computed now).

**One wandb metric key also changed:** `wm_grad_steps` -> `prnn_grad_steps` (`b02aaaf`).
Environment steps (`frames`) are untouched, so `wandb_compare`, which matches on env
steps, works across the line unchanged.

---

## `d275149` — `recurrence` removal · 2026-08-23

**What changed.** Removed a dead `recurrence` parameter and, with it, a code path that
silently dropped one transition on odd epochs. Policy minibatch composition changed, so
the update statistics and the weights changed with them.

**What it invalidates.** Any comparison of policy-update quantities across this line.
The world-model rollout is NOT affected: measured with `tests/golden/capture_golden.py`,
round 0's `curious_rewards`, `advantages`, `actions`, `log_probs`, `SRs` and `locs` are
bitwise unchanged; only the update and what follows it moved (17 leaves).

**How to tell which side an artifact is on.** Before: `37aaa1b` and earlier. After:
`d275149` and later.

**Fixtures.** `tests/golden/golden_v1.pt` is valid up to `37aaa1b`; `golden_v2.pt` is
the baseline from `d275149` on. `../experiment-curiousgeorge/tests/golden_omt/golden_omt_v1.pt` was re-captured at
the time; the training-path fixture was missed and stayed stale for three days because
no test ran it. `tests/golden/test_golden.py` now does.

---

## `0d60df7` — stranded CUDA-graph buffers · 2026-08-23

**What changed.** `on_device` now restores `param.grad` and registered **buffers**, not
just `param.data`. Before it, a captured graph read `thRNN_5win`'s phase masks
(`inMask_f` / `actMask_f` / `outMask_f`) from freed memory after any spatial eval moved
the model to CPU and back, so the graph scaled its inputs by whatever occupied that
memory.

**What it invalidates.** *"Every arm launched before it was found is invalid"* — any
run with `predNet.cuda_graph=True` that also ran a spatial eval. The failure is silent
and the loss curve looks healthy, because the forward reads parameters (correct) and
only the masked inputs were wrong: replayed loss 0.038778 → 0.006764 on identical data,
gradient norm 0.0603 → 0.0156. A graphed 20.48M-step arm held sRSA at 0.0238 against its
eager control's 0.5158.

**How to tell.** Before: `2c83318` and earlier with graphs on. After: `0d60df7` and
later.

---

## `4cfd8cb` — captured floats · 2026-08-24

**What changed.** Recorded (did not yet fix) that a captured CUDA graph freezes every
scalar hyperparameter passed as a Python float at its capture-time value.
`GraphPolicyTrainer._region` and `_GraphWMTrainer._capture` bake them in; the graphed
path never re-reads them.

**What it invalidates.** Any schedule under `rl.cuda_graph=True`. Confirmed dead:
`rl.entropy_coef_final`. Same exposure: `clip_eps`, `value_loss_coef`, and both
optimizers' `lr`. Specific runs: `10482137` ("ramp 0.001→0.01") ran pinned at 0.001;
`10483973` ("ramp 0→0.001") ran pinned at 0, a duplicate of the `ent=0` arm. `10483974`
(constant 0.0003) is valid, because a constant captures correctly.

**Status: OPEN.** The fix — a 0-dim device tensor mutated with `.fill_()` — is proven to
work under capture but has not landed. Until it does, every scalar schedule is dead
under graphing.

---

## `UpdateLogs` epoch averaging — 2026-08-25

**What changed.** The eager path re-created its log accumulators inside the PPO epoch
loop (inherited verbatim from `torch_ac/algos/ppo.py:32-35`), so it reported the mean
over the **last** epoch only — one quarter of the gradient steps at `ppo_epochs=4`. The
graphed path averaged **all** epochs. Eager now matches graphed.

**Measured, and it is reporting-only.** `tests/golden/capture_golden.py` moved
**exactly six leaves** — `policy_loss`, `value_loss` and `grad_norm` in both rounds —
and left every weight and every rollout tensor bit-identical:

```
rounds[0].grad_norm    2.1485 -> 3.6079
rounds[0].value_loss   0.2125 -> 0.2373
rounds[0].policy_loss -0.4792 -> -0.4626
```

All three in the direction the mechanism predicts, so the old convention understated
`grad_norm` by 41% on this fixture.

**What it invalidates.** Policy diagnostics from every eager arm, including the
2026-07 reference `pRNN_curious_26-07-08-16-04-37`: `policy_entropy`, `value_loss`,
`grad_norm`, `policy_loss`. The direction of the bias is systematic, not noise — eager
reports lower entropy, lower `value_loss`, lower `grad_norm` and larger-magnitude
`policy_loss` than graphed for the identical update.

**What it does NOT invalidate.** `pRNN loss`, `sRSA`, `SWdist` and `SI` do not pass
through `UpdateLogs`. The entire world-model line is untouched.

**Fixture.** `golden_v3.pt`. **Gate.** `tests/test_update_logs_semantics.py` recomputes
the statistic from every `LossTerms` the update produced, so it holds for any config and
any device; all 9 of its tests go red on the old placement.

**Already contaminated by it:** `speed-30min` §9's eager-vs-graphed entropy comparison
("g8 EAGER 0.5927 at 60M against g8 graphed 1.3357") mixes an estimator difference with
a real one, in an unmeasured proportion.

---

## 2026-08-27 — the placement rules changed, so generated pools did too

**What changed.** Three defaults, together, to widen the range of relative object
positions a room set spans:

```
OFFSET_RADIUS              4  ->  3     (envs/layouts.py)
RoomRules.min_anchor_separation  6 -> 3
RoomRules.min_testable_offsets  40 -> 20
common_offsets             now excludes offsets SHARED between anchor windows
```

**Why.** Measured: under the old rules the L-room admitted 6,908 placements with only
**13 distinct separation signatures** — 83% of them had their closest pair at exactly 6.
The set was essentially one configuration sampled many ways. `min_anchor_separation=6`
was what forced that, and it was there to stop anchor-centred offset windows overlapping
and correlating for purely geometric reasons.

Excluding the shared offsets removes that confound *directly*, which lets the separation
rule come back down. The smaller window is what makes the exclusion affordable: at
separation 6, radius 3 leaves 86% of the window usable against 67% at radius 4, and at
separation 7 or more it leaves all of it.

**Result:** 19,820 placements, separations 4–9, **43 distinct signatures**. Note the
realised minimum is 4, not 3: too-close anchors lose most of their window to the
exclusion and fail `min_testable_offsets` on their own, so the offset count now enforces
the separation and nobody had to pick 4.

**What it invalidates.** Any run whose rooms came from `Curated` or `Uniform` — the same
`(n, seed)` now returns DIFFERENT rooms. A pool is only reproducible against the rule
defaults in force when it was drawn.

**What it does NOT invalidate.** `Frozen` / `Committed` sets are literal tuples in
`layouts.py` (`ROOMS_RUN1`, `ROOMS_SQUARE`) and are unchanged, so every run built on the
committed three-room sets is untouched. Single-room runs (`EnvDefault`) never consult
these rules at all — including the 2026-08-27 Mila parity runs.

**Also note.** `n_testable_offsets` is now a smaller number for the SAME room, because it
excludes shared offsets and the window shrank. Do not compare it across the change.

**Cost.** `admissible_placements` went 1.6 s -> 17 s: ~97k candidate triples instead of
12k, each doing an exclusion pass. Runs once at startup, against a bank build measured in
minutes.

**Gate.** `tests/test_env_layouts.py`, `tests/test_location_entropy_support.py`.

### Same day, follow-up: `min_testable_offsets` was measured in the wrong room

Caught while pre-building banks for Q3's 10 training rooms. `enumerate_anchor_triples`
checked the offset count against the shape's walkable set — the room BEFORE its objects
are placed — and built its candidate `Layout`s without the `impassable` flag, so the
count never noticed that an object takes its own cells out of the walkable set.

With `min_testable_offsets=20` the generated rooms had as few as **11** offsets the
agent could actually be measured over. The constraint did not constrain what it named.

Now measured against `layout.walkable(base)`, and the enumerator is told which stencils
block. The honest admissible set for impassable objects is **9,074 placements / 25
distinct separation signatures**, against 19,820 / 43 for walkable ones — objects
genuinely rule out half the configurations, and that was previously hidden.

**What it invalidates.** Nothing that ran: no impassable-object run existed before this.
Any pool drawn between the two 2026-08-27 commits would be wrong, and there is none.

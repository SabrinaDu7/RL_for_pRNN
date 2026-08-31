2026-08-31 · branch `sdu/optim-pred` (repo) + `sdu/ce-loss` (prnn fork) · overnight autonomous execution

# Plan: multienv impassable training under a categorical (CE) prediction loss

## The goal, in one paragraph

Produce multi-environment training on impassable-object rooms whose pRNN makes
GOOD predictions — inspectable by eye against the true observation — while
keeping sRSA high, SWdist low, and SI high. The loss moves from pixel-MSE to
per-tile cross-entropy, because MSE demonstrably struggles: under whitened
advantages the pale-era entropy sweep (`mx-impassable-n5-*-norm-e*`, project
`curious-george`, 2026-08-30) shows MI_policy and sRSA moving in OPPOSITE
directions monotonically (e0.024 → MI 0.15–0.35, room sRSA 0.51–0.57; e0.12 →
MI 0.025, room sRSA 0.712) — every bit of policy commitment under pixel-MSE
curiosity costs representation quality, and the random agent beats every
curious arm (random room sRSA 0.872 / SI 0.968 / loss 0.0033 vs best curious
0.819 / 0.824 / 0.0052, walkable n5, 2026-08-30). **The success criterion is
breaking that anticorrelation**: a committed policy (healthy MI) whose
sRSA/SI sit at-or-above its own random baseline. sRSA high, SWdist low, SI
high matter MORE than MI, which varies a lot.

## Deliverables by morning (2026-08-31), in priority order

1. **bright-CE multienv-impassable curious run** on the 8-room set, wandb
   project `curious-george-multienv`, with prediction figures a human can
   check by eye.
2. **bright-MSE multienv-impassable curious baseline**, same shape, same
   project — 🔴 launched and reported BEFORE any bright-CE multienv run
   exists in that project.
3. **Behavioral baselines on bright-CE**: random forward-weighted, random
   uniform, and count-based, through the same collector, same project — so
   the learned policy's exploration metrics have references.
4. **Extensive documentation**: a results doc per phase (see § Documentation),
   every number pointing at its run and command.

Do not stop until the deliverables exist or the user wakes. If blocked, write
down exactly where and why, and continue on the next unblocked item.

## Ground rules

- **Era comparability.** `pRNN loss` is NOT comparable across the loss swap
  or across the 2026-08-30 rendering line (`docs/invalid-runs.md`). sRSA,
  SI, MI_policy and SWdist ARE treated as comparable across eras — same room
  geometry, same h-based computation (`evaluation/spatial.py`) — and the
  plan's first empirical question is whether that holds (Phase 1 arm A vs B).
- **One switch.** MSE ↔ CE must swap via ONE typed config field (see § CE
  implementation), with every downstream consumer (loss, readout shape,
  curiosity reward, prediction figures) following from it. No second knob, no
  half-states.
- **Minimum tricks.** Nothing from `docs/claude_logs/rl_tricks_2026-08-29.md`
  is spent until a measured symptom names it (see § Contingencies).
- **Small experiments first.** Intuition-checks run in `throwaway/` on the
  local 4060 (valid for curve-gating — the dev box reproduces cluster curves)
  BEFORE any Mila launch. No result may depend on a throwaway script.
- **Gates before launches.** The full pytest suite green (baseline at HEAD
  `888cebf`: 568 passed, 1 deselected), goldens bitwise at the MSE default,
  fork gates green, before each launch wave.

## Phase 0 — prerequisites (repo, before any run)

1. **Thread `probe_seed`** from `EvalCfg` into
   `evaluate_spatial_representation` / `evaluate_multi_room_representation`.
   Confirmed unthreaded at HEAD (grep: defined at `evaluation/spatial.py:125`,
   passed by nothing). Without it sRSA carries per-run bands of 0.068–0.114 —
   the anchor run's own series reads 0.34 → 0.68 → 0.60 across three events.
   Default `None` (today's behaviour, goldens untouched); presets pass a
   seed derived from `run.seed`. Every comparison below assumes this landed.
2. **Whitened advantages become the default**: `normalize_advantage=True`,
   `entropy_coef=0.035` in the presets this plan runs (`parity`-shaped and
   `multienv-fast`). 0.035 is a STARTING point, not a measured knee — the
   pale sweep shows the value trades MI against sRSA continuously, so Phase 1
   carries a scan. Default flips change dynamics → recapture goldens
   (fixture v6, `capture_golden.py` FIXTURE VERSIONS updated) +
   `docs/invalid-runs.md` entry, in the same commit.
3. **Reward normalization flag** (`train_policy.normalize_reward`, default
   False): divide the combined reward by a running std estimate in
   `compute_gae` (`rl/update/advantage.py`), per Burda et al. — the tricks
   doc §1 specifies it and its controls. Under CE the reward scale changes
   again, so this flag is expected to earn its keep; Phase 1 measures it.
   - Positive control: with it ON, `k_curious x10` must change nothing.
   - Negative control: with it OFF, `k_curious x10` must change behaviour.
   - Extreme: `k_curious=0` must degrade gracefully (the epsilon test).

## Preflights (throwaway/, local 4060)

- **Tile vocabulary extraction**: enumerate the exact per-tile RGB byte
  values over the rebuilt obs banks (both affordances, all Selected rooms +
  the default single L-room + FloorBright). Hypothesis: a small closed set
  (measured POV vocabulary on one room: floor (76,76,76), wall
  (146,146,146), agent tile (135,76,76), landmark colours e.g. blue
  (76,76,255)). Assert closure; this set IS the class alphabet C.
- **CE unit sanity**: a hand-built (logits, target) pair per class;
  masked-row semantics asserted (see implementation constraint 3).
- **Reward-scale preview**: run a handful of rollouts with the CE reward on
  an untrained and a briefly-trained net; record surprisal mean/std per step
  so `k_curious`/normalization start from measured numbers, not guesses.

## CE implementation (prnn fork branch `sdu/ce-loss`, off `sdu/rl-integration`)

Authority: granted 2026-08-31. Push the branch; re-pin here via
`[tool.uv.sources]` branch key + `uv lock`; the re-pin protocol and gate list
is `docs/claude_logs/rl_tricks_2026-08-29.md` §8.

**Reference implementation: `../grid-predict/`** (user-supplied, 2026-08-31) —
the same task (categorical next-observation prediction over a grid) on a
different architecture. Confirmed useful pieces: `src/train/__init__.py::
prediction_loss` (`F.cross_entropy` on `(B, C, S, S)` logits vs `(B, S, S)`
int64 targets, plus the 3-line focal variant behind `focal_gamma: float |
None` — copy that flag shape for the contingency), `src/obs/__init__.py::
obs_to_onehot` and `src/utils/__init__.py::get_lr_lambda` (warmup+cosine, if
the LR-warmup contingency fires), `src/vis/` (per-class probability
rendering). Go back to this repo when stuck on the CE mechanics.

Design constraints, each load-bearing:

1. **One home for the switch.** A typed field (e.g. `arch_prnn.loss: MSE|CE`)
   selects upstream `losstype` (`predCE` joins `lossOptions`), the readout
   head (`output_size = 49*C` logits, no sigmoid, vs the existing 147-sigmoid),
   and the adapter's curiosity reward (per-step SUMMED per-tile surprisal in
   nats under CE; the existing pixel MSE under MSE). `LOSSES`-style registry,
   never an if-chain spread across files.
2. **Explicit vocabulary.** The class alphabet is an explicit input derived
   from the obs banks (preflight above), passed to the loss/readout at
   construction, with a loud runtime assert on any unseen tile value. Never
   discovered lazily from a first batch.
3. **Masked rows.** Upstream masks timesteps by MULTIPLYING predictions and
   targets by zero — only sound because MSE(0,0)=0. Confirmed: mainline
   `thRNN_5win` nets default `outMask` all-True, so `predCE` ASSERTS
   all-True and needs no mask arithmetic. If that assert ever fires, stop
   and re-design; do not multiply logits by masks.
4. **RNG-neutral construction, own optimizer group appended LAST** — the
   `b_out` playbook (commit 4e224e1's three measured traps), so goldens stay
   bitwise at the MSE default and checkpoints restore by index correctly.
5. **Checkpoint drift assertion.** Checkpoints are whole-object pickles; an
   old checkpoint silently unpickles as the old architecture (tricks doc §8).
   Land a load-time assert that the restored readout matches what the current
   config constructs — CE-vs-MSE mixing in one `checkpoint_series` must fail
   loudly.
6. **Prediction figures adapter**: under CE render argmax class → palette
   RGB (plus a per-tile entropy map — new, and better than blurry pixels).
   This is the "check predictions by eye" deliverable.
7. **Capture safety**: the CUDA-graph WM trainer captures
   `predict + loss_fn + backward + step`; CE must be capture-safe (no
   data-dependent shapes). Gates: existing `tests/test_cuda_graph_wm.py`
   bitwise at MSE; a CE-arm capture-equivalence test beside it.

## Phase 1 — single L-room (default landmarks: plus6, x6, triangle6 — bright)

Project: `curious-george` (the existing one). Shape: the `parity`-derived
single-room configuration; budget under ~30 min per run on an L40S; batched
world-model training (`episodes_per_grad_step=8` pooling) as today.

**Arms** (seed s2; add s3 to any arm whose reading is decisive):

| arm | loss | agent | purpose |
|---|---|---|---|
| A `p1-mse-bright` | MSE | curious | bright-MSE control: era effect on metrics, isolated |
| B *(historical anchor, no launch)* | MSE | curious | `mila-off1-e0.003-s2_curious_26-08-29-21-09-36` (pale, raw adv, e=0.003, offset 1): sRSA 0.64 (last-2; series 0.34→0.68→0.60), SWdist 0.049, mean SI 0.90, MI 0.042, loss 0.00439 |
| C `p1-ce` | CE | curious | the intervention |
| D `p1-ce-rand` | CE | RANDOM (forward-weighted) | 🔴 the loss-effect-on-representation control at fixed data distribution |
| E `p1-mse-rand` | MSE | RANDOM (forward-weighted) | completes the 2x2: loss x agent |
| C-scan | CE | curious | entropy_coef ∈ {0.024, 0.035, 0.07}; C is the 0.035 member |
| C-rnorm | CE | curious | reward normalization ON, at C's best entropy |

**Reading order.** A vs B first: if sRSA/SWdist/SI/MI sit in the same
ballpark, brightness alone did not move the metrics and B remains a valid
anchor; if not, NOTE IT in the results doc and carry BOTH A (bright-MSE) and
B (pale-MSE) as controls for every CE reading. Then D vs E: does CE change
what the pRNN's h encodes when the DATA is held fixed? Then C against A and
against D.

**Phase-1 gate** (all from seeded probes): CE arms hold sRSA high, SWdist
low, SI high, in the ballpark of the MSE controls — MI is reported but not
gating. Hypothesized: D ≈ E on sRSA/SI (the loss swap alone should not
degrade the spatial code — if it does, that finding gates everything and the
CE-native remedies apply, see Contingencies), and C ≥ A with C's MI above
A's (the whole point).

**If it fails, the debugging ladder** — ask in this order, and answer with a
measurement, not a retry:
1. *Implementation?* — goldens at MSE default; CE unit tests; class-target
   assert; loss curve shape (CE should start near ~~`49*ln(C)` nats/step
   summed~~ `ln(C)` nats/tile - the loss reduces with a mean over tiles;
   appendix delta 8 - and fall).
2. *Evaluation?* — probe seeded? same probe across arms? argmax-render
   checked by eye against a true observation?
3. *Baselines?* — what do D and E say? A curious-arm problem that D shares
   is not a policy problem. D is the single most diagnostic run in Phase 1.

## Phase 2 — multienv, IMPASSABLE objects

Project: **`curious-george-multienv`** — used ONLY from this phase on.
🔴 Ordering: bright-MSE baselines land in that project BEFORE any bright-CE
run. All arms whitened; entropy from Phase 1's scan winner.

**The room set — source indices 0, 14, 31, 35, 126, 144, 169, 191** (index
83 dropped: it differs from index 0 by one cell — `envs/layouts.py`'s
recorded warning; 191 promoted from the spares; 195 remains spare).
⚠️ Source indices ≠ tuple positions: those are `ROOMS_SELECTED` positions
(0, 1, 2, 3, 5, 6, 7, 8). Implement as an optional position-selection field
on `Selected` (typed, CLI-visible, default = first n — today's behaviour);
NEVER reorder the committed `ROOMS_SELECTED` tuple. `Selected` applies
`impassable=True` on top; anchors are pinned so the walkable twin stays
expressible.

**Arms**, `multienv-fast`-derived shape, impassable, ~30 min each:

| arm | loss | agent |
|---|---|---|
| `mx8-mse` | MSE | curious — the mandatory-first baseline |
| `mx8-ce` | CE | curious — the headline run |
| `mx8-ce-randfwd` | CE | RANDOM forward-weighted (`RAND_ACT_PROBA`) |
| `mx8-ce-randuni` | CE | RANDOM uniform (0.25 x4) |
| `mx8-ce-count` | CE | count-based bonus agent |

Seeds: s2 everywhere; s3 on `mx8-ce` and `mx8-mse` if the night allows.

**Instrumentation per arm** — all of it already landed, none optional:
- Exploration metrics (`exploration/*` keys) with the per-room denominators;
  read `_reached` before `_steps` (censoring). Reference values regenerated
  at the launch tree: `uv run python -m curious_george.envs.action_graph` —
  run it, never transcribe (the committed set drifted once already).
- Sanity on every RANDOM launch: `exploration/coverage` near the
  calibration's matching row from the first log points, else stop — wiring.
- Trajectory visualization / statistics: paths over the room render for a
  few episodes per arm + the occupancy maps, looking for the named
  pathologies (wall-hugging, spinning, object-edge shuffling). Note what is
  seen either way.
- Prediction figures by eye: argmax renders + entropy maps against true
  observations, per room.
- Bump-event diagnostic (impassable-specific): prediction error at
  table-refused forward poses vs free moves — the transition tables know
  which is which; an affordance-learning pRNN should separate them.

**Reading.** `mx8-ce` vs `mx8-mse` on: seeded per-room sRSA, SWdist, SI,
prediction quality by eye, and the exploration metrics against the three
behavioral baselines. The pale-era walkable/impassable gap (~0.18 room sRSA)
is context, not a target — the target is CE ≥ MSE within-era, policy
committed, representation at-or-above its own random baseline.

## Contingencies — named triggers, minimum spend

- **Policy trouble** (collapse or uniform pinning): FIRST check advantage
  and reward normalization are live and sane (log `entropy_coef` per update
  — tricks §2's dead-under-graphs trap; verify the whitening controls).
  Then, in order, one at a time: entropy re-scan; random-pretrain warm start
  (forward-weighted, phase-1-of-budget, already wired via
  `run.prnn_ckpt`); policy LR warmup (tricks doc; note it is ABSENT today
  and dead-if-captured — needs the 0-dim-tensor pattern).
- **CE working really poorly** (D << E on sRSA/SI, or landmark-class error
  stalls while background saturates): focal loss on fork branch
  `sdu/focal-loss`, flag-gated like CE. Finicky — timebox it, do not chase.
  Before focal, try the cheaper CE-native remedies: label smoothing,
  temperature, class weights.
- **A/B ambiguity from noise**: add seed s3 before adding tricks.

## Documentation (the "so I can verify" contract)

- One results doc per phase in `docs/`:
  `ce-single-room-2026-08-31.md`, `ce-multienv-impassable-2026-08-31.md` —
  methods, arm tables with EXACT launch commands, wandb run names, commits,
  the A-vs-B era-comparability verdict, figures (paths), what broke and what
  was done about it. Every number carries its run name.
- `docs/invalid-runs.md` entries for: the defaults flip (Phase 0), and the
  CE re-pin when it lands.
- Golden FIXTURE VERSIONS updated with each recapture and its reason.
- The plan's own deviations recorded IN this file as an appendix, dated.

## Budget arithmetic (derived, not asserted)

Read `TrainingSchedule.summary()` at launch for every arm; the half-budget
multienv shape (`wm 21968` grad steps, policy 4x) measured 29.9–32.0 min on
L40S across 2026-08-30 runs, and the parity single-room full budget 27.6 min.
GPU type is load-bearing (L40S vs Quadro measured 52.45 vs 30.88 grad/s);
`slurm/multienv.sh` / `train_fast.sh` headers carry the launch reasoning.
Cluster babysitting per the hpc skill; local 4060 for curve-gating only.


---

## Appendix: how the night actually went (written 03:45, at `3453c44`)

Every morning deliverable exists; the two results docs carry the numbers,
figures and verdicts. The deltas from the plan as written:

1. Golden v6 was never needed - the defaults flip landed at preset level
   (dataclass defaults untouched), so train/eval fixtures stayed bitwise.
2. The fork needed one unplanned fix (output_size double-pass TypeError).
3. 🔴 The first launch wave was mislaunched through a stale cluster working
   tree (labels + EXTRA flags silently dropped); cancelled, junk runs tagged
   `mislaunched-drop`, resubmitted. Lesson recorded: reset the Mila shared
   checkout to the launch branch BEFORE sbatch - the runbook said so and the
   preflight checked the wrong thing (origin ref, not working tree).
4. CE arms run `--train-policy.normalize-reward` as a measured necessity
   (critic 200x behind raw-surprisal returns), not a maybe; ceRaw is the
   recorded ablation.
5. Phase 1 exceeded its gate (sRSA 0.871 committed-policy vs 0.682 own
   random baseline - the anticorrelation broke in Phase 1 already); Phase 2
   met the deliverable (CE > MSE on room sRSA everywhere) with two honest
   negatives: CE-curious has not separated from CE-count, and SWdist favours
   MSE at n=2. Full-budget ce8 is the ranked next step, blocked ~03:10 on a
   first-ever Mila email-OTP prompt (never retried; user notified).
6. The seed-3 replications of ce8/mse8 ran; Phase-1 arms are n=1 (seed 2)
   by design - the scan + 2x2 spent that budget instead.
7. `probe_seed` landed as a CONSTANT in the presets (10007), not derived
   from `run.seed` as planned: a shared probe makes cross-run sRSA
   differences attributable to the representation alone, at the cost of one
   shared probe-noise draw (see `EvalCfg.probe_seed`'s docstring).
8. *(2026-08-31 audit)* The ladder's "CE should start near `49*ln(C)`" was
   wrong as written: `predCE` reduces with a MEAN over tiles, so the curve
   starts near `ln(C)` ≈ 1.95 nats/tile - which is what every logged run
   shows. The curiosity REWARD is the per-step summed surprisal; the logged
   wm loss is per-tile.

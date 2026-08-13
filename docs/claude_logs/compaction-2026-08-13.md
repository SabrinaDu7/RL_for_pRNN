# Handoff — 2026-08-13

Supersedes `compaction.md` (2026-08-11) for everything after that date; that file is still
the reference for the mechanism and the pre-08-11 experiments, and its §8 was updated in
place. Branch `sdu/object-into-hidden-state`. Nothing here is pushed — `git push` for
`RL_for_pRNN` was blocked by a permission gate all session.

**Read §1 and §7 first.** §1 is the one result everything else follows from; §7 is what I am
and am not confident in.

---

## 1. The mechanism — CERTAIN

The object is written into `W_out`, the linear prediction head, **~89% position-independently**.

```
green change at (4,7)'s view cell, BEFORE the object ever goes there:  +0.0395 ± 0.011 (t=9.57, p<1e-4)
green change at (4,7)'s view cell, WHILE the object is there:          +0.0445 ± 0.008
```

`W_out` has one row per *view cell*, applied to every hidden state, so it learns "boost green
at this object-centred view offset" — not "there is an object at (4,7)". **Nothing
location-specific ever has to be learned, so nothing is.** Supporting: readout +0.0625 vs
dynamics +0.0157 gain-corrected (9/9 runs); place code unchanged, per-unit map r ≈ 0.98.

n=8 on the sequential-displacement runs, replicated. This predicts every null below.

---

## 2. Experiments — what, why, result

### Pre-08-11 (detail in `../results/`, verdicts unchanged)

| experiment | why | result |
|---|---|---|
| L-room + novel object (OMT) | baseline: does exposure change `h`? | `h` unchanged; change is in `W_out` |
| `lr_trials=[2,0,8]` — freeze `W_out`, scale `W_in` | deny the readout shortcut, force it into the dynamics | **the only positive on `h` ever.** Presence decoding ph0 0.6947→0.7214 (t=14.4, p<1e-5, n=3 v 10); survives ONE masked step (ph1 0.5834→0.5938, p=3e-5); decays ~2x/step to chance by ph4. An **encoding** gain, not memory |
| A — object identity varies (4 colours) | force a general "object" code rather than one texture | null, n=3 |
| D — weight-decay sparsity | pressure toward sparse codes | **untestable** — L2 shrinks everything, place code collapses r 0.97→0.73 |
| random object position, i.i.d. per batch | stop the readout memorising one offset | null, n=10 |
| `k_curious` = 5, 20 | is the curiosity weight the limiter? | null |
| C — occlusion (`see_through_walls=False`) | make the object genuinely leave view, so holding it is the only way to predict | **FALSE POSITIVE** — see §5 |
| sequential displacement A→B→C→removed | Moser-style displacement in the L-room | **null**, n=8. Hidden fractions 0.028/0.039/0.044, all *below* the 0.05 chance rate; (7,11) significantly below (t=−3.81, p=0.0066). This is where the 89% was measured |

### 2026-08-12 — symmetric room, Moser session sequence

**Why.** The standing top recommendation. Reasoning: the L-room gives the agent an L-shaped
wall plus a triangle/plus/x, so it localises from geometry and never needs the object. Remove
every landmark and the object becomes the only cue breaking the symmetry, making an
object-coding cell mechanically *required*.

**Design.** `MiniGrid-SquareRoom-v0`, four-fold symmetric, trained **from scratch**, sessions
`no object → 6 positions → no object` after Tsao/Moser 2013.

**Results — null, and partly confounded.**
- Object cells never above the 5% chance rate (best 6.4%, p=0.09).
- Trace cells null. I first reported 17.0% at p<1e-4 — **my error**, see §5.
- **Quadrant decode from `h`: 0.81–0.85 with the object and identical without it**
  (delta ~0.000, trajectory-level CV). The net localises fine with no object.
- The arena really is symmetric: six (cell, heading) pairs vs their 90° rotations,
  `max|obs diff| = 0`.

**Hypothesis for why it failed.** Not the room — the *assumption*. Removing visual landmarks
left a second route to position: **trajectory history / path integration**. So there are now
**two demonstrated escape routes**, and every design so far has closed at most one:

| room | how it localises without the object |
|---|---|
| L-room | geometry — L-wall, triangle, plus, x |
| square room | trajectory history — dead reckoning |

**Confound that limits the null.** The policy collapsed from session 4: entropy 1.54 → 0.17
bits (max log2(4)=2.0), loc_entropy 6.04 → 4.55, and the world model degraded (within-session
loss started *rising*). Sessions 4–6 carry object positions (12,3), (3,9), (11,12) — exactly
the collapsed ones. **Those nulls are uninformative, not negative.** Specific to the square
room: the L-room reference holds entropy 1.30–1.53 over 80k gradient steps. Likely cause is
`entropy_coef: 0.0` in `Configs/algo/ppo.yaml`; `performance=ultra` already sets 0.01.

The quadrant-decode conclusion is *not* affected — it uses a random-action probe.

---

## 3. Infrastructure — 2026-08-11/12

- **Merged `perf/device-resident-rollouts`.** Kept sequential GAE and an explicit
  `log_softmax` so `tests/golden_omt` stays **bitwise** (191 fields, 0 differ); both cost
  ~0.1%. Env optimisations are worth **1.94x** on gradient steps/s.
- **The gradient sweep killed pooling.** 16x the data per step buys 5% lower loss, while
  bigger batches give *fewer* gradient steps/s (0.98 → 0.57). B=128 pooled has 20x the FPS of
  serial and one sixth its gradient steps/s. **FPS is the wrong objective.** `batched_wm`
  stays off; we run 8 envs collecting in parallel but **8 separate gradient steps per update,
  batch 1 each**.
- **Schedule refactor.** Run length is `rl.episodes_total` trajectories; total steps, updates
  and both optimizer budgets are derived and printed at startup. All cadences are in
  environment **steps** (`logging.*_every_steps`). Previously they counted *updates*, so under
  `ultra` a whole run was 78 updates and save/analysis/plot never fired once.
- **Metrics.** Sampling now draws without replacement from a fixed seed (sRSA/SWdist are
  reproducible); the spatial eval is a fixed probe (`probe_seed`) and runs in `eval_mode`
  (dropout off, noise kept — see `../sab_context/open_choices.md` 1b).
- **Verification passed.** `verify-lroom-0812` reached pRNN loss 0.01092 at 12,000 gradient
  steps; reference 0.01131 at 10,000.
- **CUDA graphs measured at 2.8x (3.80 → 10.53 steps/s) and deliberately NOT used** — there is
  a documented silent-corruption incident (a 2026-07-22 cluster run died with
  `obs.direction`==184 after a device round-trip freed captured parameters).

---

## 4. New this session — object-vector work

**Why the pivot.** Every test so far hunted a *trace* — a field where an object used to be.
Two arguments that this was the wrong target:
1. **The objective forbids traces and demands vectors.** Predicting the next observation under
   masking requires knowing your metric relation to what is visible. A trace would push the
   readout to predict an object that is not there, which *raises* the loss.
2. **The permanent landmarks are objects and were never treated as such.**
   `scripts/trace/trace_objvector_test.py` measured the *change in* object-relative tuning from
   *exposure to the novel object* — never pre-existing tuning to the landmarks that have been
   present for all 80k gradient steps.

**Built:**
- `MiniGrid-LRoom-SmallObj-v0` (minigrid `48c2925`, pushed): solid 2×2 landmarks at (5,5),
  (11,5), (5,11). `LEnv` default geometry is byte-identical (`grid.encode()` sha1
  `6f48cd7fbcae481d`), so the 80k checkpoint stays probeable.
- **A trained checkpoint for it**: `outputs/smallobj-lroom_curious_26-08-12-19-21-53`, 40k
  gradient steps, healthy (policy entropy 1.392, loc_entropy 6.626, sRSA 0.53–0.67, loss
  0.00446). Lower loss than the reference at half the steps because the room has 12 coloured
  tiles instead of 69.
- `scripts/trace/ovc_metric.py` + `tests/test_ovc_metric.py` (8 tests) and
  `scripts/trace/ovc_eval.py`. Plan in `../exp_instructions/instructions-OVC.md`.

**Key simplification:** the anchor-centred offset map is the rate map re-indexed,
`R_A(dx,dy) = M[A + (dx,dy)]`, so the three per-anchor maps are three **crops of one map**. A
vector cell has the same bump in all three; a place cell has one bump in one crop.

**Status: Stage 0 does not pass, so no E1 result exists.** Detector is SPECIFIC but not
SENSITIVE:

```
negative (odd/even, no injection) : 0.014   want <= 0.05   PASS
specificity, place field amp 1/4  : 0.012 / 0.006          PASS
positive, amp 1.0 / 2.0 / 4.0     : 0.062 / 0.210 / 0.575  FAIL (want >= 0.90)
```

Next step is to detrend each map before cropping — the common mode is what inflates each
unit's null.

---

## 5. Failure modes encountered

Each cost a retracted claim. The first five predate this session.

1. **Pooling** (3 separate times) — averaging across the variable the mechanism runs on. The
   third produced a "structurally impossible" conclusion reported twice.
2. **n=1 effect sizes** — a "+142%" became non-significant at n=10.
3. **Reading a running job** — `max(steps)` picked step 0 of an unfinished run.
4. **Implausibly exact agreement is a bug signal** — two numbers matching to four decimals
   meant one checkpoint loaded twice.
5. **Confusing readout and hidden-state measurements.**
6. **A null that grows with the number of chances.** The trace criterion ORs over every
   previously-used object position, so chances grow with session index. Against a flat 5% it
   read 17.0% at p<1e-4 and looked like the result the project was after. Against the correct
   `1-0.95^n`, and against an empirical null run on positions the object never occupied, every
   count was BELOW chance. **Whenever a criterion is an OR over a growing set, build the null
   empirically from control locations.**
7. **Two more wrong nulls, in the OVC work.** (a) Permuting trajectory↔position pairing
   destroys *all* spatial structure, so it tests "is there structure" not "is it anchored" —
   negative control 0.130, place fields scoring 0.146. (b) Random anchor triples with a
   *population* threshold: p99 = 0.933, so nothing can clear it. Only a **within-unit** null
   works. Building the null is harder than building the metric.
8. **Quoting wandb `summary` values.** Those are the last single logged point. I claimed "the
   object lowers prediction loss" from them; windowed means killed it.
9. **Unit errors.** I flagged `policy_entropy` above ln(4) as "impossible" — it is logged in
   **bits** (max 2.0). My error, not the metric's.
10. **Row-level CV on temporally correlated data.** The first quadrant decode leaked, since
    samples from one trajectory appeared in train and test. Trajectory-level splits gave the
    same answer here, but it was luck.
11. **Operational: waiter loops keyed on a formatted string.** Seven background loops span for
    up to 16 h because `grep "meanSI  mean="` had two spaces and the output had three. Wait on
    **process exit**, not on output text.

---

## 6. Traps in the repo

- **`docs/` was gitignored until 2026-08-12.** Now tracked except `docs/sab_context/`. Older
  docs survived only because they predate the rule; the 2026-08-12 results had no committed
  record until the reorganisation.
- **`*.png` is still globally ignored** (`.gitignore:12`), so the figures the tracked docs
  point at are themselves untracked.
- **`scripts/` was reorganised** into `legacy/`, `trace/`, `moser/`. Imports and doc paths were
  rewritten; the sweep missed module paths inside **string literals** (`unittest.mock.patch`).
- **`uv run --no-sync` is obsolete.** It existed while minigrid was editable; minigrid is
  pinned from git again, so `--no-sync` now gives a *stale* environment.
- **`prnn` is pinned by BRANCH** in `pyproject.toml`; it moved on its own once during an
  ordinary `uv run`.
- Gate baseline: **136 passed, 18 skipped, 0 failed, 7 deselected.** The 18 skips are
  `test_view_coords.py` needing `trajectories.pt` from the deleted pre-migration dumps. The
  older "146 passed" baseline included those.

---

## 7. CERTAIN vs UNCERTAIN

### Certain (measured, replicated, or gated)

- The `W_out` mechanism and its ~89% position-independence (n=8).
- `h` is unchanged by object exposure — map r ≈ 0.98, null in three reference frames.
- Sequential displacement is a null at n=8, with fractions *below* chance.
- The occlusion result was a false positive; its own location control killed it.
- The square room is observation-symmetric (`max|obs diff| = 0`, six pairs).
- Quadrant decodes at ~84% from `h` **with and without** the object, trajectory-level CV.
- The policy collapsed in the square room from session 4, and this is specific to that room.
- Pooled world-model training loses to serial on loss per gradient step.
- The rebuilt stack reproduces the reference learning curve.
- `LEnv` default geometry is unchanged by the small-object work (sha1 verified).
- The small-object checkpoint trained healthily.
- The pRNN's rate maps carry strong room-wide common structure — random anchor triples
  correlate at p99 = 0.933. Measured, and it is why a population-level null fails.

### Uncertain

- **Whether object-vector cells exist here.** No E1 result. The detector is specific but
  recovers only 57% of injected fields at 4x mean rate, so it cannot yet bound a null.
- **Whether the square-room object/trace nulls mean anything for sessions 4–6** — confounded
  by the policy collapse. Treat as untested.
- **The occlusion (12,7) control, "0.0835 vs 0.0441."** Load-bearing for that retraction, and
  it exists only as a one-line summary — no cache, not re-derivable.
- **`fig_scenarioC_gradient.png` has no generating script.** From a throwaway; not reproducible.
- **The per-session entropy table in `compaction.md` §8** could not be reproduced from wandb
  under any aggregation, because I never recorded which one I used. The collapse conclusion is
  independently confirmed; the exact numbers are not.
- **Whether this architecture can do vector coding at all.** If the detector passes Stage 0 and
  E1 is still null, that is the answer, and the target — not the room — was wrong.
- **Whether "trace cell" was ever the right target.** LEC cells show little spatial modulation
  in an empty field; these units have mean SI ≈ 0.9 and are place-cell-like. We may have been
  hunting LEC phenomenology in something closer to MEC. Worth a conversation with the PI
  before more GPU.
- **Whether RL matters at all** for any of this. Untested; a random-action checkpoint exists
  (`RAND_CKPT_DIR`) so it is free to check.

---

## 8. Immediate next steps

1. **Detrend the maps** in `ovc_metric`, re-run Stage 0. If it passes, E1 on both checkpoints —
   the 6×6 baseline and the small-object net. That pair is itself a control: object-vector
   cells appearing only with small landmarks would be direct evidence the 6×6 patches were
   coded as boundaries.
2. **Re-run the symmetric room with `rl.entropy_coef > 0`** and a `loc_entropy` gate, to turn
   the confounded sessions into a real test.
3. **Multi-environment training** — the standing recommendation, now with a mechanistic
   argument: the same integrated trajectory maps to a different absolute position in a
   different room, so dead reckoning cannot resolve position.
4. `git push origin sdu/object-into-hidden-state` — needs you; the gate blocks me.

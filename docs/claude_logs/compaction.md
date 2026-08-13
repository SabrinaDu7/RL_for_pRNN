# Handoff — object / trace-cell work in the pRNN

**Read this first.** Written 2026-08-11. Branch `sdu/object-into-hidden-state` (pushed).
Everything below is committed. Full detail in
`exp_object_trace_cells_2026-07-30.md`, `exp_object_into_hidden_state_2026-08-01.md`,
`exp_trace_cell_scenarios_2026-08-11.md`; index in `README_object_experiments.md`.

---

## 1. The goal, and the answer so far

Produce **object-trace cells** in the pRNN hidden state `h` — units with a place field where a
novel object used to be (`docs/ref-trace-cells.png`, Tsao/Moser 2013).

**They do not appear, and we now know why.**

### The mechanism (this is the key result)

The object is written into `W_out`, the linear prediction head — **and almost
position-independently**. Measured at n=8 on the sequential-displacement runs:

```
green change at (4,7)'s own view cell, BEFORE the object ever goes there:  +0.0395 ± 0.011  (t=9.57, p<1e-4)
green change at (4,7)'s own view cell, WHILE the object is there:          +0.0445 ± 0.008
```

**~89% of the apparent "object signal" at a location is generalisation from exposure
elsewhere.** `W_out` has ONE row per view cell, applied to every hidden state, so it learns
"boost green at this object-centred view offset" rather than "there is an object at (4,7)".

Consequences, and they explain every result in the project:
- nothing location-specific needs to be learned → **every hidden-state metric is null**
- the readout does change → **the pixel-space metric always found something**
- the readout signal **tracks the present object and collapses when it moves** (transfer, not
  trace): (7,2) 0.0490 → 0.0196 after departure (p=0.0007); (4,7) 0.0445 → 0.0205 (p=0.0043)

### Supporting facts

- Object memory is decoder-localised: readout +0.0625 vs dynamics +0.0157 gain-corrected, 9/9 runs.
- The place code is unchanged by exposure: median per-unit map correlation **r ≈ 0.98**.
- Null in **three** reference frames: allocentric maps, object-location modulation, object-vector.
- The object is ~**27%** of the in-view per-step reward (NOT ~2% — that earlier claim was wrong;
  its pixel error is ~40× the average pixel's).
- Behaviour is driven purely by the **reward**: the policy's input is `h`, which never changes.
  In-view timesteps pay ~1.8× the reward of other timesteps.
- The net never learns the object well: predicts **0.56** against a target of **0.99** after
  3000 trajectories, so in-view error stays elevated indefinitely and the reward keeps paying.

---

## 2. Everything tried, and its verdict

| experiment | verdict |
|---|---|
| standard OMT exposure | object → `W_out`, `h` unchanged |
| `lr_trials=[2,0,8]` (freeze readout, `W_in`×4) | **only positive on `h`**: encoding 0.6947 → 0.7214, p<1e-5, n=3. Memory unchanged. |
| A — object identity varies (4 colours) | null, n=3 |
| D — weight-decay sparsity pressure | **untestable**: L2 shrinks all weights, place code collapses r 0.97→0.73. Sparsity hypothesis still open. |
| random object position (i.i.d. per batch) | null, n=10 |
| `k_curious` = 5, 20 | null |
| C — occlusion | **FALSE POSITIVE.** Looked strong (excess +0.0563, p=0.001, 8/8 seeds, graded) but the peak does **not** follow the object: moving it to (12,7) leaves (14,7) highest (0.0835 vs 0.0441). |
| sequential displacement, 4-phase + REMOVED | **null**, n=8. Nothing forms during exposure at all (0.028–0.044, below the 0.05 chance rate). |
| symmetric room (Moser sequence, from scratch) | **null**, 2026-08-12. Object cells never above chance (best 6.4%, p=0.09); trace cells null once tested against the right null. **The room did not bind**: quadrant decodes from `h` at ~84% *with and without* the object (delta ~0.000), so position arrives by trajectory history, not vision. Details in `sab_context/goal_2026-08-12.md` |
| multi-environment pretraining | **NOT BUILT — now the top recommendation, see §4** |

---

## 3. The metric, and the rules for using it

`scripts/trace/trace_metric.py`.

```
field_gain g(u,c) = mean rate in a radius-2 disc at c / unit's mean rate   (scale-free)
trace score dg(u,c) = g_post - g_pre
```

Unit score = the object cell's **percentile among all 172 walkable cells** (within-unit null);
object cell if > 95; population test = binomial vs 0.05.

**Validation (report with every result):** negative control (odd vs even probe trajectories)
**0.0601, p=0.174**; positive control **100% recall at a field 1% of mean rate in 10% of units**.
That sensitivity is what makes each null a bound rather than a shrug.

### Two rules any successor must follow

1. **The peak must move with the object.** `location_control_matrix` — score every candidate
   location under every run. The within-unit null CANNOT see drift shared across units, which
   is what produced the false-positive scenario C.
2. **Never use (14,7) as an object location in LRoom.** It has produced spurious effects in
   three independent analyses and reads elevated even before the object arrives.

---

## 4. Recommended next step

**SUPERSEDED 2026-08-12 — the symmetric room was built and is null.** Keep reading for why the
reasoning was right and the assumption was wrong; the replacement recommendation is at the end
of this section.

**Build the symmetric room.** Every failure has the same shape: the object is *predicted* but
never *needed*. The L-room has a triangle, plus, x and asymmetric walls, so the agent localises
perfectly without the object — its position never has to enter the state estimate, only the
output. Make the room symmetric so the object is the **only** disambiguating cue, and the agent
cannot know where it is without encoding where the object is. That is the one untested design
where a trace cell is mechanistically *required* rather than merely permitted.

Second: **multi-environment pretraining**. `h`'s rigidity (r ≈ 0.98) may be because one room
admits a fixed lookup; real place cells remap because they must. Both need a new `main_train`
baseline (~11.6 h for a full one at the measured 0.52 s/traj; a 3.5 h / 31k-trajectory baseline
was verified to still have a good place code — SI median 0.690 vs 0.759).

---

## 5. Operational notes

- **Checkpoint provenance.** Mila's `.env` sets `CUR_CKPT_DIR` to a RELATIVE path that resolves
  inside `$SLURM_TMPDIR` and does not exist there. `slurm/otc_seq.sh` now pins it and asserts
  sha256 `c1e43a6b…` — the baseline every result used. Do the same in any new slurm script.
- **Run-name collisions.** Run dirs are timestamped to the second; parallel slurm jobs collide.
  rsync with the job-ID parent directory, not the run dir.
- **Occlusion.** `Ball`/`Box` do NOT occlude (`see_behind()` is True); only
  `see_through_walls=False` does. The obs bank is keyed on `grid.encode()`, which does not
  capture occlusion — an `-occl` fingerprint suffix was added or it would serve non-occluded
  observations to an occluded env.
- **Mila needs an OTP** for ssh; the user supplies it.

---

## 6. Recurring failure modes (each cost a retracted claim)

1. **Pooling.** Three times — the reward map binned by agent position, input-driven vs
   memory-only timesteps, and the five input-mask phases — averaging across the variable the
   mechanism runs on turned a real effect into a flat null. The third produced a
   "structurally impossible" conclusion reported **twice** before being caught.
2. **n=1 effect sizes.** A "+142%" result became non-significant at n=10.
3. **Reading a running job.** `max(steps)` picked step 0 of an unfinished run and produced a
   fabricated "catastrophic" result.
4. **Implausibly exact agreement is a bug signal** — two measurements matching to four decimals
   meant I had loaded the same checkpoint twice.
5. **A null that grows with the number of chances.** The trace criterion is a cumulative OR
   over every previously-used object position, so the number of chances to score a hit grows
   with session index. Tested against a flat 5% it produced 17.0% at p<1e-4 and looked like the
   result the whole project was after; against the correct `1-0.95^n`, and against an empirical
   null run on positions the object never occupied, every count was BELOW chance. **Whenever a
   criterion is an OR over a growing set, the null grows too — build it empirically from control
   locations rather than assuming the per-test rate.**
6. **Confusing readout and hidden-state measurements.** They answer different questions; the
   readout generalises across locations, the hidden state does not change at all.

---

## 7. Architecture facts that dictate any analysis

- `thRNN_5win` is a `MaskedRNN` with **`inMask = [True, False×5]`**: the observation reaches the
  net only **1 timestep in 6**, while it must predict at every step. ph0 = input-driven,
  ph1–ph5 = memory. **Never pool them.**
- `h` is **(1, T, 500)**, not (5, T, 500) — `MaskedRNN` never sets `self.k`, so every
  `torch.mean(h, dim=0)` in the tree is a no-op.
- `predict(batched=True)` takes **3-D (B, L, X)** and is correct on the current pin; it injects
  fresh noise every call, so replays must seed or zero `trainNoiseMeanStd`.
- `outlayer = Linear(500→147, no bias) → Sigmoid`; the 147 outputs are 7×7×3 **pixel
  intensities**, not classifications.
- The object is a non-blocking `FloorBright` tile; `see_through_walls=True`; action space
  `Discrete(4)` with **no interaction primitive**.

Gate: `uv run pytest` → **126 passed, 0 failed, 7 deselected**. Keep it there.

---

## 8. Replacement recommendation (2026-08-12), after the symmetric room came back null

The symmetric-room reasoning was right about the *shape* of the problem and wrong about the
escape route. Removing visual landmarks did not make the object necessary, because the network
localises by **trajectory history / path integration** instead: quadrant decodes at ~84% from
`h` with the object absent, and the object adds nothing (delta ~0.000 at every session,
trajectory-level CV). The arena really is observation-symmetric — checked directly, six
(cell, heading) pairs against their 90-degree rotations, `max|obs diff| = 0`.

So there are now **two** demonstrated escape routes, and every design so far has closed at most
one:

| room | how the net localises without the object |
|---|---|
| L-room | geometry — an L-shaped wall plus a triangle, plus, x |
| square room | trajectory history — dead reckoning from the start of the episode |

### What to try next, in order

1. **Multi-environment training.** This is the one manipulation that attacks the history route
   *without* new machinery. The same integrated trajectory maps to a different absolute
   position in a different room, so dead reckoning alone cannot resolve position — the net has
   to identify *which room it is in* from what it sees, which is exactly the pressure to bind a
   visual feature to a place. It was already the standing second recommendation; the quadrant
   decode is now a mechanistic argument for it rather than an analogy to remapping.

2. **Break path integration directly**: teleport the agent mid-episode, and/or randomise the
   initial hidden state per trajectory. Cheap pre-check before building anything — re-run
   `scripts/moser/moser_decode_quadrant.py` with the start cell randomised and hidden from the
   decoder. If accuracy collapses toward 0.25, history is confirmed as the route.

3. **Combine (1) or (2) with `lr_trials=[2,0,8]`** (freeze the readout). That is the only
   manipulation that has ever moved `h` (encoding 0.6947 -> 0.7214, p<1e-5, n=3) and it has
   never been combined with a room where the object is actually needed. Freezing `W_out`
   removes the absorbing route; the room removes the localisation route. Both halves of the
   mechanism, closed at once.

4. **Make the object task-relevant.** Nothing in the current objective requires knowing where
   the object is — curiosity pays for *visiting*, and the net never learns the object anyway
   (predicts 0.56 against a target of 0.99), so the bounty is permanent and never forces a
   representational change.

### One framing question worth raising before more GPU

Tsao/Moser's LEC cells show *little* spatial modulation in an empty field. These pRNN units have
mean SI ~0.9 — they are strongly spatial, i.e. place-cell-like. We may be looking for LEC
phenomenology in an architecture closer to MEC/hippocampus, in which case the target itself,
not the room, is what needs rethinking.

### ⚠️ The 2026-08-12 symmetric-room null is CONFOUNDED — found 2026-08-12 on review

The policy collapsed partway through that run, and the world model degraded with it.

```
session        s0     s1     s2     s3     s4     s5     s6     s7
policy entropy 1.54   1.35   0.94   0.98   0.25   0.31   0.17   0.23     (max ln4 = 1.386)
loc entropy    6.04   6.58   6.84   6.59   5.15   5.29   4.55   5.11
pRNN loss      .0133  .0084  .0070  .0069  .0082  .0091  .0091  .0086
```

Everything turns at **session 4**: the policy goes near-deterministic, the agent stops covering
the room, and the pRNN loss stops improving and starts rising — within-session it now gets
WORSE (s4 0.00636 -> 0.00876; s7 0.00818 -> 0.00927).

It is specific to this room, not systemic. The L-room reference holds policy entropy 1.30-1.53
and loc entropy 6.20-6.62 across **80,000** gradient steps; the 12k verification run likewise.
The square room collapses by 16,000.

**Why this matters scientifically:** object positions for sessions 4, 5 and 6 are
(12,3), (3,9), (11,12) — exactly the sessions with collapsed exploration. If the agent rarely
visited them, no object representation could form there whatever the architecture allows. Those
sessions' nulls are therefore uninformative, and the trace counts that depend on them inherit
the problem.

**Likely cause and the cheap fix:** `Configs/algo/ppo.yaml` has `entropy_coef: 0.0`, so nothing
resists policy collapse. In the L-room the curiosity reward is spatially structured and keeps
the policy responsive; in a symmetric room the reward landscape is far more degenerate, so PPO
can drift to a deterministic policy with no gradient pulling it back. Note
`Configs/performance/ultra.yaml` already sets `entropy_coef: 0.01` — the perf work had reached
the same conclusion from a different direction.

**Before re-running anything in the symmetric room:** set `rl.entropy_coef > 0` and gate on
`loc_entropy` staying flat across sessions. That gate is cheap and it is now a precondition,
not an optional check.

**Units — a correction.** `policy_entropy` is logged in BITS, not nats:
`losses.py:51` sets `policy_entropy_bits = policy_entropy / ln2` and `updater.py:155`
accumulates that field. The maximum for four actions is therefore `log2(4) = 2.0`, and the
L-room's 1.49-1.58 is entirely legal. An earlier note here called those values impossible by
comparing bits against `ln(4)`; that was a unit error on my part, not a defect in the metric.

In the right units the collapse is sharper, and it is a collapse rather than a starting
condition:

```
L-room reference   1.53 bits = 76.5% of maximum
square room s0     1.54 bits = 77.0% of maximum   <- starts exactly as healthy
square room s6     0.17 bits =  8.5% of maximum   <- near-deterministic
```

So the symmetric room does not begin degenerate. It begins indistinguishable from the L-room
and collapses during training, which points at the optimisation (`entropy_coef: 0.0`) rather
than at anything intrinsic to a symmetric arena.

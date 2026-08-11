# Handoff — object / trace-cell work in the pRNN

**Read this first.** Written 2026-08-11. Branch `sdu/object-into-hidden-state` (pushed).
Everything below is committed. Full detail in
`exp_object_trace_cells_2026-07-30.md`, `exp_object_into_hidden_state_2026-08-01.md`,
`exp_trace_cell_scenarios_2026-08-11.md`; index in `README_object_experiments.md`.

---

## 1. The goal, and the answer so far

Produce **object-trace cells** in the pRNN hidden state `h` — units with a place field where a
novel object used to be (`docs/trace-cells-spatial-tuning.png`, Tsao/Moser 2013).

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
| symmetric room | **NOT BUILT — top recommendation** |
| multi-environment pretraining | **NOT BUILT** |

---

## 3. The metric, and the rules for using it

`scripts/trace_metric.py`.

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
5. **Confusing readout and hidden-state measurements.** They answer different questions; the
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

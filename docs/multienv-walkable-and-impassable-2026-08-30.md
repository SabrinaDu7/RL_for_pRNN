2026-08-30 · branch `sdu/multienv` · commit `56cd4e8` · Mila L40S

# Multi-environment training: 5 walkable rooms, then 5 and 10 impassable

**Both deliverables are met.** A walkable 5-room run reaches per-room sRSA **0.80** —
above the single-room baseline's 0.72 — in **29.9 minutes**, and the impassable arms land
in **30.3 / 32.0** minutes. No arm collapses. The open question is a real one:
impassability costs ~0.18 of per-room sRSA, reproducibly.

## Results

Tail means (per-room sRSA and SWdist over the last 2 analysis points, loss over the last
500 gradient steps, `MI_policy` over the last 200 updates). `parity-defaults` is the
single-room control, run today with no overrides.

| run | min | pRNN loss | per-room sRSA | SWdist | MI | % updates < 1.0 bits |
|---|---|---|---|---|---|---|
| **walkable n=5 s2, half budget** | **29.9** | 0.00526 | **0.8009** | 0.01637 | 0.0419 | 0.0% |
| **walkable n=5 s3, half budget** | **29.9** | 0.00527 | **0.7761** | 0.01795 | 0.0409 | 0.0% |
| impassable n=5 s2, half budget | 30.3 | 0.00561 | 0.6178 | 0.00964 | 0.0448 | 0.0% |
| impassable n=10 s2, half budget | 32.0 | 0.00619 | 0.6349 | 0.01704 | 0.0449 | 0.0% |
| walkable n=5 s2, full budget | 52.6 | **0.00433** | 0.7992 | 0.04013 | 0.0360 | 0.0% |
| impassable n=5 s2, full budget | 52.7 | 0.00460 | 0.6549 | 0.01410 | 0.0403 | 0.0% |
| impassable n=10 s2, full budget | 55.9 | 0.00524 | 0.6847 | 0.01725 | 0.0402 | 0.0% |
| *single-room `parity-defaults`* | *27.6* | *0.00438* | *0.7235* | *0.09719* | *0.0363* | *0.0%* |

Read against the noise floor in `docs/entropy-sweep-and-noise-floor-2026-08-29.md`
(n=5, one config): loss CV **1.7%**, sRSA CV **4.5%**, SWdist CV **28.3%**.

**Against the brief:**

- *loss low* — the full-budget walkable arm matches the single-room control exactly
  (0.00433 vs 0.00438, well inside 1.7%). The half budget costs ~20% (0.00526).
- *per-room sRSA high* — **0.80 and 0.78 across two seeds, above the 0.72 single-room
  control.** Multi-room training does not degrade the place code; it improves it.
- *SWdist low-ish* — 0.016-0.018 against the single-room 0.097, so **better**, not merely
  comparable. ⚠️ At a 28.3% CV this is the least trustworthy of the four.
- *~30 min* — yes, at half budget.
- *mutual information* — 0.041-0.045, at or above the single-room 0.036.
- *no collapse* — **0.0% of updates below 1.0 bits in all seven runs**, and `loc_entropy`
  medians 7.11-7.16 against the single-room 7.13.

## The finding: impassability costs sRSA, and it is not a coverage artifact

Walkable 0.80/0.78 against impassable 0.62 (n=5) and 0.63 (n=10) — a gap of ~0.18, far
outside sRSA's 4.5% seed CV, and it reproduces at both room counts and at both budgets
(full budget: 0.799 vs 0.655/0.685).

**It is not simply that the agent visits less.** `loc_entropy` medians are 7.16 (impassable)
against 7.11 (walkable) — the impassable agent covers *slightly more* of its support, not
less, and neither arm collapses. The impassable rooms have 153 standable cells against 172,
so the support itself is smaller; the metric is measured over what each arm can reach.

*Inferred, and the next thing to test:* the pRNN must now predict an observation stream
whose transitions are blocked in room-specific places, so identical (position, head
direction) states have different consequences per room. That is a harder prediction problem
and shows up as both higher loss (0.0056 vs 0.0053 at matched budget) and lower sRSA.

## Why half the budget

Multi-room runs at **35,120 environment steps/s** against single-room parity's ~54,300 —
~35% slower per step before any evaluation — so the full 89,980,928-step budget is ~43 min
of training plus ~2 min per analysis event.

Per-room sRSA over the full-budget walkable run:

```
0.617 -> 0.713 -> 0.786 -> 0.795 -> 0.804     (5 analysis events)
```

It is already 0.7945 at 47M steps, above the single-room control. So half the budget
(21,968 world-model / 87,872 policy gradient steps, 44,990,464 environment steps) buys
nearly all of the representation quality and lands at 30 min. The cost is prediction loss:
0.00526 against 0.00433. **If loss matters more than wall clock, run the full budget** —
`sbatch slurm/multienv.sh false 5 2` with no fifth argument.

## Methodology

**The room set is pinned by ANCHOR, not by index, and that is load-bearing.** The ten rooms
named resolve against the *impassable* pool - all 10 `Layout.key` hashes match
`Uniform(n=200, seed=7)` at `impassable=True`. But `admissible_placements` takes `content`,
and impassable landmarks admit 9,074 placements against walkable's 19,820, so the two pools
are different sequences: **0 of the 5 indices name the same room in both**. `ROOMS_SELECTED`
therefore commits the anchors and `Selected(n, impassable)` applies the affordance, so one
flag changes one variable.

**Verified before any training** (`tests/test_selected_rooms.py`):

| check | result |
|---|---|
| committed keys == the pool's at (0,14,31,35,83,126,144,169,191,195) | 10/10 |
| anchors identical across the affordance flip | 10/10 |
| observations identical across the flip, every (position, direction) | **0 of 3,440 differ** |
| impassable removes exactly its landmark cells | 172 -> 153 standable |

Visual check: `outputs/figures/selected_rooms_walkable_vs_impassable.png`, agent pinned at
(7,6) so the sprite is constant across all ten panels.

**Tricks used: none.** The brief asked for the *minimum viable* set, and the minimum is zero
until a baseline says what is broken. Nothing was broken: no arm collapsed, loss fell
normally, sRSA exceeded the single-room control. `docs/claude_logs/rl_tricks_2026-08-29.md`
ranks four candidates and they remain unspent — advantage normalization first, then the
LR warmup (which is 🔴 absent in BOTH repos and would be silently dead under
`train_policy.cuda_graph`, exactly as `entropy_coef_final` was).

## ⚠️ What is NOT established

- **Rooms 0 and 83 differ by ONE cell** — both `plus@(3,10) block3@(12,5)`, `x` at (5,3) vs
  (6,3). The 5-room set carries less configuration diversity than "five rooms" suggests.
- **n=2 seeds on the walkable arm only** (0.801, 0.776). Every impassable number is n=1
  against a 4.5% sRSA CV, so the 0.18 walkable-vs-impassable gap is solid but the
  5-vs-10-room difference (0.618 vs 0.635) is not.
- **SWdist has a 28.3% CV** and its estimator noise on a frozen checkpoint (0.117) exceeds
  every difference here. Do not read it.
- **The random-agent baseline has now RUN** (2026-08-31). The config-space blocker described
  here originally - three pairwise-conflicting `Config` constraints - was removed by
  `22e7c0d`, which routed random actions through the one collection path;
  `tests/test_random_agent_baseline.py` pins it. Measured, both at n=1:

  | rooms | learned (raw, e=0.003) | random walker |
  |---|---|---|
  | walkable | 0.00526 / 0.8009 | **0.00347 / 0.8595** |
  | impassable | 0.00561 / 0.6178 | **0.00388 / 0.6440** |

  ⚠️ **The loss column is NOT a fair comparison**: the two agents generate different training
  distributions, and a broad random sweep is genuinely easier to predict - lower loss under a
  random walker is partly the curiosity objective working as intended. sRSA IS fair, since
  both are scored by the same probe, and there the raw learned policy is *behind* by 0.059
  and 0.026 against a 4.5% CV. A tuned policy does better - see the normalization section.
- **`remapping_index` is deliberately ignored**, per instruction.
- The spatial probe is still **unseeded** (`probe_seed` accepted, passed by nothing), so
  every sRSA point carries its own rollout noise on top of the seed noise.
  *(Dated 2026-08-30. Seeded the next day - and the first threading leaked
  the seed into the training streams; docs/invalid-runs.md has the line.)*

## Advantage normalization: the right direction, and it needed 40x more entropy

Added 2026-08-31 (`train_policy.normalize_advantage`, branch `sdu/optim-pred`). Impassable
n=5, half budget:

| whitened `entropy_coef` | loss | per-room sRSA | MI | % pe<1.0 |
|---|---|---|---|---|
| 0.024 (s2/s3) | 0.00627 / 0.00703 | 0.5050 / 0.5275 | 0.13-0.15 | 1.3% / 2.6% |
| 0.035 | 0.00609 | 0.5683 | 0.107 | 0.0% |
| 0.07 | 0.00556 | 0.6420 | 0.048 | 0.0% |
| **0.12 (s2/s3)** | 0.00545 / 0.00541 | **0.6735 / 0.6940** | 0.030 | 0.0% |
| 0.20 | 0.00527 | 0.6210 | 0.019 | 0.0% |
| 0.35 | **0.00515** | 0.6417 | 0.012 | 0.0% |
| *raw, e=0.003 (s2/s3)* | *0.00561 / 0.00562* | *0.6178 / 0.6419* | *0.045* | *0.0%* |

**At e=0.12 it beats raw**: sRSA 0.6838 against 0.6299 (n=2 each, +0.054 ~ 2.3 sd at a 4.5%
CV) and loss 0.00543 against 0.00562.

🔴 **A first attempt reported it as clearly WORSE, and that was a tuning error.** The ratio
that governs exploration is `entropy_coef / |adv|`, and whitening RAISES |adv| from ~0.087 to
1, so the raw knee of 0.003 becomes ~40x too weak. Those arms collapsed for 67-70% of
updates. Nothing about whitening was being measured.

**Loss and sRSA optimize at DIFFERENT coefficients** - loss falls monotonically to 0.35 while
sRSA peaks at 0.12 and drops by 0.20. Higher entropy flattens the visiting distribution
toward something easier to predict but less spatially structured. Report both or neither.

**And at e=0.12 the learned policy finally beats the random walker on sRSA** (0.684 against
0.644) - the first evidence in this work that the policy contributes anything. n=2 against
n=1; suggestive, not settled.

⚠️ The mechanism justification is weaker than the result. `advantages_std` really does halve
over a run (0.087 -> 0.040, so the effective coefficient doubles), but the predicted symptom -
raw `policy_entropy` drifting upward - is absent: it is flat at 1.84-1.89 against a 2.000
ceiling, i.e. saturated, with nowhere to drift. And the transfer argument does not apply
here: walkable and impassable have essentially identical advantage scales (0.0871 both), so
the knee would carry between them even raw.

## 🔴 A rendering hazard found on the way

`Grid.tile_cache` (MiniGrid) is a **process-global class dict** keyed on
`obj.encode() + (agent_dir, highlight, tile_size)`. Running
`pytest tests/test_novel_object.py tests/test_selected_rooms.py` made the
observation-identity assertion report **209 of 688 observations differing**, while either
file alone reports 0 — a previous test's cache entries change what a later one renders.

The underlying claim is independently TRUE: rendering a `Floor` and an `Obstacle` of the
same colour directly, with the cache cleared, gives byte-identical tiles at tile_size 1
and 8, cold and warm. So the clean-process measurement is the correct one and the 209 was
the artifact.

**The hazard is wider than the test.** `ObsBank` builds each observation bank once and
caches it to DISK keyed on the grid fingerprint, so a bank built in a process whose tile
cache was already dirty would be wrong and then reused for every subsequent run. Nothing
currently clears the cache before a bank is built. `tests/test_selected_rooms.py` now
clears it; `envs/obs_bank.py` does not, and probably should.

## Reproduce

```bash
sbatch slurm/multienv.sh false 5 2 sdu/multienv 21968   # walkable, ~30 min
sbatch slurm/multienv.sh true  5 2 sdu/multienv 21968   # impassable, same rooms
sbatch slurm/multienv.sh true 10 2 sdu/multienv 21968   # ten rooms
sbatch slurm/multienv.sh false 5 2                      # full budget, ~53 min
```

Gate at this commit: **509 passed, 1 deselected**.

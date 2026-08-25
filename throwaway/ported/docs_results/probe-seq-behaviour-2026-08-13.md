2026-08-13

# Did behaviour follow the object through A -> B -> C -> REMOVED?

Fills a gap in `result-summary-2026-08-12.md` §3: the sequential-displacement runs
report hidden state and readout, and **no behaviour at all**. The behavioural probe
(that document's Figure 1.4) exists only for the three-location design in §1.

## The wandb runs cannot answer it

`blake-richards/curious-george-otc` holds 16 SEQ4 runs, 8 finished (the 8 seeds) and 8
failed with empty summaries. The finished runs log 27 keys - `Train/curious_rewards`,
`Train/entropy`, `Train/value`, the phase markers `Train/displacement_phase` /
`obj_x` / `obj_y`, and so on. **Not one of them is a position or an occupancy.** No
`loc_entropy`, no `subroom_ids`, no agent position. The behavioural question is not
recoverable from the logs.

## What was done instead

The per-phase checkpoints survive (`outputs/seq4/OTC_seq_*/SEQ4-*/phase{0..3}_992`,
pRNN + AC model). So the trained policy is rolled out at each phase-end checkpoint, in
that phase's own environment, and every candidate location is scored with the same
within-run percentile `scripts/trace/trace_behavior.py` uses for §1: the behavioural
measure at all 172 walkable cells, reporting the target's percentile.

8 seeds x 4 phases x 24 rollouts x 256 steps.

## Result: occupancy percentile

Rows are where the object WAS; columns are the location scored; `*` marks the
object's own location for that phase.

```
  phase                      (7, 11)      (7, 2)      (4, 7)
  0: (7, 11)                  63.2 *      18.2        61.3
  1: (7, 2)                   36.4        86.4 *      69.3
  2: (4, 7)                   63.9        40.4        87.6 *
  3: REMOVED                  70.7        20.2        44.5
```

Figures: `outputs/summary/fig_seq_behavior.png` (the control) and
`outputs/summary/fig_seq_occupancy.png` (the same result as occupancy maps).

The diagonal is the highest entry in its row in every phase where an object exists.
Read alone, that says behaviour follows the object.

## But the location control says only ONE location survives

The diagonal being highest is not enough - it is exactly the reasoning that produced
the (14,7) false positive. The statistic that matters is the EXCESS: the value at L
when the object is at L, minus the mean at L when it is anywhere else, so that pure
location bias cancels. Paired across the 8 seeds:

```
  location  own phase  other phases    EXCESS  paired t        p
   (7, 11)       63.2          57.0      +6.2      0.76   0.4715
    (7, 2)       86.4          26.3     +60.1     11.76   0.0000
    (4, 7)       87.6          58.3     +29.2      5.38   0.0010
```

- **(7,2): decisive.** +60.1 percentile points, t=11.76.
- **(4,7): significant.** +29.2, t=5.38, p=0.0010.
- **(7,11): nothing.** +6.2, p=0.47. It reads ~63rd percentile whatever the object does
  - a structurally high-traffic cell, the same class of trap as (14,7).

So: **two of three locations survive the control, one does not** - not the blanket
"behaviour follows the object" the raw diagonal suggests.

## The agent does NOT linger where the object was

The specific question. C = (4,7), during its own phase versus after removal:

```
occupancy percentile   87.6  ->  44.5      t=5.25  p=0.0012
raw occupancy          0.213 ->  0.043     5.0x drop, in 8 of 8 seeds
```

It abandons C immediately. 44.5 is a median cell - the location becomes unremarkable
the moment the object leaves. There is no behavioural trace.

## Why this matters

Three independent measurements now show the same signature:

| measurement | while the object is present | after it leaves |
|---|---|---|
| readout (§0, §3) | elevated | collapses (drop 0.024-0.030, p<0.005) |
| behaviour (here) | elevated | collapses (5.0x, 8/8 seeds) |
| hidden state (§3) | never elevated at all | nothing to collapse |

Behaviour tracks the *present* object and abandons the departed one, exactly as the
readout does - which is what the §1 interpretation predicts, since the curiosity
reward is prediction MSE and therefore a function of the readout. The policy's input
is `h`, and `h` never changes. This is the behavioural confirmation of "transfer, not
trace".

## Reproducibility: fixed, and it strengthened the result

The first pass ran with unseeded rollouts and the numbers moved between invocations
((4,7) excess +32.5 then +17.5). The fix was **three** generators, not the one line I
first guessed:

- `torch.manual_seed` - action sampling and the pRNN's per-call noise injection;
- `np.random.seed` - anything on the numpy global;
- `env.env.reset(seed=)` - the gymnasium generator that owns `place_agent`, which
  neither of the others reaches. Same trap documented in
  `curious_george/evaluation/spatial.py::evaluate_spatial_representation`.

**Every number in this document is from the seeded version.** Removing the noise made
the result stronger, not weaker:

```
                      unseeded            seeded
(4,7) excess       +17.5 (p=0.23)   +29.2 (p=0.0010)
(7,2) excess       +71.6 (p<1e-4)   +60.1 (p<1e-4)
C departure         7/8 (p=0.082)     8/8 (p=0.0012)
```

⚠️ Back-to-back calls that REUSE one agent object still diverge from timestep 2, so
full bitwise determinism is not established. Start positions ARE now reproducible
(they were not). The analysis constructs a fresh net and agent per checkpoint, which is
the case that matters here, but the residual should be chased before anyone relies on
byte-level reproducibility.

## Other caveats
1. n=8 seeds, one environment, one object type.
2. The measure is occupancy/in-view of a radius-2 disc, not "approach". An agent that
   passes through often scores the same as one that dwells.

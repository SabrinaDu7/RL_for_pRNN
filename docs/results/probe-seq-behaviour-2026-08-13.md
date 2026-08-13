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
  0: (7, 11)                  71.8 *      15.5        56.5
  1: (7, 2)                   40.1        86.5 *      65.4
  2: (4, 7)                   69.6        12.4        89.5 *
  3: REMOVED                  75.4        15.0        49.1
```

The diagonal is the highest entry in its row in every phase where an object exists.
Read alone, that says behaviour follows the object.

## But the location control says only ONE location survives

The diagonal being highest is not enough - it is exactly the reasoning that produced
the (14,7) false positive. The statistic that matters is the EXCESS: the value at L
when the object is at L, minus the mean at L when it is anywhere else, so that pure
location bias cancels. Paired across the 8 seeds:

```
  location  own phase  other phases    EXCESS  paired t        p
   (7, 11)       65.5          64.7      +0.8      0.14   0.8962
    (7, 2)       85.5          13.8     +71.6     11.54   0.0000
    (4, 7)       77.9          60.4     +17.5      1.33   0.2264
```

- **(7,2): decisive.** +71.6 percentile points, t=11.54.
- **(4,7): suggestive, not significant.** +17.5, p=0.23.
- **(7,11): nothing.** +0.8. It reads ~65th percentile whatever the object does - a
  structurally high-traffic cell, the same class of trap as (14,7).

So the honest claim is **behaviour follows the object at one of three locations
decisively, one weakly, and one not at all** - not the clean "behaviour follows the
object" the raw diagonal suggests.

## The agent does NOT linger where the object was

The specific question. C = (4,7), during its own phase versus after removal:

```
occupancy percentile   77.9  ->  48.2      t=2.03  p=0.082
raw occupancy          0.191 ->  0.046     4.2x drop, in 7 of 8 seeds
```

It abandons C immediately. 48.2 is a median cell - the location becomes unremarkable
the moment the object leaves. There is no behavioural trace.

## Why this matters

Three independent measurements now show the same signature:

| measurement | while the object is present | after it leaves |
|---|---|---|
| readout (§0, §3) | elevated | collapses (drop 0.024-0.030, p<0.005) |
| behaviour (here) | elevated | collapses (4.2x, 7/8 seeds) |
| hidden state (§3) | never elevated at all | nothing to collapse |

Behaviour tracks the *present* object and abandons the departed one, exactly as the
readout does - which is what the §1 interpretation predicts, since the curiosity
reward is prediction MSE and therefore a function of the readout. The policy's input
is `h`, and `h` never changes. This is the behavioural confirmation of "transfer, not
trace".

## Caveats

1. **The rollouts are not seeded deterministically.** `collect_policy_rollouts` seeds
   its start-direction rng but not torch, so action sampling and pRNN noise vary
   between invocations. Two runs of this analysis gave (7,2) excess +72.2 and +71.6
   (stable) but (4,7) +32.5 and +17.5 (not). Per-seed paired tests are within a single
   invocation and unaffected; the reported means carry that noise. Seeding torch in
   that helper would fix it and is a one-line change worth making before this is
   quoted.
2. n=8 seeds, one environment, one object type.
3. The measure is occupancy/in-view of a radius-2 disc, not "approach". An agent that
   passes through often scores the same as one that dwells.

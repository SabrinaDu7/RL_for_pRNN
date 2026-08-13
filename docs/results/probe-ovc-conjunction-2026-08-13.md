2026-08-13

# Idle-time probe: an object-vector lead, and the control that killed it

**Status: RETRACTED.** A promising signal (14.4% against 5% chance) passed a
random-position location control and was then killed by the cross-over test in §5. The
effect is room GEOMETRY, not object-vector coding. Kept because the way it failed is the
transferable part. Nothing in the multi-room build was steered by it.

## 1. What prompted it

`scripts/trace/ovc_eval.py` Stage 0 does not pass: the committed criterion (mean pairwise
correlation of the three anchor-centred offset maps) is specific but recovers only ~57% of
injected vector fields at 4× mean rate, so a null from it would be a weak bound.

Diagnosing that earlier today showed the cause was not what
`../claude_logs/compaction-2026-08-13.md` §4 recorded. That note said a unit's own null
sits high; measured, the real-anchor score is **median 0.023** while the per-unit null p95
is **median 0.567**. The null is not shifted, it is **wide** — a correlation over ~47
spatially autocorrelated offsets has few effective degrees of freedom. Detrending, the
next step that note proposed, does not address that and was dropped.

## 2. The statistic

Replacing the generic correlation with one shaped like the alternative. Per unit, z-score
each anchor's offset map across offsets, then

    score = max over offsets of ( min over anchors of z_A(offset) )

A vector cell is elevated at the *same* offset for *every* anchor, so the min survives; a
place cell is elevated for one anchor, so the min kills it. Measured against the committed
correlation, same maps and same null draws:

| condition | correlation | conjunction |
|---|---|---|
| uninjected (chance 0.05) | 0.014 | 0.086 |
| injected vector field, amplitude 1.0 | 0.072 | 0.268 |
| injected vector field, amplitude 2.0 | 0.226 | 0.564 |
| injected vector field, amplitude 4.0 | 0.612 | 0.888 |
| injected PLACE field, amplitude 4.0 (want ≤ 0.05) | 0.006 | 0.066 |

Roughly double the sensitivity at every amplitude, specificity retained.

## 3. The lead, and the control that matters

Scored on the 80k-step L-room checkpoint's rate map, 493 of 500 units passing the rate
screen, against each unit's own null of anchor triples.

**Matching the null on anchor separation strengthens it.** The real landmarks sit 6/7/6
apart; an unmatched null mixes in triples close enough that their radius-4 crops overlap
and correlate for purely geometric reasons, which raises the threshold.

```
null                             draws   frac > own p95   chance
unmatched (any triple)             150            0.089     0.05
matched: min separation >= 6       150            0.144     0.05
```

**The location control passes.** A within-unit null controls for a unit's own map
structure but cannot see structure shared *across* units at particular locations — which
is exactly what produced the (14,7) false positive. So the population fraction was
recomputed at control anchor triples: matched on separation, placed where this net has no
landmarks.

```
real landmarks    (4,3) (10,4) (4,10)   ->  0.154
small-obj anchors (5,5) (11,5) (5,11)   ->  0.089   (no landmarks there in this net)
40 matched control triples              ->  median 0.051   p90 0.089   max 0.138
real landmarks percentile among controls:  100
```

The controls' median of 0.051 against a nominal 0.05 says the null is calibrated. The real
landmarks exceed all 40 controls.

At this point it looked like the first object-coding signal in this project to survive its
own location control. **That reading was wrong**, and §5 is why.

## 4. Caveats recorded before the decisive test

1. **The margin is thin.** 0.154 against a control max of 0.138. With 40 controls the
   percentile bound is p < 1/41 ≈ 0.024, which is real but not overwhelming.
2. **The conjunction statistic has not been through Stage 0.** It has a specificity check
   (place field 0.066) and a sensitivity curve, but the committed gate in `ovc_eval.py`
   was written around the correlation criterion and has not been re-run on this one.
3. **n = 1 network, 1 map estimate, 1 checkpoint.** No seeds, no split-half.
4. **The map's provenance is unverified.** `outputs/trace/maps_dense.npz` has no
   generating script anywhere in the repo — nothing under `scripts/` writes it. It is
   assumed to be the 80k L-room checkpoint replayed through the standard probe, and the
   assumption is supported by its agreement with the committed Stage 0 number (0.575 vs
   0.55–0.61 re-derived here) but not established.
5. **A landmark-driven effect must follow the landmarks.** Run below.

## 5. The cross-over test, and the retraction

Two nets with landmarks in DIFFERENT places, each scored at both anchor sets. If the
signal is landmark-driven each net scores high only at its own anchors.

```
net         scored at              frac > own p95    own?
Lroom-6x6   Lroom-6x6 anchors               0.142     OWN
Lroom-6x6   small-2x2 anchors               0.073
small-2x2   Lroom-6x6 anchors               0.203
small-2x2   small-2x2 anchors               0.164     OWN

mean at OWN landmarks    0.153
mean at OTHER landmarks  0.138          chance 0.05
```

**The small-object net scores HIGHER at the L-room's landmark positions (0.203) than at
its own (0.164)** — and it never had a landmark at those positions. Own and other are
effectively equal (0.153 vs 0.138). The signal tracks where those cells sit in the room,
not what is painted there.

The L-room net on its own looks convincing (0.142 own vs 0.073 other). One net cannot
distinguish the two accounts; it takes two nets whose landmarks disagree.

## 6. The transferable lesson

**A location control drawn from RANDOM positions is weaker than one drawn from
geometrically comparable positions.** The 40 matched control triples in §3 were random
walkable triples: separation-matched, but not matched on the thing that actually mattered
- lying in the three lobes of the L, at similar wall distances. The small-object
landmarks ARE matched that way, and they score just as high.

This is the third time in this project that a within-unit null has been fooled by
structure shared across units at particular room locations - after (14,7) and the
occlusion gradient. The generalisable rule: **score the criterion under a second network
whose landmarks are somewhere else.** A control that only moves the scored location within
one network cannot separate "coding this object" from "this place is special".

## 7. What survives

- The diagnosis in §1: Stage 0's positive control fails because the null is WIDE, not
  high. Detrending, the step `compaction-2026-08-13` §4 proposed next, does not address
  that and should not be attempted.
- The conjunction statistic in §2 is genuinely ~2x more sensitive than the committed
  correlation criterion at matched specificity. It is a better detector; it just has
  nothing to detect here.
- The measurements were made with throwaway scripts and are not committed. Re-deriving
  them means re-running the cross-over, which is the only one worth keeping.

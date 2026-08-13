2026-08-13

# Idle-time probe: an object-vector lead that survives its location control

**Status: INDICATIVE, not a result.** Run while waiting on the multi-room jobs, on an
existing checkpoint. Every caveat in §4 is load-bearing. Nothing in the multi-room
build was steered by this.

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

**This is the first object-coding signal in this project to survive its own location
control** — the control that killed the occlusion result and the trace-cell count.

## 4. Why this is still not a result

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
5. **A landmark-driven effect should follow the landmarks.** The decisive test is the
   cross-over: score the small-object checkpoint at ITS anchors and at the L-room's. If
   each net scores high only at its own landmarks, the effect is landmark-driven; if both
   score high at the same places, it is geometry. **Not yet run** — it needs a GPU replay
   of the small-object checkpoint.

## 5. If it holds up

It would mean the earlier nulls were a probe problem rather than an absence: the trace
hunt was aimed at the wrong cell class, and the object-vector class the prediction
objective actually demands was never measured with a sensitive enough statistic. That is
the reading `../exp_instructions/instructions-OVC.md` set out in advance as
"E1 passes" — and it still needs E2, the landmark swap, to separate genuine vector coding
from place cells at coincidental offsets.

## 6. Reproducing

The measurements here were made with throwaway scripts and are **not** committed, so this
document is a lead to re-derive rather than a checkable result. Promoting the conjunction
statistic into `scripts/trace/ovc_metric.py` and running it through Stage 0 is the next
step; until then treat every number above as indicative.

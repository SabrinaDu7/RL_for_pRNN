# Object / trace-cell / object-vector experiments — index

Entry point for the whole novel-object line. **Read §"The one-paragraph version"
first** — the `W_out` mechanism predicts every null that follows.

Each document carries its own methods, results, bounds, and an explicit list of
claims retracted along the way. Where a claim is marked ⚠️ or RETRACTED, that is
load-bearing: this project has produced four false positives, and every one was
killed by a control rather than by inspection.

## Read in this order

| # | document | what it is |
|---|---|---|
| 1 | this file, §"The one-paragraph version" | the mechanism, in a paragraph |
| 2 | [`result-summary-2026-08-12.md`](result-summary-2026-08-12.md) **§0 then §6** | **the grounding document.** §0 is the mechanism; §6 is the verdict table with per-result reliability. The only doc with `[cache]`/`[doc]`/`[live]` evidence markers throughout |
| 3 | same doc, §1–§5 | the experiment series. **§4 is the most transferable** — the (14,7) false positive and the location control that caught it |
| 4 | [`../claude_logs/compaction-2026-08-13-multienv.md`](../claude_logs/compaction-2026-08-13-multienv.md) §1–3, §7 | why the five multi-room runs exist; certain vs uncertain |
| 5 | [`probe-remapping-2026-08-13.md`](probe-remapping-2026-08-13.md) | do multi-room maps become room-specific? |
| 6 | [`probe-e1-multianchor-2026-08-13.md`](probe-e1-multianchor-2026-08-13.md) | object / object-vector cells across five checkpoints |
| 7 | [`probe-e1-final-2026-08-18.md`](probe-e1-final-2026-08-18.md) **§3** | the same test on the finished runs. §3 is the transferable part — a location control that reads as a hit until it is run over every anchor triad |

## Every result document

| document | question | headline |
|---|---|---|
| [`result-summary-2026-08-12.md`](result-summary-2026-08-12.md) | the whole object/trace investigation, 07-30 → 08-12 | **the grounding doc.** Ten designs, every hidden-state result null; the mechanism explains all of them |
| [`exp_object_trace_cells_2026-07-30.md`](exp_object_trace_cells_2026-07-30.md) | does `h` hold a trace at a removed object's location? | **No** — object memory is decoder-localised (`W_out`); place code unchanged, r ≈ 0.98 |
| [`exp_object_into_hidden_state_2026-08-01.md`](exp_object_into_hidden_state_2026-08-01.md) | can RL-side changes move the representation into `h`? | **Partly** — `lr_trials=[2,0,8]` raises encoding 0.695 → 0.721; memory unchanged |
| [`exp_trace_cell_scenarios_2026-08-11.md`](exp_trace_cell_scenarios_2026-08-11.md) | do object identity (A), weight decay (D) or occlusion (C) force it into `h`? | **No.** A null n=3; D untestable (L2 collapses the place code r 0.97→0.73); **C was a false positive** — the peak does not follow the object |
| [`probe-seq-behaviour-2026-08-13.md`](probe-seq-behaviour-2026-08-13.md) | does behaviour follow the object through A→B→C→removed? | **Yes, and it abandons the departed one** — (4,7) occupancy 0.213 → 0.043 on removal, 5.0×, 8/8 seeds. Two of three locations survive their location control |
| [`probe-remapping-2026-08-13.md`](probe-remapping-2026-08-13.md) | does multi-room training build room-SPECIFIC maps? | **Informative null.** Per-room sRSA → 0.79 while the remapping index stays at 0.01–0.03 against a derived +0.3904 for true remapping. A better *shared* map. ⚠️ `ROOMS_RUN1` is a translation set, so the L-room arm is a weak manipulation |
| [`probe-e1-multianchor-2026-08-13.md`](probe-e1-multianchor-2026-08-13.md) | object / object-vector cells, five checkpoints | **Null everywhere**, location control flat, place code healthy. ⚠️ **A weak bound** — the detector fails its own Stage 0 gate (0.575 recall at 4× injected amplitude) |
| [`probe-e1-final-2026-08-18.md`](probe-e1-final-2026-08-18.md) | the same test on the four FINISHED multi-room runs, all at 482,344,960 steps | **Still null.** The fraction rises in every arm and two reach chance, but the location control run over every (rollout room × anchor triad) pair shows the score tracks the TRIAD, not whether its landmarks are present. ⚠️ Same weak bound; ⚠️ the top-SI units are now wall-aligned bands, so E5 is the binding gap |
| [`probe-ovc-conjunction-2026-08-13.md`](probe-ovc-conjunction-2026-08-13.md) | a promising OVC lead | 🔴 **RETRACTED.** 14.4% vs 5% chance, passed a random-position control, killed by the cross-over. Kept because the way it failed is the transferable part |
| [`exp_reward_alignment_next_obs.md`](exp_reward_alignment_next_obs.md) | curiosity-reward alignment | see doc |
| [`../legacy/omt_post_refactor_runs.md`](../legacy/omt_post_refactor_runs.md) | (superseded) OMT runnability after the migration | kept for the refactor record only |

**Plans, not results:** [`../exp_instructions/instructions-objectAndOVC.md`](../exp_instructions/instructions-objectAndOVC.md)
(the Høydal/Tsao eval plan) and [`../exp_instructions/instructions-multi-env.md`](../exp_instructions/instructions-multi-env.md).

**Reference images:** `../ref-trace-cells.png` (Tsao 2013 — a cell's field
persisting after object removal), `../ref-object-cells.png`,
`../ref-objectVector-cells.png` (Høydal 2019 Fig. 2 — the vector-field generalisation
this work is trying to reproduce).

## The one-paragraph version

Exposure writes the object into the **linear readout** `W_out`, not into the
recurrent dynamics, because the object's location is a deterministic function of
position and `h` is already a place code — so adjusting a linear map is the
cheaper gradient direction. `h` is *redundant* with the object, not ignorant of
it. Consequently there is no object structure in the rate maps to decay after
removal, in any of three reference frames. Freezing `W_out` and scaling `W_in`
does push measurably more object information into `h`, but only as a transient
input-driven code: it decays ~2× per masked step and is near chance within three.

## Methods, and where each lives

| method | module |
|---|---|
| Fixed measurement probe (collect once, replay per checkpoint) | `scripts/trace/trace_probe.py` |
| Occupancy-masked rate maps, Skaggs + uniform SI, shuffle null, split-half, object modulation | `scripts/trace/trace_maps.py` |
| Object/trace-cell metric: field gain, **within-unit percentile**, location-control matrix | `scripts/trace/trace_metric.py` |
| Rate-map panels (`unit_panel`, `trace_panel`, `occupancy_figure`) | `scripts/trace/trace_figure.py` |
| Readout-vs-dynamics chimaera swap, and its gain-corrected version | `scripts/trace/trace_readout_test.py`, `trace_readout_gaincorrected.py` |
| Object-vector tuning in the **egocentric view frame** | `scripts/trace/trace_objvector_test.py` |
| On-policy behaviour with a within-run null over all 172 cells | `scripts/trace/trace_behavior.py` |
| Curiosity-reward maps, by position and conditioned on visibility | `scripts/trace/trace_reward_map.py`, `trace_reward_inview.py` |
| Presence decoding from `h`, **split by input-mask phase** | `scripts/trace/trace_presence_decoder.py` |
| Object-vector criterion: offset maps, `vector_score`, within-unit `vector_percentile`, `radial_score`, `spatial_screen` | `scripts/trace/ovc_metric.py` |
| OVC Stage 0 controls + E1 driver (refuses to report E1 unless Stage 0 passes) | `scripts/trace/ovc_eval.py` |
| E1 with its location control built in (OWN vs OTHER anchor positions) | `scripts/trace/e1_multi_anchor.py` |
| E1 as spatial tuning: candidates, offset maps, top-SI gallery | `scripts/trace/e1_cell_figure.py` |
| Same units, displaced triad — the definitional test | `scripts/trace/e1_across_rooms.py` |
| Detector sensitivity curve with the measured result on it | `scripts/trace/ovc_power_figure.py` |
| Which run produced a cached map set (replay-and-correlate, pipeline asserted first) | `scripts/trace/verify_map_provenance.py` |
| Moser session sequence; object vs trace cells as **independent** populations | `scripts/moser/moser_sessions.py`, `moser_analysis.py` |
| Quadrant decoding from `h`, object present vs absent (**the diagnostic that explained the square-room null**) | `scripts/moser/moser_decode_quadrant.py` |
| Multi-room: per-room + pooled spatial metrics across a checkpoint series | `scripts/multienv/checkpoint_curve.py`, `remapping_figure.py` |
| Remapping measurements validated against synthetic `H_position` / `H_room` | `scripts/multienv/remapping.py` (`--validate`) |
| Reference sRSA/SWdist/SI with their noise floor; SI threshold audit | `scripts/spatial_baseline.py`, `scripts/si_threshold_audit.py` |

Regeneration commands live with each result document; the summary doc's are in
its "Regenerating every figure in this document" section.

## Four things to carry into any follow-up

**0. There are TWO demonstrated ways the net localises without the object**, and
every design so far closed at most one. The L-room gives it geometry; the square
room removes that and it localises by trajectory history instead (quadrant
decodes ~84% from `h` with the object absent). Multi-room training attacks the
second without new machinery — but see the `ROOMS_RUN1` translation caveat in
`probe-remapping-2026-08-13.md` §4.

**1. Pooling is this project's recurring failure mode.** Three separate times —
binning the curiosity-reward map by position, pooling input-driven with
memory-only timesteps, and pooling the five input-mask phases — averaging across
the variable the mechanism runs on turned a real effect into a flat null. The
third produced a "structurally impossible" conclusion reported twice, wrongly.
Split first.

**2. The mask structure is load-bearing.** `thRNN_5win` has
`inMask = [True, False×5]`: the observation reaches the network only 1 timestep
in 6 while it must predict at every step. Any analysis of "what does `h` know"
must separate the input-driven phase from the five memory-only phases.

**3. A within-unit null cannot see structure shared ACROSS units at one
location.** That is what produced (14,7), the occlusion gradient and the
retracted OVC lead. **Score every criterion at control locations where this
network has no landmark**, matched on geometry rather than drawn at random — a
random-position control passed the OVC lead that a geometry-matched one killed.

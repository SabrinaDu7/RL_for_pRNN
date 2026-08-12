# Object / trace-cell experiments — index

Entry point for the novel-object work. Two documents, both rewritten 2026-08-03 from running
logs into structured form; each carries its own methods, results, bounds and an explicit list
of claims that were retracted along the way.

| document | question | headline |
|---|---|---|
| [`exp_object_trace_cells_2026-07-30.md`](exp_object_trace_cells_2026-07-30.md) | Does the pRNN hidden state hold a trace at a removed object's location? | **No** — object memory is decoder-localised (`W_out`), the place code is unchanged (r ≈ 0.98) |
| [`exp_object_into_hidden_state_2026-08-01.md`](exp_object_into_hidden_state_2026-08-01.md) | Can RL-side changes move the representation into `h`? | **Partly** — `lr_trials=[2,0,8]` raises encoding 0.695 → 0.721; memory unchanged |
| [`exp_trace_cell_scenarios_2026-08-11.md`](exp_trace_cell_scenarios_2026-08-11.md) | Do object identity (A), weight decay (D) or occlusion (C) force the object into `h`? | **No.** A null n=3; D untestable (L2 collapses the place code r 0.97→0.73); **C was a false positive** — the peak does not follow the object |
| [`omt_post_refactor_runs.md`](omt_post_refactor_runs.md) | (superseded) runnability of OMT after the migration | kept for the refactor record only |
| `sab_context/goal_2026-08-12.md` **(untracked — see note)** | Does a four-fold-symmetric room, run through the Moser session sequence, produce object or trace cells? | **No.** Object cells never above chance (best 6.4%, p=0.09); trace cells null against the correct null. The room did not bind — `h` decodes quadrant at ~84% *with and without* the object |

> ⚠️ `docs/*` is gitignored, so any NEW document here is untracked unless force-added; the
> existing ones survive only because they were tracked before that rule. The 2026-08-12 Moser
> results therefore live in an untracked file, and their only committed record is the commit
> messages on `scripts/moser_*.py`. Worth deciding whether `docs/` should be tracked by default.

**Reference image:** `trace-cells-spatial-tuning.png` — the biological result being targeted
(a single cell's rate map across days, field persisting after object removal).

## The one-paragraph version

Exposure writes the object into the **linear readout** `W_out`, not into the recurrent
dynamics, because the object's location is a deterministic function of position and `h` is
already a place code — so adjusting a linear map is the cheaper gradient direction. `h` is
*redundant* with the object, not ignorant of it. Consequently there is no object structure in
the rate maps to decay after removal, in any of three reference frames. Freezing `W_out` and
scaling `W_in` does push measurably more object information into `h`, but only as a transient
input-driven code: it decays ~2× per masked step and is near chance within three steps.

## Methods, and where each lives

| method | module | used by |
|---|---|---|
| Fixed measurement probe (688 trajectories, collect once, replay per checkpoint) | `scripts/trace_probe.py` | both |
| Occupancy-masked rate maps, Skaggs + uniform SI, shuffle null, split-half, object modulation | `scripts/trace_maps.py` | trace cells |
| Rate-map panels (`unit_panel`, `trace_panel`, `occupancy_figure`) | `scripts/trace_figure.py` | both |
| Readout-vs-dynamics chimaera swap | `scripts/trace_readout_test.py` | trace cells |
| ...gain-corrected version (`‖h‖` differs ~9% between nets) | `scripts/trace_readout_gaincorrected.py` | trace cells |
| Object-vector (egocentric view-frame) tuning | `scripts/trace_objvector_test.py` | trace cells |
| On-policy behaviour with a within-run null over all 172 cells | `scripts/trace_behavior.py` | trace cells |
| Curiosity-reward maps — by agent position, and conditioned on visibility | `scripts/trace_reward_map.py`, `trace_reward_inview.py` | trace cells |
| Presence decoding from `h`, **split by input-mask phase** | `scripts/trace_presence_decoder.py` | hidden state |
| Stochastic object presence / random object position during training | `tasks/otc/` | hidden state |
| Moser session sequence (no object → 6 positions → no object), trained from scratch | `scripts/moser_sessions.py` | symmetric room |
| Object cells vs trace cells as **independent** populations, per Tsao/Moser | `scripts/moser_analysis.py` | symmetric room |
| Session panels, counts, field gain, map drift | `scripts/moser_figures.py` | symmetric room |
| Quadrant decoding from `h`, object present vs absent (**the diagnostic that explained the null**) | `scripts/moser_decode_quadrant.py` | symmetric room |
| Reference sRSA/SWdist/SI with their noise floor | `scripts/spatial_baseline.py` | metrics |
| Is the SI activity threshold a bias correction or does it delete real fields? | `scripts/si_threshold_audit.py` | metrics |

## Regenerating every figure

Nothing requires retraining; all figures rebuild from caches in `outputs/trace/`.

```bash
uv run python scripts/otc_figures.py plot     # fig_otc_phases / _encoding / _tradeoff
uv run python scripts/otc_figures.py maps     # fig_otc_maps / _maps_diff
uv run python scripts/trace_cell_figures.py   # the six trace-cell figures
```

To recompute the caches themselves (needs the checkpoints under `outputs/`):
`uv run python scripts/otc_figures.py collect`.

## Two things to carry into any follow-up

**0. There are now TWO demonstrated ways the net localises without the object**, and every
design so far closed at most one. The L-room gives it geometry (an L-shaped wall plus a
triangle, plus, x). The square room removes that, and it localises by trajectory history
instead — quadrant decodes at ~84% from `h` with the object absent, and the object adds
nothing. So "make the object the only cue" needs BOTH routes shut: a room without landmarks
AND something that breaks dead reckoning (teleport mid-episode, or a randomised initial hidden
state). Multi-environment training does this without new machinery, because the same
integrated trajectory maps to a different absolute position in a different room.

**1. Pooling was this project's recurring failure mode.** Three separate times — binning the
curiosity-reward map by agent position, pooling input-driven with memory-only timesteps, and
pooling the five input-mask phases — averaging across the variable the mechanism runs on turned
a real effect into a flat null. The third caused a "structurally impossible" conclusion that
was reported twice and was wrong. Split first.

**2. The architecture's mask structure is load-bearing.** `thRNN_5win` has
`inMask = [True, False×5]`: the observation reaches the network only 1 timestep in 6, while it
must predict at every step. Any analysis of "what does `h` know" must separate the input-driven
phase from the five memory-only phases, or it measures a mixture of two different things.

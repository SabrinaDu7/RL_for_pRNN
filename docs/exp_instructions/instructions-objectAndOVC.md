# Detecting Object Cells and Object-Vector Cells in the pRNN Hidden State

An eval plan following Høydal et al. 2019 (*Nature* 568:400) as closely as the
substrate allows, with Tsao et al. 2013 (*Curr. Biol.* 23:399) for the object /
trace distinction.

Substrate facts this plan assumes (confirmed with you):

- `h` is 500-d, activations are **ReLU** (`prnn/utils/thetaRNN.py:178`), so
  activations are non-negative and Skaggs applies directly with no rectification
  hack. Each unit is treated as one "cell"; 500 cells is the same order as
  Høydal's 1,100.
- pRNN inputs: head direction, observation on **1 in k timesteps** (blind
  otherwise), next action ∈ {forward, left, right, stay}.
- Tiles can be placed arbitrarily **at episode start**, never moved mid-episode.
  This matches Høydal exactly: their object also never moved within a trial.
- Rooms: L-room and square, MiniGrid, 4 discrete headings, 7×7 egocentric view.
- The trajectory/hidden-state dumper **needs building** (§2).

---

## 1. Operational definitions

The whole logic is a sieve: define positively with one criterion, then knock out
each alternative cell type with a dedicated manipulation. Nothing here is
"looks like a field near an object" — that is exactly what a place cell in a
fixed room looks like.

### 1.1 Notation

For a unit *u*, an episode with anchor (tile) *a* at world position `p_a`:

- **Room map** `R_u(x, y)` — mean activation of *u* over timesteps at position
  `(x, y)`, occupancy-normalised.
- **Anchor-centred offset map** `V_u^a(Δx, Δy)` — mean activation of *u* at
  offset `(Δx, Δy) = pos − p_a`, in **room (allocentric) coordinates**, not
  egocentric. This is Høydal's vector map; Cartesian rather than polar because
  a discrete grid makes distance×bearing bins ragged (see §3.3).
- Report the field's **centre of mass** in polar terms (distance `d`, allocentric
  bearing `θ`, 0° = east in the room frame) only for plotting, matching Fig. 1e.

### 1.2 Object cell (Tsao / LEC-style)

A unit whose field sits **on** the anchor.

- Field centre of mass at `d ≤ 1` tile from the anchor.
- Elevated activation at the anchor relative to matched control locations
  (Tsao z, §3.4), and this holds **across anchor positions**.
- Høydal's cut was 0–4 cm from object centre in an 80 cm box; scaled to a 16×16
  grid, `d ≤ 1` tile. State the threshold explicitly, and report the full
  distance distribution rather than only the binary count.

### 1.3 Object-vector cell (Høydal / MEC-style)

A unit whose field sits at a **fixed allocentric offset** from any anchor.

Must satisfy **all** of:

| # | Criterion | Høydal analogue |
|---|---|---|
| OVC-1 | Field displaced from the anchor: `d > 1` tile | object–field distance > 4 cm |
| OVC-2 | Field appears with the anchor: activation change vs. no-anchor episodes | "expressed new firing fields when the object was introduced" |
| OVC-3 | **Offset-map stability across anchor positions**: `corr(V_u^{a@A}, V_u^{a@B})` exceeds the 99th percentile of the circular-shift null | trial-2 vs trial-3 vector-map correlation > shuffled 99th pct (r = 0.42) |
| OVC-4 | Spatial information in the offset frame above the 95th percentile of the null | SI > 95th pct shuffled |
| | ⚠️ **MEASURED VACUOUS (2026-08-13): 500/500 units pass** on the 3-room L-room net. A trajectory-shuffle null destroys all spatial structure, so every spatially-tuned unit clears it and the screen changes no denominator. Either drop it, or replace the null with one that preserves spatial structure (§3.6). | |
| OVC-5 | Peak activation above a floor (see §3.5) | peak rate ≥ 2 Hz |

Plus **at least one** generalisation result, which is what separates a
reference frame from a memorised location:

| # | Criterion | Høydal analogue |
|---|---|---|
| OVC-G1 | **Multi-anchor consistency**: same offset around ≥2 simultaneous distinct tiles in the same episode | Fig. 2a,b — 51/52 object–cell pairs, median 5.8° / 3.4 cm pairwise difference |
| OVC-G2 | **Novel-room transfer**: offset preserved in a held-out room, first episode | Fig. 2c–e — vector-map r = 0.87 across rooms |

### 1.4 Exclusions — what must be ruled out

| Alternative | How it is excluded | Høydal analogue |
|---|---|---|
| **Place cell** | Fails OVC-3. A room-anchored field decorrelates in the offset frame when the anchor moves. | the entire trial-2/trial-3 criterion |
| **Egocentric bearing cell** | Field must survive on the **out-of-view / blind** timestep subset (§4.3) and show low egocentric directional modulation | Ext. Data Fig. 6 — only 10/162 exceeded the egocentric null |
| **Border / boundary-vector cell** | Field must not be wall-anchored: compare offset-map stability to wall-offset-map stability; check field is not wall-parallel | Fig. 4, Ext. Data Fig. 8c,d, Fig. 10 |
| **Path-obstruction cell** | **Free by construction** — floor tiles do not block the agent. This is their suspended-object control (32/32 OVCs kept firing, r = 0.80) with no extra machinery. | Fig. 4c,d |
| **Head-direction cell** | Report HD score per unit; check offset tuning survives conditioning on heading | Ext. Data Fig. 4b |
| **Grid cell** | For multi-field units, extrapolate a lattice from the field pair and Z-score firing at predicted extra vertices | Ext. Data Fig. 4e,f — Z ≈ 0, median −0.12 |
| **Behaviour artifact** | Random-action agent as primary condition; occupancy normalisation; speed/dwell controls | Tsao Fig. 4 |
| **Room geometry — "this PLACE is special", not "this object is"** | Score the identical criterion at anchor positions where the network has **no landmark**, matched on separation and wall distance. OWN must clearly exceed OTHER. | no analogue — Høydal moves the object instead, which we cannot do in a fixed-room checkpoint |

⚠️ **The geometry row is the one that has killed every positive this project has
produced** — (14,7), the occlusion gradient, and the 2026-08-13 OVC lead, which
reached 14.4% against 5% chance and passed a random-position control before the
cross-over killed it: the small-object net scored **0.203 at the L-room's anchor
positions against 0.164 at its own**, having never had a landmark there
(`../results/probe-ovc-conjunction-2026-08-13.md`). A within-unit null controls
for a unit's own map structure but **cannot see structure shared across units at
one location**, which is exactly what this is.

The transferable rule: **a location control drawn from RANDOM positions is weaker
than one drawn from geometrically comparable positions.** The multi-room
checkpoints supply the strong version for free — the other rooms' anchor
positions are landmark-free in the current episode and comparable by
construction, so the control lives inside one network rather than needing two.
`scripts/trace/e1_multi_anchor.py` reports OWN vs OTHER with every result.

---

## 2. The dumper (build this first)

One npz/parquet per (checkpoint, room, anchor-config, agent, seed). Per timestep
*t*:

```
h            (T, 500)   float32   hidden state, post-ReLU
pos          (T, 2)     int       agent (x, y)
hd           (T,)       int       0..3
action       (T,)       int
obs_present  (T,)       bool      did the pRNN see an observation this step
steps_since_obs (T,)    int       0 .. k-1
pred_error   (T,)       float32   per-step prediction loss (behaviour/sanity control)
speed        (T,)       float32   or dist travelled per step
```

Per episode, in metadata:

```
room_id, room_layout (walkable mask, W×H bool)
anchors: list of {tile_type, colour, (x, y)}
checkpoint_id, agent_type, k, seed
```

Notes:

- Dump **both** the AC agent and `RandomActionAgent` rollouts for every
  configuration. Random is the **primary** analysis condition (Høydal's mice
  were foraging for crumbs, not goal-directed); the AC agent is a secondary
  condition to check the code survives on-policy occupancy.
- Store the walkable mask per room — needed for occupancy masking and for the
  L-room coverage intersection (§3.3).
- Reuse `get_view_coords_batch` from `tasks/omt/metrics.py` to compute a
  per-anchor `in_fov (T,)` bool. Note: in-FoV and `obs_present` are different
  things — you need both.

**Sample size.** Target ≥ 200 visits per occupied `(Δx, Δy)` bin after pooling.
Report the per-bin visit count map for every analysis and mask bins below a
minimum (suggest 20 visits); Skaggs is positively biased with sparse sampling
and will manufacture tuning otherwise.

---

## 3. Shared machinery

### 3.1 Rate maps

```
R_u(x, y) = Σ_t h_u(t) · 1[pos(t) = (x,y)]  /  Σ_t 1[pos(t) = (x,y)]
```

Occupancy-normalised by construction. Smooth with a small 2D Gaussian
(σ = 1 tile) as Høydal did (σ = 2 bins = 4 cm), and **report results at two
smoothing levels** — halving the kernel changed their cell count by one, which
is the kind of robustness check worth reproducing.

### 3.2 Field detection

Høydal's exact procedure, ported: iteratively contour down from the unit's peak
until reaching 2 SD of all bins in the map. A field is a contiguous region of
≥ N bins with peak above the activation floor. Their thresholds were 16 bins
(= 64 cm² in an 80×80 box, ~1% of area) and 2 Hz; on a 16×16 grid, scale N to
≈ 1–2% of walkable area (so N ≈ 3–4 tiles) and set the floor per §3.5.

### 3.3 Offset maps and the L-room coverage problem

`V_u^a(Δx, Δy)` is built the same way as `R_u` but binned on `pos − p_a`.

**Critical, L-room specific.** The set of reachable offsets depends on where in
the L the anchor sits. If you correlate `V_u^{a@A}` against `V_u^{a@B}` without
care, you are correlating occupancy geometry, not tuning.

- Mask to the **intersection of well-sampled bins** across the two maps.
- Report `n_shared_bins` for every pair; drop pairs below a floor (suggest 30).
- Sanity check: run the same correlation on a **position-shuffled** control to
  confirm the masked geometry alone yields r ≈ 0.

### 3.4 Tsao z-score (for object cells)

```
z = (X̄_in − X̄_out) / (s / √n)
```

`X̄_in` = mean activation in the patch containing and surrounding the anchor;
`X̄_out`, `s` = mean and SD outside. **`X̄_out` must be computed from a matching
number of randomly drawn outside bins**, not the whole remainder of the room —
otherwise the null is over a much larger sample and z inflates. Tsao used a
22.5×22.5 cm patch in a 100×100 cm box; scale to a 3×3 or 5×5 tile patch.

The generalisation worth making: compute this at **every** offset bin, not just
at the anchor. That *is* the offset map. Object cell = peak at `d ≈ 0`;
OVC = peak further out. One analysis, one number distinguishes the classes.

### 3.5 The activation floor

Høydal's 2 Hz floor excludes units too quiet to have a meaningful field. `h` has
no natural units, so define the floor **per checkpoint** from the population:
e.g. peak activation must exceed the 50th percentile of all units' peak
activations, or a fixed multiple of the unit's own mean. Whatever you pick,
**re-run the whole pipeline at two floor values** and report both counts, as
Høydal did (2 Hz → 162 cells; 1.5 Hz → 163 cells).

### 3.6 The shuffle — **do not get this wrong**

Høydal: 200 permutations per cell, spike train **circularly time-shifted along
the animal's path** by a random interval ≥ 20 s, session wrapped.

⚠️ **A time shift is the WRONG null for the anchoring criteria here, and this has
already been measured.** Shifting `h_u(t)` against `(pos, hd, anchor)` destroys
the activity–position correspondence entirely, so it tests "is this unit
spatially tuned at all" — not "is its tuning anchored to the landmark". Every
spatial unit clears it. Measured with exactly that null in this repo: the
negative control read **0.130** and injected **PLACE** fields passed as vector
cells at **0.146** — the detector was unusable
(`scripts/trace/ovc_eval.py::shuffle_thresholds`,
`../claude_logs/compaction-2026-08-13.md` §5.7).

**Use instead, for every anchoring criterion (OVC-3, OVC-G1, E1, E2):** leave the
rate map exactly as it is and re-run the criterion at **random walkable anchor
triples**. That asks the question that matters — is the agreement SPECIFIC to
the real landmarks, or does any triple do as well — and it is the same
location-control logic that exposed (14,7).

**And it must be WITHIN-UNIT.** These maps carry strong room-wide common
structure (smooth gradients, wall effects) shared across units, so *any* three
neighbourhoods correlate: measured, random anchor triples reach a **population
99th-percentile vector score of 0.933**, which no injected field can exceed. A
population threshold is therefore unclearable in principle. Scoring each unit
against triples drawn for that same unit controls for its own global structure
and puts chance at a known 5% by construction.
`ovc_metric.vector_percentile` owns this.

A time shift remains appropriate for the criteria that are genuinely about
"is there structure" rather than "is it anchored": room-frame SI, offset-frame
SI, egocentric directionality, HD score. For those:

- 200 permutations per unit; shift ≥ 50 timesteps, wrapped, and **measure `h`'s
  autocorrelation time first** (S0.5) to justify the interval.
- Thresholds: 99th percentile for correlation criteria, 95th for SI, per Høydal.
- For OVC-3, keep the anchor at its **first** position as the reference across
  all permutations — Høydal is explicit about this.

**Why this differs from the biology.** RNN hidden states are massively
autocorrelated *and* share a global spatial gradient across units. Høydal's
shuffle works on spike trains because a real place cell decorrelates under a
time shift; here the shared structure survives it.

### 3.7 Multiple comparisons

500 units × several criteria. Report FDR-corrected counts alongside raw counts,
and always report the **proportion** of units passing against the proportion
expected under the null, not just the raw number.

---

## 4. General evals — run on every checkpoint

### Stage 0 — controls that gate everything else

Your `ovc_eval.py` already has this shape (refuses to report E1 unless Stage 0
passes). Keep that discipline. Stage 0 must include:

- **S0.1 Untrained-pRNN negative control.** Same rooms, same rollouts, randomly
  initialised pRNN. Any criterion that passes here at above-null rates is
  measuring the analysis, not the network. (An untrained RNN driven by
  observations *will* have some object-driven units — this is the number your
  trained result has to beat.)
- **S0.2 Synthetic positive control.** You already have
  `inject_vector_field` / `inject_place_field` in `ovc_metric.py`. Inject into
  real rollout data at the real occupancy statistics and confirm the vector
  field passes OVC-1..5 and the place field fails OVC-3. This is a power
  analysis: it tells you the minimum effect size the pipeline can detect at
  your sample sizes.
- **S0.3 Position-shuffle geometry control.** §3.3 — confirms L-room coverage
  masking is not itself producing correlation.
- **S0.4 Sampling adequacy.** Per-bin visit counts; fraction of walkable
  offsets above the visit floor; per-unit activation sparsity (units that are
  near-silent or near-saturated everywhere should be excluded and counted).
- **S0.5 Autocorrelation time of `h`,** to justify the shuffle interval (§3.6).

### E1 — Multi-anchor consistency (Fig. 2a,b) — **run this first**

The strongest test available on existing checkpoints with zero new environment
code, because you already have three tiles per room.

1. For each unit, build `V_u^a` separately around each of the three anchors in
   the same episodes.
2. Detect the field in each; for each pair of anchors, compute the difference in
   offset distance and allocentric bearing between the nearest-matching fields
   (Høydal: "found the two fields in the pair of maps with the nearest
   centre-to-centre distance").
3. **Benchmark:** their median pairwise difference was 5.8° orientation and
   3.4 cm distance (= ~0.7 tiles on your scale), across 51/52 object–cell pairs.
4. Report `vector_score` = mean pairwise correlation of the three anchor-centred
   maps (you already have this in `ovc_metric.py`), against the circular-shift
   null.

**Interpretation.** Same offset around all three tiles despite different
colour/shape/position ⇒ reusable offset code. Three unrelated offsets ⇒ three
place fields, and you are done.

**Caveat to state in the writeup:** in fixed-room checkpoints, anchor identity
is perfectly confounded with anchor position, so E1 there tests "same offset from
three *locations*" — which is still informative but weaker than the randomised-room
version. Say so.

### E2 — Displaced-anchor stability (the definitional criterion, Fig. 1b)

The trial triplet, mapped:

| Høydal | Here |
|---|---|
| Trial 1: no object | Episode set, anchor removed |
| Trial 2: object at A | Episode set, anchor at A |
| Trial 3: object displaced to B | Episode set, anchor at B |
| r(V², V³) > 99th pct shuffled | Same |

Improvement over the paper: run **N ≥ 8 anchor placements**, giving N(N−1)/2
correlation pairs per unit instead of one. Report the distribution, not a single
r. Bias placements toward the room interior so that large offsets stay inside
the walkable area — Høydal did exactly this ("a bias towards the box centre on
the first trial, to capture fields with large offsets").

**Out-of-distribution guard.** For the 1-room and 3-room checkpoints, a moved
tile is OOD. Every displaced-anchor eval must be paired with a sanity check —
does per-step prediction error stay within the training range, does the
room-frame position decoder still work? If not, a decorrelated offset map means
"network is confused," not "no OVC code," and must be reported as inconclusive.

### E3 — The 1/k blindness ladder (better than Ext. Data Fig. 6)

Your best control, and it is free.

1. Bin every timestep by `steps_since_obs` ∈ {0, …, k−1}.
2. Recompute the offset map and OVC criteria within each bin.
3. Also split by per-anchor `in_fov`, and analyse the **out-of-view** subset
   separately.

**Prediction.** A sensory/egocentric response dies at `steps_since_obs = 1` and
on the out-of-view subset. A genuine allocentric vector code (a belief state)
degrades gracefully as path-integration uncertainty accumulates.

⚠️ **Expect the decay, not the graceful degradation — this is already measured
for OBJECT information.** `../results/result-summary-2026-08-12.md` §2 ran exactly
this split for object *presence* decoding from `h`, and got ph0 0.695 → ph1 0.583
→ ph2 0.541 → ph3 0.524 → chance by ph4–ph5: **~2× decay per masked step, in all
13 conditions**, including the `[2,0,8]` net built to force object information
into the dynamics, and unchanged when replayed with the injected noise zeroed —
"the memory was never written, not erased". So the belief-state reading is
already disfavoured for object content specifically.

E3 is still worth running, for two reasons that are genuinely open: the §2 result
is about object PRESENCE, not about offset tuning, which could decay differently;
and the HD-score-rises signature (Høydal Ext. Data Fig. 7d–g) has never been
looked for here and would be a real convergence if it appeared.

This is simultaneously Høydal's darkness experiment (Ext. Data Fig. 7d–g), where
spatial information, coherence, peak rate and mean rate all dropped, while
**HD tuning increased**. Check for the same signature: report SI, coherence,
peak activation, and HD score as functions of `steps_since_obs`. If HD score
rises as vision drops out, that is a striking convergence worth its own figure.

If you have checkpoints at multiple `k`, that is a second dose-response axis.

### E4 — Egocentric exclusion (Ext. Data Fig. 6)

1. Build egocentric tuning curves: activation vs. movement direction relative to
   the anchor. Høydal used 20° bins with 0° = heading toward the object; you
   have 4 headings, so this is a 4-bin (or, using relative bearing of the anchor
   in the 7×7 view, a coarse angular) version.
2. Egocentric directionality index = mean vector length of the tuning curve;
   threshold at the 99th percentile of the circular-shift null.
3. **Benchmark:** only 10/162 of their OVCs exceeded it.
4. **Heading invariance.** Conditioned on `(Δx, Δy)`, decompose activation
   variance into position, heading, and interaction terms. Report as a variance
   decomposition, not a binary — with only 4 headings and heading-dependent
   observations, *some* heading dependence is unavoidable and expected.

### E5 — Border / boundary-vector exclusion (Fig. 4, Ext. Data Fig. 8c,d)

Path obstruction is already excluded by construction (tiles are walkable). What
remains is wall-anchoring:

1. Build **wall-offset maps**: distance and allocentric direction to the nearest
   wall, and to each of the L-room's wall segments.
2. For each unit, compare offset-map stability across anchor placements
   (OVC-3) against wall-map stability. A unit whose field is better explained by
   a wall than by a tile is a border cell.
3. Field shape: fit the field and report the aspect ratio. Høydal's OVCs had
   median 1.6 (Ext. Data Fig. 3) — compact blobs, not the wall-parallel bands a
   BVC produces. Check the long axis is not parallel to any wall
   (their BVC criterion: long axis ≥ 70% of a wall's length **and** within 10°
   of parallel).
4. Report the field-to-anchor vs. field-to-wall distance distributions, as in
   Ext. Data Fig. 10d (21.4 cm vs 7.6 cm).

The L-room's concave corner is a useful natural stimulus here: an interior
corner is a boundary feature, so check for units anchored to it.

### E6 — Grid / HD / other-class exclusion (Ext. Data Fig. 4)

1. Compute grid score, border score, HD score, and speed score for every unit,
   on **anchor-absent** episodes, and compare OVC-passing units against the
   remaining population (Mann–Whitney, two-sided, as they did).
2. **Benchmark directionality:** their OVCs had *lower* border and grid scores
   than the general population; HD score was *higher* without an object but
   dropped significantly when the object appeared.
3. **Multi-field units:** extrapolate a hexagonal lattice from each field pair,
   Z-score activation at the predicted extra vertices. Their median was −0.12,
   i.e. not grids.

### E7 — Population / factorisation analyses

Single-unit metrics are supplementary; `h` is a distributed code and a perfectly
factorised representation need not be axis-aligned to units.

- **F1 Cross-condition decoding.** Train a linear/ridge decoder for agent
  position from `h` on anchor-at-A episodes, test on anchor-at-B. Then the
  mirror: train a decoder for anchor-relative `(Δx, Δy)` on anchor A, test on
  anchor B. Report cm/tiles of error, and the transfer gap.
- **F2 Parallelism of displacement vectors.** Compute
  `Δ(x,y) = h̄(pos, anchor present) − h̄(pos, anchor absent)` across positions.
  Under factorisation these are roughly parallel; report the distribution of
  pairwise cosine similarities. This is the cleanest single number for
  "is anchor presence a consistent translation in `h`-space."
- **F3 Additivity.** Fit `h ≈ f(pos) + g(Δ_to_anchor)` and report the variance
  the interaction term needs. Large interaction = conjunctive coding.
- **F4 Frame comparison.** Skaggs SI computed in the room frame vs. the
  anchor-offset frame, per unit. Plot the ratio. Place cells win in room
  coordinates; OVCs win in offset coordinates; object cells win in offset
  coordinates with mass at `d ≈ 0`. This makes the frame explicit instead of
  smuggling it in as an assumption.

### E8 — Behaviour controls (Tsao Fig. 4)

Your policy is curiosity-driven, so it *approaches high-prediction-error
locations, which are the tiles*. Dwell time near anchors is systematically
elevated and correlated with anchor position. Any "object unit" could be an
occupancy artifact.

- Random-action agent as the primary condition; AC agent secondary.
- Plot, for both agents: time spent at the anchor location, speed at the anchor
  location, and unit activation as a function of speed — separately for
  anchor-present and anchor-absent episodes. Tsao's exact figure.
- Report occupancy maps alongside every rate map, always.
- Explicitly check that OVC-passing counts do not differ between the two agents;
  if they do, the code is entangled with policy and that is itself a result.

---

## 5. Checkpoint-specific evals

The four families are not four replications — they are a **dose-response ladder
on training diversity**, and the ladder is a better claim than "we found OVCs."

| Checkpoint | Role | Prediction | Evals |
|---|---|---|---|
| **1 L-room, large tiles (x, +, △)** | Negative control / floor | Anchor identity is perfectly confounded with position. No pressure to factorise. Expect conjunctive place-like coding; E2 likely OOD. | E1, E3–E8; E2 **only with the OOD guard**, report as inconclusive if it trips |
| ~~1 L-room, small tiles~~ | **DROPPED** (2026-08-13) | `LEnv_small_obj`'s three landmarks are identical `block2` 2×2 squares differing only in colour, so anchor identity cannot vary independently of position — the opposite of what E1 needs. It is also corner-anchored while every other room is centre-anchored, so its offsets do not line up. The tile-size comparison it was meant to supply would have been confounded with shape set, anchor convention, room count and training length simultaneously. | none |
| **3 L-rooms** (x / plus / block3) | Intermediate rung | Weak factorisation. ⚠️ `ROOMS_RUN1`'s three rooms are EXACT TRANSLATIONS of one configuration (room 0→1 by (0,3), 0→2 by (3,0)), all at separation signature (6,6,6). A network can cover all three with one map plus a translation, so remapping is never *required* here and a null is weak. See `curious_george/envs/layouts.py`. | Full battery; plus **cross-room offset consistency** — same unit, same offset across the three rooms (the Fig. 2f,g analogue) |
| **500 randomised L-rooms** | **The main event** | Tiles vary in colour and position across batch elements, so the network *must* bind anchor position in-episode. This is where OVC structure should appear, and where E2 is in-distribution. | Full battery + G1 + G2 below |
| **Square rooms** (x / plus / block3) | Geometry control (Ext. Data Fig. 7a–c), and the STRONGER arm — `ROOMS_SQUARE` has distinct separation signatures (6,6,6)/(6,6,9)/(6,6,8), so its triads differ in geometry rather than merely in position | Høydal: orientation preferences preserved between square and circular compartments. | Cross-geometry offset-map correlation and Δ peak bearing, per unit, against the L-room net |

⚠️ **Training length is NOT matched across these rows and must be stated with any
cross-checkpoint comparison.** As of 2026-08-13 the archives sit at: 3-room L-room
94.4M, 500-room L-room pool 52.4M, 3-room square 31.5M, 500-room square pool 21.0M
environment steps. The square arm has roughly a third of the L-room arm's training,
so a difference in OVC pass rate between arms confounds geometry with training
length. Compare at matched step counts, or say plainly that you did not.

### G1 — Held-out room, zero-shot (500-room net only)

Høydal's headline generalisation result: 14 cells recorded on the mouse's
**first exposure** to room B, novel box and novel object, and all cells with
vector fields in A had them in B; rotation-aligned vector-map r = 0.87.

⚠️ **A held-out set cannot be cut by extending the seed.** Measured:
`generate_layouts(n=600, seed=20260813)` does NOT extend `generate_layouts(n=500,
seed=20260813)` — its first 500 differ, so `[500:]` is not a held-out tail of the
training pool. Build the admissible set with `enumerate_anchor_triples`, subtract
the 500 training layouts explicitly, and sample from the remainder.

Port: hold out rooms from the 500 at training time, or generate fresh ones.
Frozen pRNN and policy. First episode. Do the offset maps transfer? This is the
single most important result if it works, because it is the difference between a
reference frame and a memorised configuration.

### G2 — Novel anchor colour/type (500-room net only)

Their Fig. 2a,b logic taken to the limit: place a tile of a colour or shape not
in the training distribution. Do the same units express the same offset? If yes,
the frame is genuinely identity-independent.

### G3 — Anchor-count parametrics

Their Fig. 3 was object *size*; yours can be object *count*. Run 1, 2, 3, 4
anchors and check whether offsets remain consistent and whether the fraction of
responding units saturates.

---

## 6. Figures to generate

Ordered as a paper would be.

1. **Fig. 1 — Existence.** For 3–5 example units: room map with anchor marked
   (anchor at A / anchor at B / anchor absent), offset map below each, matching
   Høydal Fig. 1b. Plus population counts: % object cells, % OVCs, % grid-like,
   % border-like, per checkpoint (their Fig. 1c).
2. **Fig. 1e analogue.** Polar scatter of bearing vs. distance for every detected
   offset field, pooled across units, with the distance histogram beside it.
3. **Fig. 2 — Generalisation.** (a) Example unit's three anchor-centred maps in
   one episode. (b) Box plots of pairwise offset/bearing differences across
   anchors, against their 5.8° / 3.4 cm benchmark. (c) Held-out-room example
   maps. (d) Cross-room offset-map correlation distribution.
4. **Fig. 3 — The blindness ladder.** SI, coherence, peak activation, and HD
   score vs. `steps_since_obs`; offset-map correlation vs. `steps_since_obs`;
   in-view vs. out-of-view offset maps side by side for an example unit.
5. **Fig. 4 — Exclusions.** Egocentric directionality index distribution vs.
   shuffled null (their Ext. Data Fig. 6e layout); border/grid/HD/speed score
   box plots, OVC units vs. rest (their Ext. Data Fig. 4b layout); field-to-anchor
   vs. field-to-wall distance box plot.
6. **Fig. 5 — Population geometry.** Cross-condition decoding transfer matrix;
   Δh cosine-similarity histogram; room-frame vs. offset-frame SI scatter, one
   point per unit, diagonal marked.
7. **Fig. 6 — The ladder.** OVC pass rate, F2 parallelism, and F1 transfer gap
   as functions of training-room diversity (1 → 3 → 500). The headline figure.
8. **Controls panel.** Occupancy maps, per-bin visit counts, Tsao Fig. 4
   behaviour panels, untrained-network pass rates, synthetic-injection recovery.

---

## 7. Reading the outcome

Do not frame this as a binary. Four distinguishable results, all publishable:

1. **OVCs present and graded with diversity.** E1/E2 pass, E3 shows graceful
   degradation, F2 parallelism high, and all of it increases 1 → 3 → 500 rooms.
   The strong result: factorisation is a learned consequence of environmental
   variability.
2. **Vector coding present but distributed.** Single units fail, F1/F2/F4 pass.
   Say so plainly — an axis-misaligned factorised code is a real finding and the
   population analyses are the appropriate instrument.
3. **Object cells only.** Fields at `d ≈ 0`, nothing displaced. The network
   detects anchors but does not use them as reference frames. This is the
   LEC-like regime, and it is exactly what Høydal's introduction says was known
   before their paper.
4. **Conjunctive place coding.** E2 fails everywhere including the 500-room net.
   Then next-observation prediction alone does not induce factorisation, and the
   interesting follow-up is what objective would.

Two things to guard against throughout: reporting E1 on a fixed-room checkpoint
without flagging the identity/position confound, and reading an OOD failure in
E2 as a negative result. Both are easy mistakes and both invert the conclusion.

---

## 8. Suggested build order

1. Dumper (§2) + Stage 0 controls (S0.1–S0.5).
2. Offset-map primitive + circular-shift null (§3.3, §3.6) — everything depends
   on these two being right.
3. **E1** on all checkpoints. Cheapest, strongest, no new env code.
4. **E3** — also nearly free, and it is the control that makes E1 interpretable.
5. E2 with the OOD guard, starting with the 500-room net.
6. E7 population analyses.
7. E4–E6 exclusions.
8. G1/G2 on the 500-room net.
9. E8 behaviour controls (run early in draft form; finalise last).

Existing code to fold in rather than rewrite: `ovc_metric.py`
(`vector_score`, `radial_score`, `peak_ratio`, `vector_percentile`,
`inject_vector_field`, `inject_place_field`), `trace_maps.py::view_frame_maps`
(egocentric frame — becomes the E4 comparison rather than the primary frame),
`moser_analysis.py::analyse`, and `tasks/omt/metrics.py::get_view_coords_batch`
for the `in_fov` flag. Fix the hardcoded
`CFG = "/home/sabrina/…/Configs"` in `trace_objvector_test.py` to
`Path("Configs").resolve()` while you are in there.

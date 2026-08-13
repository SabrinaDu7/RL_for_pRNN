2026-08-12

# OVC — object-vector and object cells in the pRNN hidden state

## Question

Does any pRNN hidden unit code an **object-vector relation** — firing at a fixed
allocentric offset from an anchor, generalising across anchors — or an **object cell**
relation — firing within a small radius of any anchor regardless of direction?

Answerable yes/no per unit, with a population fraction and a chance rate.

## Motivation

Every object-coding result in this project has been null, and the mechanism is understood:
the object is absorbed into `W_out` ~89% position-independently, so nothing location-specific
must be learned (`../claude_logs/compaction.md` §1). But every one of those tests looked for a
**trace** — a field at a place an object *used to be*.

Two reasons that was the wrong target.

**1. The objective forbids traces and demands vectors.** The pRNN predicts the next
observation, sometimes with the observation masked. To do that it must know its metric
relation to whatever is visible — which is exactly object-vector computation. A trace, by
contrast, would push the readout to predict an object that is not there, which *raises* the
loss. So the objective actively selects against the thing we were hunting, and plausibly
selects *for* the thing we never measured.

**2. The permanent landmarks are objects, and we never treated them as such.** The L-room
contains three coloured floor shapes. They have been present for all ~80,000 gradient steps of
`main_train`. If vector coding forms anywhere in this system it forms to those, not to a
`FloorBright` tile shown for a few hundred trajectories at the end. `scripts/trace/trace_objvector_test.py`
does not cover this: it measured the *change in* object-relative tuning caused by *exposure to
the novel object*. Pre-existing tuning to the permanent landmarks has never been measured.

A framing worth keeping in view: a usable spatial representation plausibly needs a slow
coordinate system coding the invariances of space, plus a faster mechanism binding current
content to location. **This architecture has one weight set trained at one timescale.** If
content-binding requires a faster pathway than map-learning, there is no such pathway here —
which would be a result about the architecture, not about the room.

## Why this design

### Anchors: the permanent landmarks, not the novel object

| anchor | colour | span | centroid |
|---|---|---|---|
| triangle | blue | x1–6, y1–6 | (4.3, 2.7) |
| plus | red | x8–13, y1–6 | (10.5, 3.5) |
| x | yellow | x2–7, y8–13 | (4.5, 10.5) |

Measured from the env, not the config. Separations 6.2 / 7.8 / 9.2 cells; 172 walkable cells.

### Cartesian offset maps, not polar

Moser bins firing as rate(distance, allocentric orientation) at 2 cm × 5°. On a 14×14 grid
that binning is pathological: at distance 1 there are 8 neighbours spread over 360°, so 5°
bins are almost all empty, and bin occupancy varies wildly with radius.

We use the **allocentric Cartesian offset map** instead: `R_A(dx, dy)` = mean rate over probe
samples where the agent is at `A + (dx, dy)`. Same information, native to the grid, uniform
bin occupancy. Direction and distance are both recoverable from it.

### Generalisation across anchors replaces object displacement

Moser's whole design hinges on trial 3 — move the object, does the vector map survive? A
room-anchored place field decorrelates; a vector field does not.

**We cannot move a landmark without retraining.** But we have three of them, which gives the
same dissociation from a different direction: a place cell has a field at one *room* location,
so its offset map differs per anchor; a vector cell has the *same* offset map for all three.

This is the load-bearing choice, and it is weaker than Moser's in one specific way: three
fixed anchors cannot fully break the confound between "vector tuning to anchor A" and "place
field that happens to sit at that offset from A". Agreement across three anchors makes
coincidence unlikely but does not exclude it. **E2 (the swap) exists to close that gap.**

### The testable offset window is set by geometry

An offset only counts if `A + (dx, dy)` is walkable for **all three** anchors. Computed:
**56 of 169 offsets** in a ±6 window qualify, spanning Chebyshev distance **0–4**. That is a
hard constraint on the experiment, not a parameter. It is also the right scale: Moser's median
field-to-object distance was ~20% of arena width, which is ~3 cells here.

## Stage 0 — Evaluation setup (MUST PASS BEFORE ANY RESULT COUNTS)

No experiment is interpretable until the detector is characterised. Mirrors what made the
trace metric trustworthy (negative control 0.0601 p=0.174; positive control 100% recall).

### The pipeline

Per unit, following Høydal et al. with the adaptations above:

1. **Spatial screen.** SI above the 95th percentile of a trajectory-level shuffle null
   (`scripts/trace/trace_maps.py::shuffle_null_si`). Rationale: a unit with no spatial
   structure at all cannot have vector structure.
2. **Rate screen.** Peak offset-map rate ≥ some multiple of the unit's own mean rate. Scale-free
   analogue of Moser's ≥2 Hz, since our units have no physical rate.
3. **Exclude the anchor itself.** Drop offsets with Chebyshev distance ≤ 1. Moser discarded
   fields 0–4 cm from the object centre as LEC-style object cells rather than vector cells. We
   test that class **separately** (below) rather than discarding it.
4. **Vector criterion.** Mean pairwise correlation of the three offset maps
   `r(R_blue,R_red)`, `r(R_blue,R_yellow)`, `r(R_red,R_yellow)`, computed over the 56 commonly
   valid offsets. Pass if above the 99th percentile of the shuffle null.
5. **Object-cell criterion, tested separately.** Is `R_A(dx,dy)` well predicted by a function
   of `|offset|` alone — i.e. radially symmetric with a peak near zero? Compare variance
   explained by a radial model against the full 2-D map. An object cell passes this and fails
   the directional part of (4).

### Three controls, all required

| control | construction | required outcome |
|---|---|---|
| **negative** | run the whole pipeline on odd vs even probe trajectories | pass rate ≈ the nominal chance rate |
| **positive** | inject a synthetic vector field into real `h` — a Gaussian bump at offset (+3,+2) from **every** anchor — at a range of amplitudes | recall → 100%, and the recovered offset must be (+3,+2) |
| **specificity** | inject a synthetic **place** field — a bump at one fixed *room* location, not an offset | must **NOT** be called an OVC |

The specificity control is the one that matters. It is the direct analogue of Moser's trial-3
logic, and without it a "vector cell" claim cannot be distinguished from a place cell that
happens to sit at a convenient offset.

Report all three with every result, as a sensitivity bound rather than a shrug.

## Experiments, in order

### E1 — Landmark OVC on the existing checkpoint
No new training. Replay the fixed probe (`scripts/trace/trace_probe.py`) through
`outputs/ckpts/pRNN_curious_26-07-23-10-06-25`, build the three offset maps per unit, run the
pipeline. **This is the whole question, on data we already have.**

### E2 — Landmark swap (the displacement analogue)
Swap the red plus and the yellow x, replay the *same* net at test time. A vector cell's field
follows the landmark; a place cell's field stays at the room location. Closes the confound E1
cannot.

Caveat to state in the result: the net never saw the swapped room, so this is out of
distribution. If the representation simply degrades, the test is uninformative rather than
negative — check map quality first. Colour and shape move together, so a field that follows
identity vs position is also distinguishable.

### E3 — Novel-object triplet in the square room
The `FloorBright` tile is point-like and can be absent, placed, and displaced — Moser's exact
trial structure, which the fixed landmarks cannot give. Uses `MiniGrid-SquareRoom-v0` and
`exp.new_obj_pos`. Lower prior: the net has little experience of this object. Run after E1/E2
so the criterion is already validated.

### E4 — RL vs random (free)
Same pipeline on the random-action checkpoint (`RAND_CKPT_DIR`). The world-model objective is
identical; only the trajectory distribution differs. Tests whether OVC formation depends on
behaviour at all. Expected small; costs nothing.

## Controls against the alternatives Moser ruled out

Run on whatever passes E1, in this order of value:

- **Egocentric bearing.** Build tuning against bearing-to-anchor *relative to heading* using
  the probe's `agent_dir`; directionality index vs shuffle. Moser found only 10/162 passed —
  the code is allocentric. If our cells are egocentric they are a different phenomenon.
- **Border / boundary-vector.** Our anchors are near walls, so this is a live confound.
  Compare field-to-anchor against field-to-wall distance, and check field shape — a BVC gives a
  wall-parallel band, an OVC a compact blob (Moser: median aspect ratio 1.6).
- **Multi-field / grid.** Only if multi-field units appear.

Moser's suspended-wall dissociation is the cleanest experiment in that paper and has no cheap
analogue here — our walls cannot be lifted, and "path obstruction" is not a variable the agent
experiences differently.

## Known disanalogies — state these with any result

1. **Our anchors are 6×6 walkable floor patches (21–28 tiles), not small obstacles.** The agent
   walks over them. "Distance from the object" is fuzzy for an object spanning a third of the
   room, and vector coding to a large region is closer to boundary-vector coding than to
   object-vector coding.
2. **Three fixed anchors, never moved during training.** Moser had pseudo-random displacement
   every session.
3. **The offset window is Chebyshev ≤ 4**, forced by geometry.
4. **Our units are strongly spatial** (mean SI ≈ 0.9) whereas LEC cells show little spatial
   modulation in an empty field. This population is place-cell-like, which is an argument for
   expecting MEC-style vector coding rather than LEC-style object or trace coding — the whole
   premise of this experiment.

## What each outcome would mean

| outcome | reading |
|---|---|
| E1 passes, E2 field follows the landmark | object-vector coding exists here; the earlier nulls were a probe problem, and the trace hunt was aimed at the wrong cell class |
| E1 passes, E2 field stays put | place cells at coincidental offsets. E1's three-anchor agreement was not enough; report as negative |
| E1 null, object-cell criterion passes | LEC-style object coding without a vector code |
| both null, positive control at 100% | this architecture does not do vector coding, and the target itself was wrong — not the room. Strongest argument for the slow/fast timescale point above |

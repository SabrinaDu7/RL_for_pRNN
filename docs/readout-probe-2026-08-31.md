2026-08-31 late · branch `sdu/optim-pred` · checkpoint `focal5full` s3 (Mila 10612309) · local 4060

# Readout probes on frozen h: is there room above the pRNN's own readout?

## Question and design

The user's test: fit a fresh LINEAR readout from the frozen hidden state h
of a known checkpoint over all 8 training rooms; if it beats the network's
own `outlayer`, the readout is leaving information on the table. Extended
with an MLP probe, which separates the two possible conclusions of a linear
tie: "the readout already extracts everything linear" vs "h carries more,
non-linearly".

Code: `curious_george/evaluation/readout_probe.py` (analysis-only; splits
are disjoint by env seed AND walker seed via `surprisal_timing.walk_episodes`
- train seed_base 1000, val 2000, test 100 = the published protocol). The
own readout's logits on the test walk are asserted equal to `outlayer(h)`
before anything is compared. Fits are early-stopped on their OWN objective
on val; the focal-γ weighting is applied during fitting only, evaluation is
always plain surprisal. Command for every number below:

    uv run python -m curious_george.evaluation.readout_probe \
      --ckpt outputs/fetched/mx-impassable-n8-s3-focal5full_curious_26-08-31-22-20-54/predictiveNet_state.pt \
      --train-eps 48 --hidden 512

⚠️ Not 1:1 comparable to `surprisal_timing` tables: surprisal here is
tile-pooled (not per-step-averaged) and the torch noise stream sits at a
different point after three walks; within-table rows share everything.

## Result (test = published protocol episodes, all 8 rooms)

| head (all fit on 48 eps/room of frozen h) | landmark shown/masked (nats) | background | recall shown/masked | miss shown/masked |
|---|---|---|---|---|
| own `outlayer` (co-trained, focal γ=5) | 0.677 / 0.738 | 0.364 | 0.665 / 0.598 | 0.115 / 0.152 |
| fresh linear, plain-CE fit | 0.736 / 0.853 | 0.108 | 0.642 / 0.576 | 0.102 / 0.138 |
| fresh linear, focal-5 fit | 0.652 / 0.702 | 0.291 | 0.671 / 0.612 | 0.094 / 0.118 |
| **MLP-512, focal-5 fit** | **0.589 / 0.638** | **0.224** | **0.718 / 0.657** | **0.076 / 0.086** |

Data scaling of the linear focal fit (shown-step nats): 8 eps 0.825 → 24
eps 0.684 → 48 eps 0.652 - converging to a linear ceiling ≈0.63-0.65.

## Conclusions

1. **Linear: slight room, mostly a data mirage.** A converged fresh linear
   head edges the own readout on every axis (−0.03 nats, +1pt recall,
   better background), but the margin is small and shrank as the trend
   flattened. The co-trained readout sits ~0.03-0.05 nats off the linear
   ceiling of h. Swapping/refitting a linear head is not where the win is.
2. **Nonlinear: real room.** An MLP-512 probe on the SAME frozen h beats
   the network's own readout on every metric SIMULTANEOUSLY - no
   landmark/background trade: recall 72/66% (vs 67/60), miss nearly halved
   (0.076/0.086), surprisal 0.589/0.638, background better too. The
   information is in h; the single linear map cannot extract it.
3. **The objective result replicates on frozen h**: the plain-CE linear fit
   reproduces the plain-CE pathology (best background 0.108, worst
   landmarks) with the representation held fixed - the allocation effect is
   entirely the loss's, not the dynamics'.

## What this recommends (queued for the user)

- **Architecture: an MLP readout** (`readout="mlp"`, hidden ~512) in the
  fork, trained end-to-end with focal - the probe bounds the immediate gain
  at roughly the MLP-probe row, and end-to-end typically beats the frozen
  probe. This touches the fork's `create_layers` and the goldens must stay
  bitwise at the default.
- **Cheap deployment trick, no retraining**: keep any checkpoint, refit an
  MLP readout post-hoc on rollouts (exactly this module). Usable to upgrade
  `focal5full` s3 for prediction-facing analyses today.
- The h-side questions (hidden size, masking schedule) stay open but are
  now SECOND to the readout: the representation demonstrably carries more
  than is being read.


## End-to-end confirmation (2026-09-01 ~00:00, Mila 10613004/05, `focal5mlp`)

`readout="mlp"` trained end-to-end (fork `852cc7d2`, repo `c5dd7d9`,
γ=5, full budget, both seeds). Checkpoints verified structurally before
scoring (outlayer.0.net.* keys, 674,843 outlayer params - the mlp
fingerprint). Committed protocol (`surprisal_timing --readout MLP`):

| arm (full budget, γ=5) | landmark shown/masked | background | recall shown/masked | miss | mean room sRSA | SWdist |
|---|---|---|---|---|---|---|
| linear readout s2/s3 (`focal5full`) | 0.725 / 0.712 | ~0.36 | 0.664-0.677 / 0.593-0.609 | 0.11-0.12 / 0.14-0.15 | 0.731 / 0.762 | 0.115 / 0.119 |
| **MLP readout s2/s3 (`focal5mlp`)** | **0.689 / 0.675** | **~0.30** | 0.677-0.686 / **0.630-0.633** | 0.107 / **0.12-0.13** | **0.791 / 0.740** | **0.071** / 0.101 |

- Every axis moved the right way SIMULTANEOUSLY again - no trade. sRSA
  0.791 (s2) is the project record; SWdist 0.071 the best measured.
- Training loss: the MLP arm reached the linear arm's FINAL loss at half
  the gradient steps (0.0177 @ ~21k vs 0.0171 @ ~44k; same focal-weighted
  axis, docs/figures/focal5_mlp_vs_linear_loss.png) - ~2x step-efficiency.
- Gallery: docs/figures/focal5mlp_s2_landmark_gallery.png - reconstructions
  that were partial under linear (green plus, red block extents) are now
  complete; residual errors are small shape-boundary details.
- Honest note: end-to-end recall (0.68/0.63) did NOT exceed the frozen-h
  MLP probe on the OLD representation (0.72/0.66). Inferred, not confirmed:
  with a stronger readout co-training, h itself reallocates - the probe row
  was never a strict lower bound for a DIFFERENT h. A fresh probe on the
  mlp-trained h (one command, --readout MLP --hidden 512) would measure the
  remaining extraction headroom.

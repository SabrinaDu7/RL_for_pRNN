# A LayerNorm before the pixel readout — 2026-10-01

## Why

Examining prediction quality on `MSE1024` (questions repository, the 2026-10-01 discussion
after Q18): the targets take four values (0.298 floor, 0.573 wall, 1.0 for a landmark's
colour channel); the trained readout predicts the floor and wall values with a ±0.05 haze
and the saturated landmark channels at a median of 0.63 — and under-shoots them by 0.23
even at the phase of the input cycle where the observation is the network's own input, so
the dimness is not uncertainty. The readout bias (2026-08-30) is in place and trained
(own group, unscaled lr, no decay), and the readout weights' RMSprop decay coefficient is
2.8e-7, so neither "add a bias" nor "remove the decay" changes anything. The 2026-08-30
note measured why `h` is a poor basis for the readout - its mean moves every timestep and
half of it is exactly zero. This run normalises what the readout sees and nothing else.

## What changed (commit below)

- `arch_prnn.readout_norm` (default False, inert): `models/readout_norm.py` wraps
  `pRNN.outlayer` as `Sequential(LayerNorm(hidden), outlayer)` before any checkpoint load,
  keeping the Linear's tensors (the `W_out` / `b_out` aliases and their optimizer groups),
  and gives the norm's two parameters their own optimizer group (`ReadoutNorm`, unscaled
  lr, no decay). The recurrent dynamics, the state the policy reads and every analysis of
  `h` are untouched. `tests/test_readout_norm.py`.
- A checkpoint saved by a normalised network carries the Linear under `outlayer.1.0.*`; it
  loads only into a network built with the flag (provenance records it).

## Launch

The MSE1024 recipe, seed 2, plus `--arch-prnn.readout-norm`:

```bash
sbatch slurm/multienv.sh true 8 2 sdu/mixed-count-mse '' '' '' '' 0,1,2,3,5,6,7,8 mse-h1024-lnread '' \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 --arch-prnn.readout-norm \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

Comparison against `MSE1024` over training: `pRNN loss`, `multiroom/mean_room_sRSA`,
`multiroom/pooled_SWdist` (wandb, `curious-george-multienv`).

- 2026-10-01: submitted as Mila job **11033768** from commit `4556da3` (smoke-tested locally
  at a one-rollout budget first: norm installed, exit 0). Outcome appended below.

## The branch, and the second variant (MSE with the MLP readout)

This work lives on `sdu/readout-norm`; `sdu/mixed-count-mse` was reset to `2a3180d` on
2026-10-01 so the production branch carries none of it. On naming: the historical MSE
head is `Linear -> Sigmoid`, a generalised linear readout (one affine map through a fixed
squash); "linear readout" in earlier notes means that.

`arch_prnn.readout = MLP` is now accepted under MSE (`configs.py`, `storage.py::
prediction_loss_kwargs`): the upstream decode stack (ResidualMLP -> LayerNorm -> Linear,
no squash) regresses the pixels directly. MSE + LINEAR is unchanged and still pinned by
the goldens; `tests/test_ce_boundary.py` carries the new contract.

```bash
sbatch slurm/multienv.sh true 8 2 sdu/readout-norm '' '' '' '' 0,1,2,3,5,6,7,8 mse-h1024-mlpread '' \
    --arch-prnn.loss MSE --arch-prnn.readout MLP --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

Compared against `MSE1024` and the readout-norm run on the same three metrics.

- 2026-10-01: MSE + MLP submitted as Mila job **11033877** from `sdu/readout-norm` at
  `1318cb1` (smoke-tested locally at a one-rollout budget, exit 0). The two pre-existing
  failures in `tests/test_references_resolve.py` (a questions-repository session log and
  a throwaway figure named by older notes; `curious_george.__path__`) predate this branch.

## Two more variants, same branch (commit `7d1ad5e`)

- `train_prnn.readout_lr_scale` — the readout weights' RMSprop group runs at lr / sqrt(hidden)
  = 9.4e-5 while the readout bias runs at 3e-3; this multiplies the weights' lr after
  construction. Run: `--train-prnn.readout-lr-scale 32` (the weights at 3e-3), label
  `mse-h1024-lr32read`, Mila job **11034075**.
- `arch_prnn.saturated_pixel_weight` — `models/pixel_weights.py` replaces the upstream
  `predMSE` with a weighted MSE for the TRAINING loss only: target values ≥ 0.9 (a
  landmark's colour channel, 1.8% of values) weighted `w`, floor and wall at 1,
  normalised by the mean weight; the curiosity reward (the adapter's device pass) is the
  plain per-step error and does not move. Run: `--arch-prnn.saturated-pixel-weight 10`,
  label `mse-h1024-pixw10`, Mila job **11034076**.

Both smoke-tested locally (one rollout, exit 0). First result, the readout norm
(job 11033768): final `pRNN loss` 0.0089 against `MSE1024`'s 0.0123; mean-room sRSA
0.602 against 0.624; pooled SWdist 0.085 against 0.071 — the loss improves, both spatial
metrics give a little. The rest is appended once the five-run comparison lands.

## Results so far (2026-10-02 00:30)

Training metrics (wandb, tail mean of the last 5% of logged points; the spatial metrics
are logged five times per run, so "tail" is the last point) and the fixed-probe
prediction quality (Q15's random-walk probe, noise off, final checkpoint; `mse` over all
pixels, `sat` = the saturated landmark colour channels, 1.8% of values):

| run | train loss | sRSA (high) | SWdist (low) | probe mse | sat abs err | sat pred median |
| --- | --- | --- | --- | --- | --- | --- |
| MSE1024 (s2) | 0.0092 | 0.624 | 0.071 | 0.00697 | 0.331 | 0.68 |
| MSE1024 s3 / s4 | 0.0091 / 0.0091 | 0.676 / 0.608 | 0.115 / 0.075 | 0.00735 / – | 0.345 / – | 0.67 / – |
| MSE2048 (s2) | 0.0076 | 0.683 | 0.027 | 0.00599 | 0.296 | 0.73 |
| readout norm (lnread) | 0.0091 | 0.602 | 0.085 | 0.00665 | 0.311 | 0.71 |
| MLP readout (mlpread) | 0.0109 | 0.737 | 0.128 | 0.00768 | 0.325 | 0.69 |
| readout lr x32 (lr32read) | 0.0094 | 0.256 | 0.011 | 0.00720 | 0.289 | 0.74 |
| pixel weight 10 (pixw10) | 0.0194* | 0.747 | 0.334 | 0.01463 | 0.140 | 0.90 |

\* the weighted loss, not comparable. Reading: width (2048) improves every column at
once; the readout norm's gains are inside the seed spread; the MLP readout buys sRSA with
worse prediction and SWdist; the fast readout collapses sRSA; the pixel weight is the
only change that makes the landmark channels crisp (0.90 against 0.68) and it pays with
twice the background error and five times the SWdist. Pending: 2048 + readout norm,
pixel weight 3, and seeds 3/4 of the 2048 and readout-norm recipes.

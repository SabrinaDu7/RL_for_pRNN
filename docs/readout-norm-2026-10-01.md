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

"""Is zeroing the SI of low-activity units a bias correction, or does it delete real fields?

`PredictiveNet.calculateSpatialMetrics` sets SI to 0 for any unit active in
fewer than `active_time_threshold` samples. Two readings of that rule:

  bias correction  a near-silent unit carries ~no spatial information, and the
                   mutual-information estimator is upward-biased at small
                   sample counts, so 0 is closer to the truth than the estimate.
  destroys signal  a sharply tuned place cell fires in FEW timesteps by
                   definition, so an activity threshold preferentially removes
                   the most spatially selective units.

The discriminator is split-half map reliability, which the SI estimator's
small-sample bias does NOT survive: an inflated SI from noise does not
reproduce across an independent half of the trajectories, a real field does.
So for each activity bin we report SI alongside odd/even map correlation.

    uv run python scripts/si_threshold_audit.py

Reads the shared probe under outputs/trace/ and the main_train checkpoint in
.env (CUR_CKPT_DIR). Writes nothing; prints a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The threshold under audit, as it is spelled in prnn.
ACTIVE_TIME_THRESHOLD = 200
PROBE = Path("outputs/trace/probe_lroom_noobj")


def audit() -> None:
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames

    from curious_george import get_pN, make_env
    from curious_george.utils.dev_env import get_ckpt_env_vars
    from scripts.trace import trace_maps as tm, trace_probe as tp

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)

    probe = tp.load_probe(PROBE)
    prnn_ckpt, _ = get_ckpt_env_vars()   # defaults to AgentType.AC -> CUR_CKPT_DIR
    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=prnn_ckpt)

    h = tp.replay_checkpoint(pN=pN, probe=probe).detach().numpy()   # (B, T, H)
    pos = probe.agent_pos[:, : h.shape[1], :]                       # (B, T, 2)

    _report("probe scale (all trajectories)", h=h, pos=pos, env=env, tm=tm)

    # The threshold is an ABSOLUTE count, so what it means depends entirely on
    # how many samples were pooled. The training loop's spatial eval pools
    # exp.eval_trajs trajectories, far fewer than the offline probe, so the
    # same constant is far more aggressive there. Audit both.
    n_eval = int(args.exp.get("eval_trajs", 8))
    _report(f"training-eval scale (exp.eval_trajs={n_eval})",
            h=h[:n_eval], pos=pos[:n_eval], env=env, tm=tm)


def _report(title: str, *, h, pos, env, tm, onset: int = 20) -> None:
    """Zeroed-vs-masked comparison at one pooling size.

    h:   (B, T, H) replayed activity.
    pos: (B, T, 2) the position each row was recorded at.
    """
    h_rows = h[:, onset:, :].reshape(-1, h.shape[-1])
    pos_rows = pos[:, onset:, :].reshape(-1, 2).astype(np.float64)

    maps, occupancy, _ = tm.occupancy_and_maps(h=h_rows, pos=pos_rows, env=env)
    si = tm.spatial_info(maps=maps, occupancy=occupancy)
    stability = tm.split_half_stability(h=h, pos=pos, env=env)

    num_active = (h_rows > 0).sum(axis=0)
    n_samples, n_units = h_rows.shape
    below = num_active < ACTIVE_TIME_THRESHOLD
    print(f"\n=== {title} ===")
    print(f"{n_samples} pooled samples, {n_units} units; "
          f"threshold {ACTIVE_TIME_THRESHOLD} = {100 * ACTIVE_TIME_THRESHOLD / n_samples:.2f}% of samples")
    print(f"units zeroed: {below.sum()} / {n_units} ({100 * below.mean():.1f}%)")

    print(f"{'':>18} {'units':>6} {'median SI':>10} {'median split-half r':>20}")
    for label, sel in [("below threshold", below), ("above threshold", ~below)]:
        if not sel.any():
            print(f"{label:>18} {0:>6} {'-':>10} {'-':>20}")
            continue
        print(f"{label:>18} {sel.sum():>6} "
              f"{np.nanmedian(si[sel]):>10.4f} {np.nanmedian(stability[sel]):>20.4f}")

    zeroed = np.nan_to_num(np.where(below, 0.0, si)).mean()
    masked = np.nanmean(np.where(below, np.nan, si))
    print(f"mean SI, zeroed convention : {zeroed:.4f}")
    print(f"mean SI, masked convention : {masked:.4f}   "
          f"(difference {100 * (masked - zeroed) / max(zeroed, 1e-12):+.1f}%)")


if __name__ == "__main__":
    audit()

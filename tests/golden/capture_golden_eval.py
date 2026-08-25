"""Golden fixture for ONE EVALUATION from REFERENCE WEIGHTS.

The counterpart to `capture_golden.py`, and it answers a different question.
That one builds from a seed and trains two updates: it pins "same seed => same
trajectory", which catches a refactor that moves RNG consumption. This one
LOADS a trained checkpoint, runs ONE collect under `eval_mode`, and pins the
five metrics computed from that rollout. It catches a refactor that changes
what the metrics MEAN, without needing training to be reproducible at all.

That distinction matters for cosmetic work: a rename cannot move RNG, so both
fixtures hold; but a change to how a metric is assembled shows up here and
nowhere else.

    prnn_loss   mean per-frame pRNN prediction MSE - the curiosity reward IS
                that error (rl/update/rewards.py), so the rollout already has it
    mi_policy   mutual_info_policy over the (HD, x, y, action) joint the
                rollout accumulates
    SI          per-unit spatial information, plus the coverage triple
    sRSA        representational similarity against space
    SWdist      sleep-wake distance

All five come from the SAME rollout: the hidden states and positions it
recorded are handed straight to `pN.calculateSpatialMetrics`, which takes
PRECOMPUTED activity, so nothing is collected twice.

EVAL MODE is set here, not in the collector. `collect_rollout` only wraps its
forwards in `torch.no_grad()`, which stops gradients and not dropout;
`models/device.py::eval_mode` is what disables the pRNN's input dropout, and it
is what every other eval path in the repo uses (evaluation/probe.py,
evaluation/spatial.py). The injected noise is deliberately KEPT - it is the
model's dynamics, and it is what generates the sleep activity SWdist compares
against.

Run (GATE - compares, never writes):
    uv run python tests/golden/capture_golden_eval.py

Re-baseline, only with a reviewed reason:
    uv run python tests/golden/capture_golden_eval.py --recapture

Against different reference weights:
    GOLDEN_EVAL_CKPT_DIR=/path/to/run uv run python tests/golden/capture_golden_eval.py --recapture

REFERENCE WEIGHTS. The default is the checkpoint tracked in this repo, so the
gate runs on a clean clone. It is the only one that is: cluster runs rsync to
`$SCRATCH` on Mila and are not here. To pin against a specific run (e.g.
`fast-single-e0.001to0.01-g8-p2048-s2-graphall-roll_curious_26-08-24-19-30-37`),
rsync its directory down, point `GOLDEN_EVAL_CKPT_DIR` at it, `--recapture`,
and TRACK the checkpoint - a fixture whose weights are untracked cannot run for
anyone else, which is the defect that left the OMT gate unrunnable on a clean
clone.
"""

import os
from pathlib import Path

import numpy as np
import torch

import curious_george as RLutils
from curious_george import ACModelSR, PredictivePPOAlgo
from curious_george.evaluation.on_policy import mutual_info_policy
from curious_george.evaluation.spatial import si_coverage
from curious_george.models.device import eval_mode, on_device
from curious_george.utils.checkpoints import StatusCkptKeys
from prnn.utils import PredictiveNet, load_pN

from tests.golden.capture_golden import compare_fixtures

SEED = 2
# 2048 frames in 256-step segments: the SAME sample count and segment length the
# production spatial eval uses (n_trajs=8 x traj_timesteps=seqdur). Measured at
# 256 frames these metrics come back NaN - too few samples leaves empty distance
# bins in calculateRSA_space, so sRSA and SWdist divide by zero. A fixture that
# pins NaN is worse than none: torch.equal is False for NaN, so it would fail
# every run while looking like a real regression.
FRAMES = 2048
SEQDUR = 256
#: Steps dropped at the start of each segment; the production eval's default.
ONSET_TRANSIENT = 20
DEVICE = torch.device("cpu")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "golden" / "golden_eval_v1.pt"

#: Reference weights. Pinned, not the ambient CUR_CKPT_DIR, so the oracle does
#: not move when you switch working checkpoints.
CKPT_DIR = Path(
    os.environ.get(
        "GOLDEN_EVAL_CKPT_DIR",
        str(REPO / "outputs" / "ckpts" / "pRNN_lroom_cur_noObs_26-02-15-17-33-11"),
    )
)


def build_fixture() -> dict:
    """Load reference weights, run ONE collect in eval mode, compute the five."""
    RLutils.seed(SEED)

    env = RLutils.make_env(
        env_key="MiniGrid-LRoom-v0", input_type="pRNN", seed=SEED + 10000, act_enc="SpeedHD",
    )
    obs_space, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)

    pN = PredictiveNet(
        env, hidden_size=500, pRNNtype="thRNN_5win", learningRate=3e-3, bptttrunc=1e8,
        weight_decay=3e-3, neuralTimescale=2, dropp=0.15, trainNoiseMeanStd=(0, 0.05),
        f=0.5, wandb_log=False,
    )
    pN.env_shell.hd_trans = np.array([-1, 1, 0, 0])
    load_pN(
        model_ckpt_filepath=str(CKPT_DIR / "predictiveNet_state.pt"),
        device=DEVICE, pRNNtype="thRNN_5win", predictive_net=pN,
    )

    # with_obs=False matches the checkpoint's actor (no conv tower).
    acmodel = ACModelSR(obs_space, env.action_space, pN.hidden_size, False, True, True)
    status = torch.load(CKPT_DIR / "status.pt", map_location=DEVICE, weights_only=False)
    acmodel.load_state_dict(status[StatusCkptKeys.MODEL_STATE.value])
    acmodel.to(DEVICE)

    algo = PredictivePPOAlgo(
        env, acmodel, pN, DEVICE,
        num_frames=FRAMES, discount=0.98, lr=3e-4, gae_lambda=0.95, entropy_coef=0.0,
        value_loss_coef=1, max_grad_norm=0.5, adam_eps=1e-8, clip_eps=0.2, epochs=4,
        batch_size=256, preprocess_obss=preprocess_obss, train_pN=True, noise_mu=0,
        noise_std=0.05, prnn_seqdur=SEQDUR, intrinsic=False, k_int=1, pastSR=True,
        curious_agent=True, k_curious=1,
    )

    # ONE collect, in eval mode. Nothing is trained here - the weights are the
    # fixture's input, not its output.
    with eval_mode([pN, acmodel]):
        _, logs = algo.collect_experiences()

    # The curiosity reward IS the pRNN's per-frame prediction MSE, so the
    # rollout already carries the loss; no second forward.
    prnn_loss = torch.as_tensor(logs["curious_rewards"]).double().mean()
    mi_policy = float(mutual_info_policy(logs["joint_dist"]))

    # Spatial metrics from THIS rollout's activity - calculateSpatialMetrics
    # takes precomputed (h, positions), so nothing is collected twice.
    #
    # ONSET TRIM, and it is not cosmetic. The tracker's SR is EXACTLY ZERO on
    # the first step of every segment, because a reset zeroes the hidden state -
    # measured, 8 all-zero rows in 2048 at seqdur=256. calculateRSA_space uses
    # COSINE distance, which is undefined on a zero vector, so a single such row
    # turns sRSA and SWdist into NaN. `collect_pooled_activity` drops
    # `onset_transient` steps per trajectory for exactly this reason; reusing a
    # training rollout has to do the same or it is not measuring the same thing.
    keep = np.ones(FRAMES, dtype=bool)
    for start in range(0, FRAMES, SEQDUR):
        keep[start : start + ONSET_TRANSIENT] = False
    h = algo.SRs.detach().cpu().numpy()[keep]
    positions = np.asarray(logs["locs"], dtype=np.float64)[keep]
    assert np.abs(h).sum(axis=1).min() > 0, "a zero SR row survived the onset trim"

    with eval_mode([pN]), on_device([pN], "cpu"):
        metrics = pN.calculateSpatialMetrics(
            h, positions, env,
            sleepstd=0.03, sleep_timesteps=500, active_time_threshold=200,
            rng=None, wandb_nameext="_goldenEval",
        )
    coverage = si_coverage(h, active_time_threshold=200, SI=metrics["SI"])

    return {
        "meta": {
            "seed": SEED, "frames": FRAMES, "seqdur": SEQDUR,
            "ckpt": CKPT_DIR.name, "torch": torch.__version__,
        },
        "metrics": {
            "prnn_loss": prnn_loss,
            "mi_policy": mi_policy,
            "sRSA": float(metrics["sRSA"]),
            "SWdist": float(metrics["SWdist"]),
            "SI_per_unit": torch.as_tensor(
                np.asarray(metrics["SI"]["SI"], dtype=np.float64)
            ),
            **{k: float(v) for k, v in coverage.items()},
        },
        # The rollout the metrics were computed from, so a diff says WHERE it
        # diverged - a metric moving with an identical rollout is a metric bug,
        # a rollout moving is a collection bug.
        "rollout": {
            "actions": algo.actions.clone(),
            "curious_rewards": algo.curious_rewards.clone(),
            "SRs": algo.SRs.clone(),
            "locs": list(logs["locs"]),
        },
    }


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--recapture", action="store_true",
                    help="OVERWRITE the baseline. Only with a reviewed reason.")
    args = ap.parse_args()
    path = Path(args.out)

    assert CKPT_DIR.is_dir(), f"reference weights not found: {CKPT_DIR}"
    fixture = build_fixture()
    m = fixture["metrics"]

    if path.exists() and not args.recapture:
        ref = torch.load(path, weights_only=False)
        bad = compare_fixtures(ref["metrics"], fixture["metrics"], "metrics")
        bad += compare_fixtures(ref["rollout"], fixture["rollout"], "rollout")
        if ref["meta"]["ckpt"] != fixture["meta"]["ckpt"]:
            print(f"NOTE: reference weights differ ({ref['meta']['ckpt']} -> "
                  f"{fixture['meta']['ckpt']}); a diff below is expected.")
        if bad:
            print(f"GOLDEN EVAL MISMATCH vs {path} ({len(bad)} leaves):")
            for b in bad[:20]:
                print("  ", b)
            sys.exit(1)
        print(f"GOLDEN EVAL OK - bitwise identical to {path}")
        return

    if path.exists():
        print(f"WARNING: --recapture given; OVERWRITING baseline {path}")
    torch.save(fixture, path)
    print(f"saved {path}  (weights: {fixture['meta']['ckpt']})")
    print(f"  prnn_loss={float(m['prnn_loss']):.6e}  mi_policy={m['mi_policy']:.6f}")
    print(f"  sRSA={m['sRSA']:.6f}  SWdist={m['SWdist']:.6f}")
    print(f"  SI zeroed {int(m['SI_units_zeroed'])}/{int(m['SI_units_total'])}"
          f"  mean(active)={m['SI_mean_active_only']:.6f}")


if __name__ == "__main__":
    main()

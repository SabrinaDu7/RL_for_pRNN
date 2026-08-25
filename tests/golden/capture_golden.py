"""Capture a golden fixture of the current (pre-refactor) training behavior.

Mirrors the construction path of trainRL_Adel.py for the mainline config
(pRNN + ACModelSR + PredictivePPOAlgo, curious agent, pRNN co-trained), but:
- forced to CPU for determinism,
- small frames/seqdur so it runs in seconds,
- no hydra/wandb.

Run (GATE - compares, never writes):
    uv run python tests/golden/capture_golden.py
Exits 1 and prints the diverging leaves on any bitwise difference.

Re-baseline (explicit, and only with a reviewed reason for the dynamics to
have changed):
    uv run python tests/golden/capture_golden.py --recapture

A baseline that silently re-baselines is not a gate: this script used to
`torch.save` unconditionally, so running it the obvious way overwrote
`golden_v1.pt` with whatever the current code produced and the "gate" then
passed vacuously forever. Compare is now the default.

The code must reproduce these tensors exactly (same seed => same RNG
consumption order) while the `reward_alignment=legacy` default holds.

FIXTURE VERSIONS - each bump is a REVIEWED dynamics change, never a repair:

    golden_v0.pt  pre-migration stack (SabrinaDu7 prnn). Kept for the legacy tree.
    golden_v1.pt  post-migration, valid up to and including `37aaa1b`.
    golden_v2.pt  from `d275149` on. <- current

`d275149` ("rl: remove dead `recurrence`") removed a code path that silently
dropped one transition on odd epochs, so the policy minibatches changed and
with them the update statistics and the weights. Measured 2026-08-25: v1
compares OK at `d275149^` and mismatches on 17 leaves at `d275149`, on the
IDENTICAL 17 leaves at `ba87d81` - so that commit is the only one since that
moved these numerics. Round 0's ROLLOUT is bitwise unchanged (curious_rewards,
advantages, actions, log_probs, SRs, locs); only the update and everything
downstream of it moved.

The bump went unnoticed for three days because nothing ran this file.
`tests/golden/test_golden.py` now does.
"""

import numpy as np
import torch

import curious_george as RLutils
from curious_george import ACModelSR, PredictivePPOAlgo
from prnn.utils import PredictiveNet

SEED = 2
FRAMES = 64
SEQDUR = 32
UPDATES = 2
DEVICE = torch.device("cpu")
OUT = "tests/golden/golden_v2.pt"  # see FIXTURE VERSIONS in the module docstring


def build_fixture() -> dict:
    RLutils.seed(SEED)

    env = RLutils.make_env(
        env_key="MiniGrid-LRoom-v0",
        input_type="pRNN",
        seed=SEED + 10000,
        act_enc="SpeedHD",
    )

    obs_space, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)

    predictiveNet = PredictiveNet(
        env,
        hidden_size=500,
        pRNNtype="thRNN_5win",
        learningRate=3e-3,
        bptttrunc=1e8,
        weight_decay=3e-3,
        neuralTimescale=2,
        dropp=0.15,
        trainNoiseMeanStd=(0, 0.05),
        f=0.5,
        wandb_log=False,
    )
    predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0, 0])

    acmodel = ACModelSR(
        obs_space,
        env.action_space,
        predictiveNet.hidden_size,
        False,  # with_obs
        True,   # rgb
        True,   # with_HD
    )
    acmodel.to(DEVICE)

    pastSR = "prevAct" not in predictiveNet.pRNNtype
    assert pastSR

    algo = PredictivePPOAlgo(
        env,
        acmodel,
        predictiveNet,
        DEVICE,
        num_frames=FRAMES,
        discount=0.98,
        lr=3e-4,
        gae_lambda=0.95,
        entropy_coef=0.0,
        value_loss_coef=1,
        max_grad_norm=0.5,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=16,  # frames=64 -> 4 minibatches
        preprocess_obss=preprocess_obss,
        train_pN=True,
        noise_mu=0,
        noise_std=0.05,
        prnn_seqdur=SEQDUR,
        intrinsic=False,
        k_int=1,
        pastSR=pastSR,
        curious_agent=True,
        k_curious=1,
    )

    rounds = []
    for _ in range(UPDATES):
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps=exps, update_params=True)
        rounds.append(
            {
                "curious_rewards": algo.curious_rewards.clone(),
                "rewards": algo.rewards.clone(),
                "advantages": algo.advantages.clone(),
                "values": algo.values.clone(),
                "actions": algo.actions.clone(),
                "log_probs": algo.log_probs.clone(),
                "SRs": exps.SR.clone(),
                "locs": list(logs1["locs"]),
                "policy_loss": logs2["policy_loss"],
                "value_loss": logs2["value_loss"],
                "grad_norm": logs2["grad_norm"],
            }
        )

    fixture = {
        "meta": {
            "seed": SEED,
            "frames": FRAMES,
            "seqdur": SEQDUR,
            "updates": UPDATES,
            "torch": torch.__version__,
        },
        "rounds": rounds,
        "acmodel_state": {k: v.clone() for k, v in acmodel.state_dict().items()},
        "prnn_state": {k: v.clone() for k, v in predictiveNet.pRNN.state_dict().items()},
    }
    return fixture


def compare_fixtures(ref, new, path: str = "") -> list[str]:
    """Bitwise-compare two fixtures; returns a list of mismatch descriptions."""
    bad: list[str] = []
    if isinstance(ref, dict):
        missing = set(ref) ^ set(new)
        if missing:
            bad.append(f"{path}: key set differs ({sorted(missing)})")
        for k in set(ref) & set(new):
            bad += compare_fixtures(ref[k], new[k], f"{path}.{k}")
    elif isinstance(ref, (list, tuple)):
        if len(ref) != len(new):
            bad.append(f"{path}: length {len(ref)} != {len(new)}")
        else:
            for i, (a, b) in enumerate(zip(ref, new)):
                bad += compare_fixtures(a, b, f"{path}[{i}]")
    elif torch.is_tensor(ref):
        if not torch.equal(ref, new):
            bad.append(f"{path}: max|d|={(ref - new).abs().max().item():.3e}")
    elif ref != new:
        bad.append(f"{path}: {ref!r} != {new!r}")
    return bad


def main():
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--recapture",
        action="store_true",
        help="OVERWRITE the baseline. Only with a deliberate, reviewed reason "
        "for the dynamics to have changed - a baseline that silently "
        "re-baselines is not a gate.",
    )
    args = ap.parse_args()
    path = Path(args.out)

    fixture = build_fixture()
    r0 = fixture["rounds"][0]

    # DEFAULT IS COMPARE, NEVER WRITE. Re-baselining must be explicit.
    if path.exists() and not args.recapture:
        ref = torch.load(path, weights_only=False)
        bad = compare_fixtures(ref["rounds"], fixture["rounds"], "rounds")
        bad += compare_fixtures(ref["acmodel_state"], fixture["acmodel_state"], "acmodel")
        bad += compare_fixtures(ref["prnn_state"], fixture["prnn_state"], "prnn")
        if ref["meta"]["torch"] != fixture["meta"]["torch"]:
            print(
                f"NOTE: torch {ref['meta']['torch']} -> {fixture['meta']['torch']}; "
                "a bitwise diff across torch versions may be expected."
            )
        if bad:
            print(f"GOLDEN MISMATCH vs {path} ({len(bad)} leaves):")
            for b in bad[:20]:
                print("  ", b)
            if len(bad) > 20:
                print(f"   ... and {len(bad) - 20} more")
            sys.exit(1)
        print(f"GOLDEN OK - bitwise identical to {path}")
        return

    if path.exists():
        print(f"WARNING: --recapture given; OVERWRITING baseline {path}")
    torch.save(fixture, path)
    print(f"saved {path}")
    print(f"round0 curious_rewards mean={r0['curious_rewards'].mean():.6e}")
    print(f"round0 advantages mean={r0['advantages'].mean():.6e}")
    print(f"round0 first 5 locs: {r0['locs'][:5]}")


if __name__ == "__main__":
    main()

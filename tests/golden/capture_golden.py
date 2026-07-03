"""Capture a golden fixture of the current (pre-refactor) training behavior.

Mirrors the construction path of trainRL_Adel.py for the mainline config
(pRNN + ACModelSR + PredictivePPOAlgo, curious agent, pRNN co-trained), but:
- forced to CPU for determinism,
- small frames/seqdur so it runs in seconds,
- no hydra/wandb.

Run:  uv run python tests/golden/capture_golden.py
Writes: tests/golden/golden_v0.pt

The refactored code must reproduce these tensors exactly (same seed => same
RNG consumption order) while the `reward_alignment=legacy` default holds.
"""

import numpy as np
import torch

import RLutils
from RLutils import ACModelSR, PredictivePPOAlgo
from prnn.utils import PredictiveNet

SEED = 2
FRAMES = 64
SEQDUR = 32
UPDATES = 2
DEVICE = torch.device("cpu")
OUT = "tests/golden/golden_v0.pt"


def main():
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

    pastSR = not ("prevAct" in str(predictiveNet.pRNN))
    assert pastSR

    algo = PredictivePPOAlgo(
        env,
        acmodel,
        predictiveNet,
        DEVICE,
        FRAMES,          # num_frames
        0.98,            # discount
        3e-4,            # lr
        0.95,            # gae_lambda
        0.0,             # entropy_coef
        1,               # value_loss_coef
        0.5,             # max_grad_norm
        1,               # recurrence
        1e-8,            # adam_eps
        0.2,             # clip_eps
        4,               # epochs
        16,              # batch_size (frames=64 -> 4 minibatches)
        preprocess_obss,
        None,            # place_cells
        None,            # cann
        True,            # train_pN
        0,               # noise_mu
        0.05,            # noise_std
        SEQDUR,          # prnn_seqdur
        False,           # intrinsic
        1,               # k_int
        pastSR,
        True,            # curious_agent
        1,               # k_curious
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
    torch.save(fixture, OUT)
    r0 = rounds[0]
    print(f"saved {OUT}")
    print(f"round0 curious_rewards mean={r0['curious_rewards'].mean():.6e}")
    print(f"round0 advantages mean={r0['advantages'].mean():.6e}")
    print(f"round0 first 5 locs: {r0['locs'][:5]}")


if __name__ == "__main__":
    main()

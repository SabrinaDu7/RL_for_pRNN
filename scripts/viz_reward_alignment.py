# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: RL_for_pRNN
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reward alignment visualization (legacy vs next_obs, pastSR True vs False)
#
# For each visualized timestep t this shows: the env state where action a_t was
# taken, the SR (pRNN hidden state) the policy used to pick a_t, and the
# reward-pass target/prediction/hidden-state that produce the curiosity reward
# credited to a_t under the chosen alignment.
#
# ## Information flow, pastSR=True (mainline: thRNN_5win + SpeedHD)
#
# One rollout step t:
# 1. Policy forward: `acmodel(obs_t, SR=SR_t)` — SR_t is the pRNN hidden state
#    as of the END of step t-1 (zeros right after an episode reset).
# 2. `a_t = dist.sample()`.
# 3. `obs_{t+1}, r_t = env.step(a_t)`.
# 4. Hidden-state update (in code AFTER env.step, but consuming the PRE-action
#    obs): `h <- cell(h, [obs_t * inMask[phase], SpeedHD(a_t, HD_t) * actMask[phase]])`,
#    phase advances; `SR_{t+1} = h`. obs_{t+1} enters the pRNN only at step
#    t+1's update — the SR lags one position behind the agent ("past" SR).
# 5. On episode done: `reset_state(randInit)` (h <- noise), SR <- zeros,
#    `env.reset()`.
# 6. Curiosity rewards, RETROSPECTIVELY per episode: one `predict()` pass;
#    prediction row i targets obs_i and its input step is (obs_i*inMask, a_i).
#    - legacy: reward for a_i = row i (error on the PRE-action obs).
#    - next_obs: reward for a_i = row i+1 (error on the obs a_i produced; the
#      final row comes from an appended zero-action step feeding last_obs).
#
# ## pastSR=False (thRNN_5win_prevAct + SpeedNextHD)
#
# Step 4 differs: the update consumes the POST-action obs,
# `h <- cell(h, [obs_{t+1}*inMask, enc(a_t, HD_{t+1})])` (the architecture
# right-shifts actions internally, actOffset=1), so SR_{t+1} aligns to the
# agent's NEW position. In the reward pass, row i's input action is a_{i-1}:
# legacy then credits a_i with an error driven by a_{i-1}, while next_obs
# (row i+1: predict obs_{i+1} with a_i as the shifted action input) is the
# causally matched choice for BOTH families.
#
# Note: only the pastSR=True family has a trained checkpoint in .env
# (PRNN_CUR_CKPT, thRNN_5win). The prevAct flow uses a fresh net, so its
# predictions are untrained — the indexing/flow is what to look at there.

# %%
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames, PredictiveNet, load_pN

from curious_george import AgentInputType, get_env_var, make_env
from curious_george import seed as seed_everything
from curious_george.envs.access import (
    ACTION_NAMES,
    hidden_image,
    obs_image,
    pred_image,
    render_env,
)
from curious_george.rl.update.rewards import REWARD_ALIGNMENTS
from curious_george.world_model.adapter import PRNNAdapter, infer_past_sr

plt.rcParams["figure.dpi"] = 150

DEVICE = torch.device("cpu")
SAVE_DIR = Path("../outputs/reward_alignment_viz")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def build(past_sr: bool, use_ckpt: bool = True, hidden_size: int = 500, seed: int = 2):
    """Env + pRNN + adapter for one convention. Trained ckpt exists only for
    the pastSR=True (thRNN_5win) family."""
    seed_everything(seed)
    prnn_type = "thRNN_5win" if past_sr else "thRNN_5win_prevAct"
    act_enc = ActionEncodingsEnum.SpeedHD.value if past_sr else "SpeedNextHD"

    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=act_enc,
        seed=seed + 10000,
    )
    pN = PredictiveNet(
        env,
        hidden_size=hidden_size,
        pRNNtype=prnn_type,
        trainNoiseMeanStd=(0, 0.05),
        wandb_log=False,
    )
    pN.env_shell.hd_trans = np.array([-1, 1, 0, 0])
    if use_ckpt and past_sr:
        load_pN(
            model_ckpt_filepath=get_env_var("PRNN_CUR_CKPT"),
            device=DEVICE,
            pRNNtype=prnn_type,
            predictive_net=pN,
        )
    pN.pRNN.eval()

    assert infer_past_sr(pN) is past_sr
    adapter = PRNNAdapter(pN, DEVICE, past_sr)
    return env, adapter


@dataclass
class Stream:
    """One collected episode: everything needed to visualize any timestep."""

    obss: list  # obss[t] = obs BEFORE action t
    acts: np.ndarray  # acts[t]
    last_obs: dict  # obs after the final action
    renders: list  # renders[t] = env frame where action t was taken (+ final frame)
    srs: list  # srs[t] = SR the policy would use to pick action t


def collect_stream(env, adapter: PRNNAdapter, ep_len: int, seed: int = 0) -> Stream:
    """Random-action episode, updating the SR exactly like the collector does
    (pre-action obs for pastSR nets, post-action obs for prevAct nets)."""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    adapter.reset_state()
    sr = adapter.init_sr(obs)

    obss, acts, srs, renders = [], [], [], [render_env(env)]
    for _ in range(ep_len):
        a = np.array([int(rng.integers(0, 4))])
        obss.append(obs)
        acts.append(a[0])
        srs.append(sr.clone())

        obs_next = env.step(a)[0]
        sr = adapter.next_sr(a, obs if adapter.pastSR else obs_next)
        obs = obs_next
        renders.append(render_env(env))

    return Stream(obss, np.array(acts), obs, renders, srs)


def show_alignment(
    env,
    adapter: PRNNAdapter,
    stream: Stream,
    alignment: str,
    timesteps: list[int],
    save: bool = True,
):
    """5 rows x len(timesteps) cols:
    (1) env state where a_t was taken, (2) SR used to pick a_t,
    (3) reward-pass TARGET obs, (4) reward-pass PREDICTED obs (+ reward),
    (5) reward-pass hidden state at that prediction row."""
    off = REWARD_ALIGNMENTS[alignment]
    L = len(stream.acts)
    assert all(0 <= t < L for t in timesteps), f"timesteps must be in [0, {L - 1}]"

    pred, target, hidden, mses = adapter.episode_prediction_rows(
        stream.obss,
        stream.acts,
        stream.last_obs,
        target_offset=off,
    )

    n = len(timesteps)
    fig, axes = plt.subplots(5, n, figsize=(3.2 * n, 13.5), squeeze=False)
    for col, t in enumerate(timesteps):
        tgt_idx = t + off
        tgt_label = f"obs[{tgt_idx}]" if tgt_idx < L else f"last_obs (obs[{L}])"

        axes[0, col].imshow(stream.renders[t])
        axes[0, col].set_title(
            f"t={t}   a_t={ACTION_NAMES[stream.acts[t]]}", fontsize=10
        )

        axes[1, col].imshow(hidden_image(stream.srs[t]), cmap="viridis")
        axes[1, col].set_title(f"SR used to pick a_{t}", fontsize=9)

        axes[2, col].imshow(pred_image(adapter.pN.env_shell, target[t]))
        axes[2, col].set_title(f"target: {tgt_label}", fontsize=9)

        axes[3, col].imshow(pred_image(adapter.pN.env_shell, pred[t]))
        axes[3, col].set_title(f"prediction  (reward={mses[t]:.4f})", fontsize=9)

        axes[4, col].imshow(hidden_image(hidden[t]), cmap="viridis")
        axes[4, col].set_title("h at prediction row", fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for row, lbl in enumerate(
        [
            "env state",
            "policy SR",
            "reward target",
            "reward prediction",
            "reward-pass h",
        ]
    ):
        axes[row, 0].set_ylabel(lbl, fontsize=10)

    fig.suptitle(
        f"alignment={alignment}   pastSR={adapter.pastSR}   episode length={L}",
        fontsize=12,
    )
    fig.tight_layout()
    if save:
        fig.savefig(
            SAVE_DIR / f"pastSR{adapter.pastSR}_{alignment}.png", bbox_inches="tight"
        )
    return fig


# %% [markdown]
# ## Configuration — play with these
# %%
EP_LEN = 15  # episode length (rewards computed over this one episode)
TIMESTEPS = [10, 11, 12, 13]
STREAM_SEED = 0
USE_CKPT = True  # trained thRNN_5win ckpt for the pastSR=True flow

# %% [markdown]
# ## pastSR = True (mainline, trained ckpt): legacy vs next_obs
#
# Same stream for both alignments — compare column by column: under `legacy`
# the target at t is the PRE-action obs; under `next_obs` it is the obs the
# action produced (at t = EP_LEN-1 the target is `last_obs`, predicted via the
# appended zero-action step).
# %%
if __name__ == "__main__":
    env_p, adapter_p = build(past_sr=True, use_ckpt=USE_CKPT)
    stream_p = collect_stream(env_p, adapter_p, EP_LEN, seed=STREAM_SEED)

# %%
if __name__ == "__main__":
    show_alignment(env_p, adapter_p, stream_p, "legacy", TIMESTEPS)
    plt.show()

# %%
if __name__ == "__main__":
    show_alignment(env_p, adapter_p, stream_p, "next_obs", TIMESTEPS)
    plt.show()

# %% [markdown]
# ## pastSR = False (thRNN_5win_prevAct + SpeedNextHD, fresh net)
#
# Untrained predictions — look at the indexing, not the pixels: the SR row now
# aligns to the agent's NEW position, and in the reward pass row i's action
# input is a_{i-1}.
# %%
if __name__ == "__main__":
    env_n, adapter_n = build(past_sr=False, use_ckpt=True)
    stream_n = collect_stream(env_n, adapter_n, EP_LEN, seed=STREAM_SEED)

# %%
if __name__ == "__main__":
    show_alignment(env_n, adapter_n, stream_n, "legacy", TIMESTEPS)
    plt.show()

# %%
if __name__ == "__main__":
    show_alignment(env_n, adapter_n, stream_n, "next_obs", TIMESTEPS)
    plt.show()

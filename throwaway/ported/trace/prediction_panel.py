"""What does the pRNN's next-observation prediction actually look like, and where?

The world model is trained on one number - next-observation MSE - and every
speed and regime decision in docs/exp_speed_cuda_graph_2026-08-19.md was gated
on it. This renders what that number is actually measuring: ground truth beside
prediction, at many DISTINCT room locations rather than along one trajectory.

Distinct locations matter. `PredictiveNet.plotSampleTrajectory` (which is what
wandb's "Observation Sequence" shows) samples consecutive timesteps of a single
rollout, so its panels are usually a few neighbouring cells facing the same way.
That answers "is prediction working", not "is it working EVERYWHERE".

Usage:
    uv run python scripts/trace/prediction_panel.py \\
        --ckpt outputs/<run>/predictiveNet_state.pt --env lroom --n-locs 12

Writes outputs/summary/fig_prediction_panel[_<tag>].png plus a sibling .json
carrying the checkpoint path, env, the chosen cells and the per-cell MSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float

from prnn.utils.checkpoints import CkptKeys

from curious_george.world_model.device import eval_mode, on_device


def load_net(ckpt: str, env_cfg: str, layouts: str | None, room: int):
    """The run's own pRNN and env shell, built through the ordinary setup path.

    hiddensize is read FROM the checkpoint rather than the config: a config
    mismatch here silently builds the wrong-shaped net and load_pN then fails
    with a shape error that reads like a corrupt file.
    """
    from hydra import compose, initialize_config_dir

    hidden = int(torch.load(ckpt, map_location="cpu", weights_only=False)[CkptKeys.HIDDEN_SIZE])
    overrides = [f"env={env_cfg}", "logging.wandb_log=false"]
    if layouts:
        overrides.append(f"exp.layouts={layouts}")
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    cfg.predNet.hiddensize = hidden

    # Multi-room envs refuse to build without landmarks. Resolve them with the
    # SAME resolve_layouts the training loop uses and hand them to the shell
    # directly, exactly as scripts/multienv/checkpoint_curve.py::build does -
    # naming a room by hand is how a square run gets scored in an L-room.
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import get_pN, make_env
    from curious_george.envs.layouts import resolve_layouts

    rooms = resolve_layouts(cfg)
    kw = {"landmarks": list(rooms[room].landmarks)} if rooms else {}
    env = make_env(
        env_key=cfg.exp.env_name, input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, **kw,
    )
    pN = get_pN(args=cfg, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False
    return pN, env, cfg


def collect(pN, env, steps: int, seed: int):
    """One random-action rollout, with predictions and per-step ground truth."""
    from curious_george.storage import get_agent
    from curious_george.utils.enums import AgentType

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Same forward-biased random policy checkpoint_curve.py uses for its probe.
    # A uniform random walk turns on the spot and barely leaves its start cell,
    # which would defeat the point of asking about DIFFERENT parts of the room.
    agent = get_agent(env=env, agent_Type=AgentType.RANDOM,
                      rand_act_prob=np.array([0.15, 0.15, 0.6, 0.1]))
    with eval_mode([pN]), on_device([pN], "cpu"), torch.no_grad():
        obs, act, state, render = pN.collectObservationSequence(
            env, agent, steps, includeRender=True
        )
        obs_pred, obs_next, _ = pN.predict(obs, act)
    if isinstance(obs_pred, tuple):
        obs_pred = obs_pred[1]
    return obs_pred, obs_next, state, render


def pick_distinct(pos: Float[np.ndarray, "T 2"], dirs, n: int) -> list[int]:
    """Timestep indices at n distinct (cell, heading) pairs, spread over the room.

    Greedy farthest-point on grid position, so the panel covers the L-room's
    arms rather than clustering wherever the random walk dawdled.
    """
    seen: dict[tuple, int] = {}
    for t, (p, d) in enumerate(zip(pos, dirs)):
        seen.setdefault((int(p[0]), int(p[1]), int(d)), t)
    cells = np.array([[k[0], k[1]] for k in seen], dtype=float)
    idx = list(seen.values())
    if len(idx) <= n:
        return sorted(idx)

    chosen = [int(np.argmax(cells[:, 0] + cells[:, 1]))]  # a corner, deterministically
    while len(chosen) < n:
        d = np.min(
            np.linalg.norm(cells[:, None, :] - cells[None, chosen, :], axis=-1), axis=1
        )
        chosen.append(int(np.argmax(d)))
    return sorted(idx[c] for c in chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", default="lroom")
    ap.add_argument("--layouts", default=None, choices=("one", "rooms", "pool"),
                    help="multi-room envs only; picks the layout SET")
    ap.add_argument("--room", type=int, default=0, help="which room of that set")
    ap.add_argument("--n-locs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pN, env, cfg = load_net(args.ckpt, args.env, args.layouts, args.room)
    obs_pred, obs_next, state, render = collect(pN, env, args.steps, args.seed)

    pos, dirs = np.asarray(state["agent_pos"]), np.asarray(state["agent_dir"])
    n_t = obs_pred.shape[1]
    pos, dirs = pos[:n_t], dirs[:n_t]
    ts = pick_distinct(pos, dirs, args.n_locs)

    truth = env.pred2np(obs_next, timesteps=ts)
    pred = env.pred2np(obs_pred, timesteps=ts)
    mse = [float(np.mean((pred[i] - truth[i]) ** 2)) for i in range(len(ts))]

    n = len(ts)
    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.2), squeeze=False)
    # NOT env.show_state(): its signature is show_state(self, render, t, **kwargs)
    # and the body is plt.imshow(render[t]) - it accepts `ax`/`fig` and silently
    # ignores them, drawing on the global current axes. Passing ax there leaves
    # the intended subplot blank, which is how the first version of this figure
    # came out with an empty top row.
    for i, t in enumerate(ts):
        if render is not None:
            axes[0][i].imshow(render[t])
        axes[0][i].set_title(f"({pos[t][0]},{pos[t][1]}) d{dirs[t]}", fontsize=7)
        axes[1][i].imshow(truth[i])
        axes[2][i].imshow(pred[i])
        axes[2][i].set_xlabel(f"MSE {mse[i]:.4f}", fontsize=6)
        for r in range(3):
            axes[r][i].set_xticks([])
            axes[r][i].set_yticks([])
    for r, name in enumerate(["room + agent", "TRUE next obs", "PREDICTED next obs"]):
        axes[r][0].set_ylabel(name, fontsize=8)

    head = args.label or Path(args.ckpt).parent.name
    fig.suptitle(
        f"pRNN next-observation prediction at {n} distinct locations\n"
        f"{head}   |   mean MSE over shown cells {np.mean(mse):.5f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = Path("outputs/summary")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"fig_prediction_panel{'_' + args.tag if args.tag else ''}"
    png = out / f"{stem}.png"
    fig.savefig(png, dpi=160)
    (out / f"{stem}.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt),
                "env": args.env,
                "layouts": args.layouts,
                "room": args.room,
                "env_name": cfg.exp.env_name,
                "hiddensize": int(cfg.predNet.hiddensize),
                "seed": args.seed,
                "rollout_steps": args.steps,
                "cells": [
                    {"t": int(t), "pos": [int(pos[t][0]), int(pos[t][1])],
                     "dir": int(dirs[t]), "mse": m}
                    for t, m in zip(ts, mse)
                ],
                "mean_mse_shown": float(np.mean(mse)),
            },
            indent=2,
        )
    )
    print(f"wrote {png}")
    print(f"mean MSE over {n} shown cells: {np.mean(mse):.6f}  "
          f"(min {min(mse):.6f}, max {max(mse):.6f})")


if __name__ == "__main__":
    main()

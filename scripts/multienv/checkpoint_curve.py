"""Prediction loss and per-room spatial metrics across a run's checkpoint series.

Reads the step-tagged archive a multi-room run writes under `<run>/checkpoints/`
and replays ONE fixed probe through every checkpoint, so differences between
points are in the weights and nothing else. That is the same "replay, don't
re-collect" rule the trace work runs on (`scripts/trace/trace_probe.py`).

Two jobs:
  - the local gate: is prediction loss actually going down?
  - babysitting a cluster run: are sRSA still high, SWdist still low, and is the
    prediction still correct - without waiting for the job to finish.

    uv run python scripts/multienv/checkpoint_curve.py --run outputs/<run-dir>

Reports one row per checkpoint. A RUNNING job's newest checkpoint is a
checkpoint, not a result: the row is printed with `(latest)` so it is never
mistaken for a finished number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROBE_SEED = 20260813
ONSET = 20


def archived(run_dir: Path) -> list[tuple[int, Path]]:
    """(environment step, path) for every archived checkpoint, oldest first."""
    out = []
    for p in sorted((run_dir / "checkpoints").glob("predictiveNet_state_step*.pt")):
        out.append((int(p.stem.split("step")[-1]), p))
    return out


def build(*, env_name: str, landmarks, ckpt: str, hiddensize: int | None = None):
    from hydra import compose, initialize_config_dir
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import get_pN, make_env

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    if hiddensize is not None:
        # get_pN builds the net from the CONFIG, so a checkpoint trained at a
        # different width silently fails to load without this.
        args.predNet.hiddensize = hiddensize
    env = make_env(
        env_key=env_name, input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, landmarks=list(landmarks),
    )
    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False
    return pN, env, args


def fixed_probe(*, pN, env, layout, n_trajs: int, steps: int):
    """Rollouts collected ONCE per room and reused for every checkpoint.

    Observations depend on the room and the actions, never on the network, so a
    single collection is valid for the whole series - which is what makes two
    checkpoints comparable rather than each carrying its own rollout noise.
    """
    from curious_george.utils.enums import AgentType
    from curious_george.storage import get_agent

    env.env.unwrapped.landmarks = list(layout.landmarks)
    env.env.reset(seed=PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    agent = get_agent(env=env, agent_Type=AgentType.RANDOM,
                      rand_act_prob=np.array([0.15, 0.15, 0.6, 0.1]))
    rolls = []
    for _ in range(n_trajs):
        obs, act, state, _ = pN.collectObservationSequence(env, agent, steps, discretize=True)
        rolls.append((obs, act, state["agent_pos"]))
    return rolls


def score(*, pN, rolls, env) -> tuple[float, np.ndarray, np.ndarray]:
    """(prediction loss, pooled h rows, pooled positions) over the probe."""
    losses, h_rows, pos_rows = [], [], []
    with torch.no_grad():
        for obs, act, pos in rolls:
            obs_pred, obs_next, h = pN.predict(obs, act)
            losses.append(float(pN.loss_fn(obs_pred, obs_next, h)))
            h_rows.append(torch.mean(h, dim=0)[ONSET:].numpy())
            pos_rows.append(pos[ONSET:-1, :])
    return float(np.mean(losses)), np.concatenate(h_rows), np.concatenate(pos_rows).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--layouts", default="rooms", choices=("rooms", "pool"))
    ap.add_argument("--pool-size", type=int, default=500)
    ap.add_argument("--pool-seed", type=int, default=20260813)
    ap.add_argument("--rooms-scored", type=int, default=3, help="pool runs: rooms to score")
    ap.add_argument("--n-trajs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--spatial", action="store_true", help="also compute sRSA/SWdist (slower)")
    ap.add_argument("--hiddensize", type=int, default=None,
                    help="override predNet.hiddensize to match the checkpoint")
    a = ap.parse_args()

    from curious_george.envs.layouts import ROOMS_RUN1, base_walkable, generate_layouts

    run_dir = Path(a.run)
    points = archived(run_dir)
    if not points:
        raise SystemExit(f"no archived checkpoints under {run_dir / 'checkpoints'}")

    if a.layouts == "rooms":
        rooms = list(ROOMS_RUN1)
    else:
        rooms = generate_layouts(walkable=base_walkable(), n=a.pool_size,
                                 seed=a.pool_seed)[: a.rooms_scored]
    print(f"{len(points)} checkpoints, {len(rooms)} rooms scored, "
          f"{a.n_trajs} probe trajectories x {a.steps} steps")

    pN, env, _ = build(env_name="MiniGrid-LRoom-Multi-v0",
                       landmarks=rooms[0].landmarks, ckpt=str(points[0][1]),
                       hiddensize=a.hiddensize)
    probes = [fixed_probe(pN=pN, env=env, layout=r, n_trajs=a.n_trajs, steps=a.steps)
              for r in rooms]

    header = f"{'step':>12} " + " ".join(f"{'loss@' + r.key[:4]:>12}" for r in rooms)
    if a.spatial:
        header += f" {'mean sRSA':>10} {'pooled':>8} {'remap':>8} {'SWdist':>8}"
    print(header)

    rows = []
    for i, (step, path) in enumerate(points):
        pN, env, _ = build(env_name="MiniGrid-LRoom-Multi-v0",
                           landmarks=rooms[0].landmarks, ckpt=str(path),
                           hiddensize=a.hiddensize)
        losses, hs, poss = [], [], []
        for r, probe in zip(rooms, probes):
            env.env.unwrapped.landmarks = list(r.landmarks)
            env.env.reset(seed=PROBE_SEED)
            loss, h, pos = score(pN=pN, rolls=probe, env=env)
            losses.append(loss); hs.append(h); poss.append(pos)
        line = f"{step:>12,} " + " ".join(f"{v:>12.6f}" for v in losses)
        row = {"step": step, "loss": losses}

        if a.spatial:
            per = [pN.calculateSpatialMetrics(h, p, env, wandb_nameext="")["sRSA"]
                   for h, p in zip(hs, poss)]
            pooled = pN.calculateSpatialMetrics(
                np.concatenate(hs), np.concatenate(poss), env, wandb_nameext="")
            mean_r = float(np.mean(per))
            line += (f" {mean_r:>10.4f} {float(pooled['sRSA']):>8.4f} "
                     f"{mean_r - float(pooled['sRSA']):>+8.4f} {float(pooled['SWdist']):>8.4f}")
            row |= {"mean_room_sRSA": mean_r, "pooled_sRSA": float(pooled["sRSA"]),
                    "remapping_index": mean_r - float(pooled["sRSA"]),
                    "SWdist": float(pooled["SWdist"])}
        if i == len(points) - 1:
            line += "   (latest)"
        print(line)
        rows.append(row)

    first, last = np.mean(rows[0]["loss"]), np.mean(rows[-1]["loss"])
    print(f"\nprediction loss {first:.6f} -> {last:.6f}  "
          f"({'DECREASING' if last < first else 'NOT decreasing'})")
    out = run_dir / "checkpoint_curve.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

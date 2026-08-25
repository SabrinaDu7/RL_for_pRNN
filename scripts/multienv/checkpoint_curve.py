"""Prediction loss and per-room spatial metrics across a run's checkpoint series.

Reads the step-tagged archive a multi-room run writes under `<run>/checkpoints/`
and replays ONE fixed probe through every checkpoint, so differences between
points are in the weights and nothing else. That is the same "replay, don't
re-collect" rule the probe runs on (`curious_george/evaluation/probe.py`).

Two jobs:
  - the local gate: is prediction loss actually going down?
  - babysitting a cluster run: are sRSA still high, SWdist still low, and is the
    prediction still correct - without waiting for the job to finish.

    uv run python scripts/multienv/checkpoint_curve.py --run outputs/<dir> --env lroom_multi
    uv run python scripts/multienv/checkpoint_curve.py --run outputs/<dir> --env squareroom_multi

THE ROOM COMES FROM THE RUN'S OWN CONFIG, NOT FROM THIS FILE. `--env` names the
config the job was launched with (`slurm/multienv.sh`'s second argument), and
every room-dependent quantity - the multi-room env id, the base room's walls,
the layout set, the pool size and seed, D4 dedup - is then read from it through
the SAME `resolve_layouts` the training loop uses. Re-specifying any of them
here is how a square run gets scored in an L-room: the two differ in walls, in
layout set and in whether layouts are deduplicated under the symmetries of the
square, and a mismatched score looks entirely plausible.

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


def checkpoint_hiddensize(ckpt: Path) -> int:
    """The width the checkpoint was trained at, read from the checkpoint.

    `get_pN` builds the net from the CONFIG, so a run at a different width fails
    to load with an opaque stack of torch shape mismatches. `save_pN` writes
    `hidden_size` into every checkpoint, so the answer is already in the file
    and does not need to be a flag the caller must know to pass.
    """
    from prnn.utils.checkpoints import CkptKeys

    return int(torch.load(ckpt, map_location="cpu", weights_only=False)[CkptKeys.HIDDEN_SIZE])


def run_config(*, env_cfg: str, layouts: str | None, hiddensize: int):
    """The launched run's config - the single source of room and layout set.

    Composed once: hydra's `initialize_config_dir` is not reentrant, and a
    per-checkpoint compose would also make the layout set a function of when it
    was asked for rather than of the config.
    """
    from hydra import compose, initialize_config_dir

    overrides = [f"env={env_cfg}"] + ([f"exp.layouts={layouts}"] if layouts else [])
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    cfg.predNet.hiddensize = hiddensize
    return cfg


def build(*, cfg, landmarks, ckpt: str):
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import get_pN, make_env

    env = make_env(
        env_key=cfg.exp.env_name, input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, landmarks=list(landmarks),
    )
    pN = get_pN(args=cfg, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False
    return pN, env


def fixed_probe(*, pN, env, layout, n_trajs: int, steps: int):
    """Rollouts collected ONCE per room and reused for every checkpoint.

    Observations depend on the room and the actions, never on the network, so a
    single collection is valid for the whole series - which is what makes two
    checkpoints comparable rather than each carrying its own rollout noise.
    """
    from curious_george.utils.enums import AgentType
    from curious_george.log_and_store.storage import get_agent

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
    ap.add_argument("--env", default="lroom_multi",
                    choices=("lroom_multi", "squareroom_multi"),
                    help="the env config the run was LAUNCHED with; everything "
                         "room-dependent is read from it")
    ap.add_argument("--layouts", default=None, choices=("one", "rooms", "pool"),
                    help="override exp.layouts; default is whatever the env config sets")
    ap.add_argument("--rooms-scored", type=int, default=None,
                    help="rooms to score; default is the config's exp.eval_rooms_max")
    ap.add_argument("--n-trajs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--spatial", action="store_true", help="also compute sRSA/SWdist (slower)")
    a = ap.parse_args()

    from curious_george.envs.layouts import BASE_ROOM_ID, resolve_layouts

    run_dir = Path(a.run)
    points = archived(run_dir)
    if not points:
        raise SystemExit(f"no archived checkpoints under {run_dir / 'checkpoints'}")

    cfg = run_config(env_cfg=a.env, layouts=a.layouts,
                     hiddensize=checkpoint_hiddensize(points[0][1]))
    layouts = resolve_layouts(cfg)
    if not layouts:
        raise SystemExit(f"env={a.env} resolves to no layouts (exp.layouts={cfg.exp.get('layouts')})")
    # A fixed PREFIX, matching what the training loop scores, so this series and
    # the wandb one are the same measurement.
    n_scored = a.rooms_scored or int(cfg.exp.get("eval_rooms_max", 8))
    rooms = layouts[:n_scored]
    print(f"{len(points)} checkpoints in {cfg.exp.env_name} "
          f"(base room {cfg.exp.get('room_id', BASE_ROOM_ID)}), "
          f"exp.layouts={cfg.exp.layouts}: {len(rooms)}/{len(layouts)} rooms scored, "
          f"{a.n_trajs} probe trajectories x {a.steps} steps")
    for r in rooms:
        print(f"    {r.key}  {r.describe()}")

    pN, env = build(cfg=cfg, landmarks=rooms[0].landmarks, ckpt=str(points[0][1]))
    probes = [fixed_probe(pN=pN, env=env, layout=r, n_trajs=a.n_trajs, steps=a.steps)
              for r in rooms]

    header = f"{'step':>12} " + " ".join(f"{'loss@' + r.key[:4]:>12}" for r in rooms)
    if a.spatial:
        header += f" {'mean sRSA':>10} {'pooled':>8} {'remap':>8} {'SWdist':>8}"
    print(header)

    rows = []
    for i, (step, path) in enumerate(points):
        pN, env = build(cfg=cfg, landmarks=rooms[0].landmarks, ckpt=str(path))
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
    # Metadata, so the series can say WHICH run it is. Without it a curve of
    # sRSA against step is indistinguishable between the L-room and square runs,
    # and between 3 rooms and a 500-room pool - which is exactly the confusion a
    # figure built from this file would inherit.
    out.write_text(json.dumps({
        "meta": {"run": run_dir.name, "env_config": a.env, "env_name": cfg.exp.env_name,
                 "room_id": cfg.exp.get("room_id", BASE_ROOM_ID),
                 "layouts": str(cfg.exp.layouts), "n_layouts": len(layouts),
                 "n_rooms_scored": len(rooms), "room_keys": [r.key for r in rooms],
                 "n_trajs": a.n_trajs, "steps": a.steps},
        "rows": rows}, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

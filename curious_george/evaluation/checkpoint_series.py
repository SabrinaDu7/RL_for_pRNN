"""Prediction loss and per-room spatial metrics across a run's checkpoint series.

Reads the step-tagged archive a multi-room run writes under `<run>/checkpoints/`
and replays ONE fixed probe through every checkpoint: the rollouts are collected
once per room and reused, so no point carries its own BEHAVIOURAL noise.

WHAT IS NOT PINNED, and it is not cosmetic. `score` wraps the forward in
`torch.no_grad()` alone, which stops gradients and NOT dropout. Every checkpoint
is therefore scored under a fresh dropout mask (predNet.dropout, 0.15), a fresh
noise draw (predNet.noisestd, 0.05) and an unpinned initial hidden state.
`probe.py` measures that wobble at ~0.4 in h between two identical calls, so
row-to-row differences here are NOT weights alone. `probe.py::replay_checkpoint`
is the same idea carried through - `eval_mode` around the forward, torch seeded
immediately before it so every checkpoint sees one realisation, and a fixed
initial state. Moving this file onto it is an OPEN ITEM, not an oversight.

Two jobs:
  - the local gate: is prediction loss actually going down?
  - babysitting a cluster run: are sRSA still high, SWdist still low, and is the
    prediction still correct - without waiting for the job to finish.

    uv run python -m curious_george.evaluation.checkpoint_series --run <run> --room lroom
    uv run python -m curious_george.evaluation.checkpoint_series --run <run> --room squareroom --source uniform

THE ROOM COMES FROM THE RUN'S OWN CONFIG, NOT FROM THIS FILE. `--room` and
`--source` name what the job was launched with, and every room-dependent
quantity - the multi-room env id, the base room's walls, the room set, the pool
size and seed, D4 dedup - is then read from the `Config` they build, through the
SAME `resolve_layouts` the training loop uses. Re-specifying any of them here is
how a square run gets scored in an L-room: the two differ in walls, in room set
and in whether rooms are deduplicated under the symmetries of the square, and a
mismatched score looks entirely plausible.

Reports one row per checkpoint. A RUNNING job's newest checkpoint is a
checkpoint, not a result: the row is printed with `(latest)` so it is never
mistaken for a finished number.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from curious_george.envs.layouts import BASE_ROOM_ID, SQUARE_ROOM_ID

PROBE_SEED = 20260813
ONSET = 20


def archived(run_dir: Path) -> list[tuple[int, Path]]:
    """(environment step, path) for every archived checkpoint, oldest first."""
    out = []
    for p in sorted((run_dir / "checkpoints").glob("predictiveNet_state_step*.pt")):
        out.append((int(p.stem.split("step")[-1]), p))
    return out


def archived_policies(run_dir: Path) -> dict[int, Path]:
    """environment step -> the archived POLICY checkpoint at that step.

    EMPTY FOR RUNS FINISHED BEFORE 2026-08-28, when `save_checkpoint` archived
    only the pRNN and kept the policy in one rolling file. That is why this is a
    lookup rather than a second column on `archived`: for an older run the pRNN
    series exists and the policy series does not, and a caller has to be able to
    see the difference instead of silently pairing a step-N world model with the
    last policy the run happened to write.

    An on-policy readout of an archived step is available exactly when this maps
    that step.
    """
    out: dict[int, Path] = {}
    for p in sorted((run_dir / "checkpoints").glob("policy_state_step*.pt")):
        out[int(p.stem.split("step")[-1])] = p
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


#: `--room` values -> the base room they name. The room ids themselves live in
#: layouts.py; this only maps the short name a launcher types.
ROOMS: dict[str, str] = {"lroom": BASE_ROOM_ID, "squareroom": SQUARE_ROOM_ID}

#: `--source` values -> how the room set is drawn, in the vocabulary
#: `curious_george.configs` actually uses. These used to be `one/rooms/pool`,
#: which named Hydra config groups that no longer exist. `selected` is the
#: committed production set (`multienv-fast`); pair it with `--impassable`
#: for that arm.
SOURCES: tuple[str, ...] = ("frozen", "one", "uniform", "selected")


def run_config(*, room: str, source: str, hiddensize: int, impassable: bool = False):
    """The launched run's config - the single source of room and room set.

    Built once, so the room set is a function of the config rather than of when
    it was asked for.
    """
    from dataclasses import replace

    from curious_george.configs import (
        Config,
        EnvBackend,
        EnvCfg,
        EnvShape,
        EvalCfg,
        EvalKind,
        Frozen,
        Selected,
        Uniform,
    )

    # `Frozen` picks the committed set that belongs to THIS shape, so the room
    # and its set cannot disagree - handing the L-room's set to a square room
    # was silently wrong when this branched on the room id itself.
    env = EnvCfg(
        shape=EnvShape(ROOMS[room]),
        source=(
            Uniform() if source == "uniform"
            else Selected(n=5, impassable=impassable) if source == "selected"
            else Frozen()
        ),
        indices=(0,) if source == "one" else None,
    )
    base = Config(
        env=env,
        collect=replace(Config().collect, backend=EnvBackend.DEVICE),
        eval=EvalCfg(evals=frozenset({EvalKind.SPATIAL_MULTIROOM})),
    )
    return replace(base, arch_prnn=replace(base.arch_prnn, hidden_size=hiddensize))


@dataclass(frozen=True)
class SeriesContext:
    """Every config read the series needs, resolved once.

    These reads used to sit inline in `main`, which is why four of them kept
    their pre-2026-08-26 spelling (`env.base_room`, `env.layouts`,
    `env.eval_rooms_max`, and `env_name.value` on what is now a plain `str`)
    and nothing caught it: reaching them required a run directory full of
    checkpoints, so no test ever did. Resolving them here makes them reachable
    from a `Config` alone - see tests/test_checkpoint_series.py.
    """

    env_name: str
    room: str
    source: object
    rooms: list
    n_layouts: int

    @classmethod
    def resolve(cls, cfg, *, rooms_scored: int | None) -> "SeriesContext":
        from curious_george.envs.layouts import resolve_layouts

        layouts = resolve_layouts(cfg)
        if not layouts:
            raise ValueError(f"{cfg.env.source!r} resolves to no rooms")
        # A fixed PREFIX, matching what the training loop scores, so this series
        # and the wandb one are the same measurement.
        n = rooms_scored or cfg.eval.rooms_max
        return cls(
            env_name=cfg.env.env_name,
            room=cfg.env.shape.room,
            source=cfg.env.source,
            rooms=list(layouts[:n]),
            n_layouts=len(layouts),
        )

    def describe(self, *, n_points: int, n_trajs: int, steps: int) -> str:
        return (
            f"{n_points} checkpoints in {self.env_name} (base room {self.room}), "
            f"source={self.source!r}: {len(self.rooms)}/{self.n_layouts} rooms "
            f"scored, {n_trajs} probe trajectories x {steps} steps"
        )

    def meta(self) -> dict:
        """Metadata so a curve can say WHICH run it is. Without it a plot of
        sRSA against step is indistinguishable between the L-room and square
        runs, and between 3 rooms and a 500-room pool."""
        return {
            "env_name": self.env_name,
            "room_id": self.room,
            "source": repr(self.source),
            "n_layouts": self.n_layouts,
            "n_rooms_scored": len(self.rooms),
            "room_keys": [r.key for r in self.rooms],
        }


def build(*, cfg, landmarks, ckpt: str):
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import get_pN, make_env

    env = make_env(
        env_key=cfg.env.env_name, input_type=AgentInputType.H_PO.value,
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
    from curious_george.log_and_store.storage import RAND_ACT_PROBA, get_agent

    env.env.unwrapped.landmarks = list(layout.landmarks)
    env.env.reset(seed=PROBE_SEED)
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    agent = get_agent(env=env, agent_Type=AgentType.RANDOM,
                      rand_act_prob=RAND_ACT_PROBA)
    rolls = []
    for _ in range(n_trajs):
        obs, act, state, _ = pN.collectObservationSequence(env, agent, steps, discretize=True)
        rolls.append((obs, act, state["agent_pos"]))
    return rolls


def exploration_points(run_dir: Path) -> list[tuple[int, Path, Path]]:
    """(step, pRNN checkpoint, policy checkpoint) where BOTH were archived.

    The on-policy readout condition: pairing a step-N world model with any
    other step's policy silently measures an agent that never existed
    (`archived_policies` says why the two series can differ).
    """
    policies = archived_policies(run_dir)
    return [(s, p, policies[s]) for s, p in archived(run_dir) if s in policies]


def score_exploration_series(
    *, cfg, run_dir: Path, collects_per_point: int = 2,
    thresholds: tuple[float, ...] = (0.5, 0.9),
) -> list[dict]:
    """Exploration of the ARCHIVED policy, one row per archived (pRNN, policy) pair.

    Reuses the run machinery end to end: each point rebuilds the full training
    stack through `setup_training` with `run.prnn_ckpt` / `run.policy_ckpt`
    pointed at the archives - the same loading path a resumed run uses - and
    scores `collects_per_point` real DEVICE-backend rollouts with the same
    `rollout_summary` the run logs online, so the offline series and the wandb
    series are the same measurement.

    Protocol, per the traps this module's own docstring documents: seeded by
    `cfg.run.seed` at every point (`setup_training` seeds everything), so all
    checkpoints see the SAME spawn/room schedule and rows differ by weights
    alone; wrapped in `eval_mode` (dropout off - the wobble `score` still
    carries); and the policy SAMPLES, because argmax is a different agent and
    for an exploration metric that difference is the measurement.
    """
    from dataclasses import replace

    from curious_george.evaluation.exploration import rollout_summary
    from curious_george.models.device import eval_mode
    from curious_george.training.setup import setup_training
    from curious_george.utils.checkpoints import StatusCkptKeys

    rows = []
    for step, prnn_path, policy_path in exploration_points(run_dir):
        status = torch.load(policy_path, map_location="cpu", weights_only=False)
        if StatusCkptKeys.MODEL_STATE.value not in status:
            raise ValueError(
                f"{policy_path.name} holds no policy weights - a RANDOM-agent "
                "run archives none. Its exploration is checkpoint-independent; "
                "read it off the run's own exploration/* series or the walker "
                "calibration (python -m curious_george.envs.action_graph)."
            )
        point_cfg = replace(cfg, run=replace(
            cfg.run, prnn_ckpt=prnn_path, policy_ckpt=policy_path, wandb=False,
        ))
        comps = setup_training(point_cfg)
        algo = comps.algo
        positions, layout_ids = [], []
        with eval_mode([comps.predictiveNet, comps.acmodel]):
            for _ in range(collects_per_point):
                algo.collect_experiences()
                positions.append(algo.positions_episodes)
                layout_ids.append(torch.as_tensor(
                    algo.segment_layouts.reshape(-1),
                    device=algo.positions_episodes.device,
                ))
        summary = rollout_summary(
            positions=torch.cat(positions),
            layout_ids=torch.cat(layout_ids),
            supports=algo.room_supports,
            denominators=algo.room_walkable_counts,
            thresholds=thresholds,
        )
        rows.append({"step": step, "episodes": sum(len(p) for p in positions), **summary})
        comps.envs.close()
    return rows


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


def exploration_main(*, cfg, run_dir: Path, args) -> None:
    """The --exploration branch: one row per archived (pRNN, policy) pair."""
    from dataclasses import replace

    pairs = exploration_points(run_dir)
    if not pairs:
        raise SystemExit(
            f"no archived (pRNN, policy) PAIRS under {run_dir / 'checkpoints'} - "
            "runs finished before 2026-08-28 archived only the pRNN "
            "(see archived_policies)."
        )
    # The eval's own collection shape - a protocol choice, not the run's - and
    # no curiosity pass: this is a readout, nothing consumes rewards.
    cfg = replace(
        cfg,
        collect=replace(cfg.collect, num_envs=args.num_envs,
                        episodes_per_env=1, episode_steps=args.steps),
        train_policy=replace(cfg.train_policy, curious=False),
        run=replace(cfg.run, seed=PROBE_SEED),
    )
    print(f"{len(pairs)} archived (pRNN, policy) pairs; "
          f"{args.collects} x {args.num_envs} episodes x {args.steps} steps per point, "
          f"seed {PROBE_SEED}, eval_mode, sampled actions")
    rows = score_exploration_series(
        cfg=cfg, run_dir=run_dir, collects_per_point=args.collects,
    )
    header = (f"{'step':>12} {'episodes':>9} {'coverage':>9} {'nAUC':>7} "
              f"{'T50 reach':>10} {'T50 steps':>10} {'T90 reach':>10} {'T90 steps':>10} "
              f"{'room entropy':>13}")
    print(header)
    for i, r in enumerate(rows):
        line = (f"{r['step']:>12,} {r['episodes']:>9} "
                f"{r['exploration/coverage']:>9.3f} {r['exploration/nauc']:>7.3f} "
                f"{r['exploration/t50_reached']:>10.2%} "
                f"{r.get('exploration/t50_steps', float('nan')):>10.1f} "
                f"{r['exploration/t90_reached']:>10.2%} "
                f"{r.get('exploration/t90_steps', float('nan')):>10.1f} "
                f"{r['exploration/room_entropy_norm']:>13.3f}")
        if i == len(rows) - 1:
            line += "   (latest)"
        print(line)
    out = run_dir / "exploration_curve.json"
    out.write_text(json.dumps({
        "meta": {"run": run_dir.name, "source": repr(cfg.env.source),
                 "collects": args.collects, "num_envs": args.num_envs,
                 "steps": args.steps, "seed": PROBE_SEED},
        "rows": rows}, indent=2, default=float))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--room", default="lroom", choices=tuple(ROOMS),
                    help="the base room the run was LAUNCHED in; everything "
                         "room-dependent is read from the config this builds")
    ap.add_argument("--source", default="frozen", choices=SOURCES,
                    help="how the run drew its rooms: the committed set, its "
                         "first room alone, or a uniform pool")
    ap.add_argument("--rooms-scored", type=int, default=None,
                    help="rooms to score; default is the config's eval.rooms_max")
    ap.add_argument("--n-trajs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--spatial", action="store_true", help="also compute sRSA/SWdist (slower)")
    ap.add_argument("--impassable", action="store_true",
                    help="with --source selected: the impassable arm's room set")
    ap.add_argument("--exploration", action="store_true",
                    help="score the archived POLICY's exploration instead of the "
                         "pRNN probe: seeded on-policy rollouts per checkpoint pair")
    ap.add_argument("--collects", type=int, default=2,
                    help="exploration mode: rollouts per checkpoint (episodes = "
                         "collects x num-envs)")
    ap.add_argument("--num-envs", type=int, default=64,
                    help="exploration mode: parallel episode streams")
    a = ap.parse_args()

    run_dir = Path(a.run)
    points = archived(run_dir)
    if not points:
        raise SystemExit(f"no archived checkpoints under {run_dir / 'checkpoints'}")

    cfg = run_config(room=a.room, source=a.source, impassable=a.impassable,
                     hiddensize=checkpoint_hiddensize(points[0][1]))

    if a.exploration:
        exploration_main(cfg=cfg, run_dir=run_dir, args=a)
        return
    try:
        ctx = SeriesContext.resolve(cfg, rooms_scored=a.rooms_scored)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rooms = ctx.rooms
    print(ctx.describe(n_points=len(points), n_trajs=a.n_trajs, steps=a.steps))
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
    out.write_text(json.dumps({
        "meta": {"run": run_dir.name, **ctx.meta(),
                 "n_trajs": a.n_trajs, "steps": a.steps},
        "rows": rows}, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

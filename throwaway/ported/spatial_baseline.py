"""Reference sRSA / SWdist / SI for a checkpoint, with their noise floor.

These are the numbers a retrained network gets compared against, so the point
is not just the values but how much they move on their own:

  determinism   the same pooled activity scored twice must agree exactly, now
                that RSA subsampling draws without replacement from a
                fixed-seed generator rather than the global RNG.
  sample size   the training loop pools exp.eval_trajs trajectories; if a
                metric drifts with that, cross-run comparison is fragile no
                matter what training did.
  seed spread   across explicit subsample seeds, with the probe held fixed -
                the floor a real change has to clear.

Goes through `evaluate_spatial_representation`, the same entry point the
training loop calls, so these are the quantities wandb actually logs.

    uv run python scripts/spatial_baseline.py

Reads CUR_CKPT_DIR from .env. Prints a table; writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


TRAJ_COUNTS = (8, 16, 32, 64)
SEEDS = (0, 1, 2, 3, 4)
PROBE_SEED = 1234


def _setup():
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames

    from curious_george import get_pN, make_env
    from curious_george.storage import get_agent
    from curious_george.utils.dev_env import get_ckpt_env_vars
    from curious_george.utils.enums import AgentType

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    prnn_ckpt, _ = get_ckpt_env_vars()
    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=prnn_ckpt)
    # calculateSpatialMetrics logs to wandb whenever the loaded net carries
    # wandb_log=True; this is offline analysis, so there is no run to log to.
    pN.wandb_log = False
    # via get_agent, not RandomActionAgent(env.action_space): the agent's own
    # default probability is np.ones_like(<gym Discrete>)/n, a 0-d array, which
    # makes numpy choice() raise. It only ever works because every caller in
    # this repo passes storage.RAND_ACT_PROBA explicitly.
    agent = get_agent(env, AgentType.RANDOM)
    return args, env, pN, agent, prnn_ckpt


def _score(*, pN, env, agent, n_trajs: int, seqdur: int, rng=None) -> dict:
    """One pooled spatial eval. The probe is re-seeded so only `rng` varies."""
    from curious_george.evaluation.spatial import evaluate_spatial_representation

    m = evaluate_spatial_representation(
        pN, env, agent, n_trajs=n_trajs, traj_timesteps=seqdur, rng=rng,
        probe_seed=PROBE_SEED,
    )
    si = np.asarray(m["SI"]["SI"], dtype=float)
    return {"sRSA": float(m["sRSA"]), "SWdist": float(m["SWdist"]),
            "meanSI": float(np.nanmean(si)), "zeroSI": int((si == 0).sum())}


def _row(label: str, m: dict) -> None:
    print(f"  {label:<18} sRSA={m['sRSA']:+.5f}  SWdist={m['SWdist']:.5f}  "
          f"meanSI={m['meanSI']:.5f}  (SI==0 for {m['zeroSI']}/500 units)")


def main() -> None:
    args, env, pN, agent, ckpt = _setup()
    seqdur = int(args.predNet.seqdur)
    n_eval = int(args.exp.eval_trajs)

    print(f"checkpoint : {ckpt}")
    print(f"seqdur={seqdur}  exp.eval_trajs={n_eval}\n")

    print("1. determinism - identical probe, scored twice")
    a = _score(pN=pN, env=env, agent=agent, n_trajs=n_eval, seqdur=seqdur)
    b = _score(pN=pN, env=env, agent=agent, n_trajs=n_eval, seqdur=seqdur)
    _row("run 1", a)
    _row("run 2", b)
    print(f"  -> exactly equal: "
          f"{all(a[k] == b[k] for k in ('sRSA', 'SWdist', 'meanSI'))}\n")

    print("2. reference values vs pooled sample size")
    for n in TRAJ_COUNTS:
        _row(f"eval_trajs={n}", _score(pN=pN, env=env, agent=agent,
                                       n_trajs=n, seqdur=seqdur))
    print()

    print(f"3. subsample-seed spread at eval_trajs={n_eval} (probe fixed)")
    vals: dict[str, list[float]] = {"sRSA": [], "SWdist": [], "meanSI": []}
    for s in SEEDS:
        m = _score(pN=pN, env=env, agent=agent, n_trajs=n_eval, seqdur=seqdur,
                   rng=np.random.default_rng(s))
        for k in vals:
            vals[k].append(m[k])
    for k, raw in vals.items():
        v = np.asarray(raw)
        print(f"  {k:<8} mean={v.mean():+.5f}  sd={v.std(ddof=1):.5f}  "
              f"range=[{v.min():+.5f}, {v.max():+.5f}]")


if __name__ == "__main__":
    main()

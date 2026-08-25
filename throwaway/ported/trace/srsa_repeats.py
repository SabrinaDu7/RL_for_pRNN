"""sRSA / SWdist across a checkpoint series, REPEATED and averaged.

Two things the training loop's single in-run number cannot tell you:

1. **How much of the wobble is the metric.** The loop pools `exp.eval_trajs`
   trajectories once per analysis event, and that estimate is noisy: an
   UNTRAINED net scores sRSA 0.062 +/- 0.040 across inits, and a trained
   single-room run swings 0.45-0.63 between adjacent events. Repeating the eval
   with different probe seeds separates "the representation changed" from "the
   estimator moved".

2. **Whether a decline is the representation or the policy.** wandb logs
   `sRSA_onPolicy`, evaluated with the TRAINED agent, so as the policy sharpens
   it visits fewer cells and the metric is computed over a narrower, clumpier
   set of positions. `--agent random` scores the same checkpoints with a random
   walker, whose coverage does not change as training proceeds. On-policy
   falling while off-policy holds means the policy narrowed, not the map.
   (This is also why `scripts/multienv/checkpoint_curve.py` curves differ from
   the wandb ones - it has always used a random agent.)

    uv run python scripts/trace/srsa_repeats.py \\
        --ckpt-dir outputs/ckpts/<run> --env lroom --repeats 5

Writes outputs/summary/srsa_repeats[_<tag>].json and prints mean +/- sd per
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def build(cfg, env_key, landmarks, ckpt):
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import get_pN, make_env

    kw = {"landmarks": list(landmarks)} if landmarks else {}
    env = make_env(
        env_key=env_key, input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, **kw,
    )
    pN = get_pN(args=cfg, env=env, device="cpu", pRNN_ckpt=str(ckpt))
    pN.wandb_log = False
    return pN, env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir of predictiveNet_state_step*.pt")
    ap.add_argument("--env", default="lroom")
    ap.add_argument("--layouts", default=None, choices=("one", "rooms", "pool"))
    ap.add_argument("--room", type=int, default=0)
    ap.add_argument("--agent", default="random", choices=("random", "policy"),
                    help="random = coverage independent of training (the point)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-trajs", type=int, default=8, help="matches exp.eval_trajs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from prnn.utils.checkpoints import CkptKeys

    from curious_george.envs.layouts import resolve_layouts
    from curious_george.evaluation.spatial import evaluate_spatial_representation
    from curious_george.storage import get_agent
    from curious_george.utils.enums import AgentType

    ckpts = sorted(Path(args.ckpt_dir).glob("predictiveNet_state_step*.pt"))
    if not ckpts:
        raise SystemExit(f"no predictiveNet_state_step*.pt in {args.ckpt_dir}")

    overrides = [f"env={args.env}", "logging.wandb_log=false"]
    if args.layouts:
        overrides.append(f"exp.layouts={args.layouts}")
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    cfg.predNet.hiddensize = int(
        torch.load(ckpts[0], map_location="cpu", weights_only=False)[CkptKeys.HIDDEN_SIZE]
    )
    rooms = resolve_layouts(cfg)
    landmarks = rooms[args.room].landmarks if rooms else None

    if args.agent == "policy":
        raise SystemExit(
            "--agent policy needs the run's acmodel, which the archives do not carry; "
            "the on-policy curve is already in wandb as sRSA_onPolicy"
        )

    print(f"{len(ckpts)} checkpoints x {args.repeats} repeats, agent={args.agent}, "
          f"n_trajs={args.n_trajs}\n")
    print(f"{'env step':>12}{'sRSA mean':>11}{'sd':>8}{'SWdist':>9}{'sd':>8}{'SI':>8}")
    rows = []
    for c in ckpts:
        step = int(re.search(r"step(\d+)", c.name).group(1))
        pN, env = build(cfg, cfg.exp.env_name, landmarks, c)
        s, w, si = [], [], []
        for rep in range(args.repeats):
            torch.manual_seed(1000 + rep)
            np.random.seed(1000 + rep)
            env.env.reset(seed=1000 + rep)
            agent = get_agent(env=env, agent_Type=AgentType.RANDOM)
            m = evaluate_spatial_representation(
                pN, env, agent, sleepstd=0.03, wandb_nameext="",
                n_trajs=args.n_trajs, traj_timesteps=cfg.predNet.seqdur,
                trainDecoder=False,
            )
            s.append(float(m["sRSA"]))
            w.append(float(m["SWdist"]))
            si.append(float(np.nanmean(m["SI"])))
        rows.append({"step": step, "sRSA": s, "SWdist": w, "SI": si})
        print(f"{step:>12,}{np.mean(s):>11.4f}{np.std(s):>8.4f}"
              f"{np.mean(w):>9.4f}{np.std(w):>8.4f}{np.mean(si):>8.4f}")

    out = Path("outputs/summary")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"srsa_repeats{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(
        {"ckpt_dir": args.ckpt_dir, "env": args.env, "layouts": args.layouts,
         "agent": args.agent, "repeats": args.repeats, "n_trajs": args.n_trajs,
         "rows": rows}, indent=2))
    first, last = np.mean(rows[0]["sRSA"]), np.mean(rows[-1]["sRSA"])
    print(f"\nsRSA {first:.4f} -> {last:.4f}  ({'RISING' if last > first else 'FALLING'})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

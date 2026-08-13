"""Train a pRNN from scratch through the Moser object/trace-cell paradigm.

Replicates the session structure of Tsao, Moser & Moser (2013, Curr Biol,
doi 10.1016/j.cub.2013.01.036) - see docs/ref-trace-cells.png:

    no object -> position 1 -> ... -> position N -> no object

in a SQUARE, four-fold-symmetric room (`MiniGrid-SquareRoom-v0`), where an
object is the only cue that breaks the symmetry. The L-room used for every
previous experiment gives the agent an L-shaped wall plus three floor shapes,
so it can localise without ever encoding the object - which is the leading
explanation for why every hidden-state result so far has been null.

The paper's two cell classes are distinct populations, and the analysis
(scripts/moser_analysis.py) tests them separately:

  object cell  fires when the object is present at its location.
  trace cell   fires at a location where the object USED to be, and
               "generally did not respond to the object when it was present".

Each session is a separate `main_train.py` process that loads the previous
session's checkpoint, so weights, both optimizers and the frame counter carry
over while the environment changes. `rl.episodes_total` is CUMULATIVE for that
reason: the loop runs until the total, starting from the loaded frame count.

    uv run python scripts/moser_sessions.py --dry-run
    uv run python scripts/moser_sessions.py

MiniGrid-SquareRoom-v0 comes from the pinned minigrid in uv.lock; no editable
install is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Six object positions in the 14x14 interior (walkable x,y are 1..14).
# Chosen scattered and off-centre, none related to another by the square's
# symmetry group, so a field at one is never confusable with a field at another.
POSITIONS: list[tuple[int, int] | None] = [
    None,       # session 0: empty room - establish the place code
    (5, 4),
    (10, 6),
    (6, 11),
    (12, 3),
    (3, 9),
    (11, 12),
    None,       # final session: object removed - where do the traces sit?
]

# Episodes per session. The first is long because the net starts from random
# init and has to learn the room before an object can mean anything.
EPISODES = [8000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]

SEQDUR = 256
RUN_ROOT = Path("outputs")


def session_name(i: int, pos) -> str:
    return f"moser-s{i}-" + ("none" if pos is None else f"{pos[0]}_{pos[1]}")


def main() -> None:
    global POSITIONS, EPISODES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="moser")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--wandb", default="true")
    ap.add_argument("--smoke", action="store_true",
                    help="3 tiny sessions: proves chaining before a long run")
    args = ap.parse_args()

    assert len(POSITIONS) == len(EPISODES), "one episode count per session"

    if args.smoke:
        POSITIONS, EPISODES = [None, (5, 4), None], [40, 40, 40]

    manifest: list[dict] = []
    prev_ckpt: str | None = None
    cumulative = 0

    for i, (pos, n_ep) in enumerate(zip(POSITIONS, EPISODES)):
        cumulative += n_ep
        name = f"{args.tag}-s{i}-" + ("none" if pos is None else f"{pos[0]}_{pos[1]}")
        overrides = [
            f"exp.exp_name={name}",
            f"exp.seed={args.seed}",
            "exp.env_name=MiniGrid-SquareRoom-v0",
            f"exp.num_envs={args.num_envs}",
            "exp.device_env=true",
            "predNet.batched_curiosity=true",
            "predNet.batched_wm=false",          # serial: num_envs grad steps/update
            f"rl.episodes_total={cumulative}",   # CUMULATIVE - see module docstring
            f"logging.wandb_log={args.wandb}",
            # one save exactly at the end of this session
            f"logging.save_every_steps={n_ep * SEQDUR}",
            f"logging.analysis_every_steps={max(1, n_ep // 4) * SEQDUR}",
            "logging.plot_every_steps=0",
        ]
        if pos is not None:
            overrides.append(f"exp.new_obj_pos=[{pos[0]},{pos[1]}]")
        if prev_ckpt is not None:
            overrides += ["logging.load_worldmodel=true", "logging.load_acmodel=true"]

        env = dict(os.environ)
        if prev_ckpt is not None:
            env["CUR_CKPT_DIR"] = prev_ckpt

        cmd = [sys.executable, "main_train.py", *overrides]
        print(f"\n=== session {i}: object={pos}  episodes={n_ep}  "
              f"cumulative={cumulative} ===", flush=True)
        print("  " + " ".join(overrides), flush=True)
        if prev_ckpt:
            print(f"  CUR_CKPT_DIR={prev_ckpt}", flush=True)

        if args.dry_run:
            manifest.append({"session": i, "object": pos, "episodes": n_ep,
                             "cumulative": cumulative, "run_dir": None})
            prev_ckpt = "<dry-run>"
            continue

        before = set(RUN_ROOT.glob(f"{name}_*"))
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            raise SystemExit(f"session {i} failed with code {r.returncode}")
        made = sorted(set(RUN_ROOT.glob(f"{name}_*")) - before)
        if not made:
            raise SystemExit(f"session {i} produced no run directory under {RUN_ROOT}")
        run_dir = made[-1]
        prev_ckpt = str(run_dir)
        manifest.append({"session": i, "object": pos, "episodes": n_ep,
                         "cumulative": cumulative, "run_dir": str(run_dir)})
        Path(RUN_ROOT / f"{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  -> {run_dir}", flush=True)

    print("\nmanifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

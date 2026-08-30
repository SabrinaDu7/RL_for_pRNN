"""Is tile COLOUR in the hidden state, or never encoded? [throwaway]

The decision this informs: a categorical readout can only help if the colour is
already in `h` and the regression head is discarding it. If `h` does not carry
it, the recurrent network never encoded colour and a new head cannot invent it.

METHOD. One fixed probe rollout per checkpoint. For each of the 49 tiles in the
7x7 view, fit a LINEAR 6-way classifier from `h` (500 units) to that tile's
colour, and score it held-out. Compare against two things measured on the SAME
rows:

  readout accuracy   what the network actually predicts today - its 3 RGB floats
                     for that tile, snapped to the nearest of the 6 palette
                     colours. This is the number the categorical head would
                     have to beat.
  chance             the majority class. Reported because the palette is very
                     imbalanced - floor and dark are ~90% of tiles - so raw
                     accuracy flatters everything and BALANCED accuracy is the
                     honest statistic.

READ IT THIS WAY.
  probe >> readout   the colour is in `h` and the readout throws it away.
                     A categorical head is the right fix.
  probe ~= readout   the readout is already extracting what is there. The
                     bottleneck is the representation, and a new head will not
                     move it.

The interesting rows are the OBJECT colours (blue/green/red), which are ~2% of
tiles each. Floor and dark are easy and will saturate both numbers.

    uv run python throwaway/scripts/colour_in_hidden_state.py CKPT [CKPT ...]

Throwaway: no committed result may depend on this (see CLAUDE.md). If it decides
the design, the measurement moves into `curious_george/evaluation/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

#: The closed palette, measured over all 5 selected rooms and both affordances.
#: Order is fixed so class indices mean the same thing in every report.
PALETTE = np.array([
    [0.2980, 0.2980, 0.2980],   # 0 dark / unseen
    [0.5725, 0.5725, 0.5725],   # 1 floor
    [0.5294, 0.2980, 0.2980],   # 2 the agent's own triangle
    [0.2980, 0.2980, 0.6471],   # 3 blue   (x)
    [0.2980, 0.6471, 0.2980],   # 4 green  (plus)
    [0.6471, 0.2980, 0.2980],   # 5 red    (block3)
])
NAMES = ["dark", "floor", "agent", "blue x", "green plus", "red block3"]
OBJECTS = (3, 4, 5)


def to_class(rgb: np.ndarray) -> np.ndarray:
    """Nearest palette member. Exact for targets; nearest for predictions."""
    d = ((rgb[:, None, :] - PALETTE[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def collect(ckpt: str, steps: int):
    """(hidden states, target classes, readout classes) over one fixed probe."""
    from dataclasses import replace

    from prnn.utils import ActionEncodingsEnum

    from curious_george import AgentInputType, get_pN, make_env
    from curious_george.configs import Config
    from curious_george.envs.layouts import (
        BASE_ROOM_ID, MULTI_ROOM_ID, ROOMS_SELECTED, with_affordance,
    )
    from curious_george.evaluation.checkpoint_series import (
        PROBE_SEED, checkpoint_hiddensize,
    )
    from curious_george.log_and_store.storage import RAND_ACT_PROBA, get_agent
    from curious_george.models.device import eval_mode
    from curious_george.utils.enums import AgentType

    room = with_affordance(ROOMS_SELECTED, impassable=False)[0]
    env = make_env(
        env_key=MULTI_ROOM_ID[BASE_ROOM_ID], input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0,
        landmarks=list(room.landmarks),
    )
    cfg = Config()
    cfg = replace(cfg, arch_prnn=replace(cfg.arch_prnn,
                                         hidden_size=checkpoint_hiddensize(Path(ckpt))))
    pN = get_pN(args=cfg, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False

    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    env.env.reset(seed=PROBE_SEED)
    agent = get_agent(env=env, agent_Type=AgentType.RANDOM, rand_act_prob=RAND_ACT_PROBA)
    with eval_mode(pN.pRNN), torch.no_grad():
        obs, act, _, _ = pN.collectObservationSequence(env, agent, steps, discretize=True)
        pred, tgt, h = pN.predict(obs, act)
    # theta mean, as every other spatial readout in the repo takes it
    H = torch.mean(h, dim=0).numpy()
    P = pred.squeeze(0).numpy().reshape(len(H), 49, 3)
    T = tgt.squeeze(0).numpy().reshape(len(H), 49, 3)
    y = np.stack([to_class(T[:, i, :]) for i in range(49)], axis=1)
    r = np.stack([to_class(P[:, i, :]) for i in range(49)], axis=1)
    return H, y, r


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--steps", type=int, default=3072)
    a = ap.parse_args()

    for ckpt in a.ckpts:
        H, y, r = collect(ckpt, a.steps)
        print(f"\n=== {Path(ckpt).parent.parent.name or ckpt}")
        print(f"{len(H):,} timesteps x 49 tiles")
        probe_by_class = {c: [] for c in range(6)}
        readout_by_class = {c: [] for c in range(6)}
        for tile in range(49):
            yt = y[:, tile]
            present = np.unique(yt)
            if len(present) < 2:
                continue  # a tile that never varies teaches nothing
            idx = np.arange(len(H))
            tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=yt)
            clf = LogisticRegression(max_iter=400, multi_class="multinomial").fit(H[tr], yt[tr])
            pred = clf.predict(H[te])
            for c in present:
                m = yt[te] == c
                if m.sum() < 5:
                    continue
                probe_by_class[c].append((pred[m] == c).mean())
                readout_by_class[c].append((r[te][:, tile][m] == c).mean())
        print(f"{'colour':12s} {'tiles':>6s} {'LINEAR PROBE':>13s} {'READOUT TODAY':>14s}")
        for c in range(6):
            if not probe_by_class[c]:
                continue
            mark = "  <-- object" if c in OBJECTS else ""
            print(f"{NAMES[c]:12s} {len(probe_by_class[c]):6d} "
                  f"{100*np.mean(probe_by_class[c]):12.1f}% "
                  f"{100*np.mean(readout_by_class[c]):13.1f}%{mark}")
        po = np.mean([v for c in OBJECTS for v in probe_by_class[c]])
        ro = np.mean([v for c in OBJECTS for v in readout_by_class[c]])
        print(f"\nOBJECT colours only:  probe {100*po:.1f}%   readout {100*ro:.1f}%")
        print("=> colour IS in h; the readout discards it" if po > ro + 0.15 else
              "=> the readout extracts what is there; the bottleneck is the REPRESENTATION")


if __name__ == "__main__":
    main()

"""Look at what the world model predicts, room by room, because a scalar cannot.

`multiroom/mean_room_sRSA` says the representation is spatial. It cannot say
whether the network is seeing ten different rooms at all - a run whose layout
rotation silently collapsed to one room would still report a perfectly good
per-room sRSA, because each "room" would be the same room. This draws the thing.

    from curious_george.log_and_store.storage import get_model_dir
    from curious_george.evaluation.prediction_figures import plot_run_predictions
    plot_run_predictions(run_dir=get_model_dir("<run-name>"), path="rooms.png")

`run_dir` is caller-supplied on purpose - this reads an existing run rather than
writing one - but it is resolved through `get_model_dir` so a cluster run is
found under RL_STORAGE instead of a literal `outputs/` that only exists on the
machine that trained it.

Everything is composed from `evaluation/checkpoint_series.py` - `build` for the
network and env, `fixed_probe` for the rollout - so this is the same collection
path the checkpoint series scores, not a parallel one.

`matplotlib` is imported inside the function: this module is importable from
evaluation code that must not pull a plotting stack onto the training host.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float

#: MiniGrid's egocentric view, as flattened by `env2pred`: 7x7 cells, RGB.
VIEW = (7, 7, 3)


@dataclass(frozen=True)
class RoomPrediction:
    """One room's observed and predicted views over a single rollout.

    `observed` is `predict`'s OWN target (`obs_next`), not a re-derived slice, so
    a change to the architecture's `predOffset` shows up here rather than being
    silently re-aligned by this module.
    """

    key: str
    observed: Float[np.ndarray, "T 7 7 3"]
    predicted: Float[np.ndarray, "T 7 7 3"]
    mse: Float[np.ndarray, " T"]
    #: True where `inMask` let the real observation into the input.
    shown: Bool[np.ndarray, " T"]


def _views(flat: Float[torch.Tensor, "1 T 147"]) -> Float[np.ndarray, "T 7 7 3"]:
    return flat[0].reshape(-1, *VIEW).clamp(0, 1).numpy()


def predictions_for_room(*, pN, env, layout, steps: int) -> RoomPrediction:
    """Roll out `steps` random actions in `layout` and keep target vs prediction.

    Asserts the temporal alignment rather than documenting it: under
    `predOffset=0` the target IS the current observation, so `obs_next` must
    equal `obs[:, :T]`. If a future architecture predicts the NEXT observation
    this raises, which is the point - a figure that quietly re-aligns itself
    would hide the change it exists to show.
    """
    from curious_george.evaluation.checkpoint_series import fixed_probe

    (obs, act, _), = fixed_probe(pN=pN, env=env, layout=layout, n_trajs=1, steps=steps)
    with torch.no_grad():
        obs_pred, obs_next, _ = pN.predict(obs, act)

    T = obs_next.size(1)
    if not torch.equal(obs_next, obs[:, :T, :]):
        offset = 1 if torch.equal(obs_next, obs[:, 1 : T + 1, :]) else None
        raise AssertionError(
            f"target is not obs[t]; predOffset looks like {offset if offset else '?'}. "
            "Update this module deliberately - see docs/prnn-io-alignment.md."
        )

    mask = np.asarray(pN.pRNN.inMask, dtype=bool)
    return RoomPrediction(
        key=layout.key,
        observed=_views(obs_next),
        predicted=_views(obs_pred),
        mse=((obs_pred - obs_next) ** 2).mean(dim=2)[0].numpy(),
        shown=np.resize(mask, T),
    )


def plot_predictions(rooms: list[RoomPrediction], *, steps: int, path: Path | str | None = None):
    """Rows of observed/predicted view pairs, one pair per room.

    Every panel is drawn on the same 0-1 RGB scale, so brightness is comparable
    across rooms and across the observed/predicted split.
    """
    import matplotlib.pyplot as plt

    n = len(rooms)
    fig, axes = plt.subplots(
        2 * n, steps, figsize=(1.05 * steps, 2.25 * n), squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.05},
    )
    for r, room in enumerate(rooms):
        for row, (what, frames) in enumerate((("seen", room.observed), ("pred", room.predicted))):
            for t in range(steps):
                ax = axes[2 * r + row][t]
                ax.imshow(frames[t], vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#1d6fb8" if room.shown[t] else "#d9d7d1")
                    s.set_linewidth(1.8 if room.shown[t] else 0.6)
                if row == 0 and r == 0:
                    ax.set_title(f"t={t}\n{'shown' if room.shown[t] else 'masked'}", fontsize=7)
                if t == 0:
                    ax.set_ylabel(f"{room.key}\n{what}" if row == 0 else what, fontsize=7)
        axes[2 * r][0].set_ylabel(f"room {room.key}\nseen", fontsize=7)
    fig.suptitle(
        "What the world model is given (seen) and what it produces (pred), per room\n"
        "blue border = observation shown to the network; grey = zeroed by inMask",
        fontsize=10,
    )
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_run_predictions(
    *, run_dir: Path | str, path: Path | str | None = None, steps: int = 12, step: int | None = None
):
    """Draw every room of a finished or in-flight run from its archived checkpoint.

    The config comes from the run's OWN recorded `argv`, so the rooms drawn are
    the rooms trained on rather than a re-specification that can drift.
    """
    from curious_george import configs
    from curious_george.envs.layouts import resolve_layouts
    from curious_george.evaluation.checkpoint_series import archived, build

    run_dir = Path(run_dir)
    cfg = configs.cli(json.loads((run_dir / "provenance.json").read_text())["argv"][1:])
    layouts = resolve_layouts(cfg)
    if not layouts:
        raise ValueError(f"{cfg.env.source!r} resolves to no rooms")

    points = archived(run_dir)
    if not points:
        raise FileNotFoundError(f"no archived checkpoints under {run_dir}/checkpoints")
    chosen = dict(points)[step] if step is not None else points[-1][1]

    rooms = []
    for layout in layouts:
        pN, env = build(cfg=cfg, landmarks=layout.landmarks, ckpt=str(chosen))
        rooms.append(predictions_for_room(pN=pN, env=env, layout=layout, steps=steps))
    return plot_predictions(rooms, steps=steps, path=path)

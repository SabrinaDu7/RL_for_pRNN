"""Does multi-room training build room-SPECIFIC maps, or one shared map?

The question the multi-room runs exist to answer, as a figure. Reads the
`checkpoint_curve.json` that `checkpoint_curve.py --spatial` writes, so the
numbers here and the ones printed there cannot disagree.

The remapping index is `mean(per-room sRSA) - pooled sRSA`:

    ~0 with HIGH per-room sRSA   one shared position map. Dead reckoning still
                                 wins; room identity is not bound in. This is
                                 the null the design exists to break, and it is
                                 INFORMATIVE - the map is good, it just is not
                                 room-specific.
    ~0 with LOW per-room sRSA    the map degraded; the run is UNINFORMATIVE, not
                                 negative. Which is why per-room sRSA is drawn
                                 beside the index and never omitted.
    > 0                          per-room stays high while pooled collapses:
                                 room-specific maps.

    uv run python scripts/multienv/remapping_figure.py <run_dir> [<run_dir> ...]

Writes outputs/summary/fig_remapping_index.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/summary/fig_remapping_index.png")
REF = Path("outputs/summary/remapping_reference.json")


def hypothesis_reference() -> dict:
    """Where the index lands under each hypothesis, through the REAL metric.

    Computed rather than quoted: the figure's reference lines are the scale the
    measured curve is read against, so they have to come from the same
    `calculateSpatialMetrics` the curve does. Synthetic activity from
    `remapping.synthetic` at `predNet.hiddensize` units - the room decode is a
    linear readout, so the width matters. Cached; delete the json to re-derive.
    """
    if REF.is_file():
        return json.loads(REF.read_text())

    from hydra import compose, initialize_config_dir
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames

    from curious_george import get_pN, make_env
    from scripts.multienv.remapping import synthetic

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    pN = get_pN(args=cfg, env=env, device="cpu",
                pRNN_ckpt="outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt")
    pN.wandb_log = False

    out = {}
    for hyp in ("H_position", "H_room"):
        h, pos, room, _, _ = synthetic(hypothesis=hyp, n_units=int(cfg.predNet.hiddensize))
        per = [float(pN.calculateSpatialMetrics(h[room == r], pos[room == r], env,
                                                wandb_nameext="")["sRSA"])
               for r in np.unique(room)]
        pooled = float(pN.calculateSpatialMetrics(h, pos, env, wandb_nameext="")["sRSA"])
        out[hyp] = float(np.mean(per) - pooled)
    REF.parent.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps(out, indent=2))
    return out


def describe(meta: dict, *, keys: bool = False) -> str:
    """Which run this is, from the curve's own metadata.

    A curve of sRSA against step looks identical for the L-room and the square
    room, and for 3 rooms and a 500-room pool. The figure has to say which.
    `keys=True` adds the scored rooms' ids - too long for a legend, needed for
    the caption, since which rooms were scored is a fixed prefix and reproducible
    only if named.
    """
    room = "square room" if "SquareRoom" in meta.get("room_id", "") else "L-room"
    n, mode = meta.get("n_layouts"), meta.get("layouts")
    kind = f"{n} rooms" if mode == "rooms" else (
        f"{n}-room pool" if mode == "pool" else f"single room ({mode})")
    short = f"{room}, {kind}  ({meta['n_rooms_scored']} scored)"
    return short + (f" — {', '.join(meta['room_keys'])}" if keys else "")


def series(run: Path) -> dict:
    blob = json.loads((run / "checkpoint_curve.json").read_text())
    if isinstance(blob, list):
        raise SystemExit(
            f"{run}/checkpoint_curve.json predates the metadata block, so it cannot say which run "
            "it is. Re-run: uv run python scripts/multienv/checkpoint_curve.py --run "
            f"{run} --env <lroom_multi|squareroom_multi> --spatial")
    rows = [r for r in blob["rows"] if "mean_room_sRSA" in r]
    if not rows:
        raise SystemExit(f"{run} has no spatial rows; re-run checkpoint_curve.py with --spatial")
    return {k: np.array([r[k] for r in rows], dtype=float)
            for k in ("step", "mean_room_sRSA", "pooled_sRSA", "remapping_index", "SWdist")} | {
        "loss": np.array([np.mean(r["loss"]) for r in rows]),
        "name": run.name, "label": describe(blob["meta"]),
        "detail": describe(blob["meta"], keys=True), "meta": blob["meta"]}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [series(Path(a)) for a in sys.argv[1:]] or [
        series(Path("outputs/mila-rooms-10362462-fresh/multienv-rooms_curious_26-08-13-03-17-42"))]

    colours = ["#0F5257", "#C0392B", "#6C3483", "#B7791F"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for s, c in zip(runs, colours):
        x = s["step"] / 1e6
        axes[0].plot(x, s["mean_room_sRSA"], "o-", lw=2.2, ms=6, color=c,
                     label=f"{s['label']}  mean per-room")
        axes[0].plot(x, s["pooled_sRSA"], "s--", lw=1.8, ms=5, color=c, alpha=0.65,
                     label="pooled across rooms")
        axes[1].plot(x, s["remapping_index"], "o-", lw=2.4, ms=7, color=c, label=s["label"])
        axes[2].plot(x, s["loss"], "o-", lw=2, ms=6, color=c, label=s["label"])

    axes[0].set_ylabel("sRSA  (higher = better spatial map)")
    axes[0].set_title("Is the map good?\nper-room and pooled rise together", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylim(0, 1)

    ref = hypothesis_reference()
    axes[1].axhline(ref["H_position"], color="0.4", lw=1.2, ls=":",
                    label=f"ONE shared map (H_position): {ref['H_position']:+.4f}")
    axes[1].axhline(ref["H_room"], color="#C0392B", lw=1.6, ls="--",
                    label=f"room-SPECIFIC maps (H_room): {ref['H_room']:+.4f}")
    axes[1].set_ylabel("remapping index\nmean per-room sRSA − pooled")
    axes[1].set_title("Are the maps room-SPECIFIC?\nThe question the runs exist to answer",
                      fontsize=11)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_ylim(-0.05, ref["H_room"] * 1.18)

    axes[2].set_ylabel("pRNN prediction loss")
    axes[2].set_title("Is it still learning?", fontsize=11)

    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
        ax.grid(alpha=0.25)

    who = "\n".join(
        f"{s['detail']}   |   {s['step'].min() / 1e6:.0f}–{s['step'].max() / 1e6:.0f}M steps"
        for s in runs)
    fig.suptitle(
        "Multi-room training builds a BETTER shared map, not room-specific ones: the remapping "
        "index sits at ~0 while\nper-room sRSA climbs — the informative null, because the map is "
        f"good and simply does not encode which room.\n\n{who}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    for s in runs:
        print(f"{s['name']}: remapping index {s['remapping_index'].min():+.4f} .. "
              f"{s['remapping_index'].max():+.4f} over "
              f"{s['step'].min() / 1e6:.0f}-{s['step'].max() / 1e6:.0f}M steps; "
              f"mean per-room sRSA {s['mean_room_sRSA'][0]:.3f} -> {s['mean_room_sRSA'][-1]:.3f}")


if __name__ == "__main__":
    main()

"""Visualise a sequential-displacement run: tuning changes and predictions.

Two figures, both needed to believe any number the metric produces.

1. `fields` — the units whose receptive field changed most, shown as full
   occupancy-masked rate maps over the whole environment, one column per phase.
   Object positions are marked per column, so "did a field appear where the
   object was, and does it survive the object leaving" is directly visible.

2. `predictions` — the network's predicted view at each object location during
   each phase, next to the ground truth. This is the readout side: it shows
   whether the net predicts the object where it currently is, where it used to
   be, or neither.

Everything is scored against the same pre-exposure baseline as all earlier
results, so the numbers stay comparable.

    uv run python scripts/seq_figures.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/trace")
BASE_CKPT = "outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt"
ONSET, HID = 20, 500


def _ctx():
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames
    from curious_george import make_env
    from scripts.trace.trace_probe import load_probe
    from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    probe = load_probe(OUT / "probe_lroom_noobj")
    cells = [tuple(p.tolist())
             for p in get_walkable_minigrid_positions(get_walkable_mask(env))]
    return args, env, probe, cells


def _maps(args, env, probe, ckpt):
    from curious_george import get_pN
    from scripts.trace.trace_probe import replay_checkpoint
    from scripts.trace import trace_maps as tm

    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
    h = replay_checkpoint(pN=pN, probe=probe)[:, ONSET:, :].numpy()
    pos = probe.agent_pos[:, ONSET:probe.n_steps, :].astype(np.float64)
    return tm.occupancy_and_maps(h=h.reshape(-1, HID),
                                 pos=pos.reshape(-1, 2), env=env)[0]


def _phases(run: Path):
    """[(phase_index, checkpoint_path)] in order, using each phase's END save."""
    from curious_george import resolve_prnn_ckpt

    out = []
    for d in sorted(run.iterdir()):
        if not d.is_dir() or not d.name.startswith("phase"):
            continue
        ph, step = d.name.split("_")
        out.append((int(ph[5:]), int(step), resolve_prnn_ckpt(str(d))))
    ends = {}
    for ph, step, ck in out:
        if ph not in ends or step > ends[ph][0]:
            ends[ph] = (step, ck)
    return [(ph, ends[ph][1]) for ph in sorted(ends)]


def fields(run: Path, sequence: list, n_units: int = 8) -> None:
    """Rate maps of the most-changed units, one column per phase."""
    from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions
    from scripts.trace import trace_figure as tf
    from scripts.trace.trace_metric import _disc_masks, field_gain

    args, env, probe, cells = _ctx()
    base = _maps(args, env, probe, BASE_CKPT)
    phases = _phases(run)
    maps = [base] + [_maps(args, env, probe, ck) for _, ck in phases]

    # WHAT "MOST CHANGED" MEANS, EXACTLY - the figure used to leave this to the
    # reader, and the answer used to be a ratio that ranked by quietness.
    #
    #     g(u, c)  = mean rate of unit u in a radius-2 disc at cell c
    #                / u's mean rate over the whole map            (field_gain)
    #     dg(u, c) = g_phase(u, c) - g_pre(u, c)
    #     score(u) = max over (phase, OBJECT location) of the percentile of
    #                dg(u, that location) among dg(u, ·) at ALL 172 walkable cells
    #
    # Above 95 is what makes a unit an object cell (trace_metric), so the rows
    # are the units the null test itself flagged. Two properties still bound
    # what the figure can be read to say, and both are reported per unit:
    #   - dg is signed, so a unit that LOST its field at an object location can
    #     rank high; the sign is printed and drawn;
    #   - the percentile only asks whether the object's cell is unusual FOR THIS
    #     UNIT. Structure shared ACROSS units at one location is invisible to it
    #     - that is exactly what (14,7) is - and only a location control catches
    #     that.
    locs = [tuple(c) for c in sequence if c]
    all_cells = [tuple(p.tolist())
                 for p in get_walkable_minigrid_positions(get_walkable_mask(env))]
    discs = _disc_masks(env=env, cells=all_cells, radius=2.0)
    g0 = field_gain(maps=base, discs=discs)                       # (n_cells, H)
    obj_idx = [all_cells.index(L) for L in locs]

    # WITHIN-UNIT percentile, the statistic the null test uses: how unusual is
    # the change at an object location among this unit's change at all walkable
    # cells. Ranking on the raw ratio instead put the network's QUIETEST units on
    # top - the ratio's variance scales as 1/rate - while ranking on absolute
    # change puts the loudest on top and confounds a whole map dimming with a
    # local one. Comparing a unit only against itself removes both.
    H = base.shape[0]
    n_ph = len(maps) - 1
    pct = np.full((n_ph, len(locs), H), np.nan)
    signed = np.full((n_ph, len(locs), H), np.nan)
    for p, m in enumerate(maps[1:]):
        dgp = field_gain(maps=m, discs=discs) - g0                # (n_cells, H)
        valid = np.isfinite(dgp)
        for j, ci in enumerate(obj_idx):
            ref = dgp[ci]
            pct[p, j] = 100.0 * ((dgp < ref[None, :]) & valid).sum(0) / np.maximum(valid.sum(0), 1)
            signed[p, j] = ref

    flat_pct, flat_sgn = pct.reshape(-1, H), signed.reshape(-1, H)
    score = np.nanmax(flat_pct, axis=0)
    arg = np.nanargmax(np.where(np.isfinite(flat_pct), flat_pct, -np.inf), axis=0)
    best_sgn = flat_sgn[arg, np.arange(H)]
    # percentile first, |change| as tie-break: units saturate at the top
    units = np.lexsort((-np.nan_to_num(np.abs(best_sgn)),
                        -np.nan_to_num(score, nan=-np.inf)))[:n_units]

    rate = np.nanmean(base, axis=(1, 2))
    rate_pct = np.array([100.0 * np.nanmean(rate < r) for r in rate])
    print("  ranking statistic: max over (phase, object location) of the WITHIN-UNIT percentile")
    print(f"  {'unit':>6} {'pctile':>7} {'signed dg':>10} {'phase':>6} {'at':>9} "
          f"{'mean rate':>10} {'rate pctile':>12}")
    for u in units:
        p, c = divmod(int(arg[u]), len(locs))
        print(f"  {u:>6} {score[u]:>7.0f} {best_sgn[u]:>+10.3f} {p:>6} {str(locs[c]):>9} "
              f"{rate[u]:>10.4f} {rate_pct[u]:>11.0f}%")
    signed, pct = best_sgn, rate_pct
    n_down = int((signed[units] < 0).sum())
    # Did the ranking maximum land where the object ACTUALLY was in that phase?
    # stack index p is phases[p], whose object is sequence[phases[p][0]]; a
    # maximum at any other location is map drift the statistic cannot tell from
    # an object effect, because it never asks.
    own = []
    for u in units:
        p, c = divmod(int(arg[u]), len(locs))
        here = sequence[phases[p][0]]              # [] in the REMOVED phase
        own.append(bool(here) and tuple(here) == locs[c])
    n_own = int(sum(own))
    print(f"  -> {n_down}/{len(units)} of the selected changes are DECREASES; "
          f"median rate percentile of the selection: {np.median(pct[units]):.0f}%")
    print(f"  -> {n_own}/{len(units)} maxima land at the location that ACTUALLY held the object "
          f"in that phase; the other {len(units) - n_own} are drift elsewhere")

    labels = ["pre-exposure"] + [
        f"phase {ph}: {tuple(sequence[ph]) if sequence[ph] else 'REMOVED'}"
        for ph, _ in phases]
    fig = tf.trace_panel(
        maps_by_checkpoint=maps, labels=labels, units=units, obj_xy=None,
        title=f"{run.name}\nThe {n_units} units ranked highest by the WITHIN-UNIT percentile of their "
              "change in field gain at an object location:\nhow unusual that change is among the same "
              "unit's change at all 172 walkable cells (>95 = object cell), maximised over phases.\n"
              f"Row labels give that percentile, the signed change, and the unit's rate percentile. "
              f"{n_down} of {n_units} selected changes are decreases;\n{n_own} of {n_units} land where "
              "the object actually was in that phase.")
    for r, u in enumerate(units):
        fig.axes[r * len(maps)].set_ylabel(
            f"#{u}\np{score[u]:.0f}\n{signed[u]:+.2f}\nrate p{pct[u]:.0f}", fontsize=7,
            rotation=0, labelpad=18, va="center")
    # mark each column's object position
    for c, (ph, _) in enumerate(phases, start=1):
        loc = sequence[ph]
        if not loc:
            continue
        for r in range(len(units)):
            fig.axes[r * len(maps) + c].plot(loc[0] - 1, loc[1] - 1, "w+", ms=7, mew=1.6)
    fig.savefig(OUT / f"fig_seq_fields_{run.name[-4:]}.png", dpi=150, bbox_inches="tight")
    print(f"  wrote fig_seq_fields_{run.name[-4:]}.png")


def predictions(run: Path, sequence: list) -> None:
    """Predicted view at each object location, per phase, against ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from curious_george import get_pN
    from tasks.omt.metrics import get_view_coords_batch

    args, env, probe, _ = _ctx()
    phases = _phases(run)
    locs = [tuple(c) for c in sequence if c]

    # one representative in-view timestep per location, from the shared probe
    T = probe.n_steps - ONSET
    pos = probe.agent_pos[:, ONSET:ONSET + T, :].reshape(-1, 2)
    hd = probe.agent_dir[:, ONSET:ONSET + T].reshape(-1)
    picks = {}
    for L in locs:
        vx, vy = get_view_coords_batch(L[0], L[1], pos, hd)
        ok = np.where((vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7))[0]
        picks[L] = (int(ok[len(ok) // 2]), int(vx[ok[len(ok) // 2]]), int(vy[ok[len(ok) // 2]]))

    ckpts = [("pre", BASE_CKPT)] + [
        (f"ph{ph} {tuple(sequence[ph]) if sequence[ph] else 'REM'}", ck) for ph, ck in phases]
    fig, axes = plt.subplots(len(locs), len(ckpts), figsize=(2.0 * len(ckpts), 2.2 * len(locs)),
                             squeeze=False)
    for ci, (lab, ck) in enumerate(ckpts):
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ck)
        pN.trainNoiseMeanStd = (0.0, 0.0)
        pN.pRNN.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            init = pN.pRNN.rnn.cell.actfun(torch.zeros(probe.n_trajs, 1, pN.hidden_size))
            op, _, _ = pN.predict(probe.obs, probe.act, state=init, randInit=False, batched=True)
        img = env.pred2np(op[0].permute(2, 0, 1)[:, ONSET:, :].reshape(-1, 147).unsqueeze(0))
        for ri, L in enumerate(locs):
            idx, vx, vy = picks[L]
            ax = axes[ri][ci]
            ax.imshow(np.clip(img[idx], 0, 1), interpolation="nearest")
            ax.plot(vx, vy, "w+", ms=9, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(lab, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"obj {L}", fontsize=8)
    fig.suptitle(f"{run.name}\npredicted 7x7 view at each object location, per phase "
                 "(white + = where the object projects)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_seq_pred_{run.name[-4:]}.png", dpi=150, bbox_inches="tight")
    print(f"  wrote fig_seq_pred_{run.name[-4:]}.png")

    # --- the readable version: GREEN-channel change from pre-exposure --------
    # raw predicted views are washed out (the net predicts ~0.56 against a
    # target of ~0.99), so phase-to-phase change is invisible by eye. Plot the
    # difference in the green channel instead, on a symmetric diverging scale.
    imgs = {}
    for lab, ck in ckpts:
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ck)
        pN.trainNoiseMeanStd = (0.0, 0.0)
        pN.pRNN.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            init = pN.pRNN.rnn.cell.actfun(torch.zeros(probe.n_trajs, 1, pN.hidden_size))
            op, _, _ = pN.predict(probe.obs, probe.act, state=init, randInit=False, batched=True)
        imgs[lab] = env.pred2np(
            op[0].permute(2, 0, 1)[:, ONSET:, :].reshape(-1, 147).unsqueeze(0))
    pre = imgs[ckpts[0][0]]
    later = ckpts[1:]
    fig, axes = plt.subplots(len(locs), len(later),
                             figsize=(2.0 * len(later), 2.2 * len(locs)), squeeze=False)
    vmax = 0.0
    for L in locs:
        idx, _, _ = picks[L]
        for lab, _ in later:
            vmax = max(vmax, float(np.abs(imgs[lab][idx][..., 1] - pre[idx][..., 1]).max()))
    vmax = vmax or 1e-3
    for ri, L in enumerate(locs):
        idx, vx, vy = picks[L]
        for ci, (lab, _) in enumerate(later):
            d = imgs[lab][idx][..., 1] - pre[idx][..., 1]
            ax = axes[ri][ci]
            im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.plot(vx, vy, "k+", ms=9, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(lab, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"obj {L}", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, label="green change vs pre-exposure")
    fig.suptitle(f"{run.name}\nCHANGE in predicted green vs pre-exposure, SINGLE timestep "
                 f"(red = more green). + marks the object's view cell. scale +/-{vmax:.3f}",
                 fontsize=10)
    fig.savefig(OUT / f"fig_seq_preddiff_{run.name[-4:]}.png", dpi=150, bbox_inches="tight")
    print(f"  wrote fig_seq_preddiff_{run.name[-4:]}.png")

    # --- OBJECT-CENTRED AVERAGE: the one that is actually interpretable ------
    # A single timestep is viewpoint-idiosyncratic. Average the green change
    # over EVERY in-view timestep, first rolling each 7x7 patch so the object's
    # view cell sits at the centre. Any object-locked effect lands at the
    # centre pixel; anything viewpoint-specific averages away.
    C = 3  # centre of a 7x7 patch
    fig, axes = plt.subplots(len(locs), len(later),
                             figsize=(2.0 * len(later), 2.2 * len(locs)), squeeze=False)
    panels = {}
    for L in locs:
        vx, vy = get_view_coords_batch(L[0], L[1], pos, hd)
        m = (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)
        sel = np.where(m)[0]
        for lab, _ in later:
            d = imgs[lab][:, :, :, 1] - pre[:, :, :, 1]          # (N,7,7) green change
            acc = np.zeros((7, 7))
            for i in sel:
                acc += np.roll(np.roll(d[i], C - int(vy[i]), axis=0), C - int(vx[i]), axis=1)
            panels[(L, lab)] = acc / max(len(sel), 1)
        print(f"    {L}: averaged over {len(sel)} in-view timesteps")
    vm = max(abs(v).max() for v in panels.values()) or 1e-3
    for ri, L in enumerate(locs):
        for ci, (lab, _) in enumerate(later):
            ax = axes[ri][ci]
            im = ax.imshow(panels[(L, lab)], cmap="RdBu_r", vmin=-vm, vmax=vm,
                           interpolation="nearest")
            ax.plot(C, C, "k+", ms=11, mew=2.2)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(lab, fontsize=8)
            if ci == 0: ax.set_ylabel(f"obj {L}", fontsize=8)
            ax.text(0.5, -0.12, f"centre {panels[(L,lab)][C,C]:+.3f}",
                    transform=ax.transAxes, ha="center", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, label="mean green change")
    fig.suptitle(f"{run.name}\nOBJECT-CENTRED mean change in predicted green vs pre-exposure.\n"
                 f"+ = the object's own view cell. An object-locked effect appears AT THE CENTRE. "
                 f"scale +/-{vm:.3f}", fontsize=9)
    fig.savefig(OUT / f"fig_seq_predcentred_{run.name[-4:]}.png", dpi=150, bbox_inches="tight")
    print(f"  wrote fig_seq_predcentred_{run.name[-4:]}.png")


if __name__ == "__main__":
    SEQ = [[7, 11], [7, 2], [4, 7], []]
    for r in sys.argv[1:]:
        run = Path(r)
        print(run.name)
        fields(run, SEQ)
        predictions(run, SEQ)

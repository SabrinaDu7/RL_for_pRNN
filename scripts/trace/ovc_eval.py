"""Run the object-vector pipeline on a checkpoint, with its validation controls.

Stage 0 characterises the detector on real activity before any result is
reported; E1 is the result. `main` refuses to print E1 unless Stage 0 passes,
so a broken detector cannot quietly produce a finding.

    uv run python scripts/trace/ovc_eval.py --stage0
    uv run python scripts/trace/ovc_eval.py --e1

Criterion and controls: `scripts/trace/ovc_metric.py`.
Design and rationale: `docs/exp_instructions/instructions-OVC.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int

from scripts.trace import ovc_metric as om

OUT = Path("outputs/ovc")
ONSET = 20
N_SHUFFLES = 200

ENVS = {
    "lroom": ("MiniGrid-LRoom-v0", "outputs/ckpts/pRNN_curious_26-07-23-10-06-25"),
    "smallobj": ("MiniGrid-LRoom-SmallObj-v0", None),  # resolved at runtime, newest run
}


def _newest_smallobj() -> str:
    runs = sorted(Path("outputs").glob("smallobj-lroom_curious_*"))
    if not runs:
        raise SystemExit("no smallobj-lroom run found under outputs/")
    return str(runs[-1])


def landmark_anchors(*, env) -> list[tuple[int, int]]:
    """One anchor per landmark colour: the rounded centroid of its floor tiles.

    For the small solid blocks this is the block centre; for the historical 6x6
    stencils it is the centre of mass of an irregular shape, which is the best
    available reference point and part of why those landmarks are a poor object
    (see the disanalogies in the instructions doc).
    """
    u = env.env.unwrapped if hasattr(env, "env") else env.unwrapped
    by_colour: dict[str, list[tuple[int, int]]] = {}
    for y in range(u.height):
        for x in range(u.width):
            c = u.grid.get(x, y)
            if c is not None and type(c).__name__ == "Floor":
                by_colour.setdefault(c.color, []).append((x, y))
    return [
        (int(round(np.mean([p[0] for p in pts]))), int(round(np.mean([p[1] for p in pts]))))
        for _, pts in sorted(by_colour.items())
    ]


def activity(*, env_name: str, ckpt_dir: str, n_dirs: int = 2, landmarks=None):
    """Replay the fixed probe through a checkpoint; return (h, pos, env)."""
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType
    from curious_george import get_pN, make_env
    from scripts.moser.moser_analysis import _build_probe
    from scripts.trace import trace_probe as tp

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    extra = {"landmarks": list(landmarks)} if landmarks else {}
    env = make_env(env_key=env_name, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, **extra)
    ckpt = ckpt_dir if ckpt_dir.endswith(".pt") else str(Path(ckpt_dir) / "predictiveNet_state.pt")
    pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
    pN.wandb_log = False
    probe = _build_probe(pN=pN, env=env, env_name=env_name, n_dirs=n_dirs)
    h = tp.replay_checkpoint(pN=pN, probe=probe).detach().numpy()
    pos = probe.agent_pos[:, : h.shape[1], :]
    return h, pos, env


def maps_from(*, h, pos, env, traj_slice=slice(None)):
    from scripts.trace import trace_maps as tm

    hh, pp = h[traj_slice], pos[traj_slice]
    h_rows = hh[:, ONSET:, :].reshape(-1, hh.shape[-1])
    pos_rows = pp[:, ONSET:, :].reshape(-1, 2).astype(np.float64)
    maps, occ, valid = tm.occupancy_and_maps(h=h_rows, pos=pos_rows, env=env)
    return maps


def shuffle_thresholds(
    *, maps, env, anchors, offsets, n_shuffles: int = N_SHUFFLES, seed: int = 0
) -> dict:
    """Null from RANDOM ANCHOR TRIPLES on the unmodified rate map.

    The obvious null - permuting which trajectory's activity goes with which
    trajectory's positions - is WRONG here, and measurably so: it destroys all
    spatial structure, so it answers "is there any spatial structure" rather
    than "is the structure anchored to the landmarks". Real rate maps share
    global structure (wall effects, smooth gradients) that correlates the three
    crops whatever the anchors are, so that null is far too permissive. Measured
    with it, the negative control read 0.130 and injected PLACE fields scored as
    vector cells at 0.146 - the detector was unusable.

    This null instead keeps the map exactly as it is and re-runs the criterion
    with three random walkable positions in place of the landmarks. It asks the
    question that matters: is the offset-map agreement SPECIFIC to the real
    anchors, or does any triple of points do as well? That is the same
    location-control logic that exposed the (14,7) false positive earlier in
    this project.
    """
    rng = np.random.default_rng(seed)
    walk = sorted(om.walkable_cells(env=env))
    n_off = len(offsets)
    vec, rad = [], []
    tries = 0
    while len(vec) < n_shuffles and tries < n_shuffles * 40:
        tries += 1
        pick = [walk[i] for i in rng.choice(len(walk), len(anchors), replace=False)]
        # only usable if it admits a comparable offset window
        if len(om.offset_grid(env=env, anchors=pick, radius=4)) < n_off // 2:
            continue
        o = om.offset_maps(maps=maps, anchors=pick, offsets=offsets)
        vec.append(np.nanpercentile(om.vector_score(omaps=o, offsets=offsets), 99))
        rad.append(np.nanpercentile(om.radial_score(omaps=o, offsets=offsets), 99))
    if len(vec) < 10:
        raise SystemExit("could not draw enough control anchor triples")
    return {
        "vector_p99": float(np.nanpercentile(vec, 99)),
        "radial_p99": float(np.nanpercentile(rad, 99)),
        "vector_p99_sd": float(np.nanstd(vec)),
        "n_shuffles": len(vec),
    }


def _frac_ovc(*, maps, env, anchors, offsets, n_null, pctile=95.0) -> float:
    """Fraction of units whose real-anchor score beats their OWN null."""
    pct, _ = om.vector_percentile(maps=maps, env=env, anchors=anchors,
                                  offsets=offsets, n_null=n_null)
    pr = om.peak_ratio(omaps=om.offset_maps(maps=maps, anchors=anchors, offsets=offsets))
    ok = np.isfinite(pct) & (pr >= 1.5)
    return float((ok & (pct > pctile)).sum() / max(int(ok.sum()), 1))


def stage0(*, key: str = "lroom", n_dirs: int = 2, n_shuffles: int = N_SHUFFLES) -> dict:
    """Negative, positive and specificity controls on real activity.

    Chance is 5% by construction: each unit is scored against random anchor
    triples drawn for that same unit, and called a vector cell only above its
    own 95th percentile.
    """
    env_name, ckpt = ENVS[key]
    ckpt = ckpt or _newest_smallobj()
    h, pos, env = activity(env_name=env_name, ckpt_dir=ckpt, n_dirs=n_dirs)
    anchors = landmark_anchors(env=env)
    offsets = om.offset_grid(env=env, anchors=anchors, radius=4)
    maps = maps_from(h=h, pos=pos, env=env)
    units = np.arange(maps.shape[0])
    print(f"  anchors {anchors}   testable offsets {len(offsets)}   null draws {n_shuffles}")

    kw = dict(env=env, anchors=anchors, offsets=offsets, n_null=n_shuffles)
    res = {"anchors": anchors, "n_offsets": len(offsets), "n_null": n_shuffles}

    res["negative_frac_ovc"] = _frac_ovc(maps=maps_from(h=h, pos=pos, env=env,
                                                        traj_slice=slice(0, None, 2)), **kw)
    res["positive"] = {
        amp: _frac_ovc(maps=om.inject_vector_field(maps=maps, anchors=anchors, offset=(2, 2),
                                                   units=units, amplitude=amp), **kw)
        for amp in (0.5, 1.0, 2.0, 4.0)
    }
    res["specificity"] = {
        amp: _frac_ovc(maps=om.inject_place_field(maps=maps, cell=(8, 8),
                                                  units=units, amplitude=amp), **kw)
        for amp in (1.0, 4.0)
    }
    res["passed"] = (
        res["negative_frac_ovc"] <= 0.10
        and res["positive"][4.0] >= 0.90
        and max(res["specificity"].values()) <= 0.10
    )
    return res


def e1(*, key: str, thresholds: dict, n_dirs: int = 2) -> dict:
    from scripts.trace import trace_maps as tm

    env_name, ckpt = ENVS[key]
    ckpt = ckpt or _newest_smallobj()
    h, pos, env = activity(env_name=env_name, ckpt_dir=ckpt, n_dirs=n_dirs)
    anchors = landmark_anchors(env=env)
    offsets = om.offset_grid(env=env, anchors=anchors, radius=4)
    maps = maps_from(h=h, pos=pos, env=env)

    si = tm.spatial_info(maps=maps, occupancy=np.ones(maps.shape[1:]))
    null_si = tm.shuffle_null_si(h=h[:, ONSET:, :], pos=pos[:, ONSET:, :], env=env,
                                 n_shuffles=50)
    spatial_ok = si > np.nanpercentile(null_si, 95, axis=0)

    pct, real = om.vector_percentile(maps=maps, env=env, anchors=anchors,
                                     offsets=offsets, n_null=thresholds["n_null"])
    omp = om.offset_maps(maps=maps, anchors=anchors, offsets=offsets)
    rs, pr = om.radial_score(omaps=omp, offsets=offsets), om.peak_ratio(omaps=omp)
    screen = np.isfinite(pct) & (pr >= 1.5) & spatial_ok
    is_ovc = screen & (pct > 95.0) & (rs <= 0.5)
    is_obj = screen & (pct > 95.0) & (rs > 0.5)
    c = {"vector_percentile": pct, "vector_score": real, "radial_score": rs,
         "screened": screen, "is_ovc": is_ovc, "is_object_cell": is_obj,
         "n_screened": int(screen.sum()),
         "frac_ovc": float(is_ovc.sum() / max(int(screen.sum()), 1)),
         "frac_object_cell": float(is_obj.sum() / max(int(screen.sum()), 1))}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"e1_{key}.npz", maps=maps, anchors=np.array(anchors),
                        offsets=offsets, **{k: v for k, v in c.items()
                                            if isinstance(v, np.ndarray)})
    return {"key": key, "ckpt": ckpt, "anchors": anchors, "n_offsets": len(offsets),
            "n_spatial": int(spatial_ok.sum()), **{k: v for k, v in c.items()
                                                   if not isinstance(v, np.ndarray)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--key", default="lroom", choices=list(ENVS))
    ap.add_argument("--n-dirs", type=int, default=2)
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.stage0 or a.e1:
        print(f"=== STAGE 0 ({a.key}) ===")
        s = stage0(key=a.key, n_dirs=a.n_dirs, n_shuffles=a.n_shuffles)
        print(f"  negative (odd/even, no injection) : {s['negative_frac_ovc']:.3f}  (want <= 0.05)")
        for amp, f in s["positive"].items():
            print(f"  positive  amp={amp:<5}                 : {f:.3f}")
        for amp, f in s["specificity"].items():
            print(f"  SPECIFICITY place field amp={amp:<5}   : {f:.3f}  (want <= 0.05)")
        print(f"  -> Stage 0 {'PASSED' if s['passed'] else 'FAILED'}")
        (OUT / f"stage0_{a.key}.json").write_text(json.dumps(s, indent=2, default=float))
        if not s["passed"]:
            raise SystemExit("Stage 0 failed - not reporting E1 on an uncharacterised detector")

    if a.e1:
        print(f"\n=== E1 ({a.key}) ===")
        r = e1(key=a.key, thresholds=s, n_dirs=a.n_dirs)
        print(f"  {r['n_spatial']}/500 units pass the spatial screen; "
              f"{r['n_screened']} reach the criteria")
        print(f"  object-vector cells : {r['frac_ovc']:.1%}")
        print(f"  object cells        : {r['frac_object_cell']:.1%}")
        (OUT / f"e1_{a.key}.json").write_text(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()

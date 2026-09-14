"""Is the offline sRSA estimator under-sampled at the default --n-trajs 6?

`checkpoint_series` is bitwise deterministic (same checkpoint, same command,
identical to the last digit), so its run-to-run spread is NOT measurement noise
in the re-run sense. But determinism is not precision: at the default 6
trajectories x 256 steps, minus ONSET, each room is scored from ~1,416 rows over
152-172 reachable cells - about 8-9 visits per cell. If that is too thin, part
of the spread between two runs is the estimator sampling a fixed probe too
sparsely, and more trajectories tighten it for free.

The test: collect a 48-trajectory probe ONCE per room, then evaluate sRSA on
NESTED prefixes. `fixed_probe` reseeds before collecting, so the first 6 of 48
are exactly the 6 the default would have used - the curve is a refinement of
the committed number, not a different measurement.

What decides it, on the four runs that matter:
  * REPLICATE gap (s2 clean vs s2 off0 - same seed, same config, different node)
    shrinking with n_trajs => the estimator was under-sampled.
  * It staying put => the two networks really do differ, and separating the HD
    effect needs more SEEDS, not more trajectories.

Throwaway: a one-off diagnostic. No committed result may depend on it.

    uv run python throwaway/2026-09-10/ntrajs_convergence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FETCHED = Path("outputs/fetched")
RUNS = {
    "s2 clean  HD on ": "mx-impassable-n8-s2-mse-clean_curious_26-09-09-13-29-48",
    "s2 off0   HD on ": "mx-impassable-n8-s2-mse-off0_curious_26-09-10-23-58-15",
    "s3 off0   HD on ": "mx-impassable-n8-s3-mse-off0_curious_26-09-13-15-29-36",
    "s3 off0   HD OFF": "mx-impassable-n8-s3-mse-off0-nohd_curious_26-09-13-15-45-24",
}
PREFIXES = (6, 12, 24, 48)
STEPS = 256


def mean_room_srsa_by_prefix(run_dir: Path) -> dict[int, float]:
    """mean_room_sRSA at each nested probe size, for the last checkpoint."""
    from curious_george.configs import Config
    from curious_george.envs.layouts import resolve_layouts
    from curious_george.evaluation.checkpoint_series import archived, build, fixed_probe, score

    cfg = Config.of_run(run_dir)
    layouts = resolve_layouts(cfg)
    ckpt = archived(run_dir)[-1][1]
    pN, env = build(cfg=cfg, landmarks=layouts[0].landmarks, ckpt=str(ckpt))

    per_room: dict[int, list[float]] = {k: [] for k in PREFIXES}
    for layout in layouts:
        rolls = fixed_probe(pN=pN, env=env, layout=layout,
                            n_trajs=max(PREFIXES), steps=STEPS)
        for k in PREFIXES:
            _, h, pos = score(pN=pN, rolls=rolls[:k], env=env)
            per_room[k].append(
                float(pN.calculateSpatialMetrics(h, pos, env, wandb_nameext="")["sRSA"])
            )
    return {k: float(np.mean(v)) for k, v in per_room.items()}


def main() -> None:
    out = {label: mean_room_srsa_by_prefix(FETCHED / name) for label, name in RUNS.items()}

    print("mean_room_sRSA over 8 rooms, by probe size (nested prefixes of one probe)\n")
    print(f"{'run':18s} " + " ".join(f"{'n=' + str(k):>8s}" for k in PREFIXES))
    for label, d in out.items():
        print(f"{label:18s} " + " ".join(f"{d[k]:8.4f}" for k in PREFIXES))

    print(f"\n{'gap':34s} " + " ".join(f"{'n=' + str(k):>8s}" for k in PREFIXES))
    pairs = [
        ("REPLICATE  (s2 clean vs s2 off0)", "s2 clean  HD on ", "s2 off0   HD on "),
        ("SEED       (s2 off0  vs s3 off0)", "s2 off0   HD on ", "s3 off0   HD on "),
        ("HD EFFECT  (s3 on    vs s3 off )", "s3 off0   HD on ", "s3 off0   HD OFF"),
    ]
    for name, a, b in pairs:
        print(f"{name:34s} " + " ".join(f"{abs(out[a][k] - out[b][k]):8.4f}" for k in PREFIXES))
    print("\nThe HD effect is only interpretable where it exceeds the REPLICATE gap.")


if __name__ == "__main__":
    main()

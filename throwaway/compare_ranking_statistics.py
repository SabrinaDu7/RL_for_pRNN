"""DECISION AID ONLY - no result depends on this script.

Figures 1.2 and 3.1 rank units by a SCALE-FREE ratio. This prints what the three
candidate rankings actually select, so the choice between them is made against
numbers rather than intuition:

    (c) keep object_modulation                     - the current Figure 1.2 ranking
    (a) object_modulation behind a rate screen
    (b) absolute change in mean rate inside the disc

Runs on outputs/trace/maps_all.npz alone, so it is cache-only and fast. If a
ranking is adopted it moves INTO scripts/ with the figure that uses it; this
file exists to be deleted.

    uv run python throwaway/compare_ranking_statistics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LOCS = [(7, 11), (14, 7), (7, 2)]
N_SHOW = 3


def main() -> None:
    from hydra import compose, initialize_config_dir
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames

    from curious_george import make_env
    from scripts.trace import trace_maps as tm
    from scripts.trace.trace_metric import _disc_masks

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)

    d = np.load("outputs/trace/maps_all.npz")
    base = d["baseline"]
    rate = np.nanmean(base, axis=(1, 2))
    pct = np.array([100.0 * np.nanmean(rate < r) for r in rate])
    med = float(np.nanmedian(rate))
    print(f"population mean rate: median {med:.4f}  p10 {np.nanpercentile(rate, 10):.4f}  "
          f"p90 {np.nanpercentile(rate, 90):.4f}   (n={len(rate)} units)\n")

    # Two candidate rate screens, both already conventions in this repo:
    #   - the SI/eval path drops units active in <200 samples (trace_maps.silent_units,
    #     evaluation active_time_threshold); from maps alone the analogue is a floor on mean rate
    screens = {"rate >= 10th pctile": rate >= np.nanpercentile(rate, 10),
               "rate >= 0.5x median": rate >= 0.5 * med}
    for name, m in screens.items():
        print(f"screen {name:22s} keeps {int(m.sum())}/{len(rate)} units")
    print()

    for L in LOCS:
        post = d[f"m_{L[0]}_{L[1]}"]
        disc = _disc_masks(env=env, cells=[L], radius=2.0)[0]

        dmod = (tm.object_modulation(maps=post, obj_xy=L, env=env, radius=2.0)
                - tm.object_modulation(maps=base, obj_xy=L, env=env, radius=2.0))
        in_disc = lambda m: np.nanmean(np.where(disc, m, np.nan), axis=(1, 2))
        drate = in_disc(post) - in_disc(base)

        # is (b)'s change LOCAL, or the whole map drifting? difference-in-differences
        out = lambda m: np.nanmean(np.where(disc, np.nan, m), axis=(1, 2))
        drate_out = out(post) - out(base)
        did = drate - drate_out

        # (d) the statistic the §3 NULL TEST itself uses: the object cell's
        # percentile among all walkable cells, within each unit's own distribution
        from scripts.analysis_OMT import get_walkable_mask, get_walkable_minigrid_positions
        from scripts.trace.trace_metric import trace_scores
        cells = [tuple(p.tolist())
                 for p in get_walkable_minigrid_positions(get_walkable_mask(env))]
        ts = trace_scores(maps_pre=base, maps_post=post, env=env, cells=cells, obj_cell=L)

        rankings = {
            "(c) dmod, no screen": np.where(np.isfinite(dmod), dmod, -np.inf),
            "(a) dmod, rate>=p10": np.where(np.isfinite(dmod) & screens["rate >= 10th pctile"],
                                            dmod, -np.inf),
            "(b) |d rate in disc|": np.where(np.isfinite(drate), np.abs(drate), -np.inf),
            "(d) within-unit pctile": np.where(np.isfinite(ts["percentile"]),
                                               ts["percentile"], -np.inf),
        }
        print(f"=== object at {L} " + "=" * 66)
        print(f"  {'ranking':<23} {'unit':>5} {'rate':>8} {'rate%':>6} {'dmod':>8} "
              f"{'d rate':>9} {'vs p95':>8} {'d-in-d':>9} {'pctile':>7}")
        p95 = float(np.nanpercentile(np.abs(drate), 95))
        for name, score in rankings.items():
            for i, u in enumerate(np.argsort(-score)[:N_SHOW]):
                print(f"  {name if i == 0 else '':<23} {u:>5} {rate[u]:>8.4f} {pct[u]:>5.0f}% "
                      f"{dmod[u]:>+8.3f} {drate[u]:>+9.4f} {abs(drate[u]) / p95:>7.2f}x "
                      f"{did[u]:>+9.4f} {ts['percentile'][u]:>6.0f}")
        print(f"  population p95 |d rate in disc| = {p95:.4f}   "
              f"object cells (pctile>95): {ts['n_object_cells']}/{ts['n_scored']} "
              f"= {ts['frac']:.3f}, p={ts['p_binom']:.3f}\n")


if __name__ == "__main__":
    main()

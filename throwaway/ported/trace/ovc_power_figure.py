"""How big an object-vector code would have to be for us to see it.

A null is only a bound if the detector's sensitivity is known, so this plots
Stage 0's recovery curve (`ovc_eval.py --stage0`, cached to
outputs/ovc/stage0_<key>.json) with the measured E1 result drawn on the same
axis (outputs/ovc/e1_multi_anchor_*.json).

Read it as: everything at or below the measured line is invisible to us. The
gate in `ovc_eval.main` refuses to report E1 until the amp=4 point clears 0.90,
and it does not, so the E1 line is a WEAK bound and is drawn as one.

    uv run python scripts/trace/ovc_power_figure.py

Writes outputs/summary/fig_ovc_power.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OVC = Path("outputs/ovc")
OUT = Path("outputs/summary/fig_ovc_power.png")
GATE = 0.90     # ovc_eval.stage0's positive-control requirement
CHANCE = 0.05   # by construction: within-unit 95th percentile


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((OVC / "stage0_lroom.json").read_text())
    amps = np.array(sorted(float(a) for a in s["positive"]))
    rec = np.array([s["positive"][f"{a:g}"] if f"{a:g}" in s["positive"]
                    else s["positive"][str(a)] for a in amps])

    e1 = sorted(OVC.glob("e1_multi_anchor_*.json"))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    ax.plot(amps, rec, "o-", lw=2.4, ms=9, color="#0F5257",
            label="injected vector field: fraction recovered")
    ax.axhline(GATE, color="#C0392B", lw=1.8, ls="--",
               label=f"Stage 0 gate ({GATE:.0%}) — NOT met, so E1 is a weak bound")
    ax.axhline(CHANCE, color="0.35", lw=1.2, ls=":", label=f"chance ({CHANCE:.0%}), by construction")
    ax.axhline(s["negative_frac_ovc"], color="#6C3483", lw=1.6, ls="-.",
               label=f"negative control (odd/even split): {s['negative_frac_ovc']:.3f}")
    spec = max(float(v) for v in s["specificity"].values())
    ax.plot([], [], " ", label=f"specificity, injected PLACE field: {spec:.3f}  (must stay at chance)")

    for p in e1:
        d = json.loads(p.read_text())
        ax.axhline(d["own"], color="#E67E22", lw=2.2,
                   label=f"MEASURED, own landmarks: {d['own']:.3f}")
        ax.axhline(d["other"], color="#E67E22", lw=1.4, ls=":",
                   label=f"MEASURED, control positions: {d['other']:.3f}")

    ax.set_xlabel("injected field amplitude  (multiples of the unit's own mean rate)")
    ax.set_ylabel("fraction of units called object-vector cells")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(amps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("What an object-vector code would have to look like for this detector to see it\n"
                 "The measured value sits at chance — but the detector only recovers "
                 f"{rec[-1]:.0%} of fields at {amps[-1]:g}x,\nso this bounds LARGE effects only.",
                 fontsize=11.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  recovery {dict(zip(amps.tolist(), np.round(rec, 3).tolist()))}")


if __name__ == "__main__":
    main()

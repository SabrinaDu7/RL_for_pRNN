"""Collect every OTC condition and render the figures for the writeup.

Two stages so the expensive part runs once:

    python scripts/otc_figures.py collect   # replay + decode -> outputs/trace/otc_results.npz
    python scripts/otc_figures.py plot      # npz -> outputs/trace/fig_otc_*.png

The decoding measure is `scripts.trace_presence_decoder`: a linear probe reading
object presence out of `h`, split by input-mask phase. `thRNN_5win` feeds the
observation only 1 step in 6 (`inMask = [True, False x5]`), so ph0 is
input-driven and ph1..ph5 are memory-only. Pooling them hides the effect - see
docs/exp_object_into_hidden_state_2026-08-01.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path("outputs/trace")
NPZ = OUT / "otc_results.npz"
LOC = (7, 11)
CTRL = [[2, 5], [4, 7], [11, 3], [13, 5]]
ONSET, PERIOD, HID = 20, 6, 500
BASE_CKPT = "outputs/ckpts/pRNN_curious_26-07-23-10-06-25/predictiveNet_state.pt"

# label -> (glob patterns | explicit checkpoint list)
CONDITIONS: dict[str, list[str]] = {
    "baseline": [BASE_CKPT],
    "[2,2,2] fixedpos": [
        "outputs/mila_omt_dense/omt-cur-dot-0730-165325",
        "outputs/mila_omt_dense/omt-cur-dot-0730-172405",
        "outputs/mila_omt_dense/omt-cur-dot-0730-175326",
        "glob:outputs/otcFixNorm520*",
    ],
    "[2,0,2] frozen Wout": ["outputs/otcFrozenWout3k-cur-dot-0801-091940"],
    "[2,0,4] frz+in2x": ["glob:outputs/otcFrzIn4-*"],
    "[2,0,8] frz+in4x": ["outputs/otcFrzIn8-cur-dot-0801-120212", "glob:outputs/otcFrzIn8s520*"],
    "[8,0,8] frz+all4x": ["glob:outputs/otcFrzLR8-*"],
    "[8,0,0] recurrent only": ["glob:outputs/otcFrzRecOnly-*"],
    "[2,2,2] RANDPOS": ["glob:outputs/otcRandPosNorm-*", "glob:outputs/otcRPNorm520*"],
    "[2,0,8] RANDPOS": ["glob:outputs/otcRandPos-otc-*"],
    "[2,0,8] k_cur=5": ["glob:outputs/otcKc5-*"],
    "[2,0,8] k_cur=20": ["glob:outputs/otcKc20-*"],
    "p=0.5 free": ["glob:outputs/otcP05-otc-*"],
    "p=0.5 frozen": ["glob:outputs/otcP05Frozen-otc-*"],
}


def _resolve(spec: str) -> list[str]:
    """A spec -> concrete pRNN checkpoint paths (latest step of each run dir)."""
    from curious_george import resolve_prnn_ckpt

    paths = sorted(Path().glob(spec[5:])) if spec.startswith("glob:") else [Path(spec)]
    out = []
    for p in paths:
        if p.is_file():
            out.append(str(p))
            continue
        steps = [int(x.name) for x in p.iterdir() if x.is_dir() and x.name.isdigit()]
        if steps:
            out.append(resolve_prnn_ckpt(f"{p}/{max(steps)}"))
    return out


def collect() -> None:
    from hydra import initialize_config_dir, compose
    from prnn.utils import ActionEncodingsEnum, AgentInputType, MinigridEnvNames
    from sklearn.linear_model import LogisticRegression

    from curious_george import get_pN, make_env
    from scripts.trace_probe import load_probe, replay_checkpoint
    from scripts.trace_readout_test import object_contrast, relative_weight_change
    from tasks.omt.metrics import get_view_coords_batch

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    pa = load_probe(OUT / "probe_lroom_noobj")
    po = load_probe(OUT / f"probe_lroom_obj{LOC[0]}_{LOC[1]}")
    B = pa.n_trajs

    base = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=BASE_CKPT)
    kw = dict(probe=pa, env=env, obj_loc=LOC, ctrl_locs=CTRL, n_trajs=200)
    base_contrast = object_contrast(pN=base, **kw)

    def one(ckpt: str) -> tuple[np.ndarray, float, dict]:
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
        ha = replay_checkpoint(pN=pN, probe=pa)[:, ONSET:, :].numpy()
        pN2 = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=ckpt)
        ho = replay_checkpoint(pN=pN2, probe=po)[:, ONSET:, :].numpy()
        T = ha.shape[1]
        pos = pa.agent_pos[:, ONSET:ONSET + T, :]
        hd = pa.agent_dir[:, ONSET:ONSET + T]
        vx, vy = get_view_coords_batch(LOC[0], LOC[1], pos.reshape(-1, 2), hd.reshape(-1))
        iv = ((vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)).reshape(B, T)
        ph = ((np.arange(T) + ONSET) % PERIOD)[None, :].repeat(B, 0)

        acc = np.zeros((3, PERIOD))
        for si in range(3):
            perm = np.random.default_rng(si).permutation(B)
            tr, te = perm[: B // 2], perm[B // 2:]
            for k in range(PERIOD):
                sel = ph == k

                def build(idx):
                    m = iv[idx] & sel[idx]
                    X = np.concatenate([ha[idx][m], ho[idx][m]])
                    return X, np.r_[np.zeros(int(m.sum())), np.ones(int(m.sum()))]

                Xtr, ytr = build(tr)
                Xte, yte = build(te)
                acc[si, k] = LogisticRegression(max_iter=1200, C=0.05).fit(Xtr, ytr).score(Xte, yte)
        contrast = object_contrast(pN=pN, **kw) - base_contrast
        return acc.mean(0), contrast, relative_weight_change(base=base, trained=pN)

    store: dict[str, np.ndarray] = {}
    for label, specs in CONDITIONS.items():
        ckpts = [c for s in specs for c in _resolve(s)]
        if not ckpts:
            print(f"  SKIP {label} (no runs found)")
            continue
        curves, contrasts, weights = [], [], []
        for c in ckpts:
            cur, con, w = one(c)
            curves.append(cur)
            contrasts.append(con)
            weights.append([w["W"], w["W_in"], w["W_out"]])
        store[f"curve::{label}"] = np.stack(curves)
        store[f"contrast::{label}"] = np.array(contrasts)
        store[f"weights::{label}"] = np.array(weights)
        print(f"  {label:<24} n={len(ckpts)}  ph0={np.stack(curves)[:,0].mean():.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **store)
    print(f"saved {NPZ}")


def _load() -> dict[str, dict[str, np.ndarray]]:
    d = np.load(NPZ)
    out: dict[str, dict[str, np.ndarray]] = {}
    for k in d.files:
        kind, label = k.split("::", 1)
        out.setdefault(label, {})[kind] = d[k]
    return out


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = _load()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1: decoding vs mask phase, the headline -------------------
    show = ["baseline", "[2,2,2] fixedpos", "[2,0,8] frz+in4x",
            "[2,2,2] RANDPOS", "[2,0,8] RANDPOS"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab in show:
        if lab not in R:
            continue
        c = R[lab]["curve"]
        m, s = c.mean(0), c.std(0) / max(np.sqrt(len(c)), 1)
        x = np.arange(PERIOD)
        for a, off in ((ax[0], 0.0), (ax[1], 0.5)):
            y = m - off if off else m
            a.errorbar(x, y, yerr=s, marker="o", ms=4, capsize=2,
                       label=f"{lab} (n={len(c)})")
    ax[0].axhline(0.5, color="k", ls=":", lw=1)
    ax[0].set_ylabel("presence decoding accuracy")
    ax[0].set_title("object presence decodable from $h$", fontsize=10)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("accuracy $-$ 0.5 (log)")
    ax[1].set_title("same, above chance — decay is visible", fontsize=10)
    for a in ax:
        a.set_xlabel("input-mask phase  (0 = observation fed, 1–5 = no input)")
        a.set_xticks(range(PERIOD))
    ax[0].legend(fontsize=7)
    fig.suptitle("Object information in the pRNN hidden state, by mask phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_otc_phases.png", dpi=160, bbox_inches="tight")

    # ---- Fig 2: encoding (ph0) per condition, seeds shown --------------
    labs = [l for l in CONDITIONS if l in R]
    vals = [R[l]["curve"][:, 0] for l in labs]
    order = np.argsort([v.mean() for v in vals])
    labs = [labs[i] for i in order]
    vals = [vals[i] for i in order]
    fig, a = plt.subplots(figsize=(7.5, 5))
    for i, v in enumerate(vals):
        a.scatter(v, np.full(len(v), i), s=18, zorder=3, color="C0")
        a.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="C3", lw=2, zorder=4)
    a.set_yticks(range(len(labs)))
    a.set_yticklabels(labs, fontsize=8)
    a.axvline(R["baseline"]["curve"][0, 0], color="k", ls=":", lw=1, label="pre-exposure baseline")
    a.set_xlabel("phase-0 decoding accuracy (encoding)")
    a.legend(fontsize=8)
    a.set_title("Object encoding in $h$: every seed, red bar = mean", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_otc_encoding.png", dpi=160, bbox_inches="tight")

    # ---- Fig 3: the tradeoff, and the W_in mechanism -------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab in labs:
        if lab == "baseline" or "contrast" not in R[lab]:
            continue
        x = R[lab]["contrast"].mean()
        y = R[lab]["curve"][:, 0].mean()
        ax[0].scatter(x, y, s=30)
        ax[0].annotate(lab, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        w = R[lab]["weights"].mean(0)
        ax[1].scatter(w[1], y, s=30)
        ax[1].annotate(lab, (w[1], y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0].axhline(R["baseline"]["curve"][0, 0], color="k", ls=":", lw=1)
    ax[0].axvline(0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("object contrast (prediction quality)")
    ax[0].set_ylabel("phase-0 decoding (encoding in $h$)")
    ax[0].set_title("encoding is bought at the cost of prediction", fontsize=10)
    ax[1].axhline(R["baseline"]["curve"][0, 0], color="k", ls=":", lw=1)
    ax[1].set_xlabel(r"$\|\Delta W_{in}\| / \|W_{in}\|$")
    ax[1].set_ylabel("phase-0 decoding")
    ax[1].set_title(r"$W_{in}$ is the lever", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_otc_tradeoff.png", dpi=160, bbox_inches="tight")
    print("wrote fig_otc_phases.png, fig_otc_encoding.png, fig_otc_tradeoff.png")

    # ---- summary table to stdout --------------------------------------
    print(f"\n{'condition':<24} {'n':>3} " + " ".join(f"{'ph'+str(k):>7}" for k in range(PERIOD))
          + f" {'contrast':>9}")
    for lab in CONDITIONS:
        if lab not in R:
            continue
        c = R[lab]["curve"]
        con = R[lab].get("contrast", np.array([np.nan]))
        print(f"{lab:<24} {len(c):>3} " + " ".join(f"{c[:,k].mean():>7.4f}" for k in range(PERIOD))
              + f" {con.mean():>+9.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    (collect if mode == "collect" else plot)()

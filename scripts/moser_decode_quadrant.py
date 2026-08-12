"""Quadrant decode with TRAJECTORY-level splits (no within-trajectory leakage)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from scripts.moser_analysis import _setup, _build_probe, ONSET
from scripts import trace_probe as tp
from curious_george import get_pN

man = json.load(open("outputs/moser_manifest.json"))
ENV = "MiniGrid-SquareRoom-v0"

def hidden(ckpt, obj):
    args, env = _setup(ENV, obj)
    pN = get_pN(args=args, env=env, device="cpu",
                pRNN_ckpt=f"{ckpt}/predictiveNet_state.pt"); pN.wandb_log = False
    pr = _build_probe(pN=pN, env=env, env_name=ENV, n_dirs=1)
    h = tp.replay_checkpoint(pN=pN, probe=pr).detach().numpy()
    pos = pr.agent_pos[:, :h.shape[1], :]
    B, T = h.shape[0], h.shape[1] - ONSET
    groups = np.repeat(np.arange(B), T)                 # trajectory id per row
    return h[:, ONSET:, :].reshape(-1, h.shape[-1]), pos[:, ONSET:, :].reshape(-1, 2), groups

quad = lambda p: (p[:,0] > 7.5).astype(int)*2 + (p[:,1] > 7.5).astype(int)

print(f"{'session':>7} {'object':>9} {'with obj':>9} {'no obj':>8} {'delta':>8}")
for s in man:
    if s["object"] is None: continue
    acc = {}
    for lab, o in (("with", tuple(s["object"])), ("without", None)):
        h, pos, g = hidden(s["run_dir"], o)
        sub = np.random.default_rng(0).choice(len(h), 9000, replace=False)
        X, y, gg = h[sub], quad(pos[sub].astype(float)), g[sub]
        acc[lab] = cross_val_score(LogisticRegression(max_iter=400),
                                   X, y, groups=gg, cv=GroupKFold(3)).mean()
    print(f"{s['session']:>7} {str(tuple(s['object'])):>9} {acc['with']:>9.3f} "
          f"{acc['without']:>8.3f} {acc['with']-acc['without']:>+8.3f}")
print("\nchance = 0.25; splits hold out whole trajectories")

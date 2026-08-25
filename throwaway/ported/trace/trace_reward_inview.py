"""Is prediction error elevated when the OBJECT IS IN VIEW, rather than at its cell?

The reward-map analysis binned MSE by agent position, which smears error
generated while looking at the object from a distance across all the viewing
positions. This conditions on visibility instead, and controls with the four
tasks.testing.ctrl_locs treated the same way.
"""
import numpy as np
import torch
from pathlib import Path

from hydra import initialize_config_dir, compose
from prnn.utils import MinigridEnvNames, ActionEncodingsEnum, AgentInputType
from curious_george import make_env, get_pN, resolve_prnn_ckpt
from scripts.trace.trace_probe import load_probe, PROBE_SEED
from tasks.omt.metrics import get_view_coords_batch

RUNS = {(7, 11): "outputs/mila_omt_dense/omt-cur-dot-0730-165325",
        (14, 7): "outputs/mila_omt_dense/omt-cur-dot-0730-165922",
        (7, 2):  "outputs/mila_omt_dense/omt-cur-dot-0730-165916"}
CTRL = [[2, 5], [4, 7], [11, 3], [13, 5]]
ONSET, BS = 20, 128
# Resolved from the working directory, not from one machine's home: an
# absolute path here silently breaks for every other checkout.
CFG = str(Path("Configs").resolve())


def mse_per_step(pN, probe):
    saved = pN.trainNoiseMeanStd
    pN.trainNoiseMeanStd = (0.0, 0.0)
    try:
        pN.pRNN.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, probe.n_trajs, BS):
                hi = min(lo + BS, probe.n_trajs)
                torch.manual_seed(PROBE_SEED + lo)
                init = pN.pRNN.rnn.cell.actfun(torch.zeros(hi - lo, 1, pN.hidden_size))
                op, on, _ = pN.predict(probe.obs[lo:hi], probe.act[lo:hi],
                                       state=init, randInit=False, batched=True)
                out.append(((op[0].permute(2, 0, 1) - on[0].permute(2, 0, 1)) ** 2).mean(-1))
    finally:
        pN.trainNoiseMeanStd = saved
    return torch.cat(out).numpy()


def in_view_mask(probe, cell, T):
    pos = probe.agent_pos[:, ONSET:ONSET + T, :].reshape(-1, 2)
    hd = probe.agent_dir[:, ONSET:ONSET + T].reshape(-1)
    vx, vy = get_view_coords_batch(cell[0], cell[1], pos, hd)
    return (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)


with initialize_config_dir(config_dir=CFG, version_base=None):
    args = compose(config_name="main")

for LOC, RUN in RUNS.items():
    env = make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=list(LOC),
                   input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0)
    probe = load_probe(f"outputs/trace/probe_lroom_obj{LOC[0]}_{LOC[1]}")
    steps = sorted(int(s.name) for s in Path(RUN).iterdir() if s.is_dir())
    print(f"\n=== object at {LOC}   (MSE ratio: in-view / not-in-view)", flush=True)
    print(f"{'trajs':>6} | {'object':>8} | {'ctrl mean':>10} | {'obj - ctrl':>10}", flush=True)
    for s in steps:
        pN = get_pN(args=args, env=env, device="cpu", pRNN_ckpt=resolve_prnn_ckpt(f"{RUN}/{s}"))
        e = mse_per_step(pN, probe)[:, ONSET:]
        T = e.shape[1]
        flat = e.reshape(-1)

        def ratio(cell):
            m = in_view_mask(probe, cell, T)
            return float(flat[m].mean() / flat[~m].mean())

        r_obj = ratio(LOC)
        r_ctl = float(np.mean([ratio(tuple(c)) for c in CTRL]))
        print(f"{s:>6} | {r_obj:>8.3f} | {r_ctl:>10.3f} | {r_obj-r_ctl:>+10.3f}", flush=True)
print("DONE", flush=True)

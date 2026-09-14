"""One line per run: architecture shape + deduped parameter totals."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from count_params import unique_params  # noqa: E402


def totals(state) -> int:
    return sum(p.numel for p in unique_params(state))


rows = []
for prnn_ckpt in sorted(Path("outputs").rglob("predictiveNet_state.pt")):
    run = prnn_ckpt.parent
    ck = torch.load(prnn_ckpt, map_location="cpu", weights_only=False)
    sd = ck["pRNN_state_dict"]
    n_prnn = totals(sd)
    # readout output dim = rows of the projection weight (aliased as W_out)
    out_dim = sd["W_out"].shape[0]
    depth = sum(1 for k in sd if k.startswith("outlayer.") and k.endswith(".weight"))
    readout = "mlp" if depth > 2 else "linear"

    pol = run / "policy.pt"
    if not pol.is_file():
        pol = run / "status.pt"
    n_pol, emb, conv = None, None, None
    if pol.is_file():
        st = torch.load(pol, map_location="cpu", weights_only=False)
        ms = st.get("model_state")
        if ms:
            n_pol = totals(ms)
            emb = ms["actor.0.weight"].shape[1]
            conv = any(k.startswith("image_conv") for k in ms)
    rows.append((str(run), ck["pRNNtype"], ck["hidden_size"], ck["obs_size"],
                 ck["act_size"], out_dim, readout, n_prnn, emb, conv, n_pol))

hdr = ("run", "pRNNtype", "hid", "obs", "act", "out", "readout", "pRNN params",
       "emb", "conv", "policy params")
print(f"{hdr[1]:<12} {hdr[2]:>4} {hdr[3]:>4} {hdr[4]:>3} {hdr[5]:>5} {hdr[6]:<7} "
      f"{hdr[7]:>12} {hdr[8]:>5} {hdr[9]:>5} {hdr[10]:>13}  {hdr[0]}")
for r in sorted(rows, key=lambda r: (r[7], r[0])):
    (run, ty, hid, obs, act, out, ro, np_, emb, conv, npol) = r
    print(f"{ty:<12} {hid:>4} {obs:>4} {act:>3} {out:>5} {ro:<7} {np_:>12,} "
          f"{str(emb):>5} {str(conv):>5} {('' if npol is None else f'{npol:,}'):>13}  {run}")

"""Count the unique parameters in a run's pRNN and actor-critic checkpoints.

Both state dicts register some tensors under MORE THAN ONE name, so summing
`numel()` over the dict overcounts:

  * `prnn/utils/Architectures.py:162-167` aliases `W_in`/`W`/`W_out`/`bias`
    onto `rnn.cell.weight_ih` / `weight_hh` / `outlayer.<i>.weight` / `cell.bias`;
    `:285` aliases `b_out` onto `outlayer.<i>.bias`.
  * `prnn/utils/thetaRNN.py:289-291` makes the cell's `bias` the SAME Parameter
    as the LayerNorm offset `layernorm.mu` - one tensor, three names.

Dedupe is therefore by storage identity, not by name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

#: state-dict names that are aliases; the module path is the registration home.
_CANONICAL_PREFIXES = ("rnn.", "outlayer.")


@dataclass(frozen=True)
class Param:
    """One unique tensor, with every state-dict name that points at it."""

    names: tuple[str, ...]
    shape: tuple[int, ...]
    numel: int

    @property
    def canonical(self) -> str:
        for n in self.names:
            if n.startswith(_CANONICAL_PREFIXES):
                return n
        return self.names[0]


def unique_params(state: dict[str, torch.Tensor]) -> list[Param]:
    """Collapse a state dict to its unique tensors, keyed by storage."""
    seen: dict[tuple, list[str]] = {}
    meta: dict[tuple, torch.Tensor] = {}
    for name, t in state.items():
        key = (t.untyped_storage().data_ptr(), t.storage_offset(),
               tuple(t.shape), tuple(t.stride()))
        seen.setdefault(key, []).append(name)
        meta[key] = t
    return [Param(tuple(names), tuple(meta[k].shape), meta[k].numel())
            for k, names in seen.items()]


def group_of(name: str) -> str:
    """Which sub-network a canonical parameter name belongs to."""
    if name.startswith("rnn."):
        return "recurrent core (rnn.cell)"
    if name.startswith("outlayer."):
        return "prediction readout (outlayer)"
    return name.split(".")[0]


def report(label: str, state: dict[str, torch.Tensor]) -> int:
    params = unique_params(state)
    total = sum(p.numel for p in params)
    print(f"\n{label}")
    print(f"  {len(state)} state-dict keys -> {len(params)} unique parameters")
    groups: dict[str, list[Param]] = {}
    for p in params:
        groups.setdefault(group_of(p.canonical), []).append(p)
    for gname, ps in groups.items():
        sub = sum(p.numel for p in ps)
        print(f"  {gname:32s} {sub:>10,}  ({100 * sub / total:5.1f}%)")
        for p in sorted(ps, key=lambda q: q.canonical):
            alias = ""
            if len(p.names) > 1:
                others = [n for n in p.names if n != p.canonical]
                alias = f"   [= {', '.join(others)}]"
            print(f"      {p.canonical:36s} {str(p.shape):16s} {p.numel:>9,}{alias}")
    print(f"  {'TOTAL':32s} {total:>10,}")
    return total


def main(run_dir: Path) -> None:
    prnn_ckpt = run_dir / "predictiveNet_state.pt"
    policy_ckpt = run_dir / "policy.pt"
    if not policy_ckpt.is_file():
        policy_ckpt = run_dir / "status.pt"  # pre-2026-08-28 name

    print("=" * 78)
    print(run_dir)
    print("=" * 78)

    ck = torch.load(prnn_ckpt, map_location="cpu", weights_only=False)
    print(f"pRNNtype={ck['pRNNtype']}  hidden_size={ck['hidden_size']}  "
          f"obs_size={ck['obs_size']}  act_size={ck['act_size']}")
    n_prnn = report("pRNN (world model)", ck["pRNN_state_dict"])
    n_opt = sum(len(g["params"]) for g in ck["optimizer_state_dict"]["param_groups"])
    print(f"  cross-check: optimizer covers {n_opt} params "
          f"({'matches' if n_opt == len(unique_params(ck['pRNN_state_dict'])) else 'MISMATCH'})")

    st = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
    n_pol = report(f"actor-critic policy ({policy_ckpt.name})", st["model_state"])
    print(f"\n  pRNN {n_prnn:,} + policy {n_pol:,} = {n_prnn + n_pol:,}\n")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(Path(d))

"""Three statistics `arch_prnn.action_offset` predicts, on a trained checkpoint.

The A/B reports sRSA, spatial information and SWdist. Those say WHETHER the
representation changed; these say whether it changed for the reason claimed, and
two of them measure properties of the curiosity reward nobody has looked at.

1. **MSE where the observation is SHOWN vs MASKED.** `inMask` lets the real
   observation into the input on 1 phase in 6, and the target IS that
   observation - so those rows could be solved by copying rather than
   predicting. Both circuits have this; it is measured, not fixed.

2. **MSE by within-segment position.** Under offset 0 the reward pass's tail row
   carries neither an action nor a head direction, so the segment's last action
   is scored from an input the network never sees in training. Under offset 1
   that row carries both. The end-of-segment spike should be gone; if it is not,
   the explanation for it was wrong.

3. **`fwd(a[t])` decoded from `h[t]`.** The mechanism. At offset 0, row t's input
   contains the action the agent is ABOUT to take, so `h[t]` carries a bit that
   has nothing to do with where the agent is - a nuisance variable in exactly
   the statistic sRSA measures. At offset 1 that bit is not in the row at all.
   High accuracy at offset 0 and chance at offset 1 is what makes "the offset
   cleaned up the place code" an explanation rather than a coincidence.

Read the numbers with `si_coverage`'s caution in mind: a difference in a mean is
not a difference in a distribution. `n_rows` is reported for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float

#: Matches `checkpoint_series.fixed_probe`, so the trajectory is the one the
#: rest of the evaluation code measures on. Forward-weighted: a rollout of pure
#: turns says nothing about how the forward bit is carried.
PROBE_ACTION_P = (0.15, 0.15, 0.6, 0.1)
PROBE_SEED = 20260829


@dataclass(frozen=True)
class CircuitDiagnostics:
    """One checkpoint's answer to the three questions above."""

    action_offset: int
    n_segments: int
    n_rows: int
    mse_shown: float
    mse_masked: float
    mse_by_position: Float[np.ndarray, " rows"]
    #: Held-out BALANCED accuracy, so an imbalanced forward rate cannot flatter it.
    forward_accuracy: float
    forward_chance: float

    def summary(self) -> str:
        ratio = self.mse_shown / self.mse_masked if self.mse_masked else float("nan")
        return "\n".join([
            f"action_offset={self.action_offset}  "
            f"({self.n_segments} segments, {self.n_rows} rows)",
            f"  MSE shown  {self.mse_shown:.6f}   masked {self.mse_masked:.6f}"
            f"   ratio {ratio:.3f}",
            f"  MSE by within-segment position: first {self.mse_by_position[0]:.6f}"
            f"  median {np.median(self.mse_by_position):.6f}"
            f"  LAST {self.mse_by_position[-1]:.6f}",
            f"  fwd(a[t]) from h[t]: balanced accuracy {self.forward_accuracy:.3f}"
            f"  (chance {self.forward_chance:.3f})",
        ])


def collect_segments(*, env, n_segments: int, steps: int, seed: int = PROBE_SEED):
    """Random-action segments, keeping the RAW actions and observation dicts.

    `checkpoint_series.fixed_probe` returns observations already encoded for one
    circuit, which is exactly what has to differ here - so this keeps the raw
    pieces and lets the adapter encode them per circuit.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    segments = []
    for _ in range(n_segments):
        obs = env.reset()
        obss, acts = [], []
        for _ in range(steps):
            a = int(rng.choice(len(PROBE_ACTION_P), p=PROBE_ACTION_P))
            obss.append(obs)
            acts.append(a)
            obs = env.step(np.array([a]))[0]
        segments.append((obss, np.array(acts), obs))
    return segments


def measure(*, pN, adapter, segments) -> CircuitDiagnostics:
    """Run every segment's reward pass and reduce it to the three statistics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    from curious_george.models.device import eval_mode
    from curious_george.models.prnn_adapter import FORWARD_IDX

    rows_mse, rows_h, rows_pending, rows_shown, per_position = [], [], [], [], []
    mask = np.asarray(pN.pRNN.inMask, dtype=bool)
    with eval_mode(pN.pRNN), torch.no_grad():
        for obss, acts, last in segments:
            obs_f, act_f = adapter.seq2pred(
                *adapter.reward_pass_inputs(obss, acts, last, 1)
            )
            if not adapter.action_offset:
                act_f = act_f.clone()
                act_f[:, -1, :] = 0
            pred, tgt, h = pN.predict(obs_f, act_f, state=torch.zeros(
                (1, 1, pN.hidden_size), device=obs_f.device
            ))
            mse = ((pred - tgt) ** 2).mean(dim=2)[0].cpu().numpy()
            rows_mse.append(mse)
            rows_shown.append(np.resize(mask, len(mse)))
            per_position.append(mse)
            # `h[t]` against the action the agent is ABOUT to take at step t.
            # Only rows with a pending action qualify; the tail row has none.
            usable = min(len(mse), len(acts))
            rows_h.append(h.squeeze(0)[:usable].cpu().numpy())
            rows_pending.append((acts[:usable] == FORWARD_IDX).astype(int))

    mse = np.concatenate(rows_mse)
    shown = np.concatenate(rows_shown)
    lengths = {len(m) for m in per_position}
    assert len(lengths) == 1, f"segments must be equal length, got {lengths}"

    X = np.concatenate(rows_h)
    y = np.concatenate(rows_pending)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    chance = balanced_accuracy_score(
        y_te, np.full_like(y_te, int(y_tr.mean() > 0.5))
    )
    return CircuitDiagnostics(
        action_offset=adapter.action_offset,
        n_segments=len(segments),
        n_rows=len(mse),
        mse_shown=float(mse[shown].mean()),
        mse_masked=float(mse[~shown].mean()),
        mse_by_position=np.stack(per_position).mean(axis=0),
        forward_accuracy=float(balanced_accuracy_score(y_te, clf.predict(X_te))),
        forward_chance=float(chance),
    )

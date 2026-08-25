"""Linear decodability of object presence from h, split by input-mask phase.

`thRNN_5win` is a MaskedRNN with `inMask = [True, False x5]`: the observation
reaches the network only 1 timestep in 6, while `outMask` is all-True so it
must predict the observation at every step. Phases 1-5 are therefore driven by
actions and recurrence alone - pure memory.

Pooling all timesteps conflates "the object is in the input right now" with
"the network remembers the object", which are completely different claims.
Always split.

Paired probes (object-present / object-absent) share trajectories exactly, so
each timestep contributes a matched pair. Train/test split is by TRAJECTORY.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression

from prnn.utils import PredictiveNet
from prnn.utils.Shell import FaramaMinigridShell

from scripts.trace.trace_probe import Probe, replay_checkpoint
from tasks.omt.metrics import get_view_coords_batch

MASK_PERIOD = 6  # len(pRNN.inMask) for thRNN_5win


def presence_decoding(
    *,
    pN: PredictiveNet,
    probe_absent: Probe,
    probe_present: Probe,
    obj_loc: tuple[int, int],
    onset: int = 20,
    mask_period: int = MASK_PERIOD,
    C: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Held-out accuracy decoding object presence from h, per mask phase.

    Returns {"input_phase": acc, "memory_phase": acc}. Chance is 0.5.
    The two probes must share trajectories (asserted).
    """
    assert np.array_equal(probe_absent.agent_pos, probe_present.agent_pos), (
        "probes must share trajectories for a paired comparison"
    )
    ha = replay_checkpoint(pN=pN, probe=probe_absent)[:, onset:, :].numpy()
    ho = replay_checkpoint(pN=pN, probe=probe_present)[:, onset:, :].numpy()
    B, T, _ = ha.shape

    pos = probe_absent.agent_pos[:, onset:onset + T, :]
    hd = probe_absent.agent_dir[:, onset:onset + T]
    vx, vy = get_view_coords_batch(obj_loc[0], obj_loc[1], pos.reshape(-1, 2), hd.reshape(-1))
    in_view = ((vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7)).reshape(B, T)
    # phase of each column in the ORIGINAL sequence, before the onset slice
    phase = ((np.arange(T) + onset) % mask_period)[None, :].repeat(B, 0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    train_idx, test_idx = perm[: B // 2], perm[B // 2:]

    out: dict[str, float] = {}
    for key, sel in (("input_phase", phase == 0), ("memory_phase", phase != 0)):
        def build(idx: np.ndarray):
            m = in_view[idx] & sel[idx]
            X = np.concatenate([ha[idx][m], ho[idx][m]])
            y = np.r_[np.zeros(int(m.sum())), np.ones(int(m.sum()))]
            return X, y

        Xtr, ytr = build(train_idx)
        Xte, yte = build(test_idx)
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xtr, ytr)
        out[key] = float(clf.score(Xte, yte))
    return out

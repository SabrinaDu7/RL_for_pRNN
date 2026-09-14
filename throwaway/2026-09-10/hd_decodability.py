"""Is head direction linearly decodable from the pRNN hidden state?

The question behind the `with_HD=False` ablation. The policy's only input there
is `h`, and the pRNN is fed SpeedHD - head direction IS in its input - so either
the information survives into `h` and the linear actor simply cannot read it
(a capacity problem, fixable with a bigger actor head), or it does not survive
(a representation problem, and a different fix).

Same shape as `evaluation/circuit_diagnostics.measure`: one seeded random-action
probe, identical for every checkpoint, logistic regression on `h`, held-out
BALANCED accuracy so an imbalanced head-direction distribution cannot flatter
it. Chance is 0.25 over four directions.

Throwaway: a one-off diagnostic. Nothing in the repo reads it, and no committed
result may depend on it.

    uv run python throwaway/2026-09-10/hd_decodability.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FETCHED = Path("outputs/fetched")
RUNS = {
    "s3 off0 HD on ": "mx-impassable-n8-s3-mse-off0_curious_26-09-13-15-29-36",
    "s3 off0 HD OFF": "mx-impassable-n8-s3-mse-off0-nohd_curious_26-09-13-15-45-24",
    "s2 off0 HD on ": "mx-impassable-n8-s2-mse-off0_curious_26-09-10-23-58-15",
}
N_SEGMENTS, STEPS = 8, 256


def rows_for(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(h rows, head direction per row) for one run's final checkpoint."""
    from curious_george.configs import Config
    from curious_george.evaluation.checkpoint_series import archived, build
    from curious_george.evaluation.circuit_diagnostics import collect_segments
    from curious_george.models.device import eval_mode
    from curious_george.models.prnn_adapter import PRNNAdapter
    from curious_george.envs.layouts import resolve_layouts

    cfg = Config.of_run(run_dir)
    layouts = resolve_layouts(cfg)
    pN, env = build(cfg=cfg, landmarks=layouts[0].landmarks, ckpt=str(archived(run_dir)[-1][1]))
    adapter = PRNNAdapter(pN, torch.device("cpu"),
                          action_offset=cfg.arch_prnn.action_offset)
    segments = collect_segments(env=env, n_segments=N_SEGMENTS, steps=STEPS)

    H, HD = [], []
    with eval_mode(pN.pRNN), torch.no_grad():
        for obss, acts, last in segments:
            obs_f, act_f = adapter.seq2pred(
                *adapter.reward_pass_inputs(obss, acts, last, 1)
            )
            if not adapter.action_offset:
                act_f = act_f.clone()
                act_f[:, -1, :] = 0
            _, _, h = pN.predict(obs_f, act_f, state=torch.zeros(
                (1, 1, pN.hidden_size), device=obs_f.device
            ))
            hd = np.array([int(o["direction"]) for o in obss])
            usable = min(h.shape[1], len(hd))
            H.append(h.squeeze(0)[:usable].cpu().numpy())
            HD.append(hd[:usable])
    return np.concatenate(H), np.concatenate(HD)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    print(f"head direction from h, held-out balanced accuracy (chance 0.250)")
    print(f"{'run':16s} {'rows':>7s} {'width':>6s} {'accuracy':>9s}")
    for label, name in RUNS.items():
        X, y = rows_for(FETCHED / name)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        clf = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
        acc = balanced_accuracy_score(y_te, clf.predict(X_te))
        print(f"{label:16s} {len(y):7d} {X.shape[1]:6d} {acc:9.3f}")


if __name__ == "__main__":
    main()

"""`UpdateLogs` must mean ONE thing, whichever path computed it.

The eager path declared its log accumulators inside the PPO epoch loop -
inherited verbatim, comment and all, from `torch_ac/algos/ppo.py:32-35` - so
the returned mean covered only the LAST epoch, one quarter of the gradient
steps at `ppo_epochs=4`. The graphed path (`_update_policy_epochs_graphed`)
accumulates across all of them. Two statistics under one name, and
graphed-vs-eager is the comparison the whole CUDA-graph effort rests on.

Decided 2026-08-25: all epochs. The quantity is logged once per update and an
update IS four passes over the batch. See docs/invalid-runs.md for what the
change invalidates - policy diagnostics only; `pRNN loss`, sRSA, SWdist and SI
do not pass through here.

These tests are deliberately about the STATISTIC, not about a value: they
recompute it from every `LossTerms` the update produced, so they hold for any
config, any device, and any future change to the loss itself.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from curious_george.rl.update.losses import LOSSES
from curious_george.rl.update.policy import update_policy

SEED = 5
EPOCHS = 4


@pytest.fixture(scope="module")
def algo_and_exps():
    from hydra import compose, initialize_config_dir

    from curious_george.training.setup import setup_training

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                "env=lroom", "run=multienv", "exp.device_env=True",
                "predNet.batched_wm=True", "predNet.wm_pool_group=8",
                "exp.num_envs=8", "rl.frames=2048", "rl.ppo_batch_size=256",
                f"rl.ppo_epochs={EPOCHS}", "rl.entropy_coef=0",
                "logging.wandb_log=false", f"exp.seed={SEED}",
            ],
        )
    comps = setup_training(cfg)
    exps, _ = comps.algo.collect_experiences()
    for key in ("done_indices", "last_observations"):
        if key in exps:
            del exps[key]
    return comps.algo, comps.acmodel, exps


def _run_with_spy(algo, acmodel, exps):
    """One update, recording every minibatch's LossTerms in order."""
    recorded = []
    base = LOSSES[algo.loss_name]

    def spy(dist, value, sb, **kwargs):
        loss, terms = base(dist, value, sb, **kwargs)
        recorded.append(
            {
                "entropy": float(terms.policy_entropy_bits),
                "value": float(terms.value_mean),
                "policy_loss": float(terms.policy_loss),
                "value_loss": float(terms.value_loss),
            }
        )
        return loss, terms

    logs = update_policy(
        acmodel,
        algo.optimizer,
        exps,
        loss_fn=spy,
        loss_kwargs=dict(
            clip_eps=algo.clip_eps,
            entropy_coef=algo.entropy_coef,
            value_loss_coef=algo.value_loss_coef,
        ),
        epochs=EPOCHS,
        batch_size=algo.batch_size,
        num_frames=algo.num_frames,
        max_grad_norm=algo.max_grad_norm,
        update_params=True,
        graph_trainer=None,
    )
    return logs, recorded


@pytest.fixture(scope="module")
def logs_and_terms(algo_and_exps):
    return _run_with_spy(*algo_and_exps)


@pytest.mark.parametrize("field", ["entropy", "value", "policy_loss", "value_loss"])
def test_eager_logs_average_every_epoch(logs_and_terms, field):
    """The returned mean is over ALL gradient steps in the update."""
    logs, recorded = logs_and_terms
    every = [row[field] for row in recorded]
    assert getattr(logs, field) == pytest.approx(float(np.mean(every)), rel=1e-5)


@pytest.mark.parametrize("field", ["entropy", "value", "policy_loss", "value_loss"])
def test_last_epoch_only_is_a_different_number(logs_and_terms, field):
    """The gate above has power: the two conventions genuinely disagree.

    Without this, `test_eager_logs_average_every_epoch` would pass vacuously on
    any config where the policy happens not to move within an update.
    """
    logs, recorded = logs_and_terms
    per_epoch = len(recorded) // EPOCHS
    assert per_epoch >= 1 and len(recorded) == per_epoch * EPOCHS
    last_epoch = float(np.mean([row[field] for row in recorded[-per_epoch:]]))
    assert getattr(logs, field) != pytest.approx(last_epoch, rel=1e-5), (
        f"{field}: last-epoch mean equals the all-epoch mean, so this config "
        "cannot distinguish the two conventions and the gate above is vacuous"
    )


def test_the_bias_has_the_predicted_sign(logs_and_terms):
    """Within one update the ratio starts at exactly 1 and the policy sharpens,
    so the OLD last-epoch convention reported lower `value_loss` and a
    larger-magnitude `policy_loss`. Recording the direction here means a future
    reader can tell a mechanism change from a value change."""
    logs, recorded = logs_and_terms
    per_epoch = len(recorded) // EPOCHS
    last = {
        field: float(np.mean([row[field] for row in recorded[-per_epoch:]]))
        for field in ("value_loss", "policy_loss")
    }
    assert last["value_loss"] < logs.value_loss
    assert abs(last["policy_loss"]) > abs(logs.policy_loss)

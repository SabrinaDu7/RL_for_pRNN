"""A random agent is the policy's BASELINE, so it must differ in ONE thing.

It used to be a separate serial routine that forced `num_envs == 1`, which made
a random-agent multi-room run unrepresentable: `RANDOM` wanted one instance, a
room set wants the DEVICE backend, and DEVICE wants more than one instance -
three `Config` constraints that conflicted pairwise. The constraint was about
that routine, never about random actions.

Now random actions go through `collect_rollout` like any other, so the baseline
shares the backend, the batch, the rooms and the world-model training with the
arm it is a baseline for.
"""

import dataclasses

import numpy as np
import pytest
import torch

from curious_george.configs import Config, EnvBackend, EnvCfg, EvalKind, PRESETS
from curious_george.envs.layouts import Selected
from curious_george.utils.enums import AgentType


def test_a_random_agent_multi_room_config_is_now_expressible():
    """The three-way conflict that made this impossible."""
    base = PRESETS["multienv-fast"][1]
    cfg = dataclasses.replace(
        base, arch_policy=dataclasses.replace(base.arch_policy, agent=AgentType.RANDOM)
    )
    assert cfg.collect.num_envs > 1
    assert cfg.collect.backend is EnvBackend.DEVICE
    assert isinstance(cfg.env.source, Selected)


def test_random_actions_follow_the_projects_distribution():
    """Forward-weighted, not uniform: a uniform walker mostly spins on the spot.

    Drawn from `RAND_ACT_PROBA`, the same distribution the evaluation probes use,
    so the baseline's behaviour matches what the eval machinery assumes.
    """
    from curious_george.log_and_store.storage import RAND_ACT_PROBA
    from curious_george.rl.collect.collector import _random_actions

    torch.manual_seed(0)
    draws = _random_actions(40_000, torch.device("cpu")).numpy()
    seen = np.bincount(draws, minlength=len(RAND_ACT_PROBA)) / len(draws)
    assert np.allclose(seen, RAND_ACT_PROBA, atol=0.01), f"{seen} vs {RAND_ACT_PROBA}"


@pytest.mark.parametrize("agent", (AgentType.AC, AgentType.RANDOM))
def test_both_agents_collect_through_the_same_path(agent):
    """The baseline runs the real collector, at num_envs > 1, and trains the pRNN.

    Also the regression: `algo.randomAgent_collect_exp_and_update` asserted
    `num_envs == 1`, so this would have raised for RANDOM.
    """
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config(backend=EnvBackend.SERIAL_TABLE)
    cfg = dataclasses.replace(
        cfg, arch_policy=dataclasses.replace(cfg.arch_policy, agent=agent)
    )
    assert cfg.collect.num_envs > 1, "the point is that this is not single-env"
    algo = setup_training(cfg).algo
    exps, logs = algo.collect_experiences()
    assert len(exps.action) == cfg.collect.num_envs * (
        algo.num_frames // cfg.collect.num_envs
    )
    # every action is a legal index, whichever agent produced it
    acts = np.asarray(exps.action.cpu()).reshape(-1)
    assert acts.min() >= 0 and acts.max() < 4

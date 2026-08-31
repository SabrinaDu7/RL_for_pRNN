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
    from curious_george.rl.collect.collector import _random_actions, random_action_probs

    torch.manual_seed(0)
    probs = random_action_probs(40_000, torch.device("cpu"))
    draws = _random_actions(probs).numpy()
    seen = np.bincount(draws, minlength=len(RAND_ACT_PROBA)) / len(draws)
    assert np.allclose(seen, RAND_ACT_PROBA, atol=0.01), f"{seen} vs {RAND_ACT_PROBA}"


def test_a_uniform_random_baseline_is_expressible():
    """The OTHER random baseline. The two differ ~2x on coverage (nAUC 0.108
    vs 0.224 - `python -m curious_george.envs.action_graph`), so "random"
    without its distribution is ambiguous. The distribution is a config field
    whose default is the ONE home, `configs.RAND_ACT_PROBA` - the four
    independent spellings are gone."""
    from curious_george.configs import RAND_ACT_PROBA
    from curious_george.log_and_store.storage import RAND_ACT_PROBA as stored

    base = PRESETS["multienv-fast"][1]
    cfg = dataclasses.replace(
        base,
        arch_policy=dataclasses.replace(
            base.arch_policy,
            agent=AgentType.RANDOM,
            random_action_probs=(0.25, 0.25, 0.25, 0.25),
        ),
    )
    assert cfg.arch_policy.random_action_probs == (0.25,) * 4
    assert base.arch_policy.random_action_probs == RAND_ACT_PROBA
    assert tuple(stored) == RAND_ACT_PROBA, "storage's ndarray must derive from the one home"


def test_the_collector_samples_the_configured_distribution():
    """`arch_policy.random_action_probs` reaches the sampler: uniform in,
    uniform out - not the forward-weighted default."""
    from curious_george.rl.collect.collector import _random_actions, random_action_probs

    torch.manual_seed(0)
    probs = random_action_probs(40_000, torch.device("cpu"), probs=(0.25,) * 4)
    seen = np.bincount(_random_actions(probs).numpy(), minlength=4) / 40_000
    assert np.allclose(seen, 0.25, atol=0.01), f"{seen}"


def test_a_uniform_run_walks_uniformly():
    """End-to-end: the config field reaches the rollout's action stream."""
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config(backend=EnvBackend.SERIAL_TABLE)
    cfg = dataclasses.replace(
        cfg,
        arch_policy=dataclasses.replace(
            cfg.arch_policy,
            agent=AgentType.RANDOM,
            random_action_probs=(0.25, 0.25, 0.25, 0.25),
        ),
    )
    algo = setup_training(cfg).algo
    acts = np.concatenate([
        np.asarray(algo.collect_experiences()[0].action.cpu()).reshape(-1)
        for _ in range(4)
    ])
    seen = np.bincount(acts, minlength=4) / len(acts)
    assert np.allclose(seen, 0.25, atol=0.07), f"{seen}"
    assert (acts == 2).mean() < 0.4, "0.6 forward means the DEFAULT distribution leaked through"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to capture")
def test_random_actions_survive_the_rollout_cuda_graph():
    """The regression that cost two cluster jobs.

    Building the probability table inside the captured region is a
    host-to-device copy, and capture REFUSES those - "operation not permitted
    when stream is capturing" - rather than degrading. The table is therefore
    built in `GraphRolloutStepper.prepare`, before `_capture_all`.
    """
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config(backend=EnvBackend.DEVICE)
    cfg = dataclasses.replace(
        cfg,
        collect=dataclasses.replace(cfg.collect, rollout_cuda_graph=True),
        arch_policy=dataclasses.replace(cfg.arch_policy, agent=AgentType.RANDOM),
    )
    algo = setup_training(cfg).algo
    seen = []
    for _ in range(2):  # capture on the first, replay on the second
        exps, _ = algo.collect_experiences()
        seen.append(np.asarray(exps.action.cpu()).reshape(-1))
    acts = np.concatenate(seen)
    assert acts.min() >= 0 and acts.max() < 4
    # forward-weighted, not uniform; loose because the sample is small
    assert acts.mean() > 0, "all-zero actions means the draw never ran"
    assert 0.35 < (acts == 2).mean() < 0.85, f"forward fraction {(acts == 2).mean()}"


def test_the_random_baseline_trains_its_world_model():
    """The bug that made the first baseline meaningless.

    `update_params` gated the POLICY and the world model together, so passing
    False to stop policy updates silently trained nothing - and the run scored
    sRSA 0.06, which is the UNTRAINED score and looks like a finding.
    """
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config(backend=EnvBackend.SERIAL_TABLE)
    cfg = dataclasses.replace(
        cfg, arch_policy=dataclasses.replace(cfg.arch_policy, agent=AgentType.RANDOM)
    )
    algo = setup_training(cfg).algo
    before = [p.detach().clone() for p in algo.pN.pRNN.parameters()]
    policy_before = [p.detach().clone() for p in algo.acmodel.parameters()]

    exps, _ = algo.collect_experiences()
    algo.update_parameters(exps=exps, update_params=False, update_world_model=True)

    moved = any(not torch.equal(a, b)
                for a, b in zip(before, algo.pN.pRNN.parameters()))
    frozen = all(torch.equal(a, b)
                 for a, b in zip(policy_before, algo.acmodel.parameters()))
    assert moved, "the world model did not train - the baseline would be an untrained net"
    assert frozen, "the policy trained; the baseline is supposed to hold it still"

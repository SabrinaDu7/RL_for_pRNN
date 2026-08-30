"""The typed config: budgets round-trip, and invalid states do not construct.

The point of re-rooting the budget on gradient steps is that two arms budgeted
the same way are actually comparable. That only holds if the derivation is
right, so the three presets are pinned against the numbers the YAML tree they
replace produced.
"""

import json

import pytest

from curious_george.envs.layouts import ROOMS_RUN1
from curious_george.configs import (
    PRESETS,
    AgentType,
    ArchPolicyCfg,
    CollectCfg,
    Config,
    EnvBackend,
    EvalCfg,
    EvalKind,
    Committed,
    EnvCfg,
    RunCfg,
    TrainPolicyCfg,
    TrainPrnnCfg,
    cli,
)

#: What each preset's YAML ancestor produced. Verified against the pre-migration
#: tree by tests/golden/capture_golden_setup.py, which recorded the same three
#: budgets from the composed Hydra configs.
EXPECTED = {
    #             prnn grad,  policy grad,   env steps,  ppo_batch, rollout
    "reference": (   80_000,      320_000,  20_480_000,        256,   2_048),
    "multienv":  (  240_000,    3_840_000, 491_520_000,        512,   2_048),
    "ultra":     (      625,        5_000,  20_480_000,     16_384,  32_768),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_preset_budget_matches_the_config_it_replaces(name):
    """Re-rooting on gradient steps must reproduce the old budgets exactly."""
    prnn, policy, env_steps, batch, rollout = EXPECTED[name]
    cfg = PRESETS[name][1]
    assert cfg.train_prnn.total_grad_steps == prnn
    assert cfg.train_policy.total_grad_steps == policy
    assert cfg.total_env_steps == env_steps
    assert cfg.ppo_batch_size == batch
    assert cfg.collect.env_steps_per_rollout == rollout


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_two_learners_agree_on_total_experience(name):
    """`total_fresh_env_steps` and the pRNN's `total_env_steps` are one run
    measured two ways. If they disagree, a derivation is wrong and there is no
    way to tell which from a plot."""
    cfg = PRESETS[name][1]
    assert cfg.train_policy.total_fresh_env_steps(cfg.total_env_steps) == cfg.total_env_steps


def test_processed_transitions_exceed_fresh_by_exactly_ppo_epochs():
    """The whole reason the policy has two rate properties: it replays the same
    transitions, so compute volume and experience differ by ppo_epochs."""
    cfg = PRESETS["reference"][1]
    total = cfg.total_env_steps
    assert cfg.train_policy.processed_transitions_per_grad_step(total) == (
        cfg.train_policy.fresh_env_steps_per_grad_step(total) * cfg.train_policy.ppo_epochs
    )
    assert cfg.train_policy.total_processed_transitions(total) == total * cfg.train_policy.ppo_epochs


def test_rollout_size_does_not_change_the_training_budget():
    """`frames` was never a scientific knob, and the new shape must keep it that
    way: varying the rollout 16x leaves both budgets untouched, because they are
    stated rather than derived from it."""
    base = PRESETS["reference"][1]
    wide = Config(
        collect=CollectCfg(num_envs=128, backend=EnvBackend.DEVICE),
        train_prnn=base.train_prnn,
        train_policy=base.train_policy,
    )
    assert wide.collect.env_steps_per_rollout == 16 * base.collect.env_steps_per_rollout
    assert wide.total_env_steps == base.total_env_steps
    assert wide.train_prnn.total_grad_steps == base.train_prnn.total_grad_steps
    assert wide.train_policy.total_grad_steps == base.train_policy.total_grad_steps


@pytest.mark.parametrize(
    "label,build",
    [
        ("random agent needs one instance",
         lambda: Config(arch_policy=ArchPolicyCfg(agent=AgentType.RANDOM))),
        ("intrinsic needs one instance",
         lambda: Config(train_policy=TrainPolicyCfg(intrinsic=True))),
        ("early_stop needs measurable return",
         lambda: Config(collect=CollectCfg(backend=EnvBackend.DEVICE), run=RunCfg(early_stop=True))),
        ("grad-step group must divide the rollout",
         lambda: Config(train_prnn=TrainPrnnCfg(episodes_per_grad_step=3))),
        ("a room set needs the device backend",
         lambda: Config(env=EnvCfg(source=Committed(rooms=ROOMS_RUN1)),
                        eval=EvalCfg(evals=frozenset({EvalKind.SPATIAL_MULTIROOM})))),
        ("the multi-room eval needs a room set",
         lambda: Config(env=EnvCfg(),
                        eval=EvalCfg(evals=frozenset({EvalKind.SPATIAL_MULTIROOM})))),
        ("device batches, so it needs more than one instance",
         lambda: CollectCfg(backend=EnvBackend.DEVICE, num_envs=1)),
        ("a rollout graph needs the device backend",
         lambda: CollectCfg(rollout_cuda_graph=True)),
    ],
)
def test_invalid_states_do_not_construct(label, build):
    """Each of these was a runtime failure mid-run, or a silent no-op."""
    with pytest.raises(ValueError):
        build()


def test_to_dict_is_json_serialisable_and_keeps_subclass_identity():
    """provenance.json and the wandb config record both go through this.
    `dataclasses.asdict` alone emits enums, Paths and frozensets that json
    refuses or orders nondeterministically - and would erase which environment
    source and shape were used - which the env id alone does not say."""
    for name, (_, cfg) in PRESETS.items():
        blob = json.dumps(cfg.to_dict())
        assert "${" not in blob, f"{name}: unresolved interpolation"
        assert json.loads(blob)["env"]["source"]["_type"] == type(cfg.env.source).__name__


def test_cli_selects_presets_and_applies_overrides():
    """The subcommand mechanism replaces Hydra's cross-namespace group
    selection: one name patches collection, both learners, eval and run."""
    assert cli(["reference"]).collect.backend is EnvBackend.SERIAL
    assert cli(["multienv"]).collect.backend is EnvBackend.DEVICE
    assert cli(["multienv", "--run.seed", "3"]).run.seed == 3
    assert cli(["ultra"]).collect.num_envs == 128


def test_a_ramp_under_the_policy_graph_is_refused():
    """A captured policy step bakes `entropy_coef` in, so the ramp cannot fire.

    `algo.py` builds GraphPolicyTrainer once with the coefficient as a Python
    float, `policy_graph._region` captures that float, and graphed updates route
    to `_update_policy_epochs_graphed`, which never re-reads it. The coefficient
    is not logged either, so a dead ramp is invisible - one cluster run was
    spent before this combination was made unrepresentable.
    """
    import pytest

    from curious_george.configs import TrainPolicyCfg

    with pytest.raises(ValueError, match="silently never happen"):
        TrainPolicyCfg(entropy_coef=0.001, entropy_coef_final=0.01, cuda_graph=True)

    # Either alone is fine, and so is a ramp on the eager path.
    TrainPolicyCfg(entropy_coef=0.001, entropy_coef_final=0.01, cuda_graph=False)
    TrainPolicyCfg(entropy_coef=0.003, cuda_graph=True)


def test_parity_defaults_to_the_measured_entropy_knee():
    """0.003, not the 0.001 the original reference ran.

    The knee is the lowest coefficient with a 0.0% collapse duty cycle across 5
    seeds; see docs/entropy-sweep-and-noise-floor-2026-08-29.md.
    """
    from curious_george.configs import PRESETS

    cfg = PRESETS["parity"][1]
    assert cfg.train_policy.entropy_coef == 0.003
    assert cfg.train_policy.entropy_coef_final is None
    assert cfg.arch_prnn.action_offset == 0, "the circuit must still be typed to change"

"""rl.cuda_graph: graphed PPO minibatch step equivalence.

The actor-critic has no dropout and no injected noise, so a graphed step has no
RNG of its own: given the SAME minibatch indices it must move the weights
BITWISE like the eager step it replaces.

One thing has to be held equal for that to be true. Capture needs Adam rebuilt
`capturable=True`, which keeps `step` as a device tensor and computes the bias
correction on-device - a different order of float32 operations than the default
path, worth ~3e-8 on the weights. That is Adam's difference, not the graph's, so
the eager arm is given the SAME capturable optimizer and the comparison is then
exactly bitwise. `test_capturable_adam_is_the_only_numerical_difference` pins
the size of what capturable costs on its own, so this stays honest rather than
being a loosened tolerance.

The property that matters beyond one step is that the in-graph optimizer step
ADVANCES the weights between replays - a graph replaying at frozen weights
would pass a single-step equivalence check and silently train nothing.

CUDA-only; skipped on CPU boxes.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph path requires a GPU"
)

SEED = 3
STEPS = 4


def _algo(policy_cuda_graph: bool):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from curious_george.training.setup import setup_training

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=[
            "env=lroom", "run=multienv", "exp.device_env=True",
            "predNet.batched_wm=True", "predNet.wm_pool_group=8",
            "exp.num_envs=8", "rl.frames=2048", "rl.ppo_batch_size=256",
            "rl.entropy_coef=0", "logging.wandb_log=false", "exp.seed=2",
            f"rl.cuda_graph={policy_cuda_graph}",
        ])
    comps = setup_training(cfg)
    return comps.algo, comps.acmodel, cfg


def _weights(acmodel) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in acmodel.parameters()])


def _make_capturable(optimizer):
    """Adam rebuilt capturable=True, preserving every param group."""
    groups = []
    for g in optimizer.param_groups:
        ng = {k: v for k, v in g.items() if k != "params"}
        ng["params"] = list(g["params"])
        ng["capturable"] = True
        groups.append(ng)
    return torch.optim.Adam(groups)


def _run(policy_cuda_graph: bool, index_sets, *, capturable_eager: bool = False):
    """Take len(index_sets) gradient steps on FIXED minibatches, so the eager
    and graphed arms see identical data and the comparison is about the
    mechanism rather than about the shuffle."""
    from curious_george.rl.update.updater import _index_policy_batch
    from curious_george.rl.update.losses import LOSSES

    algo, acmodel, cfg = _algo(policy_cuda_graph)
    exps, _ = algo.collect_experiences()
    for k in ("done_indices", "last_observations"):
        if k in exps:
            del exps[k]
    loss_kwargs = dict(clip_eps=algo.clip_eps, entropy_coef=algo.entropy_coef,
                       value_loss_coef=algo.value_loss_coef)
    dev = algo.device
    seen = []

    if policy_cuda_graph:
        from curious_george.rl.update.policy_graph import GraphPolicyTrainer

        trainer = GraphPolicyTrainer(
            acmodel, algo.optimizer, loss_fn=LOSSES[algo.loss_name],
            loss_kwargs=loss_kwargs, max_grad_norm=algo.max_grad_norm,
        )
        trainer.bind(exps)
        for inds in index_sets:
            trainer.step(torch.as_tensor(inds, device=dev, dtype=torch.long))
            seen.append(_weights(acmodel).clone())
        return acmodel, seen, trainer

    if capturable_eager:
        algo.optimizer = _make_capturable(algo.optimizer)
    for inds in index_sets:
        idx = torch.as_tensor(inds, device=dev, dtype=torch.long)
        sb = _index_policy_batch(exps, idx, acmodel)
        dist, value = acmodel(sb.obs, SR=sb.SR)
        loss, _ = LOSSES[algo.loss_name](dist, value, sb, **loss_kwargs)
        algo.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(acmodel.parameters(), algo.max_grad_norm)
        algo.optimizer.step()
        seen.append(_weights(acmodel).clone())
    return acmodel, seen, None


def _fixed_index_sets(n_steps: int, batch: int, num_frames: int):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(num_frames)[:batch] for _ in range(n_steps)]


def test_graphed_policy_step_bitwise_equals_eager():
    """Identical minibatches, identical optimizer => identical weights, bitwise.

    This is the gate on the GRAPH: replaying a capture must be the same
    computation as dispatching it. Adam is capturable in both arms so the only
    thing under test is the mechanism.
    """
    sets = _fixed_index_sets(STEPS, 256, 2048)
    _, w_eager, _ = _run(False, sets, capturable_eager=True)
    _, w_graph, _ = _run(True, sets)
    assert torch.equal(w_eager[-1], w_graph[-1]), (
        f"graphed != eager: max|d|={(w_eager[-1] - w_graph[-1]).abs().max().item():.3e}"
    )


def test_capturable_adam_is_the_only_numerical_difference():
    """What turning the graph on costs numerically, stated rather than assumed.

    Capture forces `capturable=True`, which is a real (tiny) change to Adam's
    arithmetic that a graphed run inherits whether or not it wanted it. Pin its
    size: same data, same schedule, only the optimizer flag differs. If this
    ever grows past float32 rounding, the graph is no longer free and the curve
    gate has to be re-run.
    """
    sets = _fixed_index_sets(STEPS, 256, 2048)
    _, w_default, _ = _run(False, sets)
    _, w_capturable, _ = _run(False, sets, capturable_eager=True)
    delta = (w_default[-1] - w_capturable[-1]).abs().max().item()
    assert delta < 1e-6, f"capturable Adam diverges by {delta:.3e}, not rounding"
    assert not torch.equal(w_default[-1], w_capturable[-1]), (
        "capturable Adam is now bitwise-identical - simplify this test and the "
        "one above, which only exists because it was not"
    )


def test_graphed_policy_advances_weights_every_replay():
    sets = _fixed_index_sets(STEPS, 256, 2048)
    _, seen, _ = _run(True, sets)
    for i in range(1, len(seen)):
        assert not torch.equal(seen[i - 1], seen[i]), f"replay {i} left weights unchanged"
    assert torch.isfinite(seen[-1]).all()


def test_ragged_minibatch_gets_its_own_graph():
    """`ppo_batch_size` need not divide `rl.frames`; the short final minibatch
    is a different shape and must capture separately."""
    sets = [np.arange(256), np.arange(100)]
    _, _, trainer = _run(True, sets)
    assert set(trainer.graphs) == {256, 100}, f"graph keys {set(trainer.graphs)}"


def test_distribution_validation_is_restored_after_capture():
    """`_no_dist_validation` touches a PROCESS-GLOBAL torch setting; leaving it
    off would silently disable arg checking for every later Categorical in the
    interpreter, including in eval code that never asked for it."""
    before = torch.distributions.Distribution._validate_args
    _run(True, _fixed_index_sets(1, 256, 2048))
    assert torch.distributions.Distribution._validate_args == before


def test_graphed_policy_survives_spatial_evals():
    """The policy graph has the same exposure as the world model's.

    `evaluate_spatial_representation` appends `agent.acmodel` to the modules it
    moves to CPU, so a captured PPO step is subject to exactly the failure that
    made a graphed world-model run train on nothing: `Module._apply` replaces
    `param.data` and buffers and mutates `param.grad` in place, and a graph
    holds all of them by address. Asserting "finite" would not catch it - the
    weights must move EXACTLY as eager moves them, across the eval.
    """
    from curious_george.rl.update.losses import LOSSES
    from curious_george.rl.update.updater import _index_policy_batch
    from curious_george.world_model.device import on_device

    sets = _fixed_index_sets(STEPS, 256, 2048)
    results = {}
    for graphed in (False, True):
        algo, acmodel, _ = _algo(graphed)
        exps, _ = algo.collect_experiences()
        for k in ("done_indices", "last_observations"):
            if k in exps:
                del exps[k]
        lk = dict(clip_eps=algo.clip_eps, entropy_coef=algo.entropy_coef,
                  value_loss_coef=algo.value_loss_coef)
        dev = algo.device
        trainer = None
        if graphed:
            from curious_george.rl.update.policy_graph import GraphPolicyTrainer

            trainer = GraphPolicyTrainer(
                acmodel, algo.optimizer, loss_fn=LOSSES[algo.loss_name],
                loss_kwargs=lk, max_grad_norm=algo.max_grad_norm,
            )
            trainer.bind(exps)
        else:
            # capturable in both arms, so only the mechanism is under test
            algo.optimizer = _make_capturable(algo.optimizer)

        deltas = []
        for inds in sets:
            idx = torch.as_tensor(inds, device=dev, dtype=torch.long)
            before = _weights(acmodel)
            if graphed:
                trainer.step(idx)
            else:
                sb = _index_policy_batch(exps, idx, acmodel)
                dist, value = acmodel(sb.obs, SR=sb.SR)
                loss, _ = LOSSES[algo.loss_name](dist, value, sb, **lk)
                algo.optimizer.zero_grad(set_to_none=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(acmodel.parameters(), algo.max_grad_norm)
                algo.optimizer.step()
            deltas.append((_weights(acmodel) - before).norm().item())
            with on_device([acmodel], "cpu"):  # the analysis event
                pass
        results[graphed] = deltas

    for k, (e, g) in enumerate(zip(results[False], results[True])):
        assert g == pytest.approx(e, rel=1e-4), (
            f"step {k} after an eval: graphed moved the weights by {g:.6e}, "
            f"eager by {e:.6e} - the policy graph is training on stranded memory"
        )

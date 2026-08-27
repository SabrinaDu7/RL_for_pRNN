"""exp.rollout_cuda_graph: graphed rollout timestep equivalence.

A CUDA graph fails SILENTLY by construction - it replays kernels against
addresses it baked in at capture, so a stale address produces plausible
numbers rather than an error. Every gate here therefore compares the ACTUAL
rollout tensors against the eager path, never a summary statistic:

- `test_graphed_rollout_bitwise_equals_eager` - with pRNN noise and dropout
  off and the policy saturated (see `_saturate`), the rollout has no RNG at
  all, so graphed and eager must agree BITWISE across several updates.
- `test_graphed_rollout_survives_repeated_spatial_evals` - the same, with an
  `on_device` round trip between rollouts. This is the shape of the failure
  that cost a full cluster run (tests/test_cuda_graph_wm.py has the story).
- `test_bitwise_gate_catches_a_poisoned_static_buffer` - the negative control.
  A gate that passes either way is worthless, so this poisons a buffer the
  graphs read and requires the comparison above to notice.
- `test_replay_draws_fresh_randomness` - from an IDENTICAL restored state,
  repeated replays must sample different actions and land on different pRNN
  states. A graph that froze its RNG would make the agent deterministic while
  every other number still looked right.

`predNet.compile_cell` is deliberately absent: it is orthogonal to capture
(the compiled cell is simply what gets recorded) and a fresh inductor compile
costs more than the rest of this file put together. The combination is
exercised by throwaway/hydra_era/perf/benchmark.py, which runs with compile_cell=layer.

CUDA-only; skipped on CPU boxes.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from curious_george.training.setup import setup_training
from tests.small_config import small_config

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph path requires a GPU"
)

# Fields of `exps` that a rollout produces and the graph could corrupt.
_EXP_FIELDS = ("SR", "action", "value", "reward", "advantage", "returnn", "log_prob")


class _Arm:
    """One algo plus its OWN random-number stream.

    Both arms live in one process and PPO shuffles its minibatches from
    numpy's global RNG, so without this the second arm to run each update
    draws a different permutation and its weights drift by a float32 ULP -
    a difference in the test harness that would read as a graph defect.
    """

    def __init__(self, algo) -> None:
        self.algo = algo
        self.rng = self._capture()

    @staticmethod
    def _capture() -> tuple:
        return (
            np.random.get_state(),
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone(),
        )

    @contextmanager
    def turn(self):
        """Run this arm's work on its own stream, then bank where it got to."""
        np_state, cpu_state, cuda_state = self.rng
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        try:
            yield self.algo
        finally:
            self.rng = self._capture()


def _config(*, graphed: bool, deterministic: bool = True, **extra):
    """Two segments per rollout per env, so a mid-rollout episode boundary -
    where the mask flips and the pRNN phase resets - is inside the gate."""
    return small_config(
        rollout_cuda_graph=graphed,
        noise_std=0.0 if deterministic else 0.05,
        dropout=0.0 if deterministic else 0.15,
        **extra,
    )


def _saturate(acmodel) -> None:
    """Make the policy's SAMPLE deterministic without touching its code path.

    `Categorical.sample` is the rollout's only RNG once pRNN noise and dropout
    are off, and a captured region draws from CUDA's graph-safe stream rather
    than eager order - so a bitwise comparison needs the sampled action to
    stop depending on the draw. Scaling the actor's output layer saturates the
    softmax to an exact float32 one-hot (exp(-1e4) underflows to 0), and
    `multinomial` returns the same index for ANY uniform draw. The same
    `Categorical` is still built and the same `multinomial` still runs.

    Checkable at the far end rather than assumed: every recorded `log_prob` is
    then exactly 0, which `_assert_saturated` requires.
    """
    with torch.no_grad():
        head = acmodel.actor[-1]
        head.weight.mul_(1e5)
        head.bias.mul_(1e5)


def _assert_saturated(exps) -> None:
    assert torch.all(exps.log_prob == 0), (
        "the policy is not saturated, so the sampled action still depends on "
        "the RNG draw and a bitwise eager/graphed comparison is meaningless"
    )


def _assert_same_rollout(expected, actual, *, where: str) -> None:
    for field in _EXP_FIELDS:
        left, right = getattr(expected, field), getattr(actual, field)
        assert torch.equal(left, right), (
            f"{where}: graphed != eager on exps.{field} "
            f"(max|d|={(left - right).abs().max().item():.3e})"
        )
    assert torch.equal(expected.obs.image, actual.obs.image), f"{where}: obs.image"
    assert torch.equal(
        expected.obs.direction, actual.obs.direction
    ), f"{where}: obs.direction"
    assert expected.done_indices == actual.done_indices, f"{where}: done_indices"


def _collect_both(eager: "_Arm", graphed: "_Arm"):
    """One rollout from each arm, on its own RNG stream.

    Compared BEFORE either update runs, because `update_parameters` consumes
    `exps.done_indices` and `exps.last_observations` off the DictList.
    """
    with eager.turn() as algo:
        expected, _ = algo.collect_experiences()
    with graphed.turn() as algo:
        actual, _ = algo.collect_experiences()
    return expected, actual


def _paired_algos(*, deterministic: bool = True, **extra):
    """An eager and a graphed run of the same config, verified to start from
    identical weights - otherwise the comparison below proves nothing.

    `extra` goes to BOTH arms, so a config that graphs the world model or the
    PPO step isolates `collect.rollout_cuda_graph` rather than confounding it.
    """
    kw = dict(deterministic=deterministic, **extra)
    eager = setup_training(_config(graphed=False, **kw)).algo
    graphed = setup_training(_config(graphed=True, **kw)).algo
    assert graphed._rollout_graph is not None, "exp.rollout_cuda_graph did not build"
    assert eager._rollout_graph is None, "the flag leaked into the eager arm"
    for name, left, right in (
        ("acmodel", eager.acmodel.parameters(), graphed.acmodel.parameters()),
        ("pRNN", eager.pN.pRNN.parameters(), graphed.pN.pRNN.parameters()),
    ):
        assert all(torch.equal(a, b) for a, b in zip(left, right)), (
            f"{name} initialisation differs between the two arms"
        )
    _saturate(eager.acmodel)
    _saturate(graphed.acmodel)
    return _Arm(eager), _Arm(graphed)


def test_graphed_rollout_bitwise_equals_eager():
    """No RNG anywhere: the graphed rollout must reproduce eager exactly.

    Three updates, with the PPO and world-model steps in between, so the
    replays run against weights that have MOVED since capture - which is the
    normal case and the one where a stale captured address would show up.
    """
    eager, graphed = _paired_algos()
    try:
        for update in range(3):
            expected, actual = _collect_both(eager, graphed)
            _assert_saturated(expected)
            _assert_same_rollout(expected, actual, where=f"update {update}")
            for arm, exps in ((eager, expected), (graphed, actual)):
                with arm.turn() as algo:
                    algo.update_parameters(exps=exps)
    finally:
        eager.algo.envs.close()
        graphed.algo.envs.close()


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="rollout_graph_only"),
        pytest.param(
            {"prnn_cuda_graph": True, "policy_cuda_graph": True}, id="all_three_graphs"
        ),
    ],
)
def test_graphed_rollout_survives_repeated_spatial_evals(extra):
    """Collection must keep WORKING across analysis events, not merely not
    crash. `on_device` moves both models to the CPU and back; it is
    address-preserving by contract (curious_george/models/device.py), and
    `GraphRolloutStepper` re-checks those addresses anyway. Either way the
    rollout it produces afterwards must still be eager's, exactly.

    The second parametrization runs it with the world-model and PPO graphs on
    in BOTH arms - three sets of captured graphs surviving the same device
    round trip, which is the configuration a real run uses."""
    from curious_george.models.device import on_device

    eager, graphed = _paired_algos(**extra)
    try:
        for update in range(3):
            expected, actual = _collect_both(eager, graphed)
            _assert_saturated(expected)
            _assert_same_rollout(expected, actual, where=f"after {update} evals")
            for arm, exps in ((eager, expected), (graphed, actual)):
                with arm.turn() as algo:
                    algo.update_parameters(exps=exps)
                    with on_device([algo.pN, algo.acmodel], "cpu"):  # analysis event
                        pass
    finally:
        eager.algo.envs.close()
        graphed.algo.envs.close()


def test_bitwise_gate_catches_a_poisoned_static_buffer():
    """The negative control for the two gates above.

    `mask_row` is a static buffer the captured graphs read and that
    `_addresses_now` deliberately does NOT watch, because production never
    rebinds it. Rebinding it here leaves the graphs reading the ORIGINAL row,
    so poisoning that row changes every recorded episode mask - and therefore
    the advantages - while the stepper's own bookkeeping looks healthy. If the
    comparison cannot see that, it cannot see a stranded address either.
    """
    eager, graphed = _paired_algos()
    try:
        expected, actual = _collect_both(eager, graphed)
        _assert_same_rollout(expected, actual, where="before poisoning")

        stepper = graphed.algo._rollout_graph
        stranded = stepper.mask_row
        stepper.mask_row = stranded.clone()  # graphs keep reading `stranded`
        stranded.fill_(0.5)

        expected, actual = _collect_both(eager, graphed)
        with pytest.raises(AssertionError):
            _assert_same_rollout(expected, actual, where="after poisoning")
    finally:
        eager.algo.envs.close()
        graphed.algo.envs.close()


def test_replay_draws_fresh_randomness():
    """Identical inputs, different draws: replay must not freeze the RNG.

    The stepper's own snapshot/restore puts the environment, the pRNN state
    and the phase back exactly where they were, so the ONLY thing that can
    differ between these replays is the random draw - `Categorical.sample` for
    the action and `generate_noise`'s `randn` for the pRNN.
    """
    algo = setup_training(_config(graphed=True, deterministic=False)).algo
    try:
        algo.collect_experiences()  # captures the graphs
        stepper = algo._rollout_graph
        snapshot = stepper._snapshot()

        actions, states = [], []
        for _ in range(4):
            stepper._restore(snapshot)
            stepper.step(mask=1.0)
            actions.append(stepper.buffers.actions[0].clone())
            states.append(stepper.sr_tracker.state.clone())

        assert len({tuple(a.tolist()) for a in actions}) > 1, (
            f"every replay sampled the same actions {actions[0].tolist()} from "
            "an identical state - the captured graph froze exploration"
        )
        assert not any(
            torch.equal(states[i], states[i + 1]) for i in range(len(states) - 1)
        ), "the pRNN's internal noise is frozen across replays"
    finally:
        algo.envs.close()

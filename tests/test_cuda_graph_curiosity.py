"""predNet.curiosity_cuda_graph: graphed batched curiosity forward.

The curiosity reward is one full-sequence pRNN forward over every equal-length
rollout segment. It is forward-only, so unlike the world-model graph nothing
has to be snapshotted and restored - but it is captured, so it fails the same
SILENT way: a replay writes to the addresses it saw at capture time and
returns plausible numbers when those addresses have gone stale.

Gates, all comparing the returned MSE tensor rather than a summary:

- `test_graphed_curiosity_bitwise_equals_eager` - with dropout and internal
  noise off the forward has no RNG, so graphed must equal eager BITWISE. Each
  round feeds a DIFFERENT batch, which is what makes a missing input `copy_`
  visible; feeding one batch repeatedly would pass either way.
- `test_graph_is_keyed_on_shape` - `target_offset=1` appends a step and
  `exp.num_envs` sets the batch, so one graph per shape. A single graph reused
  across shapes would read the wrong length.
- `test_graphed_curiosity_survives_repeated_spatial_evals` - the failure that
  made a graphed run train on nothing (0d60df7): `on_device` relocates
  `param.grad` and registered BUFFERS, not just `param.data`. This requires
  both that the answers stay exact AND that the graph was never re-captured,
  because re-capture is the safety net, not the contract.
- `test_raw_to_roundtrip_invalidates_captured_graphs` - and when a move DOES
  happen, the guard must notice and discard.
- `test_replay_draws_fresh_randomness` - dropout and noise live inside the
  captured region and must draw fresh on every replay.
- `test_flag_reaches_the_device_curiosity_path` - the end-to-end wiring, on
  the path a real run takes.

CUDA-only; skipped on CPU boxes.
"""

import pytest
import torch

from curious_george.models.prnn_adapter import PRNNAdapter
from tests.test_cuda_graph_wm import _make_pN, _one_segment

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph path requires a GPU"
)

GROUP = 4  # segments per curiosity batch


def _adapter(*, graphed: bool, dropp: float = 0.0, noise: tuple = (0.0, 0.0)):
    dev = torch.device("cuda")
    pN = _make_pN(dropp=dropp, noise=noise)
    pN.pRNN.to(dev)
    pN.pRNN.train()
    return pN, PRNNAdapter(pN, dev, action_offset=0, curiosity_cuda_graph=graphed)


def _batch(pN, adapter, *, target_offset: int, n: int = GROUP, first: int = 0):
    """`(obs_b, act_b)` in exactly the layout `_curiosity_errors` receives.

    `target_offset=1` (the next_obs alignment) appends one more observation
    row and one ZEROED action row - a different sequence length, and hence a
    different graph. `first` selects which segments, so successive rounds can
    feed genuinely different data.

    NOT reproducible for a fixed `first`: `_one_segment` reseeds torch and
    numpy but drives `pN.env_shell`, whose MiniGrid RNG carries across resets,
    so the k-th call to this returns the k-th trajectory rather than the same
    one. Two arms therefore have to consume their envs in LOCKSTEP - call it
    once per arm per round - or reuse one batch. Getting that wrong shows up
    as a ~1e-2 disagreement that reads exactly like a stranded graph.
    """
    obs_rows, act_rows = [], []
    for k in range(first, first + n):
        images, hd, act_np, last = _one_segment(pN, k)
        obs, act = adapter._episode_tensors(images, hd, act_np, last)
        if target_offset:
            obs = torch.cat([obs, obs[-1:]])
            act = torch.cat([act, torch.zeros_like(act[-1:])])
        obs_rows.append(obs)
        act_rows.append(act)
    dev = adapter.device
    return torch.stack(obs_rows).to(dev), torch.stack(act_rows).to(dev)


def _paired_adapters(**kwargs):
    """An eager and a graphed adapter over identically initialised nets."""
    eager_pN, eager = _adapter(graphed=False, **kwargs)
    graph_pN, graphed = _adapter(graphed=True, **kwargs)
    assert not eager.curiosity_cuda_graph, "the flag leaked into the eager arm"
    assert graphed.curiosity_cuda_graph, "predNet.curiosity_cuda_graph did not engage"
    assert all(
        torch.equal(a, b)
        for a, b in zip(eager_pN.pRNN.parameters(), graph_pN.pRNN.parameters())
    ), "the two arms start from different weights"
    return (eager_pN, eager), (graph_pN, graphed)


@pytest.mark.parametrize("target_offset", [0, 1])
def test_graphed_curiosity_bitwise_equals_eager(target_offset: int):
    """Dropout + noise off: replay == eager forward, bitwise, on fresh data."""
    (eager_pN, eager), (graph_pN, graphed) = _paired_adapters()
    for round_ in range(3):
        obs_e, act_e = _batch(
            eager_pN, eager, target_offset=target_offset, first=round_
        )
        obs_g, act_g = _batch(
            graph_pN, graphed, target_offset=target_offset, first=round_
        )
        assert torch.equal(obs_e, obs_g) and torch.equal(act_e, act_g)

        expected = eager._curiosity_errors(obs_e, act_e, target_offset=target_offset)
        actual = graphed._curiosity_errors(obs_g, act_g, target_offset=target_offset)
        assert torch.equal(expected, actual), (
            f"round {round_}: graphed != eager "
            f"(max|d|={(expected - actual).abs().max().item():.3e})"
        )
        assert expected.shape == (GROUP, len(act_e[0]) - target_offset)
    assert len(graphed._graph_curiosity.graphs) == 1, "one shape, one graph"


def test_graph_is_keyed_on_shape():
    """One adapter, three shapes: both alignments and a smaller batch. Each
    must get its own capture and each must still match eager."""
    (eager_pN, eager), (graph_pN, graphed) = _paired_adapters()
    shapes = ((0, GROUP), (1, GROUP), (0, GROUP - 1))
    for target_offset, n in shapes:
        kw = dict(target_offset=target_offset, n=n)
        expected = eager._curiosity_errors(
            *_batch(eager_pN, eager, **kw), target_offset=target_offset
        )
        actual = graphed._curiosity_errors(
            *_batch(graph_pN, graphed, **kw), target_offset=target_offset
        )
        assert torch.equal(expected, actual), f"shape {(target_offset, n)}"
    assert len(graphed._graph_curiosity.graphs) == len(shapes), (
        f"expected one graph per shape, got {sorted(graphed._graph_curiosity.graphs)}"
    )


def test_graphed_curiosity_survives_repeated_spatial_evals():
    """Curiosity must keep WORKING across analysis events, not merely not
    crash.

    Two assertions, because either alone is too weak. The answers must stay
    exactly eager's - and the captured graph must be the SAME OBJECT
    throughout, because `on_device` is address-preserving by contract
    (curious_george/models/device.py) and re-capture is only the net
    beneath it. If a future `on_device` stopped restoring `param.grad` or the
    `inMask_f`/`actMask_f`/`outMask_f` buffers to their addresses, the object
    identity is what notices.
    """
    from curious_george.models.device import on_device

    (eager_pN, eager), (graph_pN, graphed) = _paired_adapters()
    captured = None
    for round_ in range(3):
        expected = eager._curiosity_errors(
            *_batch(eager_pN, eager, target_offset=1, first=round_), target_offset=1
        )
        actual = graphed._curiosity_errors(
            *_batch(graph_pN, graphed, target_offset=1, first=round_), target_offset=1
        )
        assert torch.equal(expected, actual), f"after {round_} evals"

        graph = next(iter(graphed._graph_curiosity.graphs.values()))["graph"]
        if captured is None:
            captured = graph
        assert graph is captured, (
            f"the graph was re-captured after eval {round_ - 1}: `on_device` "
            "moved a parameter or buffer instead of restoring its address"
        )
        for pN in (eager_pN, graph_pN):
            with on_device([pN], "cpu"):  # the analysis event
                pass


def test_raw_to_roundtrip_invalidates_captured_graphs():
    """`Module.to()` REPLACES parameter storage, so every captured graph is
    left writing to freed memory. `on_device` no longer triggers this (see the
    test above), but the detection has to keep working for every other path -
    which is what a bare `.to()` exercises.

    The ballast is load-bearing. A bare cuda->cpu->cuda round trip usually
    hands the SAME addresses back, because nothing claimed the freed blocks in
    between - and then the graph is not stranded at all and the guard is right
    not to fire. Claiming those blocks with same-sized allocations while the
    net is away forces the genuine relocation this is about; the assertion
    below makes the test say so rather than pass vacuously.
    """
    (eager_pN, eager), (graph_pN, graphed) = _paired_adapters()
    kw = dict(target_offset=0)
    # One batch per arm, reused on both sides of the move - see `_batch`.
    obs_e, act_e = _batch(eager_pN, eager, **kw)
    obs_g, act_g = _batch(graph_pN, graphed, **kw)
    assert torch.equal(obs_e, obs_g) and torch.equal(act_e, act_g)

    graphed._curiosity_errors(obs_g, act_g, **kw)  # captures
    before = next(iter(graphed._graph_curiosity.graphs.values()))["graph"]
    ptr_before = next(graph_pN.pRNN.parameters()).data_ptr()

    sizes = [p.numel() for p in graph_pN.pRNN.parameters()]
    graph_pN.pRNN.to("cpu")
    ballast = [torch.empty(n, device="cuda") for n in sizes]
    graph_pN.pRNN.to("cuda")
    del ballast

    assert next(graph_pN.pRNN.parameters()).data_ptr() != ptr_before, (
        "the round trip did not move params - this test no longer exercises "
        "the bug and would pass whatever the guard did"
    )
    expected = eager._curiosity_errors(obs_e, act_e, **kw)
    actual = graphed._curiosity_errors(obs_g, act_g, **kw)
    after = next(iter(graphed._graph_curiosity.graphs.values()))["graph"]
    assert after is not before, "a move went undetected; the graph is stranded"
    assert torch.equal(expected, actual), "re-capture did not restore correctness"


def test_replay_draws_fresh_randomness():
    """Identical inputs, different draws. Dropout and the initial-state noise
    run inside the captured region; a graph that froze them would make the
    curiosity reward deterministic while every other number still looked
    right."""
    pN, adapter = _adapter(graphed=True, dropp=0.15, noise=(0.0, 0.05))
    obs_b, act_b = _batch(pN, adapter, target_offset=0)
    outs = [
        adapter._curiosity_errors(obs_b, act_b, target_offset=0).clone()
        for _ in range(4)
    ]
    assert not any(
        torch.equal(outs[i], outs[i + 1]) for i in range(len(outs) - 1)
    ), "every replay returned the same MSEs from identical inputs"
    assert torch.isfinite(torch.stack(outs)).all()


def test_flag_reaches_the_device_curiosity_path():
    """End-to-end wiring: `predNet.curiosity_cuda_graph` in the config must
    reach `prediction_mses_device`, the path a device_env run actually takes.
    The unit gates above prove the graph is correct; this proves it is used."""
    from dataclasses import replace

    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config()
    cfg = replace(cfg, train_prnn=replace(cfg.train_prnn, curiosity_cuda_graph=True))
    algo = setup_training(cfg).algo
    try:
        assert algo.adapter._graph_curiosity is None, "captured before any rollout"
        _, logs = algo.collect_experiences()
        graphs = algo.adapter._graph_curiosity.graphs
        assert len(graphs) == 1, f"expected one captured shape, got {sorted(graphs)}"
        assert torch.isfinite(torch.as_tensor(logs["curious_rewards"])).all()
    finally:
        algo.envs.close()

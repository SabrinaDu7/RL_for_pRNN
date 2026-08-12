"""predNet.cuda_graph: graphed per-segment world-model trainStep equivalence.

Gates (see docs/exp_cuda_graphs_wm.md):
- With dropout + internal noise OFF the graphed step has no RNG, so a graphed
  segment must move the weights BITWISE-identically to the eager trainStep it
  replaces (same 1 optimizer step, same math).
- With them ON the path must run and produce finite, changing weights (the
  in-graph RNG realizes different-but-equivalent draws; not bitwise by design).

CUDA-only; skipped on CPU boxes.
"""

import numpy as np
import pytest
import torch

from prnn.utils import (
    ActionEncodingsEnum,
    MinigridEnvNames,
    PredictiveNet,
    RandomActionAgent,
)
from curious_george import AgentInputType, make_env
from curious_george.world_model.adapter import PRNNAdapter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph path requires a GPU"
)

SEED = 5
L = 32  # segment length


def _make_pN(*, dropp: float, noise: tuple[float, float]) -> PredictiveNet:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=SEED,
    )
    return PredictiveNet(
        env, hidden_size=64, pRNNtype="thRNN_5win",
        trainNoiseMeanStd=noise, dropp=dropp, learningRate=3e-3,
        weight_decay=3e-3, bptttrunc=int(1e8), wandb_log=False,
    )


def _one_segment(pN: PredictiveNet):
    """A single deterministic episode segment: obs (L, img), hd, act."""
    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 1)
    env = pN.env_shell
    agent = RandomActionAgent(env.action_space, np.array([0.15, 0.15, 0.6, 0.1]))
    obs_dicts, acts = [env.reset()], []
    for _ in range(L):
        a = int(np.random.randint(0, 4))
        acts.append(a)
        obs_dicts.append(env.step(np.array([a]))[0])
    images = torch.stack(
        [torch.tensor(np.asarray(o["image"]), dtype=torch.float) for o in obs_dicts[:-1]]
    )
    directions = torch.tensor([o["direction"] for o in obs_dicts[:-1]])
    return images, directions, np.asarray(acts), obs_dicts[L]


def _weights(pN) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in pN.pRNN.parameters()])


def test_graphed_segment_bitwise_equals_eager_no_rng():
    """Dropout + noise off: graphed replay == eager trainStep, bitwise."""
    dev = torch.device("cuda")

    pN_eager = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN_eager.pRNN.to(dev)
    pN_eager.pRNN.train()
    images, hd, act, last = _one_segment(pN_eager)
    a_eager = PRNNAdapter(pN_eager, dev, pastSR=True, cuda_graph=False)
    w0 = _weights(pN_eager).clone()
    a_eager.train_on_episode(images, hd, act, last)
    w_eager = _weights(pN_eager)

    pN_graph = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN_graph.pRNN.to(dev)
    pN_graph.pRNN.train()
    images, hd, act, last = _one_segment(pN_graph)
    a_graph = PRNNAdapter(pN_graph, dev, pastSR=True, cuda_graph=True)
    assert a_graph.cuda_graph
    a_graph.train_on_episode(images, hd, act, last)
    w_graph = _weights(pN_graph)

    assert torch.equal(w0, _weights(pN_eager)) is False, "eager step did not move weights"
    assert torch.equal(w_eager, w_graph), (
        f"graphed != eager: max|d|={(w_eager - w_graph).abs().max().item():.3e}"
    )


def test_device_roundtrip_invalidates_captured_graphs():
    """A device round-trip REPLACES param storage, so every captured graph is
    left writing to freed memory - a use-after-free that silently corrupts
    whatever the allocator hands those blocks to next.

    This is not hypothetical: `logging.plot_every_steps`/`analysis_every_steps`
    wrap plotting and the spatial eval in `on_device([...], "cpu")`
    (training/loop.py, evaluation/spatial.py). A 2026-07-22 cluster run died 7
    updates after the update-200 event with `obs.direction`==184, because the
    freed parameter block had been reused for that tensor.

    Guarding in `PRNNAdapter.to()` is not enough: `on_device` calls
    `world_model.device._move()` straight on the PredictiveNet, bypassing the
    adapter. The graphs must notice the move themselves.
    """
    from curious_george.world_model.device import on_device

    dev = torch.device("cuda")
    pN = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    images, hd, act, last = _one_segment(pN)

    adapter.train_on_episode(images, hd, act, last)  # captures
    trainer = adapter._graph_trainer
    assert len(trainer.graphs) == 1
    ptr_before = next(pN.pRNN.parameters()).data_ptr()

    with on_device([pN], "cpu"):  # exactly what the plot / analysis path does
        pass
    assert next(pN.pRNN.parameters()).data_ptr() != ptr_before, (
        "device round-trip did not move params - test no longer exercises the bug"
    )

    # next use must notice the move, drop the stale graphs and re-capture
    adapter.train_on_episode(images, hd, act, last)
    assert len(trainer.graphs) == 1
    assert trainer._param_ptrs == tuple(p.data_ptr() for p in pN.pRNN.parameters())
    assert torch.isfinite(_weights(pN)).all()


def test_recapture_preserves_optimizer_state():
    """A mid-training re-capture must not wipe RMSprop's accumulated state:
    the warmup runs real optimizer steps, so `_capture` snapshots and RESTORES
    the state rather than zeroing it (zeroing would silently reset the
    optimizer every time a new segment length or a device move forced a
    re-capture)."""
    dev = torch.device("cuda")
    pN = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    images, hd, act, last = _one_segment(pN)

    for _ in range(8):  # accumulate real optimizer state
        adapter.train_on_episode(images, hd, act, last)
    opt = pN.optimizer
    before = {
        id(p): float(st["square_avg"].mean())
        for p, st in opt.state.items() if "square_avg" in st
    }
    assert before and any(v > 0 for v in before.values())

    # force a re-capture at a different segment length
    adapter.train_on_episode(images[:-4], hd[:-4], act[:-4], last)

    # Compare MAGNITUDE, not just non-zero: the replay right after capture runs
    # opt.step(), which repopulates square_avg either way. Continuing from the
    # snapshot keeps ~alpha (0.95) of the accumulated value; a zeroed restart
    # would leave only (1-alpha)*grad^2, i.e. orders of magnitude smaller.
    for p, st in opt.state.items():
        if id(p) in before and before[id(p)] > 0:
            after = float(st["square_avg"].mean())
            assert after > 0.5 * before[id(p)], (
                f"re-capture reset RMSprop state: {before[id(p)]:.3e} -> {after:.3e}"
            )


def test_graphed_segment_runs_with_rng():
    """Dropout + noise on: runs, finite, weights change (not bitwise by design)."""
    dev = torch.device("cuda")
    pN = _make_pN(dropp=0.15, noise=(0.0, 0.05))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    images, hd, act, last = _one_segment(pN)
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    w0 = _weights(pN).clone()
    adapter.train_on_episode(images, hd, act, last)
    w1 = _weights(pN)
    assert torch.isfinite(w1).all()
    assert not torch.equal(w0, w1)
    # a second replay must keep moving (RNG advanced, buffers refreshed)
    adapter.train_on_episode(images, hd, act, last)
    assert not torch.equal(w1, _weights(pN))

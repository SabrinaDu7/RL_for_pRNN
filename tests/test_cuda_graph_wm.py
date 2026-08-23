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


def _one_segment(pN: PredictiveNet, k: int = 0):
    """A single deterministic episode segment: obs (L, img), hd, act.

    `k` selects distinct segments for the pooled tests; k=0 is the segment the
    single-segment tests have always used.
    """
    torch.manual_seed(SEED + 1 + k)
    np.random.seed(SEED + 1 + k)
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


# --- pooled path (predNet.wm_pool_group) ------------------------------------
#
# The graph used to serve only `train_on_episode`, so the shipped config -
# batched_wm with wm_pool_group=8 - ran entirely ungraphed. These gate the
# pooled path on the SAME criterion as the serial one, plus the property the
# pooled regime actually depends on: N sequential replays must advance the
# weights the way N sequential eager steps do, because `wm_pool_group` means 16
# steps per update each starting from the previous step's weights. A graph that
# replayed at frozen weights would pass a one-step test and silently destroy
# that.

GROUP = 4      # segments pooled into one gradient step
STEPS = 3      # sequential gradient steps


def _pooled_batch(pN, adapter, n: int):
    """n equal-length segments stacked to (B, L+1, X) / (B, L, A), on device."""
    obs_rows, act_rows = [], []
    for k in range(n):
        images, hd, act, last = _one_segment(pN, k)
        o, a = adapter._episode_tensors(images, hd, act, last)
        obs_rows.append(o)
        act_rows.append(a)
    dev = adapter.device
    return torch.stack(obs_rows).to(dev), torch.stack(act_rows).to(dev)


def _run_pooled(*, cuda_graph: bool, dropp: float, noise: tuple[float, float]):
    dev = torch.device("cuda")
    pN = _make_pN(dropp=dropp, noise=noise)
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=cuda_graph)
    obs_b, act_b = _pooled_batch(pN, adapter, GROUP)
    if cuda_graph:
        assert adapter._use_graph_wm(), "pooled graph path not engaged"
        adapter._graph_trainer = None
    for _ in range(STEPS):
        if cuda_graph:
            from curious_george.world_model.adapter import _GraphWMTrainer

            adapter._graph_trainer = adapter._graph_trainer or _GraphWMTrainer(pN, dev)
            adapter._graph_trainer.train_batch(obs_b, act_b, batched=True)
        else:
            pN.trainStep(obs_b, act_b, batched=True, return_stats=False)
    return pN, adapter


def test_pooled_graphed_bitwise_equals_eager_no_rng():
    """Noise+dropout off: N sequential pooled replays == N eager pooled steps.

    Bitwise, because with no RNG the captured region is the same math. This is
    the semantics gate for wm_pool_group: it fails if the in-graph optimizer
    step does not advance the weights between replays.
    """
    pN_eager, _ = _run_pooled(cuda_graph=False, dropp=0.0, noise=(0.0, 0.0))
    pN_graph, _ = _run_pooled(cuda_graph=True, dropp=0.0, noise=(0.0, 0.0))

    w_fresh = _weights(_make_pN(dropp=0.0, noise=(0.0, 0.0)))
    w_eager, w_graph = _weights(pN_eager), _weights(pN_graph)
    assert not torch.equal(w_fresh, w_eager), "eager pooled steps did not move weights"
    assert torch.equal(w_eager, w_graph), (
        f"pooled graphed != eager: max|d|={(w_eager - w_graph).abs().max().item():.3e}"
    )


def test_pooled_graph_advances_weights_every_replay():
    """Each replay must move the weights - a graph replaying at frozen weights
    would still be bitwise-equal to eager on step 1 alone."""
    dev = torch.device("cuda")
    from curious_george.world_model.adapter import _GraphWMTrainer

    pN = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    obs_b, act_b = _pooled_batch(pN, adapter, GROUP)
    trainer = _GraphWMTrainer(pN, dev)

    seen = []
    for _ in range(STEPS):
        trainer.train_batch(obs_b, act_b, batched=True)
        seen.append(_weights(pN).clone())
    for i in range(1, len(seen)):
        assert not torch.equal(seen[i - 1], seen[i]), f"replay {i} left weights unchanged"
    assert torch.isfinite(seen[-1]).all()


def test_pooled_and_serial_graphs_do_not_collide():
    """batched=True at B=1 is a DIFFERENT layout from the serial batched=False
    call, so they must not share a graph key."""
    dev = torch.device("cuda")
    from curious_george.world_model.adapter import _GraphWMTrainer

    pN = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    obs_b, act_b = _pooled_batch(pN, adapter, 1)
    trainer = _GraphWMTrainer(pN, dev)

    trainer.train_batch(obs_b, act_b, batched=True)
    trainer.train_segment(obs_b[0].cpu(), act_b[0].cpu())
    assert len(trainer.graphs) == 2, f"expected two graph keys, got {list(trainer.graphs)}"
    assert {k[0] for k in trainer.graphs} == {True, False}


def test_ragged_final_group_gets_its_own_graph():
    """wm_pool_group need not divide the segment count; the short final group is
    a different shape and must capture separately rather than replay the wrong
    one."""
    dev = torch.device("cuda")
    from curious_george.world_model.adapter import _GraphWMTrainer

    pN = _make_pN(dropp=0.0, noise=(0.0, 0.0))
    pN.pRNN.to(dev)
    pN.pRNN.train()
    adapter = PRNNAdapter(pN, dev, pastSR=True, cuda_graph=True)
    obs_b, act_b = _pooled_batch(pN, adapter, 3)
    trainer = _GraphWMTrainer(pN, dev)

    trainer.train_batch(obs_b[:2], act_b[:2], batched=True)   # full group
    trainer.train_batch(obs_b[2:], act_b[2:], batched=True)   # ragged tail
    assert {k[1] for k in trainer.graphs} == {2, 1}
    assert torch.isfinite(_weights(pN)).all()

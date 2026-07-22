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

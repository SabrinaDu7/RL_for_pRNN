"""One way to turn a checkpoint into (hidden state, position).

Promoted out of the retired scripts/trace/ (now throwaway/ported/trace/) 2026-08-25. It was one of FOUR competing
implementations of this - the others were
evaluation/checkpoint_series.py::fixed_probe (still live, see its docstring),
moser_analysis.py::_build_probe (whose own docstring said it was the same
construction as build_probe here), and evaluation/task.py::collect_eval_rollouts.
This one won because it is the only one with a FIXED SEED (PROBE_SEED) and
documented fixed actions, which is what makes repeated scoring of one
checkpoint agree. The training collector (rl/collect/collector.py) is a
different thing and stays separate.

The trace/object-vector RESULTS this served are not trusted and are being
restarted; the MECHANISM is what every future question needs.
Fixed measurement rollouts ("the probe") for object-trace-cell analysis.

A probe is collected ONCE and replayed against every checkpoint. Its actions
come from a RandomActionAgent, which pre-generates them independently of the
network (prnn/utils/agent.py), and its observations depend only on the (fixed,
object-absent) environment - so the same trajectories are valid for every
checkpoint, arm and seed. Replaying a checkpoint is then a single batched
forward, and any map difference between two checkpoints is in the weights.

One trajectory starts from every (walkable cell, head direction) pair, so
occupancy is broad by construction: ~172 * 4 = 688 trajectories in LRoom.

Reproducibility: PredictiveNet.predict injects fresh noise on EVERY call
(trainNoiseMeanStd, default (0, 0.05)) - two identical calls differ by ~0.4 in
h. replay_checkpoint seeds torch immediately before the forward so every
checkpoint sees the same noise realisation, and can zero the noise entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from prnn.utils import PredictiveNet
from prnn.utils.Shell import FaramaMinigridShell

from curious_george import get_agent, AgentType
from curious_george.models.device import eval_mode, on_device
from curious_george.envs.access import get_walkable_mask, get_walkable_minigrid_positions

PROBE_SEED = 20260730


@dataclass
class Probe:
    """Frozen measurement rollouts: `B` trajectories of `T` steps."""

    obs: Float[torch.Tensor, "B T+1 X"]
    act: Float[torch.Tensor, "B T A"]
    agent_pos: Int[np.ndarray, "B T+1 2"]
    agent_dir: Int[np.ndarray, "B T+1"]
    seed: int
    env_name: str

    @property
    def n_trajs(self) -> int:
        return self.obs.shape[0]

    @property
    def n_steps(self) -> int:
        return self.act.shape[1]


def build_probe(
    *,
    pN: PredictiveNet,
    env: FaramaMinigridShell,
    n_steps: int = 256,
    seed: int = PROBE_SEED,
    env_name: str = "",
) -> Probe:
    """Collect one trajectory per (walkable cell, head direction) pair.

    `pN` is used only to format observations (collectObservationSequence
    delegates to the env shell); its weights do not influence the result.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = get_agent(env=env, agent_Type=AgentType.RANDOM, device=torch.device("cpu"))
    positions = get_walkable_minigrid_positions(get_walkable_mask(env))
    starts = [(tuple(p.tolist()), d) for p in positions for d in range(4)]

    obs_rows: list[torch.Tensor] = []
    act_rows: list[torch.Tensor] = []
    pos_rows: list[np.ndarray] = []
    dir_rows: list[np.ndarray] = []

    with eval_mode([pN]), on_device([pN], "cpu"):
        for pos, direction in tqdm(starts, desc="probe"):
            env.env.unwrapped.agent_start_pos = pos
            env.env.unwrapped.agent_start_dir = direction
            obs, act, state, _ = pN.collectObservationSequence(
                env=env, agent=agent, tsteps=n_steps, discretize=True
            )
            obs_rows.append(obs.squeeze(0))
            act_rows.append(act.squeeze(0))
            pos_rows.append(np.asarray(state["agent_pos"]))
            dir_rows.append(np.asarray(state["agent_dir"]))

    return Probe(
        obs=torch.stack(obs_rows),
        act=torch.stack(act_rows),
        agent_pos=np.stack(pos_rows).astype(np.int32),
        agent_dir=np.stack(dir_rows).astype(np.int32),
        seed=seed,
        env_name=env_name or str(getattr(env, "env_key", "")),
    )


def save_probe(probe: Probe, path: str | Path) -> Path:
    """Write the probe to `<path>/probe.pt` (creating the directory)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs": probe.obs,
            "act": probe.act,
            "agent_pos": probe.agent_pos,
            "agent_dir": probe.agent_dir,
            "seed": probe.seed,
            "env_name": probe.env_name,
        },
        out / "probe.pt",
    )
    return out / "probe.pt"


def load_probe(path: str | Path) -> Probe:
    """Load a probe from a directory or a direct path to `probe.pt`."""
    p = Path(path)
    if p.is_dir():
        p = p / "probe.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return Probe(**d)


def replay_checkpoint(
    *,
    pN: PredictiveNet,
    probe: Probe,
    noise: bool = True,
    seed: int = PROBE_SEED,
    batch_size: int = 128,
) -> Float[torch.Tensor, "B T H"]:
    """Hidden states of `pN` over the probe's stored observations and actions.

    noise=True keeps the trained operating regime but pins the RNG so every
    checkpoint sees the same realisation; noise=False removes it entirely.
    Trajectories are replayed in chunks of `batch_size` to bound peak memory.
    """
    saved_noise = pN.trainNoiseMeanStd
    if not noise:
        pN.trainNoiseMeanStd = (0.0, 0.0)

    try:
        with eval_mode([pN]), on_device([pN], "cpu"), torch.no_grad():
            torch.manual_seed(seed)
            init = pN.pRNN.rnn.cell.actfun(
                pN.pRNN.generate_noise(pN.trainNoiseMeanStd, (1, 1, pN.hidden_size))
            )
            chunks: list[torch.Tensor] = []
            for lo in range(0, probe.n_trajs, batch_size):
                hi = min(lo + batch_size, probe.n_trajs)
                torch.manual_seed(seed + lo)  # same draw for every checkpoint
                _, _, h = pN.predict(
                    probe.obs[lo:hi],
                    probe.act[lo:hi],
                    state=init.repeat(hi - lo, 1, 1),
                    randInit=False,
                    batched=True,
                )
                chunks.append(h[0].permute(2, 0, 1))  # (1, T, H, b) -> (b, T, H)
        return torch.cat(chunks)
    finally:
        pN.trainNoiseMeanStd = saved_noise


def pool(
    *,
    h: Float[torch.Tensor, "B T H"],
    probe: Probe,
    onset_transient: int = 20,
) -> tuple[Float[np.ndarray, "N H"], Float[np.ndarray, "N 2"]]:
    """Flatten trajectories into (activity, position) rows, dropping the onset.

    h[b, t] pairs with agent_pos[b, t] (the RGA convention used throughout:
    positions carry one extra row, so they are sliced to h's length).
    """
    T = h.shape[1]
    h_rows = h[:, onset_transient:T, :].reshape(-1, h.shape[-1])
    pos_rows = probe.agent_pos[:, onset_transient:T, :].reshape(-1, 2)
    return h_rows.detach().numpy(), pos_rows.astype(np.float64)

"""CUDA-graph capture of ONE rollout timestep (`exp.rollout_cuda_graph`).

Why: with the world model and the PPO minibatch already graphed, collection is
the largest remaining block of a training update. At `exp.num_envs=128`,
`rl.frames=32768` the rollout runs 256 sequential iterations of three tiny GPU
regions - the actor-critic forward, the device environment table, and the pRNN
single step - and is almost pure PyTorch dispatch: the GPU sits idle while
Python decides what to launch.

The whole timestep goes in, and it goes in by CALLING THE PRODUCTION
FUNCTIONS - `DeviceTableShellPool.observation_device` / `.step_device`,
`ACModelSR.forward`, `BatchedSRTrackerShim.step_device`. Nothing here mirrors
the eager body, so the two cannot drift apart. What the capture needs instead
is that every address it bakes in is stable, which costs three things:

1. **The per-timestep buffers are owned here**, for the life of the run, not
   allocated per rollout. `RolloutBuffers` is the one place their shapes and
   dtypes are written; the eager path allocates the same object per call.
2. **Python's `t` cannot index a graph.** Writes go through `index_copy_`
   against a static `t_idx` tensor the graph increments itself.
3. **The observation is re-gathered from the pool's live position/direction
   tensors** at the top of the body rather than carried in a Python variable
   from the previous step. `step_device` already returns exactly that gather,
   so this is the same value; the cost is gathering twice per timestep, which
   buys calling the real `step_device` instead of reimplementing it.

ONE GRAPH PER pRNN PHASE. `thRNN_5win` masks its observation input on a cycle
of `pN.phase_k` (`pRNN.inMask`), and `BatchedSRTracker.step_synchronized`
branches on that in Python - so the branch is baked in at capture and the
phase is part of the graph key. The episode mask, by contrast, is a static row
refilled from the host, because its value changes only at segment boundaries.

Captures happen ALL AT ONCE before a rollout's first timestep, never mid-loop.
Warmup and capture run the body for real - they step the environment, advance
the pRNN state and write buffer rows - so a mid-loop capture would corrupt
rows already collected. Up front, the only row it can touch is row 0, which
the loop is about to overwrite, and everything else is snapshotted and
restored in place (`_snapshot`/`_restore`), keeping the addresses the graphs
captured.

Not bit-comparable to the eager rollout when the policy is stochastic: a
captured region draws from CUDA's graph-safe RNG, which realizes a
different - equally valid - stream than eager order. Gated bitwise in
tests/test_cuda_graph_rollout.py with a saturated policy, which removes the
sampling RNG without touching the code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch

from curious_george.utils.cuda_graph import no_dist_validation
from curious_george.utils.timing import timer


@dataclass(frozen=True)
class RolloutBuffers:
    """Per-timestep rollout storage, indexed `[t, b]`.

    `images`/`directions`/`positions` exist only for the device-resident
    environment table, the one backend that keeps observations on-device;
    every other backend leaves them None and keeps its observations in Python.
    """

    actions: torch.Tensor  # (T, B) int32
    values: torch.Tensor  # (T, B) float32
    rewards: torch.Tensor  # (T, B) float32
    masks: torch.Tensor  # (T, B) float32
    srs: torch.Tensor  # (T, B, hidden) float32
    policy_logits: torch.Tensor  # (T, B, act_dim) float32 - LOGITS, not probs
    images: torch.Tensor | None  # (T, B, *image_shape) uint8
    directions: torch.Tensor | None  # (T, B) int64
    positions: torch.Tensor | None  # (T, B, 2) int64

    @classmethod
    def allocate(
        cls,
        *,
        num_steps: int,
        num_envs: int,
        hidden_size: int,
        act_dim: int,
        device: torch.device,
        image_shape: tuple[int, ...] | None,
    ) -> "RolloutBuffers":
        """Zeroed buffers for a `num_steps` x `num_envs` rollout."""
        t, b = num_steps, num_envs
        on_device = image_shape is not None
        return cls(
            actions=torch.zeros((t, b), device=device, dtype=torch.int),
            values=torch.zeros((t, b), device=device),
            rewards=torch.zeros((t, b), device=device),
            masks=torch.zeros((t, b), device=device),
            srs=torch.zeros((t, b, hidden_size), device=device),
            policy_logits=torch.zeros((t, b, act_dim), device=device),
            images=(
                torch.empty((t, b, *image_shape), dtype=torch.uint8, device=device)
                if on_device
                else None
            ),
            directions=(
                torch.empty((t, b), dtype=torch.long, device=device)
                if on_device
                else None
            ),
            positions=(
                torch.empty((t, b, 2), dtype=torch.long, device=device)
                if on_device
                else None
            ),
        )


class GraphRolloutStepper:
    """One captured rollout timestep per pRNN phase, replayed T times.

    Lifetime is one training run. `prepare(sr=...)` once per rollout - it
    captures on first use, re-checks that the addresses are still the captured
    ones, and rewinds `t_idx` - then `step(mask=...)` per timestep. The
    collector reads `buffers` after the loop; every consumer copies out of
    them, so the next rollout is free to overwrite.
    """

    def __init__(
        self,
        *,
        acmodel: torch.nn.Module,
        tracker,  # BatchedSRTrackerShim
        pool,  # DeviceTableShellPool
        num_steps: int,
        device: torch.device,
        random_actions: bool = False,
        random_action_probs: tuple[float, ...] | None = None,
    ) -> None:
        self.random_actions = random_actions
        #: `arch_policy.random_action_probs`; None = project default. The
        #: capture path builds its own device table in `prepare`, so the
        #: distribution has to be threaded HERE too, not only through
        #: `RolloutConfig` - this is the switch that actually runs when the
        #: rollout is graphed.
        self.random_action_probs = random_action_probs
        self._rand_probs = None
        self.acmodel = acmodel
        self.shim = tracker
        self.sr_tracker = tracker.tracker
        self.pN = self.sr_tracker.pN
        self.pool = pool
        self.num_steps = num_steps
        self.device = device
        self.phase_k = int(self.sr_tracker.phase_k)

        num_envs = len(pool)
        self.buffers = RolloutBuffers.allocate(
            num_steps=num_steps,
            num_envs=num_envs,
            hidden_size=self.pN.hidden_size,
            act_dim=int(acmodel.act_dim),
            device=device,
            image_shape=pool.image_shape,
        )
        self.t_idx = torch.zeros((1,), dtype=torch.long, device=device)
        self.mask_row = torch.ones((1, num_envs), device=device)
        self._mask_value = 1.0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._addresses: tuple[int, ...] = ()

    # --- address safety, same contract as _GraphWMTrainer --------------------
    def _addresses_now(self) -> tuple[int, ...]:
        """Every storage a captured replay reads or writes.

        A graph writes to the addresses it saw AT CAPTURE TIME, so any path
        that reallocates one of these - `Module.to()` replacing `param.data`,
        a tracker reset that rebinds rather than zeroes - leaves the graphs
        pointing at freed memory. That fails SILENTLY: the eager forward still
        reads the right values while the replay corrupts whatever the
        allocator handed the freed block to next.
        """
        return tuple(
            x.data_ptr()
            for x in chain(
                self.acmodel.parameters(),
                self.acmodel.buffers(),
                self.pN.pRNN.parameters(),
                self.pN.pRNN.buffers(),
                (
                    self.sr_tracker.state,
                    self.pool.positions,
                    self.pool.directions,
                    self.pool.stream_layout,
                    self.pool.obs_banks,
                    self.pool.next_state,
                ),
            )
        )

    # --- the captured region ------------------------------------------------
    def _sr(self) -> torch.Tensor:
        """The SR the policy consumes: a view of the tracker's ONE state
        buffer, so its address survives every reset."""
        return self.sr_tracker.sr()

    def current_sr(self) -> torch.Tensor:
        """The SR the last replay produced, as a copy the caller owns - the
        same contract as the eager `BatchedSRTrackerShim.step_device`."""
        return self._sr().clone()

    def _region(self) -> None:
        """One timestep, in the order collect_rollout runs it: policy forward,
        record, environment step, pRNN step."""
        from curious_george.rl.collect.collector import _random_actions, _device_policy_obss

        b, t, pool = self.buffers, self.t_idx, self.pool
        images, directions = pool.observation_device()
        with torch.no_grad():
            dist, value = self.acmodel(
                _device_policy_obss(images, directions, self.acmodel), SR=self._sr()
            )
        # `random_actions` is a Python constant for the run, so the branch is
        # resolved at CAPTURE time - no graph-level branching. Both arms are a
        # torch RNG op drawing from the graph-safe generator, exactly as
        # `dist.sample()` already did.
        action = (
            _random_actions(self._rand_probs)
            if self.random_actions else dist.sample()
        )

        b.masks.index_copy_(0, t, self.mask_row)
        b.srs.index_copy_(0, t, self._sr().unsqueeze(0))
        b.actions.index_copy_(0, t, action.to(b.actions.dtype).unsqueeze(0))
        b.values.index_copy_(0, t, value.unsqueeze(0))
        b.policy_logits.index_copy_(0, t, dist.logits.detach().unsqueeze(0))
        b.images.index_copy_(0, t, images.unsqueeze(0))
        b.directions.index_copy_(0, t, directions.unsqueeze(0))
        b.positions.index_copy_(0, t, pool.positions.unsqueeze(0))

        post_images, post_directions, step_rewards = pool.step_device(actions=action)
        b.rewards.index_copy_(0, t, step_rewards.unsqueeze(0))

        # The circuit, inside the capture: offset 0 feeds the pRNN the
        # observation the action was chosen from, offset 1 the one it produced.
        # `step_device` already returns both, and the branch is a Python
        # constant for the run, so it is resolved at capture time.
        if self.shim.adapter.action_offset:
            self.shim.step_device(
                actions=action, images=post_images, directions=post_directions
            )
        else:
            self.shim.step_device(actions=action, images=images, directions=directions)
        t.add_(1)

    # --- capture ------------------------------------------------------------
    def _snapshot(self) -> dict:
        return {
            "positions": self.pool.positions.clone(),
            "directions": self.pool.directions.clone(),
            "state": self.sr_tracker.state.clone(),
            "phases": self.sr_tracker.phases.copy(),
        }

    def _restore(self, snap: dict) -> None:
        """Undo warmup+capture IN PLACE, keeping the captured addresses."""
        self.pool.positions.copy_(snap["positions"])
        self.pool.directions.copy_(snap["directions"])
        self.sr_tracker.state.copy_(snap["state"])
        self.sr_tracker.phases[:] = snap["phases"]
        self.t_idx.zero_()

    def _capture_phase(self, phase: int) -> None:
        """Capture the body as it runs at `phase`.

        The phase is forced before EVERY execution because the body advances
        it: without this, warmup run 2 would record at `phase+1` and the
        capture at `phase+3`. `t_idx` is rewound the same way, so warmup can
        only ever write row 0.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.sr_tracker.phases[:] = phase
                self.t_idx.zero_()
                self._region()
        torch.cuda.current_stream().wait_stream(stream)

        self.sr_tracker.phases[:] = phase
        self.t_idx.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._region()
        self.graphs[phase] = graph

    def _capture_all(self) -> None:
        snap = self._snapshot()
        with timer.disabled(), no_dist_validation():
            for phase in range(self.phase_k):
                self._capture_phase(phase)
        self._restore(snap)
        self._addresses = self._addresses_now()

    # --- per-rollout / per-timestep -----------------------------------------
    def prepare(self, *, sr: torch.Tensor) -> None:
        """Ready the graphs for one rollout. Call before its first timestep."""
        # BEFORE `_capture_all`, deliberately: a host-to-device copy inside a
        # captured region raises "operation not permitted when stream is
        # capturing" rather than degrading, and two cluster jobs died on exactly
        # that. `sr` carries the batch, so the table needs no pool attribute.
        if self.random_actions and self._rand_probs is None:
            from curious_george.rl.collect.collector import random_action_probs

            self._rand_probs = random_action_probs(
                sr.shape[0], sr.device, probs=self.random_action_probs
            )
        if self.graphs and self._addresses_now() != self._addresses:
            self.graphs.clear()
        if not self.graphs:
            self._capture_all()
        # The body reads the tracker's state buffer, NOT the collector's
        # carried `state.sr`. They hold the same value on every path into a
        # rollout - so check it rather than assume it, once per rollout.
        assert torch.equal(sr, self._sr()), (
            "carried SR and tracker state disagree; a graphed rollout would "
            "silently start the policy from the tracker's value"
        )
        self.t_idx.zero_()

    def step(self, *, mask: float) -> None:
        """Replay one timestep. `mask` is the episode mask recorded for it."""
        if mask != self._mask_value:
            self.mask_row.fill_(mask)
            self._mask_value = mask
        phase = int(self.sr_tracker.phases[0])
        self.graphs[phase].replay()
        # Replay runs no Python, so the phase advance `step_synchronized` does
        # in eager has to happen here, or the tracker's phase would freeze.
        self.sr_tracker.phases.fill((phase + 1) % self.phase_k)

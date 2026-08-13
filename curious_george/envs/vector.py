"""Process-parallel rollout envs: AsyncVectorEnv + a shell-like adapter.

The B>1 collector's env-stepping loop was serial Python - eight MiniGrid RGB
renders per step, one after another (~half of a B=8 update). AsyncShellPool
runs the RAW wrapped minigrid envs (everything factory.make_env builds BELOW
FaramaMinigridShell) in worker processes via gymnasium's AsyncVectorEnv and
exposes exactly what the batched collector consumes: per-env obs dicts,
rewards, dones, and agent positions (shipped in step infos - no extra IPC
round-trip). Shell-level services (env2pred formatting, map bins, rendering)
are NOT available on workers; static attributes are mirrored from a separate
eval shell that also serves analysis/eval rollouts (comps.env).

Episode cuts are collector-driven (prnn_seqdur) and synchronized across envs,
so only a full reset_all is supported; a real env-signaled termination raises
(LRoom training never terminates - seqdur cuts fire long before max_steps).

Equivalence to the serial list-of-envs path is exact and tested
(tests/test_async_envs.py): same construction seeds, same reset chain, and
env transitions are deterministic given actions.
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from minigrid.wrappers import FullyObsWrapper


class DropMission(ObservationWrapper):
    """Numeric-only obs (image, direction) so AsyncVectorEnv can use shared
    memory instead of pickling dicts with a string mission through pipes.
    The mission is a constant per env; the pool re-attaches it on unstack."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {k: v for k, v in env.observation_space.spaces.items() if k != "mission"}
        )

    def observation(self, obs):
        return {k: v for k, v in obs.items() if k != "mission"}


class PosInfo(Wrapper):
    """Adds agent_pos/agent_dir to step() and reset() infos so the main
    process gets positions with the step payload (no second call round-trip)."""

    def _annotate(self, info: dict) -> dict:
        info["agent_pos"] = np.asarray(self.env.unwrapped.agent_pos)
        info["agent_dir"] = int(self.env.unwrapped.agent_dir)
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, self._annotate(info)


def _make_worker_thunk(cfg, seed_offset: int):
    """One worker env: factory.make_env minus the FaramaMinigridShell wrap,
    identical seeding (env.reset(seed=...) at construction)."""
    env_key = str(cfg.exp.env_name)
    input_type = str(cfg.exp.input_type)
    seed = int(cfg.exp.seed) + 10000 + seed_offset
    start_room = None if cfg.exp.start_rand else cfg.exp.start_room
    kwargs = dict(
        agent_start_pos=None,
        agent_start_dir=None,
        render_mode="rgb_array",
        open_all_paths=False,
        subroom_size=cfg.exp.env_subroom_size,
        door_poss=cfg.exp.door_poss,
        agent_start_room=start_room,
    )

    see_through = cfg.exp.get("see_through_walls", None)

    def thunk():
        env = gym.make(env_key, **kwargs)
        if see_through is not None:
            # must match factory.make_env, else async workers would train
            # WITHOUT occlusion while the eval shell has it
            env.unwrapped.see_through_walls = see_through
        if input_type == "Visual_FO":
            env = FullyObsWrapper(env)
        elif "pRNN" in input_type or "PO" in input_type:
            from curious_george.envs.obs_bank import (
                BankedRGBPartialObsWrapper,
                TableDrivenRGBPartialObsWrapper,
            )

            wrapper_cls = (
                TableDrivenRGBPartialObsWrapper
                if cfg.exp.get("table_env", False)
                else BankedRGBPartialObsWrapper
            )
            env = wrapper_cls(env, tile_size=1)
        else:
            raise ValueError(f"async envs not supported for input_type={input_type}")
        env.reset(seed=seed)
        return PosInfo(DropMission(env))

    return thunk


class AsyncShellPool:
    """B worker envs + a local eval shell providing static/shell services.

    Interface consumed by the collector's pool branch:
      reset_all() -> obs_dicts;  step(actions) -> (obs_dicts, rewards, dones,
      positions);  positions/last infos are (B, ...) numpy.
    `eval_shell` is a normal FaramaMinigridShell for analysis/eval/plotting
    (its own seed stream, NOT one of the workers - eval rollouts no longer
    perturb the training env streams, unlike the sync list-of-envs mode).
    """

    def __init__(self, cfg, eval_shell):
        self.B = int(cfg.exp.num_envs)
        self.eval_shell = eval_shell
        self.envs = gym.vector.AsyncVectorEnv(
            [_make_worker_thunk(cfg, seed_offset=1000 * i) for i in range(self.B)],
            shared_memory=False,  # measured faster than shm here; numeric-only obs keeps pickles small
        )
        # mission is constant per task; re-attached on unstack for downstream
        # consumers (preprocessor tokenizes it)
        self._mission = eval_shell.reset()["mission"]
        # static attrs mirrored for diagnostics/preprocessor consumers
        self.numHDs = eval_shell.numHDs
        self.width = eval_shell.width
        self.height = eval_shell.height
        self.obs_shape = eval_shell.obs_shape
        self.action_space = eval_shell.action_space
        self.observation_space = eval_shell.observation_space

    def __len__(self) -> int:
        return self.B

    def __getitem__(self, i: int):
        """envs[0] historically doubles as the analysis/eval shell; worker
        envs live in other processes and are not indexable."""
        if i == 0:
            return self.eval_shell
        raise IndexError("AsyncShellPool: only [0] (the eval shell) is addressable")

    def _unstack_obs(self, obs_batch: dict, B: int) -> list:
        images = obs_batch["image"]
        directions = obs_batch["direction"]
        return [
            {
                "mission": self._mission,
                "image": np.array(images[b]),  # copy out of shared memory
                "direction": int(directions[b]),
            }
            for b in range(B)
        ]

    @staticmethod
    def _positions(infos: dict, B: int) -> np.ndarray:
        return np.asarray([infos["agent_pos"][b] for b in range(B)])

    def reset_all(self) -> tuple[list, np.ndarray]:
        """Synchronized reset (unseeded: continues each worker's RNG chain,
        matching the serial path's plain env.reset()). Returns
        (obs_dicts, positions (B, 2))."""
        obs_batch, infos = self.envs.reset()
        return self._unstack_obs(obs_batch, self.B), self._positions(infos, self.B)

    def step(self, actions: np.ndarray) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """One parallel step. actions (B,) int -> (obs_dicts list[B],
        rewards (B,), dones (B,) bool, positions (B, 2))."""
        obs_batch, rewards, term, trunc, infos = self.envs.step(actions)
        dones = np.logical_or(term, trunc)
        assert not dones.any(), (
            "async pool got an env-signaled termination; training episode cuts "
            "are seqdur-driven and synchronized - per-env resets are not "
            "implemented (see module docstring)"
        )
        return (
            self._unstack_obs(obs_batch, self.B),
            np.asarray(rewards, dtype=np.float64),
            dones,
            self._positions(infos, self.B),
        )

    def close(self) -> None:
        self.envs.close()


class DeviceTableShellPool:
    """One batched static L-room state machine resident on the train device.

    The ordinary list-of-envs path requires a device-to-host action copy and
    then runs ``B`` Python ``env.step`` calls at every timestep.  This pool
    stores ``(x, y, direction)`` for all streams in tensors and applies the
    same exhaustive transition table with one advanced-indexing operation.

    CPU shells produce the independently seeded reset schedule before each
    rollout; all reset rows are uploaded together. The supported training
    layout has no environment-triggered reward/termination, so no tensor value
    has to be inspected by Python inside the transition loop.
    """

    def __init__(
        self,
        *,
        training_shells: list,
        eval_shell,
        device: torch.device,
        layouts: list | None = None,
        layout_seed: int = 0,
    ):
        """`layouts` holds one or more rooms the streams are drawn from.

        None keeps the historical single-room behaviour exactly. Otherwise each
        stream is assigned a layout at every synchronized episode boundary, so a
        pooled world-model gradient step averages over several rooms at once and
        the same integrated trajectory lands at a different absolute position
        depending on which room a stream is in.

        The observation bank gains a leading layout axis; the TRANSITION table
        does not, and that is asserted rather than assumed. It holds only because
        landmarks are walkable, non-occluding `Floor`, so they change what the
        agent sees and never where it can go. A landmark that blocked movement
        would make the tables diverge, and this must fail loudly if that ever
        happens.
        """
        from curious_george.envs.obs_bank import TableDrivenRGBPartialObsWrapper

        if len(training_shells) < 2:
            raise ValueError("device_env is a batched path and requires num_envs > 1")

        wrappers = []
        for shell in training_shells:
            wrapper = shell.env
            if not isinstance(wrapper, TableDrivenRGBPartialObsWrapper):
                raise ValueError(
                    "device_env requires TableDrivenRGBPartialObsWrapper; "
                    "enable the static table environment"
                )
            wrapper._ensure_bank()
            wrappers.append(wrapper)

        reference = wrappers[0]
        for wrapper in wrappers[1:]:
            if (
                wrapper._fingerprint != reference._fingerprint
                or not np.array_equal(wrapper._next_state, reference._next_state)
                or not np.array_equal(wrapper._bank, reference._bank)
            ):
                raise ValueError(
                    "device_env requires every stream to start from one static grid"
                )

        if np.any(reference._rewarding) or np.any(reference._terminated):
            raise ValueError(
                "device_env currently requires a transition table with no "
                "environment-triggered rewards/terminations; synchronized "
                "prnn_seqdur episode cuts remain supported"
            )

        self.B = len(training_shells)
        self.eval_shell = eval_shell
        self._training_shells = training_shells
        self._wrappers = wrappers
        self.device = torch.device(device)
        self._mission = reference.unwrapped.mission
        self.layouts = list(layouts) if layouts else None
        self._layout_rng = np.random.default_rng(layout_seed)

        banks = (
            [np.array(reference._bank)]
            if self.layouts is None
            else self._collect_layout_banks(reference=reference)
        )

        # Copy read-only NumPy banks before torch takes ownership. State table
        # is tiny; the uint8 observation bank is only ~0.2 MB per layout at 16x16.
        self.next_state = torch.tensor(
            np.array(reference._next_state),
            dtype=torch.long,
            device=self.device,
        )
        self.obs_banks = torch.tensor(
            np.stack(banks), dtype=torch.uint8, device=self.device
        )
        self.stream_layout = torch.zeros(self.B, dtype=torch.long, device=self.device)
        # Episodes each layout has been trained on. A layout with few episodes is
        # UNTESTED rather than negative, so this has to be reported with any
        # per-layout result.
        self.layout_episodes = np.zeros(self.n_layouts, dtype=np.int64)
        self._zero_rewards = torch.zeros(self.B, device=self.device)

        # Static shell services used by setup/diagnostics.
        self.numHDs = eval_shell.numHDs
        self.width = eval_shell.width
        self.height = eval_shell.height
        self.obs_shape = eval_shell.obs_shape
        self.action_space = eval_shell.action_space
        self.observation_space = eval_shell.observation_space
        self.positions = torch.empty((self.B, 2), dtype=torch.long, device=self.device)
        self.directions = torch.empty(self.B, dtype=torch.long, device=self.device)
        self._prepared_positions: torch.Tensor | None = None
        self._prepared_directions: torch.Tensor | None = None
        self._prepared_layouts: torch.Tensor | None = None
        self._prepared_layouts_host: np.ndarray | None = None

    def _collect_layout_banks(self, *, reference) -> list:
        """One wrapper visits every layout once, stacking the banks it builds.

        Also the place the transition-table invariant is checked: a layout whose
        landmarks changed where the agent can go would need its own table, and
        the device path holds exactly one.
        """
        banks = []
        base_state = np.array(reference._next_state)
        for layout in self.layouts:
            reference.unwrapped.landmarks = list(layout.landmarks)
            reference.reset(seed=0)
            if not np.array_equal(reference._next_state, base_state):
                raise ValueError(
                    f"layout {layout.key} changes the transition table; the device "
                    "path holds one table for all layouts, which is only valid "
                    "while landmarks are walkable and non-occluding"
                )
            banks.append(np.array(reference._bank))
        return banks

    @property
    def n_layouts(self) -> int:
        return 1 if self.layouts is None else len(self.layouts)

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Shape of one observation image, independent of the bank's layout axis."""
        return tuple(self.obs_banks.shape[4:])

    def __len__(self) -> int:
        return self.B

    def __getitem__(self, i: int):
        if i == 0:
            return self.eval_shell
        raise IndexError(
            "DeviceTableShellPool: only [0] (the eval shell) is addressable"
        )

    @property
    def mission(self) -> str:
        return self._mission

    def observation_device(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image uint8 ``(B,H,W,C)`` and direction int64 ``(B,)``."""
        images = self.obs_banks[
            self.stream_layout, self.positions[:, 0], self.positions[:, 1], self.directions
        ]
        return images, self.directions

    def _reset_streams(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw one layout and one start state per stream. (layouts, pos, dir).

        The layout is applied to the CPU shell BEFORE it resets, because
        MiniGrid's `place_agent` rejects occupied cells and the landmarks are
        objects - so where an episode can start depends on which room it is in.
        Reading the start from the shell keeps that rule owned by MiniGrid
        rather than reimplemented here.
        """
        if self.layouts is not None:
            chosen = self._layout_rng.integers(0, len(self.layouts), size=self.B)
            for b, wrapper in enumerate(self._wrappers):
                wrapper.unwrapped.landmarks = list(self.layouts[chosen[b]].landmarks)
        else:
            chosen = np.zeros(self.B, dtype=np.int64)

        for shell in self._training_shells:
            shell.reset()
        positions = np.asarray(
            [wrapper.unwrapped.agent_pos for wrapper in self._wrappers], dtype=np.int64
        )
        directions = np.asarray(
            [wrapper.unwrapped.agent_dir for wrapper in self._wrappers], dtype=np.int64
        )
        return chosen.astype(np.int64), positions, directions

    def reset_all(self) -> tuple[list, np.ndarray]:
        """Continue each CPU shell's RNG stream and upload only reset state."""
        chosen, positions, directions = self._reset_streams()
        self.positions.copy_(torch.as_tensor(positions, device=self.device))
        self.directions.copy_(torch.as_tensor(directions, device=self.device))
        self.stream_layout.copy_(torch.as_tensor(chosen, device=self.device))
        np.add.at(self.layout_episodes, chosen, 1)
        # BatchedSRTracker only needs the stream count; policy/pRNN inputs come
        # directly from observation_device().
        return [None] * self.B, positions

    def prepare_resets(self, *, count: int) -> None:
        """Precompute and upload the finite reset schedule before rollout."""
        if count <= 0:
            raise ValueError("prepared reset count must be positive")
        layout_rows, position_rows, directions_rows = [], [], []
        for _ in range(count):
            chosen, positions, directions = self._reset_streams()
            layout_rows.append(chosen)
            position_rows.append(positions)
            directions_rows.append(directions)

        self._prepared_positions = torch.as_tensor(
            np.stack(position_rows), device=self.device
        )
        self._prepared_directions = torch.as_tensor(
            np.stack(directions_rows), device=self.device
        )
        self._prepared_layouts = torch.as_tensor(
            np.stack(layout_rows), device=self.device
        )
        self._prepared_layouts_host = np.stack(layout_rows)

    def apply_prepared_reset(self, *, index: int) -> None:
        """Select an already resident reset row without any host transfer."""
        if (
            self._prepared_positions is None
            or self._prepared_directions is None
            or not 0 <= index < self._prepared_positions.shape[0]
        ):
            raise IndexError(f"prepared reset {index} is unavailable")
        self.positions.copy_(self._prepared_positions[index])
        self.directions.copy_(self._prepared_directions[index])
        self.stream_layout.copy_(self._prepared_layouts[index])
        np.add.at(self.layout_episodes, self._prepared_layouts_host[index], 1)

    def step_device(
        self, *, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply ``B`` transitions without a device-to-host copy or sync.

        Returns post-step images, directions, and the reusable all-zero reward
        vector. Episode cuts are owned by the collector's known Python
        timestep, not by data-dependent device control flow.
        """
        actions = actions.to(dtype=torch.long)
        next_rows = self.next_state[
            self.positions[:, 0],
            self.positions[:, 1],
            self.directions,
            actions,
        ]
        self.positions.copy_(next_rows[:, :2])
        self.directions.copy_(next_rows[:, 2])
        images, directions = self.observation_device()
        return images, directions, self._zero_rewards

    def close(self) -> None:
        for shell in self._training_shells:
            shell.env.close()

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
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper_HD

from curious_george.utils.enums import AgentInputType


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

    def thunk():
        env = gym.make(env_key, **kwargs)
        if input_type == "Visual_FO":
            env = FullyObsWrapper(env)
        elif "pRNN" in input_type or "PO" in input_type:
            env = RGBImgPartialObsWrapper_HD(env, tile_size=1)
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

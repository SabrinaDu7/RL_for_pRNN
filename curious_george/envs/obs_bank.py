"""Precomputed observation bank: (agent_pos, agent_dir) -> partial RGB obs.

For a static grid, RGBImgPartialObsWrapper_HD's per-step get_frame render
(~0.7 ms, the single biggest per-step cost) is a pure function of
(x, y, dir, grid). BankedRGBPartialObsWrapper precomputes the full
(W, H, 4) bank once per grid layout (~1 s for 18x18), persists it under
data/obs_bank/ (committed; the cluster copies it to $SLURM_TMPDIR with the
repo), and serves per-step observations as lookups.

Byte-equality with the live render is asserted in tests/test_obs_bank.py;
the bank is keyed by a fingerprint of grid.encode(), so a layout change
(e.g. OMT's object-present vs object-absent envs, randomized FourRooms)
transparently builds/loads a different bank file. Bank arrays are served
read-only (writeable=False) so accidental mutation raises instead of
silently corrupting the bank.
"""

import hashlib
from pathlib import Path

import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from minigrid.wrappers import RGBImgPartialObsWrapper_HD

BANK_DIR = Path(__file__).resolve().parents[2] / "data" / "obs_bank"


class BankedRGBPartialObsWrapper(RGBImgPartialObsWrapper_HD):
    """Drop-in replacement for RGBImgPartialObsWrapper_HD backed by the bank."""

    def __init__(self, env, tile_size: int = 1, bank_dir: Path = BANK_DIR):
        super().__init__(env, tile_size)
        self.bank_dir = Path(bank_dir)
        self._bank: np.ndarray | None = None  # (W, H, 4, h, w, 3) uint8
        self._fingerprint: str | None = None

    # -- bank management ---------------------------------------------------

    def _grid_fingerprint(self) -> str:
        """Fingerprint of everything that changes the rendered observation.

        grid.encode() alone is NOT enough: `see_through_walls` decides whether
        gen_obs_grid runs process_vis, so two envs with the same grid but
        different occlusion produce different observations. Without this the
        bank would serve non-occluded observations for an occluded env.
        The suffix is only appended when occlusion is ON, so every bank cached
        before this change (all see_through_walls=True) keeps its filename.
        """
        grid = self.unwrapped.grid
        fp = hashlib.sha1(grid.encode().tobytes()).hexdigest()[:16]
        if not getattr(self.unwrapped, "see_through_walls", True):
            fp += "-occl"
        return fp

    def _bank_path(self, fingerprint: str) -> Path:
        env_id = (self.unwrapped.spec.id if self.unwrapped.spec else "env").replace("/", "_")
        return self.bank_dir / f"{env_id}_tile{self.tile_size}_{fingerprint}.npz"

    def _build_bank(self) -> np.ndarray:
        """Render every (x, y, dir) once (walls included - the agent never
        stands there, but rendering them is harmless and keeps indexing flat)."""
        env = self.unwrapped
        saved_pos, saved_dir = env.agent_pos, env.agent_dir
        h, w, c = self.observation_space.spaces["image"].shape
        bank = np.zeros((env.width, env.height, 4, h, w, c), dtype=np.uint8)
        for x in range(env.width):
            for y in range(env.height):
                for d in range(4):
                    env.agent_pos = (x, y)
                    env.agent_dir = d
                    bank[x, y, d] = self.get_frame(tile_size=self.tile_size, agent_pov=True)
        env.agent_pos, env.agent_dir = saved_pos, saved_dir
        return bank

    def _ensure_bank(self) -> None:
        fingerprint = self._grid_fingerprint()
        if fingerprint == self._fingerprint:
            return
        path = self._bank_path(fingerprint)
        if path.exists():
            bank = np.load(path)["bank"]
        else:
            bank = self._build_bank()
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, bank=bank)
            print(f"obs bank built and saved: {path} ({bank.nbytes / 1e6:.1f} MB raw)")
        bank.flags.writeable = False
        self._bank = bank
        self._fingerprint = fingerprint

    # -- gym API -----------------------------------------------------------

    def reset(self, **kwargs):
        # re-key BEFORE formatting the obs: reset may regenerate the grid,
        # and the reset obs itself must come from the matching bank
        obs, info = self.env.reset(**kwargs)
        self._ensure_bank()
        return self.observation(obs), info

    def observation(self, obs: dict) -> dict:
        if self._bank is None:  # first observation arrives during reset()
            self._ensure_bank()
        env = self.unwrapped
        x, y = env.agent_pos
        return {
            "mission": obs["mission"],
            "image": self._bank[x, y, env.agent_dir],
            "direction": obs["direction"],
        }


class TableDrivenRGBPartialObsWrapper(BankedRGBPartialObsWrapper):
    """Banked observations plus a table-driven transition fast path.

    This is intentionally narrow: it supports static MiniGrid layouts whose
    action space is exactly ``left, right, forward, pickup`` and which contain
    no pickable cells.  That is the state machine used by the L-room training
    and OMT variants.  Unsupported layouts fail during reset instead of
    silently changing their dynamics.

    MiniGrid's regular ``step`` computes a fresh partial observation before
    :class:`BankedRGBPartialObsWrapper` discards it in favour of the bank.
    Here both the transition and post-step observation are table lookups, so
    no grid slicing, rotation, visibility computation, or RGB encoding occurs
    on the hot path.
    """

    _NUM_ACTIONS = 4

    def __init__(self, env, tile_size: int = 1, bank_dir: Path = BANK_DIR):
        super().__init__(env, tile_size=tile_size, bank_dir=bank_dir)
        self._transition_fingerprint: str | None = None
        self._next_state: np.ndarray | None = None
        self._rewarding: np.ndarray | None = None
        self._terminated: np.ndarray | None = None

    def _ensure_bank(self) -> None:
        super()._ensure_bank()
        if self._transition_fingerprint == self._fingerprint:
            return
        self._build_transition_tables()
        self._transition_fingerprint = self._fingerprint

    def _build_transition_tables(self) -> None:
        env = self.unwrapped
        if env.action_space.n != self._NUM_ACTIONS:
            raise ValueError(
                "table_env requires the four-action L-room action space; "
                f"got Discrete({env.action_space.n})"
            )

        for cell in env.grid.grid:
            if cell is not None and cell.can_pickup():
                raise ValueError(
                    "table_env does not support mutable/pickable grid objects"
                )

        # State is (x, y, direction). Invalid/wall states are populated too;
        # the agent never occupies them, and flat complete tables keep lookup
        # and device transfer simple.
        shape = (env.width, env.height, 4, self._NUM_ACTIONS)
        next_state = np.empty(shape + (3,), dtype=np.int16)
        rewarding = np.zeros(shape, dtype=np.bool_)
        terminated = np.zeros(shape, dtype=np.bool_)

        for x in range(env.width):
            for y in range(env.height):
                for direction in range(4):
                    for action in range(self._NUM_ACTIONS):
                        nx, ny, nd = x, y, direction
                        if action == int(env.actions.left):
                            nd = (direction - 1) % 4
                        elif action == int(env.actions.right):
                            nd = (direction + 1) % 4
                        elif action == int(env.actions.forward):
                            dx, dy = DIR_TO_VEC[direction]
                            fx, fy = x + int(dx), y + int(dy)
                            # Invalid table states on an outer boundary can
                            # point outside the grid. They remain stationary.
                            if 0 <= fx < env.width and 0 <= fy < env.height:
                                cell = env.grid.get(fx, fy)
                                if cell is None or cell.can_overlap():
                                    nx, ny = fx, fy
                                if cell is not None:
                                    rewarding[x, y, direction, action] = (
                                        cell.type == "goal"
                                        or cell.type == "fake_lava"
                                    )
                                    terminated[x, y, direction, action] = (
                                        rewarding[x, y, direction, action]
                                        or cell.type == "lava"
                                    )
                        # pickup is a no-op because pickable cells were rejected
                        next_state[x, y, direction, action] = (nx, ny, nd)

        next_state.flags.writeable = False
        rewarding.flags.writeable = False
        terminated.flags.writeable = False
        self._next_state = next_state
        self._rewarding = rewarding
        self._terminated = terminated

    @staticmethod
    def _scalar_action(action) -> int:
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError(f"expected one scalar action, got shape {action_array.shape}")
        return int(action_array.item())

    def _current_observation(self) -> dict:
        env = self.unwrapped
        x, y = env.agent_pos
        return {
            "mission": env.mission,
            "image": self._bank[x, y, env.agent_dir],
            "direction": env.agent_dir,
        }

    def step(self, action):
        if self._next_state is None:
            self._ensure_bank()

        env = self.unwrapped
        action_int = self._scalar_action(action)
        if not 0 <= action_int < self._NUM_ACTIONS:
            raise ValueError(f"Unknown action: {action_int}")

        x, y = env.agent_pos
        direction = env.agent_dir
        env.step_count += 1

        nx, ny, nd = self._next_state[x, y, direction, action_int]
        env.agent_pos = (int(nx), int(ny))
        env.agent_dir = int(nd)

        reward = (
            env._reward()
            if self._rewarding[x, y, direction, action_int]
            else 0
        )
        terminated = bool(self._terminated[x, y, direction, action_int])
        truncated = env.step_count >= env.max_steps

        if env.render_mode == "human":
            env.render()

        return self._current_observation(), reward, terminated, truncated, {}

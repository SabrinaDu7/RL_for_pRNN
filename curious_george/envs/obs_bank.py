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

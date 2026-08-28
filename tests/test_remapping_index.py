"""`multiroom/remapping_index` must detect remapping that is there by construction,
and must NOT report it for a map that is merely degraded.

Why this exists. The index is `mean(per-room sRSA) - pooled sRSA`
(`evaluation/spatial.py::evaluate_multi_room_representation`). Its docstring covers
two cases - index~0 with HIGH per-room sRSA is a real negative, index~0 with LOW
per-room sRSA is an uninformative run - and says nothing about the third, which is
what `multienv-walkable-traj` actually produced:

    sRSA peak  33.6M   mean 0.8030  ->  remapping +0.0030   (LOWEST index)
    sRSA trough 83.9M  mean 0.5773  ->  remapping +0.0619   (HIGHEST index)

A rising index while the map falls apart. If degradation alone can lift the index -
because pooling more rows is a harder test and so degrades faster - then every
remapping number this project has is uninterpretable. `test_degraded_shared_map_*`
is the arm that settles it; the other two are the positive and negative controls
that make its result readable.

Synthetic populations, not trained networks: the ground truth has to be known.
"""

import numpy as np
import pytest
import torch

SEED = 7
N_ROOMS = 4
N_ROWS = 400  # rows per room
N_CELLS = 120  # synthetic place cells
SIGMA = 2.0  # place-field width, in grid cells


def place_code(
    positions: np.ndarray, centres: np.ndarray, *, sigma: float = SIGMA
) -> np.ndarray:
    """(N, C) Gaussian place-cell rates for `positions` against `centres`."""
    d2 = ((positions[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma**2))


@pytest.fixture(scope="module")
def env():
    from prnn.utils import ActionEncodingsEnum, MinigridEnvNames
    from curious_george import AgentInputType, make_env

    return make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def pN(env):
    from prnn.utils import PredictiveNet

    net = PredictiveNet(
        env, hidden_size=N_CELLS, pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0), wandb_log=False,
    )
    return net


@pytest.fixture(scope="module")
def walkable(env) -> np.ndarray:
    """(K, 2) MiniGrid coordinates the agent can occupy.

    `get_walkable_mask` returns a WALL-EXCLUDED, SHIFTED frame - mask[x, y] is
    grid cell (x+1, y+1) - so it is padded back rather than corrected at the
    call site.
    """
    from curious_george.envs.access import get_walkable_mask

    inner = get_walkable_mask(env).numpy().astype(bool)
    mask = np.zeros((env.env.unwrapped.width, env.env.unwrapped.height), dtype=bool)
    mask[1:-1, 1:-1] = inner
    return np.argwhere(mask).astype(np.float64)


def remapping_index(*, pN, env, per_room_h, per_room_pos) -> dict:
    """`mean(per-room sRSA) - pooled sRSA`, assembled exactly as
    `evaluate_multi_room_representation` assembles it (spatial.py:315-341)."""
    from curious_george.models.device import eval_mode, on_device

    rng = np.random.default_rng(SEED)
    with eval_mode([pN]), on_device([pN], "cpu"):
        per = [
            float(pN.calculateSpatialMetrics(h, p, env, rng=rng, wandb_nameext="")["sRSA"])
            for h, p in zip(per_room_h, per_room_pos)
        ]
        pooled = float(
            pN.calculateSpatialMetrics(
                np.concatenate(per_room_h), np.concatenate(per_room_pos), env,
                rng=rng, wandb_nameext="",
            )["sRSA"]
        )
    return {"per_room": per, "mean_room": float(np.mean(per)), "pooled": pooled,
            "index": float(np.mean(per)) - pooled}


def populations(walkable: np.ndarray, *, room_specific: bool, noise: float):
    """Per-room (h, pos). `room_specific` gives each room its OWN place centres."""
    rng = np.random.default_rng(SEED)
    shared = walkable[rng.choice(len(walkable), N_CELLS)]
    hs, ps = [], []
    for _ in range(N_ROOMS):
        pos = walkable[rng.choice(len(walkable), N_ROWS)]
        centres = walkable[rng.choice(len(walkable), N_CELLS)] if room_specific else shared
        h = place_code(pos, centres)
        hs.append(h + noise * rng.standard_normal(h.shape))
        ps.append(pos)
    return hs, ps


@pytest.fixture(scope="module")
def shared_clean(pN, env, walkable):
    h, p = populations(walkable, room_specific=False, noise=0.0)
    return remapping_index(pN=pN, env=env, per_room_h=h, per_room_pos=p)


@pytest.fixture(scope="module")
def specific_clean(pN, env, walkable):
    h, p = populations(walkable, room_specific=True, noise=0.0)
    return remapping_index(pN=pN, env=env, per_room_h=h, per_room_pos=p)


@pytest.fixture(scope="module")
def shared_degraded(pN, env, walkable):
    h, p = populations(walkable, room_specific=False, noise=1.5)
    return remapping_index(pN=pN, env=env, per_room_h=h, per_room_pos=p)


# -- the negative control: one map for every room -------------------------------

def test_shared_map_has_high_per_room_srsa(shared_clean):
    assert shared_clean["mean_room"] > 0.5, shared_clean


def test_shared_map_index_is_near_zero(shared_clean):
    """The same (x, y) means the same thing in every room, so pooling costs nothing."""
    assert abs(shared_clean["index"]) < 0.05, shared_clean


# -- the positive control: a different map per room -----------------------------

def test_room_specific_map_has_high_per_room_srsa(specific_clean):
    """Each room is individually well-mapped; only the POOLED view should suffer."""
    assert specific_clean["mean_room"] > 0.5, specific_clean


def test_room_specific_map_raises_the_index(specific_clean, shared_clean):
    """The whole point. If this fails, no remapping number in this project means
    anything, because the metric cannot see remapping built by construction."""
    assert specific_clean["index"] > shared_clean["index"] + 0.05, {
        "specific": specific_clean, "shared": shared_clean,
    }


# -- the arm that settles the confound ------------------------------------------

def test_degraded_shared_map_has_low_per_room_srsa(shared_degraded):
    """Precondition: the noise really did damage the map, or the test is vacuous."""
    assert shared_degraded["mean_room"] < 0.75, shared_degraded


def test_degradation_alone_does_not_lift_the_index(shared_degraded, specific_clean):
    """🔴 The question `multienv-walkable-traj` raised.

    A degraded but still room-AGNOSTIC population must not score like a
    room-specific one. If it does, a rising index during an sRSA decline is an
    artefact and the index cannot gate anything.
    """
    assert shared_degraded["index"] < specific_clean["index"], {
        "degraded": shared_degraded, "specific": specific_clean,
    }

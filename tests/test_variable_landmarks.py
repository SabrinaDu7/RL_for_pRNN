"""A room design may hold 0 to 3 landmarks (2026-09-01).

Removal previously lived only at env construction (`make_env(landmarks=...)`);
these pin the CONFIG level: `EnvContent` accepts any kind count its palette
covers, the resolver produces the right pools - the zero-kind design admits
exactly ONE room, and asking it for more fails with the count spelled out -
and the environment, bank and walkable set are correct at every count.
"""

import numpy as np
import pytest

from curious_george import make_env
from curious_george.configs import Config, EnvCfg, EvalCfg, EvalKind
from curious_george.envs.access import get_walkable_mask
from curious_george.envs.layouts import (
    SHAPES,
    UNCONSTRAINED,
    EnvContent,
    EnvShape,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
    resolve_layouts,
    resolve_rooms,
)
from curious_george.utils.enums import AgentInputType
from prnn.utils import ActionEncodingsEnum

from dataclasses import replace


def _content(n: int) -> EnvContent:
    return EnvContent(kinds=tuple(LandmarkKind(s, impassable=True) for s in SHAPES[:n]))


def _rooms(n: int, *, pool: int):
    return resolve_rooms(
        shape=EnvShape("MiniGrid-LRoom-v0"), content=_content(n),
        source=Uniform(n=pool, seed=7),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_every_count_resolves_and_builds(n: int) -> None:
    rooms = _rooms(n, pool=1)
    room = rooms[0]
    assert len(room.landmarks) == n
    env = make_env(env_key="MiniGrid-LRoom-v0", input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0,
                   landmarks=list(room.landmarks))
    walk = int(np.asarray(get_walkable_mask(env)).sum())
    blocked = sum(len(lm.cells) for lm in room.landmarks)
    assert walk == 172 - blocked, f"walkable {walk} != 172 - {blocked}"


def test_the_empty_design_admits_exactly_one_room() -> None:
    assert len(_rooms(0, pool=1)) == 1
    with pytest.raises(ValueError, match="only 1 admissible"):
        _rooms(0, pool=2)


def test_pairwise_rules_are_unconstrained_below_two_landmarks() -> None:
    for n in (0, 1):
        room = _rooms(n, pool=1)[0]
        assert room.min_anchor_separation == UNCONSTRAINED
        assert room.min_cell_gap() == UNCONSTRAINED


def test_two_landmark_pools_still_respect_separation() -> None:
    for room in _rooms(2, pool=5):
        assert room.min_anchor_separation >= 3
        assert len(room.landmarks) == 2


def test_a_full_config_carries_a_zero_landmark_design() -> None:
    from curious_george.configs import EnvBackend

    base = Config(env=EnvCfg(content=_content(0), source=Uniform(n=1, seed=7),
                             set_rules=RoomSetRules(varies=frozenset({Vary.POSITION}))),
                  collect=replace(Config().collect, backend=EnvBackend.DEVICE),
                  eval=EvalCfg(evals=frozenset({EvalKind.SPATIAL_MULTIROOM})))
    rooms = resolve_layouts(base)
    assert len(rooms) == 1 and rooms[0].landmarks == ()


def test_the_default_design_is_untouched() -> None:
    """Three kinds, exactly as every existing checkpoint trained."""
    c = EnvContent()
    assert c.n_landmarks == 3 and c.stencils == SHAPES


def test_keep_landmarks_expresses_the_mixed_count_design() -> None:
    from curious_george.envs.layouts import Selected

    rooms = resolve_rooms(
        shape=EnvShape("MiniGrid-LRoom-v0"), content=EnvContent(),
        source=Selected(n=8, impassable=True, positions=(0, 1, 2, 3, 5, 6, 7, 8),
                        keep_landmarks=("012", "012", "012", "12", "01", "0", "2", "-")),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )
    assert [len(r.landmarks) for r in rooms] == [3, 3, 3, 2, 2, 1, 1, 0]
    from collections import Counter

    colours = Counter(lm.color for r in rooms for lm in r.landmarks)
    assert colours == {"blue": 5, "green": 5, "red": 5}


def test_keep_landmarks_misalignment_and_bad_digits_fail_loudly() -> None:
    from curious_george.envs.layouts import Selected

    kw = dict(shape=EnvShape("MiniGrid-LRoom-v0"), content=EnvContent(),
              set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})))
    with pytest.raises(ValueError, match="align by position"):
        resolve_rooms(source=Selected(n=8, impassable=True,
                                      positions=(0, 1, 2, 3, 5, 6, 7, 8),
                                      keep_landmarks=("012",)), **kw)
    with pytest.raises(ValueError, match="invalid or repeated"):
        resolve_rooms(source=Selected(n=1, impassable=True, positions=(0,),
                                      keep_landmarks=("03",)), **kw)


def test_keep_landmarks_parses_from_the_command_line() -> None:
    from curious_george.configs import cli

    cfg = cli(["multienv-fast", "--run.no-wandb", "env.source:selected",
               "--env.source.n", "8", "--env.source.impassable",
               "--env.source.positions", "0", "1", "2", "3", "5", "6", "7", "8",
               "--env.source.keep-landmarks", "012", "012", "012", "12", "01", "0", "2", "-"])
    assert cfg.env.source.keep_landmarks == ("012", "012", "012", "12", "01", "0", "2", "-")

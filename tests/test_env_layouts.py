"""Shape x content -> rooms: what varies, what is held constant, what is legal.

The point of separating shape from content is that a room set becomes a stated
experiment rather than whatever the generator happened to do. These cover the
three axes that statement is made on:

    WITHIN a room   RoomRules      - is this placement legal?
    BETWEEN rooms   RoomSetRules   - what differs, and what is held constant?
    WHERE FROM      RoomSource     - env default, committed, curated, uniform?
"""

import pytest

from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    ROOMS_RUN1,
    ROOMS_SQUARE,
    SQUARE_ROOM_ID,
    Committed,
    Curated,
    EnvContent,
    EnvDefault,
    EnvShape,
    LandmarkKind,
    RoomRules,
    RoomSetRules,
    Symmetry,
    Uniform,
    Vary,
    _d4,
    admissible_placements,
    resolve_rooms,
    separation_signature,
)

L_SHAPE = EnvShape(BASE_ROOM_ID)
SQ_SHAPE = EnvShape(SQUARE_ROOM_ID)
CONTENT = EnvContent()


def _rooms(varies, *, n=3, seed=7, shape=L_SHAPE, content=CONTENT):
    return resolve_rooms(
        shape=shape, content=content, source=Curated(n=n, seed=seed),
        set_rules=RoomSetRules(varies=frozenset(varies)),
    )


# --- what the shape knows about itself ------------------------------------


def test_symmetry_is_a_fact_about_the_walls():
    """Not a setting. `dedupe_d4` used to be a boolean derived by
    string-comparing the room id at the one call site that needed it."""
    assert SQ_SHAPE.symmetry is Symmetry.D4
    assert L_SHAPE.symmetry is Symmetry.TRIVIAL


@pytest.mark.parametrize("shape,closed", [(SQ_SHAPE, True), (L_SHAPE, False)])
def test_span_matches_the_d4_contract(shape, closed):
    """`_d4` documents "interior runs 1..span", so span is the MAX interior
    coordinate, not max+1.

    Computing it as max+1 maps every cell one step outside the room - 225
    images for a 196-cell square - which would silently corrupt the orbits the
    square-room dedup is built on. Caught exactly this way.

    The square is closed under D4 and the L-shape is not, which IS the
    symmetry classification above, measured rather than asserted.
    """
    cells = shape.walkable
    images = {_d4(c, op, shape.span) for c in cells for op in range(8)}
    assert (images <= cells) is closed


# --- BETWEEN rooms: what varies -------------------------------------------


def test_position_only_holds_identity_constant():
    """"Same objects, different places" - and it was INEXPRESSIBLE before.

    `generate_layouts` drew a fresh colour permutation inside the loop that
    chose anchors, so colour always varied with position.
    """
    rooms = _rooms({Vary.POSITION})
    identities = {tuple((lm.shape, lm.color) for lm in r.landmarks) for r in rooms}
    assert len(identities) == 1, "identity must be constant when only POSITION varies"
    assert len({r.anchors for r in rooms}) == len(rooms), "positions must differ"


def test_kind_only_holds_position_constant():
    """"Same places, different shapes" - the other single-axis design."""
    rooms = _rooms({Vary.KIND})
    assert len({r.anchors for r in rooms}) == 1, "position must be constant"


def test_colour_varies_only_when_asked():
    with_colour = _rooms({Vary.POSITION, Vary.COLOR})
    assert len({tuple(lm.color for lm in r.landmarks) for r in with_colour}) > 1


def test_curated_rejects_congruent_rooms():
    """Selecting on distance alone returned three rooms all at signature
    (6,6,6): one configuration moved around, which tests nothing about room
    identity. That is the failure ROOMS_RUN1 shipped with."""
    rooms = _rooms({Vary.POSITION}, n=4)
    sigs = [separation_signature(r.anchors) for r in rooms]
    assert len(set(sigs)) == len(sigs)


def test_the_committed_l_room_set_is_congruent_and_that_is_recorded():
    """A regression pin on a KNOWN flaw, not an endorsement.

    layouts.py records it: all three rooms are exact translations of one
    configuration, so a network can cover them with one map plus a shift and
    remapping is never REQUIRED in the L-room arm. Pinned so the day someone
    re-derives the set, this test fails and the docs get updated with it.
    """
    sigs = {separation_signature(r.anchors) for r in ROOMS_RUN1}
    assert sigs == {(6, 6, 6)}, "the recorded degeneracy changed; update layouts.py"
    assert len({separation_signature(r.anchors) for r in ROOMS_SQUARE}) > 1


# --- WITHIN a room: what is legal -----------------------------------------


def test_max_coverage_is_inert_by_default():
    """A new constraint must not move the admissible set until asked.

    Measured: the committed designs sit at 11.0% (L) and 9.7% (square), so a
    default of 0.10 would have silently excluded this project's own L-room
    design from every generated pool.
    """
    assert RoomRules().max_coverage == 1.0
    covered = len(ROOMS_RUN1[0].cells) / len(L_SHAPE.walkable)
    assert covered > 0.10, "the L-room design exceeds a 10% budget"
    assert admissible_placements(L_SHAPE, CONTENT, RoomRules(max_coverage=0.09)) == []


def test_content_parameterises_the_search():
    """`SHAPES` was a module constant read at four places inside the enumerator,
    so a different landmark set meant editing a global every layout function
    also read."""
    two = EnvContent(kinds=(LandmarkKind("x"), LandmarkKind("plus")))
    assert len(admissible_placements(L_SHAPE, two, RoomRules())) != len(
        admissible_placements(L_SHAPE, CONTENT, RoomRules())
    )


def test_content_rejects_a_palette_too_small_to_colour_its_landmarks():
    with pytest.raises(ValueError, match="colours"):
        EnvContent(kinds=tuple(LandmarkKind(s) for s in "abcde"), palette=("red", "blue"))


# --- WHERE FROM: the sources ----------------------------------------------


def test_env_default_means_the_environment_supplies_its_own_landmarks():
    """None, not an empty list: there is no room SET, the env class owns the
    content. Nearly every checkpoint in this project trained this way."""
    assert resolve_rooms(shape=L_SHAPE, content=CONTENT, source=EnvDefault()) is None


def test_committed_is_returned_verbatim_and_not_validated():
    """It records what RAN; it does not propose a design. Validating it against
    today's RoomSetRules would fail the historical set - truly, and uselessly."""
    rooms = resolve_rooms(
        shape=L_SHAPE, content=CONTENT, source=Committed(rooms=ROOMS_RUN1),
        set_rules=RoomSetRules(distinct_signatures=True),
    )
    assert tuple(rooms) == ROOMS_RUN1


def test_uniform_refuses_to_return_a_short_pool():
    """A silently smaller pool would change the experiment without changing the
    config."""
    with pytest.raises(ValueError, match="admissible"):
        resolve_rooms(shape=L_SHAPE, content=CONTENT, source=Uniform(n=10**6, seed=1))


def test_indices_subset_any_source():
    """The single-room control is a SUBSET, not its own source - so "one room
    from a curated draw" is expressible, which it was not before."""
    full = _rooms({Vary.POSITION}, n=3)
    one = resolve_rooms(
        shape=L_SHAPE, content=CONTENT, source=Curated(n=3, seed=7),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})), indices=(2,),
    )
    assert len(one) == 1 and one[0].anchors == full[2].anchors

    with pytest.raises(ValueError, match="out of range"):
        resolve_rooms(shape=L_SHAPE, content=CONTENT, source=Curated(n=3, seed=7),
                      indices=(9,))


def test_a_seed_names_the_set():
    """A run must be reproducible from its seed alone."""
    a = _rooms({Vary.POSITION, Vary.COLOR}, seed=99)
    b = _rooms({Vary.POSITION, Vary.COLOR}, seed=99)
    c = _rooms({Vary.POSITION, Vary.COLOR}, seed=100)
    assert [r.anchors for r in a] == [r.anchors for r in b]
    assert [r.anchors for r in a] != [r.anchors for r in c]


def test_frozen_resolves_the_committed_set_for_ITS_shape():
    """`Committed` names literal rooms and is room-specific; handing the
    L-room's set to a square room would be silently wrong, and it cannot come
    from a command line at all. `Frozen` picks the right one from the shape."""
    from curious_george.envs.layouts import Frozen

    l_rooms = resolve_rooms(shape=L_SHAPE, content=CONTENT, source=Frozen())
    sq_rooms = resolve_rooms(shape=SQ_SHAPE, content=CONTENT, source=Frozen())
    assert tuple(l_rooms) == ROOMS_RUN1
    assert tuple(sq_rooms) == ROOMS_SQUARE
    assert l_rooms[2].anchors != sq_rooms[2].anchors


def test_committed_with_no_rooms_refuses_rather_than_returning_nothing():
    """It is defaulted only so the union can build a CLI parser - tyro refuses a
    tuple of struct types without a default. An empty set is not a set."""
    with pytest.raises(ValueError, match="Frozen"):
        resolve_rooms(shape=L_SHAPE, content=CONTENT, source=Committed())

"""Does the hidden state distinguish the rooms, or carry one map across them?

Three measurements, each with an explicit prediction under both hypotheses. They
run offline on the archived checkpoint series - they need activity from several
rooms at once, which the training loop does not have in one place.

    H_position : the network carries ONE position map. Room identity is not
                 represented; the same (x, y) looks the same everywhere. This is
                 the null the multi-room design exists to break, and it is what
                 the square-room quadrant decode found (~84% with the object
                 absent, delta ~0.000).
    H_room     : the network binds what it sees to where it is, so each room has
                 its own map.

    measurement                     H_position      H_room
    cross-room map correlation      ~ 1             ~ 0
    room-identity decode            ~ chance        ~ 1
    cross-room position transfer    ~ within-room   ~ chance

`validate()` builds synthetic activity satisfying each hypothesis exactly and
checks every measurement lands where the table says. A measurement that cannot
separate the two is not evidence about the pRNN, so this runs first:

    uv run python scripts/multienv/remapping.py --validate
"""

from __future__ import annotations

import argparse

import numpy as np
from jaxtyping import Float, Int

CHANCE_FLOOR = 1e-9


def cross_room_map_correlation(
    *, maps: Float[np.ndarray, "n_rooms H ny nx"]
) -> Float[np.ndarray, "H"]:
    """Mean pairwise correlation of a unit's rate map between rooms.

    The standard remapping measure, and directly comparable to this project's
    r ~ 0.98 map-stability number: a unit whose field sits at the same ROOM
    location regardless of the landmarks correlates at ~1 across rooms, while a
    unit that remaps correlates at ~0. Bins invalid in either room are dropped.
    """
    n_rooms, H = maps.shape[0], maps.shape[1]
    out = np.full(H, np.nan)
    for u in range(H):
        rs = []
        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                a, b = maps[i, u].ravel(), maps[j, u].ravel()
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 8:
                    continue
                av, bv = a[ok], b[ok]
                if av.std() < 1e-12 or bv.std() < 1e-12:
                    continue
                rs.append(float(np.corrcoef(av, bv)[0, 1]))
        if rs:
            out[u] = float(np.mean(rs))
    return out


def _group_folds(groups: Int[np.ndarray, "N"], n_folds: int = 5):
    """Split by TRAJECTORY, never by row.

    Rows within a trajectory are temporally correlated, so a row-level split
    leaks: the first quadrant decode in this project did exactly that.
    """
    uniq = np.unique(groups)
    order = np.argsort(uniq)
    for f in range(n_folds):
        held = set(uniq[order[f::n_folds]].tolist())
        test = np.array([g in held for g in groups])
        if test.any() and (~test).any():
            yield ~test, test


def room_decode(
    *,
    h: Float[np.ndarray, "N H"],
    room: Int[np.ndarray, "N"],
    groups: Int[np.ndarray, "N"],
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Can a linear readout say WHICH room the agent is in, from `h` alone?

    Chance is 1/n_rooms. At chance the multi-room manipulation is not landing at
    all: the network cannot be using room identity for anything, whatever else
    it does.
    """
    from sklearn.linear_model import LogisticRegression

    accs = []
    for train, test in _group_folds(groups, n_folds):
        clf = LogisticRegression(max_iter=2000, multi_class="auto", random_state=seed)
        clf.fit(h[train], room[train])
        accs.append(float((clf.predict(h[test]) == room[test]).mean()))
    return {
        "accuracy": float(np.mean(accs)),
        "sd": float(np.std(accs)),
        "chance": 1.0 / len(np.unique(room)),
        "n_folds": len(accs),
    }


def position_transfer(
    *,
    h: Float[np.ndarray, "N H"],
    pos: Float[np.ndarray, "N 2"],
    room: Int[np.ndarray, "N"],
    groups: Int[np.ndarray, "N"],
    seed: int = 0,
) -> dict:
    """Train a position decoder in one room, test it in the others.

    High transfer means one shared spatial map - the network is path-integrating
    and the room is irrelevant to how position is encoded. Reported against the
    WITHIN-room decoder as the ceiling, because transfer is only interpretable
    relative to how well position is decodable at all.
    """
    from sklearn.linear_model import Ridge

    rooms = np.unique(room)
    within, across = [], []
    for r in rooms:
        tr = room == r
        # hold out whole trajectories inside the training room for the ceiling
        folds = list(_group_folds(groups[tr], 5))
        if not folds:
            continue
        keep, held = folds[0]
        model = Ridge(alpha=1.0, random_state=seed).fit(h[tr][keep], pos[tr][keep])
        within.append(float(np.mean(np.linalg.norm(
            model.predict(h[tr][held]) - pos[tr][held], axis=1))))
        for other in rooms:
            if other == r:
                continue
            te = room == other
            across.append(float(np.mean(np.linalg.norm(
                model.predict(h[te]) - pos[te], axis=1))))
    return {
        "within_room_error": float(np.mean(within)),
        "across_room_error": float(np.mean(across)),
        "transfer_ratio": float(np.mean(within) / max(np.mean(across), CHANCE_FLOOR)),
    }


# --------------------------------------------------------------------------
# Synthetic validation: build activity that satisfies each hypothesis exactly.
# --------------------------------------------------------------------------

def synthetic(
    *, hypothesis: str, n_rooms: int = 3, n_units: int = 60, n_trajs: int = 12,
    steps: int = 120, grid: int = 14, noise: float = 0.05, seed: int = 0,
):
    """Activity obeying `H_position` or `H_room`, plus its rate maps.

    H_position: every unit is a place field at a fixed ROOM location, identical
    in every room. H_room: the same units, but each room gets its own
    independent set of field centres - the definition of global remapping.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(1, grid, size=(n_units, 2))
    per_room = (
        [centres] * n_rooms
        if hypothesis == "H_position"
        else [rng.uniform(1, grid, size=(n_units, 2)) for _ in range(n_rooms)]
    )
    if hypothesis not in ("H_position", "H_room"):
        raise ValueError(hypothesis)

    h_rows, pos_rows, room_rows, group_rows = [], [], [], []
    gid = 0
    for r in range(n_rooms):
        for _ in range(n_trajs):
            p = rng.uniform(1, grid, size=(steps, 2))
            d = np.linalg.norm(p[:, None, :] - per_room[r][None, :, :], axis=2)
            act = np.exp(-(d ** 2) / (2 * 2.0 ** 2)) + noise * rng.normal(size=d.shape)
            h_rows.append(act); pos_rows.append(p)
            room_rows.append(np.full(steps, r)); group_rows.append(np.full(steps, gid))
            gid += 1

    h = np.concatenate(h_rows); pos = np.concatenate(pos_rows)
    room = np.concatenate(room_rows); groups = np.concatenate(group_rows)

    maps = np.full((n_rooms, n_units, grid, grid), np.nan)
    for r in range(n_rooms):
        sel = room == r
        ix = np.clip(pos[sel, 0].astype(int), 0, grid - 1)
        iy = np.clip(pos[sel, 1].astype(int), 0, grid - 1)
        flat = iy * grid + ix
        for b in np.unique(flat):
            maps[r, :, b // grid, b % grid] = h[sel][flat == b].mean(axis=0)
    return h, pos, room, groups, maps


def validate(n_units: int = 500) -> int:
    """Every measurement must land where the hypothesis table says. Returns exit code.

    `n_units` defaults to `predNet.hiddensize`, deliberately: the room decode is
    a LINEAR readout, so its accuracy under H_room depends on how many units it
    has to work with. Measured on this synthetic data - 0.701 at 60 units, 0.823
    at 200, 0.865 at 500 - while H_position stays at chance (0.315 / 0.328 /
    0.333) throughout. Validating at a width the experiment never uses would
    have set the bar against a handicapped version of the measurement.
    """
    rows = []
    for hyp in ("H_position", "H_room"):
        h, pos, room, groups, maps = synthetic(hypothesis=hyp, n_units=n_units)
        corr = float(np.nanmedian(cross_room_map_correlation(maps=maps)))
        dec = room_decode(h=h, room=room, groups=groups)
        tr = position_transfer(h=h, pos=pos, room=room, groups=groups)
        rows.append((hyp, corr, dec, tr))

    print(f"{'':<12} {'map corr':>10} {'room decode':>13} {'chance':>8} "
          f"{'within err':>11} {'across err':>11}")
    for hyp, corr, dec, tr in rows:
        print(f"{hyp:<12} {corr:>10.3f} {dec['accuracy']:>13.3f} {dec['chance']:>8.3f} "
              f"{tr['within_room_error']:>11.3f} {tr['across_room_error']:>11.3f}")

    (_, c_pos, d_pos, t_pos), (_, c_room, d_room, t_room) = rows
    checks = [
        ("map correlation separates the hypotheses", c_pos > 0.8 and c_room < 0.3),
        ("room decode is at chance under H_position", abs(d_pos["accuracy"] - d_pos["chance"]) < 0.15),
        ("room decode is above chance under H_room", d_room["accuracy"] > 0.8),
        ("position transfers under H_position",
         t_pos["across_room_error"] < 1.5 * t_pos["within_room_error"]),
        ("position does NOT transfer under H_room",
         t_room["across_room_error"] > 2.0 * t_room["within_room_error"]),
    ]
    print()
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\n{'VALIDATED' if ok else 'NOT VALIDATED'}: these measurements "
          f"{'separate' if ok else 'DO NOT separate'} the two hypotheses.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--units", type=int, default=500,
                    help="synthetic width; default matches predNet.hiddensize")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(validate(n_units=a.units))
    ap.error("nothing to do; pass --validate (checkpoint scoring lives in checkpoint_curve.py)")

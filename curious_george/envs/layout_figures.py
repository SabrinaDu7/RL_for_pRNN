"""Look at a layout set, because it is an experimental design.

A layout is three landmarks at anchor coordinates. Read as numbers, two sets can
look interchangeable and be nothing alike - one clustered in a corner, one spread
across the room - and that difference is the manipulation. So the set gets
drawn, not inferred.

Promoted from `throwaway/ported/layout_figures.py`, which also generated,
verified and timed pools; only the drawing belongs in the library. The rest was
a one-off design study and stays in throwaway.

    from curious_george.envs.layout_figures import plot_layouts
    from curious_george.envs.layouts import ROOMS_RUN1
    plot_layouts(ROOMS_RUN1, title="the frozen three")

`matplotlib` is imported inside the function: `envs/layouts.py` is on the
training hot path and must not pull a plotting stack in to resolve a room.
"""

from __future__ import annotations

from pathlib import Path

from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    MULTI_ROOM_ID,
    Layout,
    base_walkable,
)

#: Chebyshev radius the "testable offsets" count is quoted at, matching
#: `Layout.n_testable_offsets`'s own default.
OFFSET_RADIUS = 4



def render_layout(layout: Layout, *, room: str = BASE_ROOM_ID):
    """Top-down RGB frame of `layout`'s room.

    `room` names the BASE room, which owns the wall geometry; the multi-room id
    that accepts landmarks is derived from it. Passing the wrong one is how a
    square-room design once came out drawn as an L-room carrying square-room
    coordinates - every panel plausible, every panel wrong.
    """
    import gymnasium as gym

    env = gym.make(MULTI_ROOM_ID[room], landmarks=list(layout.landmarks))
    env.reset(seed=0)
    return env.unwrapped.get_frame(highlight=False, tile_size=16)


def plot_layouts(
    layouts: "list[Layout] | tuple[Layout, ...]",
    *,
    room: str = BASE_ROOM_ID,
    title: str = "",
    ncols: int = 5,
    path: str | Path | None = None,
):
    """Draw each layout with the constraints it satisfies, and return the figure.

    Returns rather than only writing, so a caller can log it to wandb or show it
    inline; `path` saves as well when given.

    The caption states the constraint MARGINS, not just the anchors: a layout
    that merely satisfies every bound is a different design from one that
    clears them comfortably, and that is invisible in coordinates alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walkable = base_walkable(room)
    layouts = list(layouts)
    nrows = -(-len(layouts) // ncols)
    # 4.9 in per row, not 3.5: the caption is five lines, and at 3.5 the next
    # row's captions were drawn over the previous row's images.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 4.9 * nrows), squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.08},
    )
    for ax in axes.ravel():
        ax.axis("off")

    for ax, layout in zip(axes.ravel(), layouts):
        ax.imshow(render_layout(layout, room=room))
        caption = "\n".join(
            [f"layout {layout.key}"]
            + [f"{lm.shape} · {lm.color} @ {lm.anchor}" for lm in layout.landmarks]
            + [
                f"anchor separation ≥ {layout.min_anchor_separation} cells",
                f"landmark gap ≥ {layout.min_cell_gap()} · "
                f"wall clearance ≥ {layout.min_wall_distance(walkable=walkable)}",
                f"{layout.n_testable_offsets(walkable=walkable)} testable offsets "
                f"(Chebyshev ≤ {OFFSET_RADIUS})",
            ]
        )
        # caption BELOW the image, so nothing collides with the row above
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, linespacing=1.35)

    if title:
        fig.suptitle(title, fontsize=13)
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_config_layouts(cfg, *, path: str | Path | None = None):
    """Draw the layout set a `Config` actually resolves to.

    The point of routing through `resolve_layouts` rather than taking a list: it
    answers "what will THIS run train on", which for a `LayoutPool` is a
    function of its size and seed and cannot be read off the config by eye.
    """
    from curious_george.envs.layouts import resolve_layouts

    layouts = resolve_layouts(cfg)
    if not layouts:
        raise ValueError(
            f"{type(cfg.env.source).__name__} specifies no room set - the "
            "environment class supplies its own landmarks"
        )
    return plot_layouts(
        layouts,
        room=cfg.env.shape.room,
        title=f"{cfg.env.shape.room} · {type(cfg.env.source).__name__} · "
              f"{len(layouts)} rooms",
        path=path,
    )

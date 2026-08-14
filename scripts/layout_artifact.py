"""Render the layout design as a self-contained HTML page.

Same numbers and same generator as `scripts/layout_figures.py` - this is the
viewing surface, not a second source of truth. Every room here is rendered from
a live env built by `curious_george.envs.layouts`, so the page cannot drift from
what training will actually construct.

    uv run python scripts/layout_artifact.py --pool 500 --gallery 24

Writes outputs/layouts/layouts.html.
"""

from __future__ import annotations

import argparse
import base64
import io

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.envs.Lroom import STENCILS, Landmark

from curious_george.envs.layouts import (
    LANDMARK_COLORS,
    OFFSET_RADIUS,
    SHAPES,
    Layout,
    enumerate_anchor_triples,
    generate_layouts,
    walkable_cells,
)
from scripts.layout_figures import (
    BASE_ENV_ID,
    ENV_ID,
    OUT,
    base_room,
    build,
    cross_layout_distance,
    pick_rooms,
)


def png_data_uri(rgb: np.ndarray, *, scale: int = 1) -> str:
    img = Image.fromarray(rgb.astype(np.uint8))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def room_uri(layout: Layout, *, tile: int = 14, env_id: str = ENV_ID) -> str:
    env = build(layout, env_id)
    return png_data_uri(env.unwrapped.get_frame(highlight=False, tile_size=tile))


def agent_view_uri(shape: str, env_id: str = ENV_ID) -> str:
    """The 7x7 observation, upscaled - the only view the pRNN receives."""
    centre = (7, 7)
    env = gym.make(env_id, landmarks=[Landmark(shape, "red", centre)])
    env.reset(seed=0)
    u = env.unwrapped
    u.agent_pos, u.agent_dir = (centre[0], centre[1] + 3), 3
    return png_data_uri(
        u.get_frame(highlight=False, tile_size=1, agent_pov=True), scale=28
    )


def card(layout: Layout, *, walkable, tile: int = 14, env_id: str = ENV_ID) -> str:
    rows = "".join(
        f"<tr><td>{lm.shape}</td><td><span class='sw' style='background:{_css(lm.color)}'></span>"
        f"{lm.color}</td><td class='num'>{lm.anchor[0]}, {lm.anchor[1]}</td></tr>"
        for lm in layout.landmarks
    )
    return f"""<figure class="room">
  <img src="{room_uri(layout, tile=tile, env_id=env_id)}" alt="L-room with three landmarks">
  <figcaption>
    <span class="key">{layout.key}</span>
    <table class="lm"><tbody>{rows}</tbody></table>
    <dl class="stats">
      <dt>anchor separation</dt><dd class="num">{layout.min_anchor_separation}</dd>
      <dt>landmark gap</dt><dd class="num">{layout.min_cell_gap()}</dd>
      <dt>wall clearance</dt><dd class="num">{layout.min_wall_distance(walkable=walkable)}</dd>
      <dt>testable offsets</dt><dd class="num">{layout.n_testable_offsets(walkable=walkable)}</dd>
    </dl>
  </figcaption>
</figure>"""


def _css(color: str) -> str:
    from minigrid.core.constants import COLORS

    r, g, b = (76 + 0.349 * COLORS[color]).astype(int)   # as Floor renders it
    return f"rgb({r},{g},{b})"


HEAD_TMPL = """<title>{title}</title>
<style>
:root {
  --ground: #F2F3F1; --panel: #FFFFFF; --ink: #16191A; --muted: #5C6567;
  --rule: #D8DBD7; --accent: #0F5257; --grid: rgba(22,25,26,.045);
  --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #101314; --panel: #171B1C; --ink: #E8EBE9; --muted: #98A2A3;
    --rule: #262D2E; --accent: #5FB0B7; --grid: rgba(232,235,233,.05);
  }
}
:root[data-theme="dark"] {
  --ground: #101314; --panel: #171B1C; --ink: #E8EBE9; --muted: #98A2A3;
  --rule: #262D2E; --accent: #5FB0B7; --grid: rgba(232,235,233,.05);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.6;
  background-image:
    repeating-linear-gradient(to right, var(--grid) 0 1px, transparent 1px 28px),
    repeating-linear-gradient(to bottom, var(--grid) 0 1px, transparent 1px 28px);
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }
.measure { max-width: 66ch; }
h1 {
  font-family: var(--serif); font-weight: 400; font-size: clamp(2rem, 4.5vw, 3rem);
  line-height: 1.12; margin: 0 0 .75rem; text-wrap: balance; letter-spacing: -.01em;
}
h2 {
  font-family: var(--serif); font-weight: 400; font-size: 1.6rem; line-height: 1.2;
  margin: 0 0 .35rem; text-wrap: balance;
}
.eyebrow {
  font-family: var(--mono); font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 .6rem;
}
p { margin: 0 0 1rem; }
.lede { font-size: 1.1rem; color: var(--muted); }
section { margin-top: 4rem; padding-top: 2rem; border-top: 1px solid var(--rule); }
section > .measure { margin-bottom: 2rem; }
.grid { display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); }
.grid.wide { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
figure.room {
  margin: 0; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 3px; overflow: hidden; display: flex; flex-direction: column;
}
figure.room img { display: block; width: 100%; height: auto; image-rendering: pixelated; }
figcaption { padding: .8rem .9rem 1rem; display: flex; flex-direction: column; gap: .6rem; }
.key { font-family: var(--mono); font-size: .7rem; color: var(--muted); letter-spacing: .06em; }
table.lm { width: 100%; border-collapse: collapse; font-size: .78rem; }
table.lm td { padding: .12rem 0; vertical-align: middle; }
table.lm td:last-child { text-align: right; color: var(--muted); }
.sw { display: inline-block; width: .68em; height: .68em; margin-right: .4em;
      border-radius: 2px; vertical-align: -.04em; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
dl.stats { display: grid; grid-template-columns: 1fr auto; gap: .1rem .8rem;
           margin: 0; font-size: .74rem; color: var(--muted);
           border-top: 1px solid var(--rule); padding-top: .55rem; }
dl.stats dt { margin: 0; } dl.stats dd { margin: 0; text-align: right; }
figure.shape { margin: 0; background: var(--panel); border: 1px solid var(--rule);
               border-radius: 3px; padding: 1rem; text-align: center; }
figure.shape img { width: 100%; max-width: 190px; height: auto; image-rendering: pixelated;
                   border-radius: 2px; }
figure.shape h3 { font-family: var(--mono); font-size: .82rem; margin: 0 0 .1rem;
                  letter-spacing: .04em; }
figure.shape p { font-size: .76rem; color: var(--muted); margin: .45rem 0 0; }
.tablewrap { overflow-x: auto; }
table.facts { border-collapse: collapse; width: 100%; font-size: .88rem; min-width: 32rem; }
table.facts th, table.facts td { text-align: left; padding: .55rem .8rem .55rem 0;
                                 border-bottom: 1px solid var(--rule); vertical-align: top; }
table.facts th { font-family: var(--mono); font-size: .7rem; text-transform: uppercase;
                 letter-spacing: .1em; color: var(--muted); font-weight: 400; }
table.facts td:last-child, table.facts th:last-child { padding-right: 0; }
code { font-family: var(--mono); font-size: .86em; background: var(--panel);
       border: 1px solid var(--rule); border-radius: 2px; padding: .06em .32em; }
.note { border-left: 2px solid var(--accent); padding-left: 1rem; color: var(--muted);
        font-size: .93rem; }
footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
         font-size: .8rem; color: var(--muted); font-family: var(--mono); }
</style>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--gallery", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--min-cell-gap", type=int, default=2)
    ap.add_argument("--min-anchor-separation", type=int, default=6)
    ap.add_argument("--min-wall-distance", type=int, default=2)
    ap.add_argument("--min-testable-offsets", type=int, default=40)
    ap.add_argument("--room", default="lroom", choices=("lroom", "square"))
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from curious_george.envs.layouts import (
        BASE_ROOM_ID, MULTI_ENV_ID, SQUARE_ROOM_ID, base_walkable)
    square = a.room == "square"
    base_id = SQUARE_ROOM_ID if square else BASE_ROOM_ID
    env_id = MULTI_ENV_ID[base_id]
    walkable = base_walkable(base_id)
    kw = dict(min_cell_gap=a.min_cell_gap, min_anchor_separation=a.min_anchor_separation,
              min_wall_distance=a.min_wall_distance,
              min_testable_offsets=a.min_testable_offsets,
              dedupe_d4=square, span=14)
    pool = generate_layouts(walkable=walkable, n=a.pool, seed=a.seed, **kw)
    triples = enumerate_anchor_triples(walkable=walkable, **kw)
    n_colour = len(LANDMARK_COLORS) * (len(LANDMARK_COLORS) - 1) * (len(LANDMARK_COLORS) - 2)
    rooms = pick_rooms(
        layouts=[Layout(tuple(Landmark(s, c, an) for s, c, an in zip(SHAPES, LANDMARK_COLORS, t)))
                 for t in triples],
        k=3, walkable=walkable, min_config_distance=a.min_anchor_separation,
        distinct_signatures=square)
    from scripts.layout_figures import assign_distinct_colors
    rooms = assign_distinct_colors(rooms=rooms, seed=a.seed)
    xroom = min(cross_layout_distance(x, y)
                for i, x in enumerate(rooms) for y in rooms[i + 1:])

    shapes_html = "".join(
        f"""<figure class="shape">
  <h3>{s}</h3><p style="margin:0 0 .6rem">{len(STENCILS[s])} cells</p>
  <img src="{agent_view_uri(s, env_id)}" alt="7x7 agent view of the {s} landmark">
  <p>as the pRNN receives it — 7&times;7, one pixel per cell</p>
</figure>"""
        for s in SHAPES
    )
    rooms_html = "".join(card(r, walkable=walkable, tile=18, env_id=env_id) for r in rooms)
    pool_html = "".join(card(p, walkable=walkable, env_id=env_id) for p in pool[: a.gallery])

    HEAD = HEAD_TMPL.replace("{title}",
        "Square rooms for multi-room training" if square else "Multi-room landmark layouts")
    html = f"""{HEAD}
<div class="wrap">
<header class="measure">
  <p class="eyebrow">Curious George · environment design</p>
  <h1>{"Square" if square else "L-shaped"} rooms for multi-environment training</h1>
  <p class="lede">Two training runs need rooms whose landmarks move: three rooms
  held simultaneously, and a pool of {a.pool} sampled at random. Every room below is
  rendered from a live environment built by the same generator training will use.</p>
  <p>The point of moving the landmarks is to break dead reckoning. In a single fixed
  room the network can locate itself by integrating its own trajectory and never needs
  to look at anything — which is what the quadrant-decoding control showed it doing.
  When the same integrated path lands in a different absolute position depending on
  which room a stream is in, position has to come from what is visible.</p>
</header>

<section>
  <p class="eyebrow">Landmark shapes</p>
  <h2 class="measure">Three shapes, at the scale that matters</h2>
  <div class="measure">
    <p>Each landmark is a patch of coloured floor. Shape, colour and position vary
    independently, so a unit responding at the same offset from all three is coding a
    metric relation rather than recognising one object.</p>
    <p class="note">The third shape is a solid 3&times;3 <code>block3</code>, chosen over a
    4-cell diamond for exactly the reason the bottom row makes visible: a diamond is
    <code>plus</code> without its centre pixel, so in the only view the network receives the
    two would have differed by one pixel out of forty-nine. Solid, sparse and cross are
    mutually distinct at every pixel. All three are anchored at their centre cell, so one
    reference point serves every landmark when offsets are measured.</p>
  </div>
  <div class="grid wide">{shapes_html}</div>
</section>

<section>
  <p class="eyebrow">Run 1 · alternating rooms</p>
  <h2 class="measure">The three rooms, trained on simultaneously</h2>
  <div class="measure">
    <p>Chosen from all {len(triples):,} admissible anchor assignments by an exact search:
    first requiring the rooms' landmark <em>configurations</em> to differ, then maximising
    the distance between their landmarks. Configuration comes first because maximising
    distance alone returns rooms that are translates of one another — and a place field
    that shifts with the room is exactly what a vector cell would look like.</p>
    <p>These three reach configuration distance 14 against a floor of
    {a.min_anchor_separation}, and their landmarks are {xroom} cells apart at the closest.
    That {xroom} is the ceiling over the whole admissible set, not a search artifact: the
    room is only 172 walkable cells.</p>
  </div>
  <div class="grid wide">{rooms_html}</div>
</section>

<section>
  <p class="eyebrow">Run 2 · random positions</p>
  <h2 class="measure">{a.gallery} of the {a.pool}-layout pool</h2>
  <div class="measure">
    <p>Drawn uniformly without replacement from the admissible set, seeded on
    <span class="num">{a.seed}</span>. A pool rather than fresh rooms per step because
    observations come from a bank keyed on the grid, so each new room costs one bank
    build; a pool pays that once and stays a fixed artefact the analysis can be pointed
    at afterwards.</p>
  </div>
  <div class="grid">{pool_html}</div>
</section>

<section>
  <p class="eyebrow">Constraints</p>
  <h2 class="measure">What every room satisfies, and why</h2>
  <div class="tablewrap">
  <table class="facts">
    <thead><tr><th>Constraint</th><th>Value</th><th>Reason</th></tr></thead>
    <tbody>
      <tr><td>all three shapes, three distinct colours</td><td class="num">—</td>
          <td>a unit firing at the same offset from landmarks sharing neither shape nor
              colour is coding a relation, not an identity</td></tr>
      <tr><td>gap between landmarks</td><td class="num">&ge; {a.min_cell_gap}</td>
          <td>touching landmarks merge into one blob in a 7&times;7 view and their anchors
              stop being separable</td></tr>
      <tr><td>anchor separation</td><td class="num">&ge; {a.min_anchor_separation}</td>
          <td>offset maps are read in a window of radius {OFFSET_RADIUS} around each anchor;
              closer anchors give windows that overlap and correlate for purely geometric
              reasons</td></tr>
      <tr><td>wall clearance</td><td class="num">&ge; {a.min_wall_distance}</td>
          <td>a landmark against a wall is part of the boundary, and vector tuning to it
              cannot be told from boundary-vector tuning</td></tr>
      <tr><td>testable offsets</td><td class="num">&ge; {a.min_testable_offsets}</td>
          <td>offsets walkable for every anchor — the only ones over which a vector code
              can be tested at all</td></tr>
    </tbody>
  </table>
  </div>
  <div class="measure" style="margin-top:2rem">
    <p>Under these, the room admits <strong>{len(triples):,}</strong> anchor assignments,
    or <strong>{len(triples) * n_colour:,}</strong> layouts once colour is assigned. Pool
    size is not a binding constraint.</p>
    <p class="note">Colours are quoted as <em>rendered</em>, not nominal.
    <code>Floor</code> paints <code>76 + 0.35 &times; colour</code>, a blend toward grey,
    so empty floor is (76,&nbsp;76,&nbsp;76) and every nominal separation reaches the
    network at about a third of face value. On that basis the closest pair in this palette
    is 89 apart and the closest colour to empty floor is 89. <code>grey</code> renders 61
    from empty floor and was visibly gone in the first rooms rendered; <code>purple</code>
    is 46 from <code>blue</code>. Both are excluded.</p>
  </div>
</section>

<footer>
  generated by scripts/layout_artifact.py · rooms from {env_id} · geometry from {base_id}
  · regenerate with <code>uv run python scripts/layout_artifact.py</code>
</footer>
</div>"""

    path = OUT / ("layouts_square.html" if square else "layouts.html")
    path.write_text(html)
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

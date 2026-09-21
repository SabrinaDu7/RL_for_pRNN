"""Draw the eight one-object rooms the `Placed` source would train on. Throwaway."""
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import curious_george  # noqa: F401  (registers the MiniGrid ids)
from curious_george.envs.layouts import MULTI_ROOM_ID, BASE_ROOM_ID, EnvContent, EnvShape, Placed, resolve_rooms

ANCHORS = ("3,3", "12,3", "12,6", "9,7", "8,12", "5,10", "7,4", "3,7")  # the user's final set, 2026-09-21
KIND, CLEARANCE = 1, 2  # green plus; one floor cell between it and any wall (RoomRules.min_wall_distance)
NOTES = ("top-left corner", "top-right corner", "right wall, above the cut-out", "inner (concave) corner",
         "bottom-right of the foot", "foot, off its top-left", "top wall, middle, one row in", "left wall, middle")
from curious_george.envs.layouts import _wall_distance
shape, content = EnvShape(BASE_ROOM_ID), EnvContent()
rooms = resolve_rooms(shape=shape, content=content, source=Placed(anchors=ANCHORS, kind=KIND))
floor = shape.walkable
print("floor extent:", min(x for x, _ in floor), "..", max(x for x, _ in floor), "x", min(y for _, y in floor), "..", max(y for _, y in floor),
      "| floor cells:", len(floor), "| standable per room:", [len(floor) - len(r.cells) for r in rooms])

fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
for ax, room, text, note in zip(axes.ravel(), rooms, ANCHORS, NOTES):
    env = gym.make(MULTI_ROOM_ID[BASE_ROOM_ID], landmarks=list(room.landmarks), agent_start_pos=None,
                   agent_start_dir=None, render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    img = u.grid.render(24, agent_pos=None, agent_dir=None)  # the room alone, no agent
    ax.imshow(img)
    (lm,) = room.landmarks
    assert all(c in floor for c in lm.cells) and all(getattr(u.grid.get(*c), "type", None) == "obstacle" for c in lm.cells)
    gap = min(_wall_distance(cell=c, walkable=floor) for c in lm.cells)
    assert gap >= CLEARANCE, (text, gap)
    ax.set_title(f"anchor ({text})  {note}\n{lm.shape} {lm.color}, impassable, wall clearance {gap}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("One-object rooms for the Placed source: 8 rooms, one green plus each, one floor cell off every wall (MiniGrid x right, y down)", fontsize=11)
fig.tight_layout(h_pad=2.0)
out = "throwaway/2026-09-21/one_object_rooms.png"
fig.savefig(out, dpi=130); print("wrote", out)

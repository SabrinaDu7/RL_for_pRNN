"""Q3 setup check: drive the agent yourself through impassable-object rooms.

Throwaway verification, not a result. Builds several variants of the L-room with
IMPASSABLE landmarks and emits a self-contained interactive page where you pick
each action - turn left, turn right, forward, stay put - and watch:

  left    the whole room top-down, the agent, and its trail
  right   what the agent sees, READ OUT OF THE OBSERVATION BANK

The bank is what training consumes on the GPU path, so checking a live render
would check the wrong object. The page does not replay a precomputed walk: the
whole bank and the whole transition table are embedded, and the page looks up
the next state and the next view exactly the way `DeviceTableShellPool` does.
That is the point - if the page moves the agent correctly, the tables training
uses are correct.

Because nothing is precomputed, the bank is verified at EVERY reachable pose
rather than at the handful some particular walk visited.

    PYTHONPATH=../minigrid uv run python throwaway/q3_setup_check.py
"""

from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from curious_george.envs.layouts import (  # noqa: E402
    BASE_ROOM_ID,
    MULTI_ROOM_ID,
    EnvContent,
    EnvShape,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
    base_walkable,
    resolve_rooms,
)
from curious_george.envs.obs_bank import TableDrivenRGBPartialObsWrapper  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs" / "q3_setup_check.html"

N_ROOMS = 6
SEED = 7
ROOM_TILE = 18  # top-down render scale; the network never sees this
SHAPES = ("x", "plus", "block3")


def build_rooms(*, impassable: bool):
    content = EnvContent(
        kinds=tuple(LandmarkKind(s, impassable=impassable) for s in SHAPES)
    )
    return resolve_rooms(
        shape=EnvShape(BASE_ROOM_ID),
        content=content,
        source=Uniform(n=N_ROOMS, seed=SEED),
        set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
    )


def verify_bank_everywhere(wrapper, walkable) -> int:
    """Compare the banked view against a live render at EVERY reachable pose.

    Stronger than checking the poses one rollout happened to hit: the page
    below can put the agent anywhere, so anywhere has to be right.
    """
    u = wrapper.env.unwrapped
    saved = (u.agent_pos, u.agent_dir)
    bad = 0
    for x, y in sorted(walkable):
        for d in range(4):
            u.agent_pos, u.agent_dir = (x, y), d
            live = u.get_frame(highlight=False, tile_size=1, agent_pov=True)
            if not np.array_equal(np.asarray(wrapper._bank[x, y, d]), live):
                bad += 1
    u.agent_pos, u.agent_dir = saved
    return bad


def room_png(env, tile: int) -> str:
    """The room with NO agent drawn - the marker is drawn in the page."""
    img = env.unwrapped.grid.render(tile, agent_pos=(-1, -1), agent_dir=None)
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def collect() -> dict:
    rooms = build_rooms(impassable=True)
    base = base_walkable(BASE_ROOM_ID)
    env = gym.make(MULTI_ROOM_ID[BASE_ROOM_ID], landmarks=list(rooms[0].landmarks))
    env.reset(seed=0)
    wrapper = TableDrivenRGBPartialObsWrapper(env, tile_size=1)
    u = env.unwrapped

    out, total_bad = [], 0
    for i, layout in enumerate(rooms):
        wrapper.unwrapped.landmarks = list(layout.landmarks)
        env.reset(seed=SEED + i)
        wrapper._ensure_bank()

        walk = layout.walkable(base)
        bad = verify_bank_everywhere(wrapper, walk)
        total_bad += bad

        # The transition table's own account of where movement is refused - not
        # a separate reimplementation - split by what did the blocking. The
        # object count is IDENTICAL across rooms by construction: min_cell_gap
        # and min_wall_distance keep the three shapes isolated, so their
        # perimeter does not depend on where they sit. Walls are constant too.
        # A change to either rule would move the object number.
        from minigrid.core.constants import DIR_TO_VEC

        nxt = np.asarray(wrapper._next_state)
        refused = by_object = 0
        for (x, y) in walk:
            for dd in range(4):
                if tuple(nxt[x, y, dd, 2][:2]) == (x, y):
                    refused += 1
                    ahead = (x + int(DIR_TO_VEC[dd][0]), y + int(DIR_TO_VEC[dd][1]))
                    by_object += ahead in layout.cells

        out.append(
            {
                "key": layout.key,
                "describe": layout.describe(),
                "png": room_png(env, ROOM_TILE),
                "bank": b64(wrapper._bank),
                "next": b64(nxt),
                "blockedCells": sorted(map(list, layout.cells)),
                "walkable": sorted(map(list, walk)),
                "start": [int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)],
                "nWalkable": len(walk),
                "nBase": len(base),
                "refusedPoses": refused,
                "refusedByObject": by_object,
                "refusedByWall": refused - by_object,
                "bankMismatches": bad,
            }
        )
        print(
            f"  room {layout.key}  walkable {len(base)}->{len(walk)}  "
            f"forward refused at {refused} of {4*len(walk)} poses "
            f"(walls {refused - by_object}, objects {by_object})  "
            f"bank mismatches {bad}"
        )

    return {
        "rooms": out,
        "width": u.width,
        "height": u.height,
        "tile": ROOM_TILE,
        "view": wrapper._bank.shape[3],
        "totalBad": total_bad,
    }


PAGE = """<title>Q3 setup check</title>
<style>
 :root{--bg:#101215;--fg:#e8e6e3;--dim:#8b9096;--line:#272b30;--panel:#15181c;
       --ok:#5fb37a;--bad:#d9605a;--accent:#6ea8d8;--agent:#f2c14e}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
 .wrap{max-width:1180px;margin:0 auto;padding:26px 20px 70px}
 h1{font-size:19px;margin:0 0 4px;font-weight:600}
 .sub{color:var(--dim);margin:0 0 20px;max-width:70ch}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
 .panes{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
 .pane{border:1px solid var(--line);border-radius:9px;padding:14px;background:var(--panel)}
 .lbl{color:var(--dim);font-size:11px;text-transform:uppercase;
      letter-spacing:.09em;margin-bottom:9px}
 canvas{display:block;image-rendering:pixelated;border-radius:4px}
 button,select{background:#1e2228;color:var(--fg);border:1px solid var(--line);
   border-radius:7px;padding:8px 13px;font:inherit;cursor:pointer}
 button:hover{border-color:var(--accent)}
 button:active{background:#252a31}
 .keys{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:12px}
 .keys button{padding:13px 10px;font-size:13px}
 .keys .k{display:block;color:var(--dim);font-size:11px;margin-top:3px}
 table{border-collapse:collapse;font-size:13px;min-width:290px}
 td{padding:4px 15px 4px 0;color:var(--dim);vertical-align:top}
 td+td{color:var(--fg)}
 .ok{color:var(--ok)} .bad{color:var(--bad)}
 .hint{color:var(--dim);font-size:12px;margin-top:14px}
 label.tog{display:inline-flex;gap:7px;align-items:center;color:var(--dim);cursor:pointer}
</style>
<div class=wrap>
<h1>Q3 setup check &mdash; drive the agent through impassable objects</h1>
<p class=sub>You pick every action. Nothing here is a replay: the whole observation
bank and the whole transition table are embedded, and this page looks up the next
state and the next view <b>exactly the way the GPU training path does</b>. If the
agent moves correctly here, those tables are correct.</p>

<div class=bar>
  <select id=room></select>
  <button id=reset>reset room</button>
  <button id=rand>random step</button>
  <label class=tog><input type=checkbox id=show> show impassable cells</label>
</div>

<div class=panes>
  <div class=pane><div class=lbl>the whole room</div><canvas id=top></canvas></div>
  <div class=pane>
    <div class=lbl>what the agent sees &mdash; from the bank</div>
    <canvas id=view width=203 height=203></canvas>
    <div class=keys>
      <button data-a=0>turn left<span class=k>&#8592;</span></button>
      <button data-a=1>turn right<span class=k>&#8594;</span></button>
      <button data-a=2>forward<span class=k>&#8593;</span></button>
      <button data-a=3>stay put<span class=k>space</span></button>
    </div>
  </div>
  <div class=pane><div class=lbl>readout</div><table id=info></table>
    <div class=hint>Arrow keys and space work too.<br>Walk into an object &mdash;
    the step should be refused and the agent should not move.</div>
  </div>
</div>
</div>
<script>
const D = __DATA__;
const ARROW=['\\u2192','\\u2193','\\u2190','\\u2191'];
const NAMES=['turn left','turn right','forward','stay put'];
const W=D.width, H=D.height, V=D.view, VS=V*V*3;

function bytes(s){const b=atob(s),a=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
D.rooms.forEach(r=>{r.bankA=bytes(r.bank);r.nextA=bytes(r.next);
  r.blocked=new Set(r.blockedCells.map(c=>c[0]+','+c[1]));});

const top_=document.getElementById('top'),tctx=top_.getContext('2d');
const view=document.getElementById('view'),vctx=view.getContext('2d');
const sel=document.getElementById('room'),info=document.getElementById('info');
const show=document.getElementById('show');
let r=0,x=0,y=0,d=0,trail=[],last=null,refused=false,bg=null,steps=0,blocks=0;

D.rooms.forEach((rm,i)=>sel.add(new Option(`room ${i+1}/${D.rooms.length} \\u00b7 ${rm.key}`,i)));
top_.width=W*D.tile; top_.height=H*D.tile;

// Index exactly as the training tables are indexed.
const viewAt=(rm,x,y,d)=>((x*H+y)*4+d)*VS;
const nextAt=(rm,x,y,d,a)=>(((x*H+y)*4+d)*4+a)*3;

function load(i){
  r=i; const rm=D.rooms[i];
  [x,y,d]=rm.start; trail=[[x,y]]; last=null; refused=false; steps=0; blocks=0;
  bg=new Image(); bg.onload=draw; bg.src='data:image/png;base64,'+rm.png;
}
function act(a){
  const rm=D.rooms[r], o=nextAt(rm,x,y,d,a);
  const nx=rm.nextA[o], ny=rm.nextA[o+1], nd=rm.nextA[o+2];
  refused = (a===2 && nx===x && ny===y);
  if(refused) blocks++;
  x=nx; y=ny; d=nd; last=a; steps++;
  if(trail[trail.length-1][0]!==x||trail[trail.length-1][1]!==y) trail.push([x,y]);
  draw();
}
function draw(){
  const rm=D.rooms[r], T=D.tile;
  tctx.clearRect(0,0,top_.width,top_.height);
  if(bg&&bg.complete) tctx.drawImage(bg,0,0,top_.width,top_.height);
  if(show.checked){
    tctx.fillStyle='rgba(217,96,90,.42)';
    rm.blockedCells.forEach(([cx,cy])=>tctx.fillRect(cx*T,cy*T,T,T));
  }
  tctx.strokeStyle='rgba(110,168,216,.6)'; tctx.lineWidth=2; tctx.beginPath();
  trail.forEach(([cx,cy],k)=>{const px=(cx+.5)*T,py=(cy+.5)*T;
    k?tctx.lineTo(px,py):tctx.moveTo(px,py);});
  tctx.stroke();
  tctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--agent');
  tctx.beginPath(); tctx.arc((x+.5)*T,(y+.5)*T,T*0.35,0,7); tctx.fill();
  tctx.fillStyle='#101215'; tctx.font=`bold ${Math.round(T*0.72)}px monospace`;
  tctx.textAlign='center'; tctx.textBaseline='middle';
  tctx.fillText(ARROW[d],(x+.5)*T,(y+.5)*T+1);

  const o=viewAt(rm,x,y,d), s=view.width/V;
  for(let vy=0;vy<V;vy++)for(let vx=0;vx<V;vx++){
    const p=o+(vy*V+vx)*3;
    vctx.fillStyle=`rgb(${rm.bankA[p]},${rm.bankA[p+1]},${rm.bankA[p+2]})`;
    vctx.fillRect(vx*s,vy*s,s,s);
  }
  const mm=rm.bankMismatches;
  info.innerHTML=`
   <tr><td>position</td><td>(${x}, ${y}) ${ARROW[d]}</td></tr>
   <tr><td>last action</td><td>${last===null?'\\u2014':NAMES[last]}</td></tr>
   <tr><td>step refused</td><td class="${refused?'bad':''}">${refused?'YES \\u2014 blocked by an object':'no'}</td></tr>
   <tr><td>on an object?</td><td class="${rm.blocked.has(x+','+y)?'bad':'ok'}">${rm.blocked.has(x+','+y)?'INSIDE ONE \\u2014 bug':'no'}</td></tr>
   <tr><td>steps / blocked</td><td>${steps} / ${blocks}</td></tr>
   <tr><td>objects</td><td>${rm.describe}</td></tr>
   <tr><td>walkable cells</td><td>${rm.nWalkable} of ${rm.nBase}</td></tr>
   <tr><td>forward refused at</td><td>${rm.refusedPoses} of ${4*rm.nWalkable} poses<br>
       <span style="color:var(--dim)">walls ${rm.refusedByWall} \u00b7 objects ${rm.refusedByObject}</span></td></tr>
   <tr><td>bank vs live render<br>at EVERY pose</td><td class="${mm?'bad':'ok'}">${mm?mm+' MISMATCHES':'identical'}</td></tr>`;
}
document.querySelectorAll('.keys button').forEach(b=>b.onclick=()=>act(+b.dataset.a));
document.getElementById('rand').onclick=()=>act(Math.floor(Math.random()*4));
document.getElementById('reset').onclick=()=>load(r);
sel.onchange=e=>load(+e.target.value);
show.onchange=draw;
addEventListener('keydown',e=>{
  const m={ArrowLeft:0,ArrowRight:1,ArrowUp:2,' ':3};
  if(e.key in m){e.preventDefault();act(m[e.key]);}
});
load(0);
</script>"""


def main() -> None:
    print(f"building {N_ROOMS} impassable-object rooms")
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"\n  bank vs live render, over every reachable pose in every room: "
          f"{data['totalBad']} mismatches")
    print(f"  wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

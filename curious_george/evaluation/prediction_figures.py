"""Look at what the world model predicts, room by room, because a scalar cannot.

`multiroom/mean_room_sRSA` says the representation is spatial. It cannot say
whether the network is seeing ten different rooms at all - a run whose layout
rotation silently collapsed to one room would still report a perfectly good
per-room sRSA, because each "room" would be the same room. This draws the thing.

    from curious_george.log_and_store.storage import get_model_dir
    from curious_george.evaluation.prediction_figures import plot_run_predictions
    plot_run_predictions(run_dir=get_model_dir("<run-name>"), path="rooms.png")

`run_dir` is caller-supplied on purpose - this reads an existing run rather than
writing one - but it is resolved through `get_model_dir` so a cluster run is
found under RL_STORAGE instead of a literal `outputs/` that only exists on the
machine that trained it.

Everything is composed from `evaluation/checkpoint_series.py` - `build` for the
network and env, `fixed_probe` for the rollout - so this is the same collection
path the checkpoint series scores, not a parallel one.

`matplotlib` is imported inside the function: this module is importable from
evaluation code that must not pull a plotting stack onto the training host.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float

#: MiniGrid's egocentric view, as flattened by `env2pred`: 7x7 cells, RGB.
VIEW = (7, 7, 3)


@dataclass(frozen=True)
class RoomPrediction:
    """One room's observed and predicted views over a single rollout.

    `observed` is `predict`'s OWN target (`obs_next`), not a re-derived slice, so
    a change to the architecture's `predOffset` shows up here rather than being
    silently re-aligned by this module.
    """

    key: str
    observed: Float[np.ndarray, "T 7 7 3"]
    predicted: Float[np.ndarray, "T 7 7 3"]
    mse: Float[np.ndarray, " T"]
    #: True where `inMask` let the real observation into the input.
    shown: Bool[np.ndarray, " T"]


def _views(flat: Float[torch.Tensor, "1 T 147"]) -> Float[np.ndarray, "T 7 7 3"]:
    return flat[0].reshape(-1, *VIEW).clamp(0, 1).numpy()


def predictions_for_room(*, pN, env, layout, steps: int) -> RoomPrediction:
    """Roll out `steps` random actions in `layout` and keep target vs prediction.

    Asserts the temporal alignment rather than documenting it: under
    `predOffset=0` the target IS the current observation, so `obs_next` must
    equal `obs[:, :T]`. If a future architecture predicts the NEXT observation
    this raises, which is the point - a figure that quietly re-aligns itself
    would hide the change it exists to show.
    """
    from curious_george.evaluation.checkpoint_series import fixed_probe

    (obs, act, _), = fixed_probe(pN=pN, env=env, layout=layout, n_trajs=1, steps=steps)
    with torch.no_grad():
        obs_pred, obs_next, _ = pN.predict(obs, act)

    T = obs_next.size(1)
    if not torch.equal(obs_next, obs[:, :T, :]):
        offset = 1 if torch.equal(obs_next, obs[:, 1 : T + 1, :]) else None
        raise AssertionError(
            f"target is not obs[t]; predOffset looks like {offset if offset else '?'}. "
            "Update this module deliberately - see docs/prnn-io-alignment.md."
        )

    mask = np.asarray(pN.pRNN.inMask, dtype=bool)
    return RoomPrediction(
        key=layout.key,
        observed=_views(obs_next),
        predicted=_views(obs_pred),
        mse=((obs_pred - obs_next) ** 2).mean(dim=2)[0].numpy(),
        shown=np.resize(mask, T),
    )


def plot_predictions(rooms: list[RoomPrediction], *, steps: int, path: Path | str | None = None):
    """Rows of observed/predicted view pairs, one pair per room.

    Every panel is drawn on the same 0-1 RGB scale, so brightness is comparable
    across rooms and across the observed/predicted split.
    """
    import matplotlib.pyplot as plt

    n = len(rooms)
    fig, axes = plt.subplots(
        2 * n, steps, figsize=(1.05 * steps, 2.25 * n), squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.05},
    )
    for r, room in enumerate(rooms):
        for row, (what, frames) in enumerate((("seen", room.observed), ("pred", room.predicted))):
            for t in range(steps):
                ax = axes[2 * r + row][t]
                ax.imshow(frames[t], vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#1d6fb8" if room.shown[t] else "#d9d7d1")
                    s.set_linewidth(1.8 if room.shown[t] else 0.6)
                if row == 0 and r == 0:
                    ax.set_title(f"t={t}\n{'shown' if room.shown[t] else 'masked'}", fontsize=7)
                if t == 0:
                    ax.set_ylabel(f"{room.key}\n{what}" if row == 0 else what, fontsize=7)
        axes[2 * r][0].set_ylabel(f"room {room.key}\nseen", fontsize=7)
    fig.suptitle(
        "What the world model is given (seen) and what it produces (pred), per room\n"
        "blue border = observation shown to the network; grey = zeroed by inMask",
        fontsize=10,
    )
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_run_predictions(
    *, run_dir: Path | str, path: Path | str | None = None, steps: int = 12, step: int | None = None
):
    """Draw every room of a finished or in-flight run from its archived checkpoint.

    The config comes from the run's OWN recorded `argv`, so the rooms drawn are
    the rooms trained on rather than a re-specification that can drift.
    """
    from curious_george import configs
    from curious_george.envs.layouts import resolve_layouts
    from curious_george.evaluation.checkpoint_series import archived, build

    run_dir = Path(run_dir)
    cfg = configs.cli(json.loads((run_dir / "provenance.json").read_text())["argv"][1:])
    layouts = resolve_layouts(cfg)
    if not layouts:
        raise ValueError(f"{cfg.env.source!r} resolves to no rooms")

    points = archived(run_dir)
    if not points:
        raise FileNotFoundError(f"no archived checkpoints under {run_dir}/checkpoints")
    chosen = dict(points)[step] if step is not None else points[-1][1]

    rooms = []
    for layout in layouts:
        pN, env = build(cfg=cfg, landmarks=layout.landmarks, ckpt=str(chosen))
        rooms.append(predictions_for_room(pN=pN, env=env, layout=layout, steps=steps))
    return plot_predictions(rooms, steps=steps, path=path)


# ---------------------------------------------------------------------------
# The circuit itself: every tensor the two networks exchange, on one clock.


#: The four actions this environment exposes. Index 3 is MiniGrid's `pickup`,
#: which cannot succeed in an L-room and leaves position and direction
#: unchanged - measured, not assumed.
ACTION_NAMES = ("turn left", "turn right", "forward", "pickup (no-op)")
#: Same four, short enough for a figure cell.
ACTION_SHORT = ("turn L", "turn R", "FORWARD", "no-op")

#: Head direction, clockwise from east. MiniGrid's `DIR_TO_VEC`, with y growing
#: DOWNWARD: 0 right (+1,0), 1 down (0,+1), 2 left (-1,0), 3 up (0,-1).
HD_NAMES = ("right", "down", "left", "up")

@dataclass(frozen=True)
class CircuitTrace:
    """Every tensor the two networks exchange over one episode, indexed by pRNN row.

    Row `t` consumes `h_prev[t]`, `obs_in[t]` and `act_in[t]` and emits `h[t]`
    and `pred[t]`, scored against `target[t]`. The policy row carries the same
    `t`, and `policy_state_label` names which hidden state it acted on - the one
    fact the two circuits disagree about.

    `obs_in` and `act_in` are `restructure_inputs`' own output, and the hidden
    states come from the production `SingleSRTracker`, so a change to `inMask`,
    `actOffset` or the tracker's observation choice shows up here rather than
    being absorbed by this module.
    """

    label: str
    encoding: str
    policy_state_label: str
    #: what chose the actions, for the caption - the two circuits diverge if
    #: each drives its own, so a comparison replays one trajectory in both.
    driven_by: str
    obs_env: Float[np.ndarray, "T 7 7 3"]
    head_direction: Int[np.ndarray, " T"]
    action: Int[np.ndarray, " T"]
    obs_in: Float[np.ndarray, "T 7 7 3"]
    act_in: Int[np.ndarray, "T A"]
    h_prev: Float[np.ndarray, "T H"]
    h: Float[np.ndarray, "T H"]
    pred: Float[np.ndarray, "T 7 7 3"]
    target: Float[np.ndarray, "T 7 7 3"]
    mse: Float[np.ndarray, " T"]
    shown: Bool[np.ndarray, " T"]
    policy_state: Float[np.ndarray, "T H"]
    policy_hd_onehot: Float[np.ndarray, "T 4"]
    policy_probs: Float[np.ndarray, "T A_env"]
    policy_value: Float[np.ndarray, " T"]
    #: index of the action whose curiosity reward is this row's MSE; -1 = none
    rewards_action: Int[np.ndarray, " T"]
    num_acts: int
    num_hd: int
    #: `inMask`'s period, so the phase column is the network's phase, not `t`.
    phase_k: int
    #: `actOffset`. Row `t`'s action block encodes `a[t - act_offset]`.
    act_offset: int
    #: real environment steps. The reward pass is ONE row longer (the tail row
    #: that scores the final action); it has no action and no policy step.
    n_steps: int


def trace_circuit(
    *,
    pN,
    acmodel,
    env,
    preprocess_obss,
    device,
    action_offset: int,
    steps: int = 12,
    seed: int = 20260829,
    actions: "np.ndarray | None" = None,
    driven_by: str = "the trained policy",
) -> CircuitTrace:
    """One episode, opened up row by row, for either circuit.

    `action_offset` IS the circuit. 0 pairs `obs[t]` with `a[t]`, the action
    chosen after seeing it, and the policy acts on `h[t-1]`. 1 pairs `obs[t]`
    with `a[t-1]`, the action that produced it, and the policy acts on `h[t]`.
    The shift is built here rather than by the architecture's `actOffset`,
    which front-pads ZEROS and so would drop `HD[0]` from row 0 and discard the
    segment's last action - both avoidable, and neither worth living with.

    `actions` replays a given sequence instead of sampling from the policy, so
    two circuits can be drawn on one trajectory. The policy is evaluated at
    every step either way, so its inputs and outputs are real.
    """
    import torch

    from curious_george.models.device import eval_mode
    from curious_george.models.prnn_adapter import (
        PRNNAdapter, encode_speed_hd_rows, make_sr_tracker,
    )

    adapter = PRNNAdapter(pN, device, pastSR=action_offset == 0)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    tracker = make_sr_tracker(adapter, device, [obs])
    obss, acts, srs, probs, values = [], [], [tracker.initial_sr()], [], []
    with eval_mode([pN.pRNN, acmodel]), torch.no_grad():
        for t in range(steps):
            dist, value = acmodel(preprocess_obss([obs], device=device), SR=srs[-1])
            probs.append(dist.probs[0].cpu().numpy())
            values.append(float(value))
            # dist.sample() is what rl/collect/collector.py:285 calls.
            a = int(actions[t]) if actions is not None else int(dist.sample().item())
            obss.append(obs)
            acts.append(a)
            obs = env.step(np.array([a]))[0]
            srs.append(tracker.step(np.array([a]), [obss[-1]], [obs]))
    last_obs, acts = obs, np.array(acts)
    seq = list(obss) + [last_obs]
    hd = np.array([int(o["direction"]) for o in seq])
    L = len(acts)

    # Both circuits get L+1 rows targeting obs[0..L], so the tail action is
    # scored and the two are row-comparable. offset 0 pads a zero action at the
    # tail (today's `target_offset=1` convention); offset 1 pads row 0 with a
    # zero SPEED and the real HD[0].
    padded = np.concatenate(([-1], acts)) if action_offset else np.concatenate((acts, [-1]))
    act_f = encode_speed_hd_rows(padded, hd, adapter.num_acts, adapter.num_hd)[None]
    if not action_offset:
        act_f[:, -1, adapter.num_acts:] = 0     # today's tail row carries no HD either
    obs_f = adapter.seq2pred(seq, acts)[0]

    h_init = torch.zeros((1, 1, pN.hidden_size))
    with eval_mode(pN.pRNN), torch.no_grad():
        x_t, _, _ = pN.pRNN.restructure_inputs(obs_in=obs_f, act=act_f)
        pred, tgt, h = pN.predict(obs_f, act_f, state=h_init)

    T = tgt.size(1)
    obs_size = pN.obs_size
    h_np = h.squeeze(0)[:T].numpy()
    pad = lambda a, fill: np.concatenate([a[:L], np.full(T - L, fill, dtype=a.dtype)])
    stack = lambda rows, w: np.concatenate(
        [np.stack(rows[:L]), np.zeros((T - L, w), dtype=np.float32)]
    )
    return CircuitTrace(
        label=pN.pRNNtype,
        encoding=getattr(pN.env_shell.encodeAction, "__name__", "?"),
        policy_state_label="h[t]" if action_offset else "h[t-1]",
        driven_by=driven_by,
        obs_env=np.stack([np.asarray(o["image"]) for o in seq[:T]]) / 255.0,
        head_direction=hd[:T],
        action=pad(acts, -1),
        obs_in=x_t[0, :T, :obs_size].reshape(T, *VIEW).clamp(0, 1).numpy(),
        act_in=x_t[0, :T, obs_size:].to(torch.int64).numpy(),
        h_prev=np.concatenate([h_init.squeeze(0).numpy(), h_np[:-1]]),
        h=h_np,
        pred=_views(pred[:, :T]),
        target=_views(tgt[:, :T]),
        mse=((pred - tgt) ** 2).mean(dim=2)[0, :T].numpy(),
        shown=np.resize(np.asarray(pN.pRNN.inMask, dtype=bool), T),
        policy_state=stack([s.squeeze(0).cpu().numpy() for s in srs], pN.hidden_size),
        policy_hd_onehot=stack([np.eye(4, dtype=np.float32)[hd[i]] for i in range(L)], 4),
        policy_probs=stack(probs, adapter.num_acts),
        policy_value=pad(np.asarray(values, dtype=np.float32), np.float32("nan")),
        # Row t's error is the surprise caused by the action row t encodes, and
        # that is a[t-1] for BOTH circuits - offset 0 reads it one row late
        # (`reward_alignment="next_obs"`), offset 1 has it in the row itself.
        rewards_action=np.where(np.arange(T) - 1 >= 0, np.arange(T) - 1, -1),
        num_acts=adapter.num_acts,
        num_hd=adapter.num_hd,
        phase_k=int(pN.phase_k),
        act_offset=action_offset,
        n_steps=L,
    )


def _encodes(trace: CircuitTrace, t: int) -> str:
    """Which action row `t`'s action block carries, and whether it survives.

    Only the forward bit survives SpeedHD/SpeedNextHD, so a turn or a no-op
    leaves the whole action block at zero: the world model is told nothing about
    which of the three it was. Naming that here is the difference between the
    figure reading as 'no data' and reading as 'no signal, by construction'.
    """
    i = t - trace.act_offset
    if i < 0:
        return "-- (front pad)"
    if i >= trace.n_steps:
        return "-- (tail row)"
    name = ACTION_SHORT[trace.action[i]]
    return f"a[{i}]={name}" + ("" if trace.action[i] == 2 else " ->0")


def alignment_table(trace: CircuitTrace) -> str:
    """The trace as text, because indices are checked numerically, not by eye."""
    a, hd = trace.num_acts, trace.num_hd
    head = (f"{'t':>3} {'phase':>5} {'HD[t]':>10} {'a[t]':>15} {'row encodes':>22} "
            f"{'obs shown':>9} {'obs-in norm':>11} {'pRNN act row':>{a + hd + 3}} {'target':>8} "
            f"{'MSE':>8} {'reward for':>10} {'policy state':>12} {'V[t]':>7}")
    rows = [f"circuit: {trace.label} + {trace.encoding}   "
            f"policy acts on {trace.policy_state_label}", head, "-" * len(head)]
    shift = 1 if trace.policy_state_label == "h[t-1]" else 0
    for t in range(len(trace.mse)):
        act, real = trace.act_in[t], t < trace.n_steps
        bits = ("[" + "".join(str(int(v)) for v in act[:a]) + "|"
                + "".join(str(int(v)) for v in act[a:]) + "]")
        reward = f"a[{trace.rewards_action[t]}]" if trace.rewards_action[t] >= 0 else "(dropped)"
        rows.append(
            f"{t:>3} {t % trace.phase_k:>5} "
            f"{str(trace.head_direction[t]) + ' ' + HD_NAMES[trace.head_direction[t]]:>10} "
            f"{(ACTION_NAMES[trace.action[t]] if real else '- (tail row)'):>15} "
            f"{_encodes(trace, t):>22} "
            f"{('yes' if trace.shown[t] else 'no'):>9} "
            f"{np.abs(trace.obs_in[t]).sum():>11.3f} {bits:>{a + hd + 3}} "
            f"{'obs[' + str(t) + ']':>8} {trace.mse[t]:>8.5f} {reward:>10} "
            f"{(f'h[{t - shift}]' if real else '-'):>12} "
            f"{(f'{trace.policy_value[t]:.3f}' if real else '-'):>7}"
        )
    return "\n".join(rows)


#: (label, colour, number of rows). The five stages of one timestep.
_BANDS = (
    ("ENVIRONMENT", "#8a7fb5", 2),
    ("pRNN INPUTS", "#1d6fb8", 4),
    ("pRNN OUTPUTS", "#c25e00", 4),
    ("POLICY INPUTS", "#2e8b57", 2),
    ("POLICY OUTPUTS", "#b03060", 2),
)


def plot_circuit(trace: CircuitTrace, *, path: Path | str | None = None, units: int = 64):
    """One column per timestep, rows grouped into the five stages of the circuit.

    Every hidden-state strip - `h[t-1]`, `h[t]`, and the policy's input - shares
    one colour scale, so the recurrent update and the policy's choice of state
    are directly comparable. Image panels share the 0-1 RGB scale. Band brackets
    are drawn from the real axes positions, so they cannot drift from the rows
    they name.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    T = len(trace.mse)
    A, HD = trace.num_acts, trace.num_hd
    rows = [
        ("obs[t]\nfrom the environment", "img", trace.obs_env),
        ("HD[t] and a[t]", "txt", "env"),
        ("obs[t] the pRNN GETS\n(black = zeroed by inMask)", "img", trace.obs_in),
        (f"action vector in\n({trace.encoding} keeps ONLY the forward bit)", "bits",
         trace.act_in),
        ("...which encodes:\npink = zeroed, so the pRNN\nis told nothing", "txt", "encodes"),
        ("h[t-1]   recurrent in", "hid", trace.h_prev),
        ("h[t]   recurrent out", "hid", trace.h),
        ("prediction  y[t]", "img", trace.pred),
        ("target  obs_next[t]", "img", trace.target),
        ("MSE, and whose reward", "txt", "mse"),
        (f"hidden state consumed:\n{trace.policy_state_label}", "hid", trace.policy_state),
        ("head direction in", "bits", trace.policy_hd_onehot),
        ("action probabilities", "prob", trace.policy_probs),
        ("executed action, V[t]", "txt", "pol"),
    ]
    assert len(rows) == sum(n for _, _, n in _BANDS)
    heights = {"img": 1.0, "bits": 0.62, "hid": 0.45, "prob": 0.62, "txt": 0.30}
    hr = [heights[k] for _, k, _ in rows]

    fig = plt.figure(figsize=(0.78 * T + 4.2, 1.55 * sum(hr) + 1.8))
    gs = GridSpec(len(rows), T, figure=fig, height_ratios=hr, hspace=0.16, wspace=0.06,
                  left=0.245, right=0.945, top=0.90, bottom=0.035)

    live = trace.policy_state[:trace.n_steps, :units]
    vmin, vmax = float(min(trace.h_prev[:, :units].min(), live.min())), \
                 float(max(trace.h_prev[:, :units].max(), live.max()))
    first_axes = []

    def cells(ax, vec, cmap, hi):
        ax.imshow(np.asarray(vec)[:, None], vmin=0, vmax=hi, cmap=cmap,
                  aspect="auto", interpolation="nearest")
        for edge in np.arange(len(vec) - 1) + 0.5:
            ax.axhline(edge, color="#cccccc", lw=0.4)

    for r, (label, kind, data) in enumerate(rows):
        blank_tail = kind in ("hid", "bits", "prob") and r >= 10
        for t in range(T):
            ax = fig.add_subplot(gs[r, t])
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                first_axes.append(ax)
            if blank_tail and t >= trace.n_steps:
                # No imshow here, so the axis keeps its default upward
                # orientation - and this is the column the slot labels sit on.
                # Match imshow's inverted axis or the labels render bottom-up.
                ax.set_facecolor("#f4f4f4")
                ax.set_ylim(len(np.atleast_1d(data[0])) - 0.5, -0.5)
            elif kind == "img":
                ax.imshow(data[t], vmin=0.0, vmax=1.0, interpolation="nearest")
            elif kind == "hid":
                ax.imshow(data[t][:units, None], vmin=vmin, vmax=vmax, cmap="magma",
                          aspect="auto", interpolation="nearest")
            elif kind == "bits":
                cells(ax, data[t], "Blues", 1)
                if r == 3:
                    ax.axhline(A - 0.5, color="#1d6fb8", lw=1.4)
            elif kind == "prob":
                cells(ax, data[t], "Greens", 1.0)
            else:
                ax.axis("off")
                real = t < trace.n_steps
                if data == "env":
                    txt = (f"HD {trace.head_direction[t]} {HD_NAMES[trace.head_direction[t]]}\n"
                           f"{ACTION_SHORT[trace.action[t]] if real else '—'}")
                elif data == "encodes":
                    txt = _encodes(trace, t).replace("=", "\n")
                elif data == "mse":
                    who = (f"→ r[a{trace.rewards_action[t]}]"
                           if trace.rewards_action[t] >= 0 else "dropped")
                    txt = f"{trace.mse[t]:.4f}\n{who}"
                else:
                    txt = (f"a = {ACTION_SHORT[trace.action[t]]}\nV = {trace.policy_value[t]:.2f}"
                           if real else "no policy\nstep")
                grey = not real and data not in ("mse", "encodes")
                ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=6.5,
                        color="#999999" if grey else
                        ("#b03060" if data == "encodes" and "->0" in txt else "black"))
            if kind in ("bits", "prob") and t == T - 1:
                names = ([*ACTION_SHORT[:A], *(f"HD {i} {HD_NAMES[i]}" for i in range(HD))]
                         if r == 3 else
                         [f"HD {i} {HD_NAMES[i]}" for i in range(HD)] if r == 11 else
                         list(ACTION_SHORT[:len(trace.policy_probs[0])]))
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=6)
                ax.yaxis.tick_right(); ax.tick_params(length=0, pad=2)
            if kind == "img":
                for sp in ax.spines.values():
                    sp.set_color("#1d6fb8" if trace.shown[t] else "#d9d7d1")
                    sp.set_linewidth(1.6 if trace.shown[t] else 0.6)
            if r == 0:
                ax.set_title(f"t={t}\n{'SHOWN' if trace.shown[t] else 'masked'}", fontsize=7,
                             color="#1d6fb8" if trace.shown[t] else "#8a8a8a")
        first_axes[r].set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center",
                                 labelpad=8)

    # Band brackets, from the axes that are actually there.
    fig.canvas.draw()
    idx = 0
    for name, colour, n in _BANDS:
        boxes = [first_axes[i].get_position() for i in range(idx, idx + n)]
        top, bottom = max(b.y1 for b in boxes), min(b.y0 for b in boxes)
        fig.add_artist(plt.Line2D([0.018, 0.018], [bottom, top], color=colour, lw=5,
                                  solid_capstyle="butt"))
        fig.text(0.035, (top + bottom) / 2, name, rotation=90, va="center", ha="center",
                 fontsize=9.5, color=colour, weight="bold")
        idx += n

    fig.suptitle(
        f"The circuit, one timestep per column — {trace.label} + {trace.encoding}, "
        f"policy acts on {trace.policy_state_label}\n"
        f"actions chosen by {trace.driven_by}, replayed identically in both circuits"
        f"   ·   hidden strips: first {units} units, one shared scale\n"
        f"h[t] is the sequence pass the world model trains on; the policy's state is the "
        f"rollout tracker's - same quantity, two code paths, so they are close but not "
        f"bit-identical",
        fontsize=10.5,
    )
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

"""Does a SMALL-SCALE INIT ON THE POLICY OUTPUT HEAD improve EARLY training?

THROWAWAY. No committed result may depend on this file (CLAUDE.md).

THE LAYER
`ACModelSR.forward` ends `x = self.actor(embedding)` -> `Categorical(...)`, and
`self.actor` is built in `ACModel.define_model` as
`Sequential(Linear(embedding_size, 64), Tanh(), Linear(64, act_dim))`. The
policy output head is therefore the LAST `nn.Linear` of `acmodel.actor`
(`actor[2]`, in_features=64, out_features=act_dim, bias=True). Every Linear is
initialised by `curious_george.models.init_params`: `normal_(0, 1)` followed by
a per-ROW division by that row's L2 norm, and `bias.fill_(0)`. So the default
head has unit-L2 rows (per-element std 1/sqrt(64) = 0.125) and a zero bias.

THE ARMS
A 2x2 in {row direction} x {row magnitude}, plus the untouched control, because
"uniform(-0.1, 0.1) helps" would otherwise conflate the two: uniform(-0.1, 0.1)
is BOTH a different draw AND a 2.17x shrink of the default row norm.

    default            untouched (row norm 1.0)          control
    uniform_0.1        nn.init.uniform_(w, -0.1, 0.1)    the proposal
    orthogonal_0.462   orthonormal rows, same row norm as uniform_0.1
    uniform_2.17e-3    iid uniform, same row norm as orthogonal_0.01
    orthogonal_0.01    nn.init.orthogonal_(w, gain=0.01) standard RL recipe

Each arm draws from a DEDICATED generator, so applying one consumes nothing
from the global RNG streams: arms at the same seed differ ONLY in the head
weight values, and every later draw (action sampling, minibatch shuffles,
world-model noise) is identical across arms until learning diverges.

WHAT IT MEASURES, per update
    entropy_pre_bits  H(a|s) in BITS under the CURRENT parameters on the fresh
                      rollout, before the update touches them - the quantity
                      this intervention acts on directly. Max 2 bits (4 actions).
    entropy_bits      the UpdateLogs value: same quantity averaged over the
                      update's minibatch gradient steps.
    mi_policy_bits    I(S;A) from the rollout's joint distribution. Bounded by
                      2 - policy_entropy, so any shift in entropy moves its cap.
    policy_loss, value_loss, grad_norm, value_mean   UpdateLogs
    loc_entropy, loc_entropy_5                       spatial coverage
    curious/advantage/value rollout statistics
    wm_loss           `predictiveNet.TrainingSaver["loss"]`, the EXACT
                      per-gradient-step world-model loss (wandb history is
                      capped at 10,000 samples and has misled this project
                      repeatedly).

REPRODUCE
    uv run python throwaway/policy_head_init.py --updates 400 \
        --arms default,uniform_0.1,orthogonal_0.01 --seeds 1,2,3 --out <dir>
    uv run python throwaway/policy_head_init.py --summarize <dir>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float

# The repo root is not encoded here: git owns the answer, so moving this file
# cannot silently break the config lookup.
REPO = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
    ).stdout.strip()
)

_LOG2 = math.log(2.0)

# Exactly the overrides the task specifies (~30k env steps/s, three CUDA
# graphs). seed and episodes_total are appended per run.
RUN_OVERRIDES = [
    "env=lroom",
    "run=multienv",
    "exp.device_env=True",
    "predNet.batched_wm=True",
    "predNet.wm_pool_group=1",
    "predNet.compile_cell=layer",
    "predNet.cuda_graph=True",
    "rl.cuda_graph=True",
    "exp.rollout_cuda_graph=True",
    "exp.num_envs=128",
    "rl.frames=32768",
    "rl.ppo_batch_size=256",
    "rl.entropy_coef=0",
    "logging.wandb_log=false",
]


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

Draw = Callable[[tuple[int, int], torch.Generator], Float[torch.Tensor, "out in"]]


@dataclass(frozen=True)
class HeadInit:
    """One re-initialisation of the actor's final Linear weight.

    `draw is None` is the control: the constructed weight is left alone.
    `row_norm` is the L2 norm each output row is expected to have, stated so
    the applied weight can be CHECKED rather than assumed.
    """

    name: str
    draw: Draw | None
    row_norm: float


def _uniform_draw(bound: float) -> Draw:
    """iid U(-bound, bound). Row of n has E||.||^2 = n*bound^2/3."""
    def draw(shape: tuple[int, int], g: torch.Generator) -> torch.Tensor:
        return nn.init.uniform_(torch.empty(shape), -bound, bound, generator=g)
    return draw


def _orthogonal_draw(gain: float) -> Draw:
    """Mutually orthogonal rows, each of L2 norm `gain`."""
    def draw(shape: tuple[int, int], g: torch.Generator) -> torch.Tensor:
        return nn.init.orthogonal_(torch.empty(shape), gain=gain, generator=g)
    return draw


def _uniform_bound_for(row_norm: float, fan_in: int) -> float:
    return row_norm * math.sqrt(3.0 / fan_in)


# fan_in of the head is 64 (the actor's hidden width, fixed in define_model).
_FAN_IN = 64
_UNIFORM_01_ROW_NORM = 0.1 * math.sqrt(_FAN_IN / 3.0)  # 0.46188

ARMS: dict[str, HeadInit] = {
    a.name: a
    for a in [
        HeadInit("default", None, 1.0),
        HeadInit("uniform_0.1", _uniform_draw(0.1), _UNIFORM_01_ROW_NORM),
        HeadInit(
            "orthogonal_0.462", _orthogonal_draw(_UNIFORM_01_ROW_NORM),
            _UNIFORM_01_ROW_NORM,
        ),
        HeadInit(
            "uniform_2.17e-3", _uniform_draw(_uniform_bound_for(0.01, _FAN_IN)), 0.01,
        ),
        HeadInit("orthogonal_0.01", _orthogonal_draw(0.01), 0.01),
    ]
}


def policy_head(acmodel: nn.Module) -> nn.Linear:
    """The actor's LAST Linear - the layer emitting the action logits."""
    head = [m for m in acmodel.actor if isinstance(m, nn.Linear)][-1]
    assert head.out_features == acmodel.act_dim, (
        f"actor's last Linear emits {head.out_features} logits, "
        f"but act_dim is {acmodel.act_dim}"
    )
    return head


def _weight_stats(w: torch.Tensor, bias: torch.Tensor | None) -> dict:
    return {
        "shape": list(w.shape),
        "std": float(w.std()),
        "mean": float(w.mean()),
        "absmax": float(w.abs().max()),
        "row_l2": [round(float(v), 6) for v in w.norm(dim=1)],
        "bias_absmax": None if bias is None else float(bias.abs().max()),
        "has_bias": bias is not None,
    }


def apply_head_init(acmodel: nn.Module, arm: HeadInit, *, seed: int) -> dict:
    """Re-initialise the policy head in place; return before/after statistics.

    Draws from a dedicated generator so no global RNG stream is advanced.
    """
    head = policy_head(acmodel)
    before = _weight_stats(head.weight.detach(), head.bias)
    if arm.draw is not None:
        # Python's hash() of a str is per-process randomized, so it cannot seed
        # anything that has to be reproducible across invocations.
        digest = hashlib.sha256(f"{arm.name}:{seed}".encode()).digest()[:4]
        g = torch.Generator().manual_seed(int.from_bytes(digest, "little"))
        w = arm.draw(tuple(head.weight.shape), g).to(head.weight.device)
        with torch.no_grad():
            head.weight.copy_(w)
    after = _weight_stats(head.weight.detach(), head.bias)
    return {
        "arm": arm.name,
        "layer": f"acmodel.actor[{list(acmodel.actor).index(head)}]",
        "layer_repr": repr(head),
        "expected_row_norm": arm.row_norm,
        "before": before,
        "after": after,
    }


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def entropy_bits_on_rollout(acmodel: nn.Module, exps) -> tuple[float, float]:
    """(mean, std) of H(a|s) in bits over a collected rollout's states.

    Pure forward under no_grad: consumes no RNG and mutates nothing, so calling
    it between collect and update cannot perturb the run.
    """
    with torch.no_grad():
        dist, _ = acmodel(exps.obs, SR=exps.SR)
        h = dist.entropy() / _LOG2
    return float(h.mean()), float(h.std())


def logit_spread_at_reset(acmodel: nn.Module, algo) -> dict:
    """Head output statistics at the algo's CURRENT (pre-rollout) state.

    At an episode boundary every stream's SR is zero, so the only input that
    varies is the head-direction one-hot: this is the most degenerate state the
    policy ever sees, and therefore the tightest check that the head re-init
    actually reached the forward pass.
    """
    from curious_george.rl.collect.collector import _device_policy_obss

    images, directions = algo.envs.observation_device()
    obs = _device_policy_obss(images, directions, acmodel)
    with torch.no_grad():
        dist, _ = acmodel(obs, SR=algo.state.sr)
        logits = dist.logits
        h = dist.entropy() / _LOG2
    return {
        "entropy_bits_mean": float(h.mean()),
        "entropy_bits_min": float(h.min()),
        "logit_absmax": float(logits.abs().max()),
        "max_action_prob": float(dist.probs.max()),
    }


def run_arm(cfg, arm: HeadInit, *, updates: int, spatial_analysis: bool) -> dict:
    """Build the stack, apply `arm` to the policy head, train `updates` updates."""
    from curious_george.evaluation.on_policy import mutual_info_policy
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    t_setup = time.perf_counter()
    comps = setup_training(cfg)
    algo = comps.algo
    init_report = apply_head_init(comps.acmodel, arm, seed=int(cfg.exp.seed))
    init_report["at_reset"] = logit_spread_at_reset(comps.acmodel, algo)
    setup_s = time.perf_counter() - t_setup
    print(json.dumps(init_report, indent=1), flush=True)

    saver_start = len(comps.predictiveNet.TrainingSaver)
    keys = (
        "entropy_pre_bits", "entropy_pre_bits_std", "entropy_bits", "mi_policy_bits",
        "policy_loss", "value_loss", "grad_norm", "value_mean",
        "loc_entropy", "loc_entropy_5", "curious_mean", "curious_std",
        "advantage_mean", "advantage_std", "return_mean", "update_s",
    )
    series: dict[str, list[float]] = {k: [] for k in keys}

    t_loop = time.perf_counter()
    for _ in range(updates):
        t0 = time.perf_counter()
        exps, logs1 = algo.collect_experiences()
        # BEFORE the update: the entropy of the parameters that produced this
        # rollout, not of the parameters the update leaves behind.
        h_mean, h_std = entropy_bits_on_rollout(comps.acmodel, exps)
        logs2 = algo.update_parameters(exps=exps)
        dt = time.perf_counter() - t0

        series["entropy_pre_bits"].append(h_mean)
        series["entropy_pre_bits_std"].append(h_std)
        series["entropy_bits"].append(logs2["entropy"])
        series["mi_policy_bits"].append(float(mutual_info_policy(logs1["joint_dist"])))
        series["policy_loss"].append(logs2["policy_loss"])
        series["value_loss"].append(logs2["value_loss"])
        series["grad_norm"].append(logs2["grad_norm"])
        series["value_mean"].append(logs2["value"])
        series["loc_entropy"].append(float(logs1["loc_entropy"]))
        series["loc_entropy_5"].append(float(logs1["loc_entropy_5"]))
        series["curious_mean"].append(float(np.mean(logs1["curious_rewards"])))
        series["curious_std"].append(float(np.std(logs1["curious_rewards"])))
        series["advantage_mean"].append(float(np.mean(logs1["advantages"])))
        series["advantage_std"].append(float(np.std(logs1["advantages"])))
        series["return_mean"].append(float(np.mean(logs1["return_per_episode"])))
        series["update_s"].append(dt)
    loop_s = time.perf_counter() - t_loop

    ts = comps.predictiveNet.TrainingSaver
    wm_loss = [float(x) for x in ts["loss"].iloc[saver_start:].tolist()]

    srsa = None
    if spatial_analysis:
        from curious_george.training.loop import run_spatial_analysis
        t_an = time.perf_counter()
        run_spatial_analysis(cfg, comps, wandb_log=False)
        srsa = {"seconds": round(time.perf_counter() - t_an, 2)}

    from curious_george.training.schedule import TrainingSchedule
    sched = TrainingSchedule.from_config(cfg)
    result = {
        "meta": {
            "arm": arm.name,
            "seed": int(cfg.exp.seed),
            "updates": updates,
            "git": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                capture_output=True, text=True).stdout.strip(),
            "device": str(get_device()),
            "torch": torch.__version__,
            "overrides": list(RUN_OVERRIDES) + [
                f"exp.seed={cfg.exp.seed}", f"rl.episodes_total={cfg.rl.episodes_total}",
            ],
            "policy_grad_steps_per_update": sched.policy_steps_per_update,
            "wm_grad_steps_per_update": sched.world_model_steps_per_update,
            "frames_per_update": int(cfg.rl.frames),
            "setup_s": round(setup_s, 1),
            "loop_s": round(loop_s, 1),
            "env_steps_per_s": round(updates * int(cfg.rl.frames) / loop_s, 1),
        },
        "head_init": init_report,
        "series": series,
        "wm_loss": wm_loss,
        "spatial": srsa,
    }
    # Captured CUDA graphs pin a memory pool for as long as the graph objects
    # live; a second run in the same process must not inherit the first one's
    # reservation on an 8 GB card.
    del comps, algo, exps
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# summary (reads JSON only - collects nothing)
# --------------------------------------------------------------------------- #

# Gradient-step checkpoints are stated in UPDATES; the per-update policy and
# world-model step counts are recorded in each run's meta.
CHECKPOINT_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


# The world-model loss is recorded ONE ROW PER GRADIENT STEP and is noisy at
# that resolution: a 256-step window (2 updates) made a pure-noise endpoint
# difference look like it had cleared the seed spread. 2560 steps = 20 updates.
# Below the mark the window is truncated, so an early mark is a prefix mean.
WM_WINDOW = 2560


def _window_mean(xs: list[float], end: int, width: int = 10) -> float:
    lo = max(0, end - width)
    return float(np.mean(xs[lo:end])) if end > lo else float("nan")


def summarize(out_dir: Path) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no run JSON under {out_dir}")
    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        by_arm.setdefault(r["meta"]["arm"], []).append(r)

    n_updates = min(len(r["series"]["entropy_bits"]) for r in runs)
    per_update = runs[0]["meta"]["policy_grad_steps_per_update"]
    marks = sorted({max(1, int(round(f * n_updates))) for f in CHECKPOINT_FRACTIONS})

    print(f"\nruns: {len(runs)}  arms: {list(by_arm)}  matched updates: {n_updates}")
    print(f"policy gradient steps/update: {per_update}   "
          f"world-model gradient steps/update: "
          f"{runs[0]['meta']['wm_grad_steps_per_update']}")

    print("\nHEAD AT INITIALISATION (weight of the actor's last Linear)")
    print(f"{'arm':<18}{'row L2 (mean)':>15}{'elem std':>12}"
          f"{'bias absmax':>13}{'H(a|s) bits @reset':>21}{'max p(a)':>10}")
    for arm, rs in by_arm.items():
        a = rs[0]["head_init"]["after"]
        at = rs[0]["head_init"]["at_reset"]
        print(f"{arm:<18}{np.mean(a['row_l2']):>15.6f}{a['std']:>12.6f}"
              f"{a['bias_absmax']:>13.1e}{at['entropy_bits_mean']:>21.5f}"
              f"{at['max_action_prob']:>10.4f}")

    for metric, fmt in [
        ("entropy_pre_bits", "{:.4f}"), ("mi_policy_bits", "{:.4f}"),
        ("policy_loss", "{:.2e}"), ("value_loss", "{:.2e}"),
        ("grad_norm", "{:.3f}"), ("loc_entropy", "{:.4f}"),
    ]:
        print(f"\n{metric}   (mean +/- sd across seeds; 10-update window ending at U)")
        header = "".join(f"{'U=' + str(m):>20}" for m in marks)
        print(f"{'arm':<18}{'n':>3}{header}")
        for arm, rs in by_arm.items():
            cells = []
            for m in marks:
                vals = [_window_mean(r["series"][metric], m) for r in rs]
                cells.append(f"{fmt.format(np.mean(vals))}+-{np.std(vals):.1e}".rjust(20))
            print(f"{arm:<18}{len(rs):>3}" + "".join(cells))

    print("\nwm_loss (predictiveNet.TrainingSaver['loss'], exact per-gradient-step)")
    n_wm = min(len(r["wm_loss"]) for r in runs)
    wm_marks = sorted({max(1, int(round(f * n_wm))) for f in CHECKPOINT_FRACTIONS})
    header = "".join(f"{'S=' + str(m):>20}" for m in wm_marks)
    print(f"{'arm':<18}{'n':>3}{header}")
    for arm, rs in by_arm.items():
        cells = []
        for m in wm_marks:
            vals = [_window_mean(r["wm_loss"], m, width=WM_WINDOW) for r in rs]
            cells.append(f"{np.mean(vals):.5f}+-{np.std(vals):.1e}".rjust(20))
        print(f"{arm:<18}{len(rs):>3}" + "".join(cells))

    print("\nEARLY WINDOW  entropy_pre_bits per update, mean across seeds"
          " (the intervention lives here)")
    early = min(15, n_updates)
    print(f"{'arm':<18}" + "".join(f"{'u' + str(u + 1):>8}" for u in range(early)))
    for arm, rs in by_arm.items():
        row = "".join(
            f"{np.mean([r['series']['entropy_pre_bits'][u] for r in rs]):>8.4f}"
            for u in range(early)
        )
        print(f"{arm:<18}{row}")
    print(f"{'(across-seed sd)':<18}" + "".join(
        f"{np.mean([np.std([r['series']['entropy_pre_bits'][u] for r in rs]) for rs in by_arm.values()]):>8.4f}"
        for u in range(early)
    ))

    print("\nENTROPY TRAJECTORY  entropy_pre_bits, H(a|s) in bits, max 2.0"
          "  (mean across seeds; 10-update window)")
    grid = sorted({max(1, int(round(f * n_updates)))
                   for f in np.linspace(0.0, 1.0, 11)[1:]})
    print(f"{'arm':<18}" + "".join(f"{'U=' + str(m):>9}" for m in grid))
    for arm, rs in by_arm.items():
        cells = "".join(
            f"{np.mean([_window_mean(r['series']['entropy_pre_bits'], m) for r in rs]):>9.4f}"
            for m in grid
        )
        print(f"{arm:<18}{cells}")

    control = "default"
    if control in by_arm:
        print(f"\nPAIRED DELTA vs {control} (same seed = same RNG stream, so the"
              " difference is the arm)")
        print("   verdict: |mean delta| must exceed the across-seed sd of that"
              " delta to be more than noise")
        for metric in ("entropy_pre_bits", "mi_policy_bits", "grad_norm",
                       "policy_loss", "value_loss", "loc_entropy"):
            print(f"\n  {metric}")
            print(f"  {'arm':<18}" + "".join(f"{'U=' + str(m):>18}" for m in marks))
            base = {r["meta"]["seed"]: r for r in by_arm[control]}
            for arm, rs in by_arm.items():
                if arm == control:
                    continue
                cells = []
                for m in marks:
                    d = [
                        _window_mean(r["series"][metric], m)
                        - _window_mean(base[r["meta"]["seed"]]["series"][metric], m)
                        for r in rs if r["meta"]["seed"] in base
                    ]
                    cells.append(f"{np.mean(d):+.4f}+-{np.std(d):.4f}".rjust(18))
                print(f"  {arm:<18}" + "".join(cells))
        print(f"\n  wm_loss (world-model loss; NEGATIVE delta = better than {control})")
        print(f"  {'arm':<18}" + "".join(f"{'S=' + str(m):>18}" for m in wm_marks))
        base = {r["meta"]["seed"]: r for r in by_arm[control]}
        for arm, rs in by_arm.items():
            if arm == control:
                continue
            cells = []
            for m in wm_marks:
                d = [
                    _window_mean(r["wm_loss"], m, width=WM_WINDOW)
                    - _window_mean(base[r["meta"]["seed"]]["wm_loss"], m, width=WM_WINDOW)
                    for r in rs if r["meta"]["seed"] in base
                ]
                cells.append(f"{np.mean(d):+.5f}+-{np.std(d):.5f}".rjust(18))
            print(f"  {arm:<18}" + "".join(cells))

    print("\nthroughput")
    for arm, rs in by_arm.items():
        print(f"  {arm:<18} setup {np.mean([r['meta']['setup_s'] for r in rs]):>6.1f}s "
              f"loop {np.mean([r['meta']['loop_s'] for r in rs]):>7.1f}s "
              f"{np.mean([r['meta']['env_steps_per_s'] for r in rs]):>8.0f} env steps/s")


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--out", type=Path, default=None,
                   help="directory for the per-run JSONs (required to collect)")
    p.add_argument("--spatial-analysis", action="store_true",
                   help="one run_spatial_analysis (sRSA) after each arm (~7 s)")
    p.add_argument("--summarize", type=Path, default=None,
                   help="print the table for an existing output directory and exit")
    args = p.parse_args()

    if args.summarize is not None:
        summarize(args.summarize)
        return
    if args.out is None:
        p.error("--out is required unless --summarize is given")

    import matplotlib
    matplotlib.use("Agg")
    from hydra import compose, initialize_config_dir

    arms = [ARMS[a] for a in args.arms.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    # episodes_total is EXPERIENCE; the loop bound is episodes_total*seqdur env
    # steps. Drive it from the update count so every arm gets the same budget.
    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        for seed in seeds:
            for arm in arms:
                dest = args.out / f"{arm.name}__seed{seed}.json"
                if dest.exists():
                    print(f"skip (exists): {dest}", flush=True)
                    continue
                cfg = compose(config_name="main", overrides=RUN_OVERRIDES + [
                    f"exp.seed={seed}", "rl.episodes_total=1",
                ])
                episodes = (
                    args.updates * int(cfg.rl.frames) // int(cfg.predNet.seqdur)
                )
                cfg.rl.episodes_total = episodes
                print(f"\n=== arm={arm.name} seed={seed} "
                      f"updates={args.updates} episodes_total={episodes} ===",
                      flush=True)
                result = run_arm(
                    cfg, arm, updates=args.updates,
                    spatial_analysis=args.spatial_analysis,
                )
                dest.write_text(json.dumps(result))
                m = result["meta"]
                print(f"wrote {dest}  loop {m['loop_s']}s  "
                      f"{m['env_steps_per_s']} env steps/s", flush=True)

    summarize(args.out)


if __name__ == "__main__":
    main()

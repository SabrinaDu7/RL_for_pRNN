"""Does linear LEARNING-RATE WARMUP improve EARLY training?  [throwaway]

Standalone: nothing in the repo imports this, and no committed result depends
on it (CLAUDE.md, "one-off scripts and throwaways").

WHAT IT DOES
    Builds the normal training stack with `setup_training(cfg)`, then MUTATES
    the optimizers it got back before stepping them. No file under
    `curious_george/`, `Configs/` or `tests/` is touched.

WHY A TENSOR LEARNING RATE, AND WHY THE BASELINE USES ONE TOO
    Both optimizer steps in this configuration run INSIDE CUDA graphs
    (`rl/update/policy_graph.py::GraphPolicyTrainer._region`,
    `world_model/adapter.py::_GraphWMTrainer._capture`). A graph bakes a float
    hyperparameter into the captured kernels, so assigning
    `param_group["lr"] = <float>` after capture is silently ignored. Torch's
    capturable optimizers read a 0-dim device tensor instead
    (`_multi_tensor_rmsprop` / `_multi_tensor_adam`, the
    `capturable and isinstance(lr, torch.Tensor)` branches), which this script
    refills in place. `--gate` proves both halves of that claim.

    Swapping float lr for tensor lr changes the ARITHMETIC ORDER of the update
    (`_foreach_div_(avg, -lr)` then `addcdiv_`, instead of `addcdiv_(value=-lr)`),
    so the no-warmup baseline runs the same tensor-lr machinery with the ramp
    pinned at 1.0. The arms then differ only in the schedule.

USAGE
    uv run python throwaway/lr_warmup.py gate
    uv run python throwaway/lr_warmup.py probe --updates 12
    uv run python throwaway/lr_warmup.py run --arm wm15 --seed 1 --updates 400 --out DIR
    uv run python throwaway/lr_warmup.py report --out DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# The fast configuration named in the task: three CUDA graphs on, world-model
# and policy gradient-step ratios matched to the 2026-07 reference
# (wm_pool_group=1, ppo_batch_size=256).
BASE_OVERRIDES = [
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

PROBE_SEED = 1234  # fixed spatial probe, so sRSA is comparable across arms


# --------------------------------------------------------------------------
# the intervention
# --------------------------------------------------------------------------
class RampedLearningRate:
    """Graph-safe handle on every param group's learning rate.

    Replaces each group's float `lr` with a 0-dim device tensor holding the
    same value, keeping the group's RELATIVE rates intact - the world model's
    RMSprop has four groups scaled by rootk_h / rootk_i / bias_lr
    (prnn PredictiveNet), and collapsing them would change the architecture's
    learning rates, not warm them up.

    The tensors, not the optimizer, are the handle: both graph trainers REBUILD
    their optimizer capturable on first use, copying group dicts, so the
    rebuilt optimizer carries these very tensors.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, device: torch.device) -> None:
        self.base: list[float] = [float(g["lr"]) for g in optimizer.param_groups]
        self.tensors: list[torch.Tensor] = [
            torch.tensor(lr, dtype=torch.float32, device=device) for lr in self.base
        ]
        for group, tensor in zip(optimizer.param_groups, self.tensors):
            group["lr"] = tensor

    def set_factor(self, factor: float) -> None:
        for tensor, base in zip(self.tensors, self.base):
            tensor.fill_(base * factor)


@dataclass(frozen=True)
class Arm:
    """One condition: what fraction of the run each optimizer warms over.

    `factor` ramps LINEARLY over `frac * n_updates` updates, from lr/w on the
    first update to the configured lr on the last warmup update. A literal
    zero would spend the first update's gradient steps doing nothing, which is
    a different intervention (a delay) than a ramp.
    """

    policy_frac: float
    wm_frac: float

    @staticmethod
    def factor(frac: float, *, update: int, n_updates: int) -> float:
        window = int(round(frac * n_updates))
        return 1.0 if window <= 0 else min(1.0, (update + 1) / window)


ARMS: dict[str, Arm] = {
    "none": Arm(policy_frac=0.0, wm_frac=0.0),
    "policy05": Arm(policy_frac=0.05, wm_frac=0.0),
    "policy15": Arm(policy_frac=0.15, wm_frac=0.0),
    "wm05": Arm(policy_frac=0.0, wm_frac=0.05),
    "wm15": Arm(policy_frac=0.0, wm_frac=0.15),
    "both15": Arm(policy_frac=0.15, wm_frac=0.15),
}


# --------------------------------------------------------------------------
# gate: a captured graph must honour an in-place tensor-lr change, and must
#       ignore a float one. A gate that passes either way proves nothing.
# --------------------------------------------------------------------------
def _capture_one_step(optimizer_of, *, tensor_lr: bool) -> tuple[float, float]:
    """Capture one graphed optimizer step; return |delta w| at lr and at 2*lr."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight = torch.nn.Parameter(torch.ones(64, 64, device=device))
    lr0 = 1e-3
    lr = torch.tensor(lr0, device=device) if tensor_lr else lr0
    opt = optimizer_of([{"params": [weight], "lr": lr, "capturable": True}])
    grad = torch.full_like(weight, 0.1)

    def region():
        opt.zero_grad(set_to_none=False)
        weight.grad = grad.clone()
        opt.step()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            region()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        region()

    # Both replays must start from the SAME optimizer state, or RMSprop's
    # evolving square_avg changes the step size on its own and the comparison
    # measures that instead of lr.
    snapshot = ([weight.detach().clone()],
                {id(p): {k: v.detach().clone() for k, v in st.items()
                         if isinstance(v, torch.Tensor)}
                 for p, st in opt.state.items()})

    def replay_delta(new_lr: float) -> float:
        with torch.no_grad():
            weight.copy_(snapshot[0][0])
            for p, st in opt.state.items():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        v.copy_(snapshot[1][id(p)][k])
        if tensor_lr:
            lr.fill_(new_lr)
        else:
            opt.param_groups[0]["lr"] = new_lr
        before = weight.detach().clone()
        graph.replay()
        torch.cuda.synchronize()
        return float((weight.detach() - before).abs().mean())

    return replay_delta(lr0), replay_delta(2 * lr0)


def gate() -> int:
    if not torch.cuda.is_available():
        print("GATE SKIPPED: no CUDA")
        return 1
    failures = 0
    for name, builder in (("Adam", torch.optim.Adam), ("RMSprop", torch.optim.RMSprop)):
        d1, d2 = _capture_one_step(builder, tensor_lr=True)
        ratio = d2 / d1
        ok = abs(ratio - 2.0) < 0.02
        failures += not ok
        print(f"  {name:8s} tensor lr : |dw|@lr={d1:.3e} |dw|@2lr={d2:.3e} "
              f"ratio={ratio:.4f}  {'PASS (mutation is honoured)' if ok else 'FAIL'}")

        f1, f2 = _capture_one_step(builder, tensor_lr=False)
        frozen = f1 == f2
        failures += not frozen
        print(f"  {name:8s} float  lr : |dw|@lr={f1:.3e} |dw|@2lr={f2:.3e} "
              f"ratio={f2 / f1:.4f}  "
              f"{'PASS (negative control: float lr is BAKED IN)' if frozen else 'FAIL'}")
    print("GATE:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return int(failures > 0)


def live_gate() -> int:
    """The gate that matters: does the ramp reach the REAL captured graphs?

    Micro-benchmarks prove torch's contract; they do not prove that this
    script's tensors survive `_ensure_capturable_optimizer`'s rebuild of both
    optimizers, which happens inside the first `update_parameters` call. So:
    run a real update at factor 0, and require that NOT ONE parameter moved -
    with the same updates at factor 1 either side as the positive control.
    """
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    cfg = build_cfg(seed=1, n_updates=4)
    comps = setup_training(cfg)
    device = get_device()
    ramps = {"policy": RampedLearningRate(comps.algo.optimizer, device=device),
             "world-model": RampedLearningRate(comps.predictiveNet.optimizer, device=device)}
    params = {"policy": list(comps.acmodel.parameters()),
              "world-model": list(comps.predictiveNet.pRNN.parameters())}

    def step(factor: float) -> dict[str, float]:
        for ramp in ramps.values():
            ramp.set_factor(factor)
        before = {k: [p.detach().clone() for p in ps] for k, ps in params.items()}
        exps, _ = comps.algo.collect_experiences()
        comps.algo.update_parameters(exps=exps)
        return {k: max(float((p.detach() - b).abs().max()) for p, b in zip(params[k], bs))
                for k, bs in before.items()}

    moved_first = step(1.0)          # update 0: captures the graphs
    print(f"  update 0 (factor 1.0, captures graphs): {moved_first}", flush=True)

    identities = {
        "policy": all(g["lr"] is t for g, t in
                      zip(comps.algo.optimizer.param_groups, ramps["policy"].tensors)),
        "world-model": all(g["lr"] is t for g, t in
                           zip(comps.predictiveNet.optimizer.param_groups,
                               ramps["world-model"].tensors)),
    }
    print(f"  ramp tensors still installed after the capturable rebuild: {identities}")

    moved = step(1.0)
    frozen = step(0.0)
    thawed = step(1.0)
    failures = 0
    for who in params:
        ok_pos = moved[who] > 0 and thawed[who] > 0
        ok_zero = frozen[who] == 0.0
        failures += (not ok_pos) + (not ok_zero) + (not identities[who])
        print(f"  {who:<12} max |dw| at factor 1.0 = {moved[who]:.3e} / {thawed[who]:.3e} "
              f"({'moves' if ok_pos else 'FAIL: did not move'}), "
              f"at factor 0.0 = {frozen[who]:.3e} "
              f"({'FROZEN - the ramp is live' if ok_zero else 'FAIL: ramp ignored'})")
    print("LIVE GATE:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return int(failures > 0)


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------
def build_cfg(*, seed: int, n_updates: int):
    """Hydra config for one run, sized so the loop is exactly `n_updates`."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=BASE_OVERRIDES + [f"exp.seed={seed}"],
        )
    # `run=multienv` sets a 240,000-update episodes_total. This script drives
    # its own loop, so that number only reaches TrainingSchedule - which prints
    # the run's gradient-step budget at startup. Derive it from n_updates so
    # the printed budget is the one actually run.
    cfg.rl.episodes_total = n_updates * (int(cfg.rl.frames) // int(cfg.predNet.seqdur))
    return cfg


def run_arm(*, arm_name: str, seed: int, n_updates: int, out: Path, spatial: bool) -> None:
    from curious_george.training.schedule import TrainingSchedule
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    arm = ARMS[arm_name]
    cfg = build_cfg(seed=seed, n_updates=n_updates)
    device = get_device()

    t_setup = time.time()
    comps = setup_training(cfg)
    schedule = TrainingSchedule.from_config(cfg)
    print(schedule.summary(), flush=True)

    policy_lr = RampedLearningRate(comps.algo.optimizer, device=device)
    wm_lr = RampedLearningRate(comps.predictiveNet.optimizer, device=device)
    print(f"arm={arm_name} seed={seed} updates={n_updates}\n"
          f"  policy lr {policy_lr.base} warmed over "
          f"{int(round(arm.policy_frac * n_updates))} updates\n"
          f"  world-model lr {wm_lr.base} warmed over "
          f"{int(round(arm.wm_frac * n_updates))} updates", flush=True)
    setup_seconds = time.time() - t_setup

    per_update: dict[str, list[float]] = {
        k: [] for k in ("entropy", "value", "policy_loss", "value_loss", "grad_norm",
                        "return_mean", "curious_mean", "loc_entropy", "seconds",
                        "policy_factor", "wm_factor")
    }
    t_loop = time.time()
    for update in range(n_updates):
        pf = Arm.factor(arm.policy_frac, update=update, n_updates=n_updates)
        wf = Arm.factor(arm.wm_frac, update=update, n_updates=n_updates)
        policy_lr.set_factor(pf)
        wm_lr.set_factor(wf)

        t0 = time.time()
        exps, collect_logs = comps.algo.collect_experiences()
        update_logs = comps.algo.update_parameters(exps=exps)
        seconds = time.time() - t0

        for key in ("entropy", "value", "policy_loss", "value_loss", "grad_norm"):
            per_update[key].append(float(update_logs[key]))
        returns = collect_logs["return_per_episode"]
        per_update["return_mean"].append(float(np.mean(returns)) if returns else np.nan)
        per_update["curious_mean"].append(float(np.mean(collect_logs["curious_rewards"])))
        per_update["loc_entropy"].append(float(collect_logs["loc_entropy"]))
        per_update["seconds"].append(seconds)
        per_update["policy_factor"].append(pf)
        per_update["wm_factor"].append(wf)

        if update % 25 == 0 or update == n_updates - 1:
            loss = comps.predictiveNet.TrainingSaver["loss"]
            print(f"  update {update:4d}  wm_steps={len(loss):6d}  "
                  f"wm_loss={float(loss.iloc[-1]):.6f}  "
                  f"entropy={update_logs['entropy']:.4f}  "
                  f"lr_factor policy={pf:.3f} wm={wf:.3f}  {seconds:.2f}s", flush=True)
    loop_seconds = time.time() - t_loop

    metrics: dict[str, float] = {}
    if spatial:
        from curious_george.evaluation.spatial import evaluate_spatial_representation

        t0 = time.time()
        # TWO fixed probes: the second costs ~5 s (the first pays a one-off CPU
        # recompile of the compiled cell) and gives a WITHIN-run probe-noise
        # estimate, which is the floor any between-arm difference must clear.
        scored = [
            evaluate_spatial_representation(
                comps.predictiveNet, comps.env, comps.ac_agent,
                n_trajs=int(cfg.exp.eval_trajs), traj_timesteps=int(cfg.predNet.seqdur),
                sleepstd=0.03, probe_seed=probe,
            )
            for probe in (PROBE_SEED, PROBE_SEED + 1)
        ]
        si = np.asarray(scored[0]["SI"]["SI"], dtype=float)
        metrics = {"sRSA": float(scored[0]["sRSA"]),
                   "sRSA_probe2": float(scored[1]["sRSA"]),
                   "SWdist": float(scored[0]["SWdist"]),
                   "meanSI": float(np.nanmean(si))}
        print(f"  spatial ({time.time() - t0:.1f}s): "
              + "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()), flush=True)

    wm_loss = comps.predictiveNet.TrainingSaver["loss"].to_numpy(dtype=np.float64)
    meta = {
        "arm": arm_name, "arm_spec": asdict(arm), "seed": seed, "n_updates": n_updates,
        "overrides": BASE_OVERRIDES, "commit": _git_commit(),
        "wm_steps_per_update": schedule.world_model_steps_per_update,
        "policy_steps_per_update": schedule.policy_steps_per_update,
        "frames_per_update": schedule.frames_per_update,
        "policy_lr_base": policy_lr.base, "wm_lr_base": wm_lr.base,
        "setup_seconds": setup_seconds, "loop_seconds": loop_seconds,
        "spatial": metrics,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{arm_name}_seed{seed}.npz"
    np.savez(path, wm_loss=wm_loss, meta=json.dumps(meta),
             **{k: np.asarray(v, dtype=np.float64) for k, v in per_update.items()})
    print(f"wrote {path}  ({len(wm_loss)} world-model steps, loop {loop_seconds / 60:.1f} min)")


def probe(*, updates: int, seed: int) -> None:
    """Throughput probe: how many updates fit in the wall-clock budget."""
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    cfg = build_cfg(seed=seed, n_updates=updates)
    t0 = time.time()
    comps = setup_training(cfg)
    device = get_device()
    RampedLearningRate(comps.algo.optimizer, device=device).set_factor(1.0)
    RampedLearningRate(comps.predictiveNet.optimizer, device=device).set_factor(1.0)
    setup_seconds = time.time() - t0
    print(f"setup: {setup_seconds:.1f}s", flush=True)

    times = []
    for update in range(updates):
        t = time.time()
        exps, _ = comps.algo.collect_experiences()
        comps.algo.update_parameters(exps=exps)
        times.append(time.time() - t)
        print(f"  update {update:3d}: {times[-1]:.3f}s", flush=True)
    steady = np.median(times[3:]) if len(times) > 3 else np.median(times)
    frames = int(cfg.rl.frames)
    print(f"\nsetup {setup_seconds:.1f}s | steady-state {steady:.3f}s/update "
          f"= {frames / steady:,.0f} env steps/s")
    for minutes in (8, 9, 10):
        print(f"  {minutes} min of loop -> {int(minutes * 60 / steady)} updates")


def diagnose(*, updates: int, seed: int) -> None:
    """Is the harness training a REAL representation, or only lowering a loss?

    The repo lost a cluster run to a CUDA graph reading stranded memory, whose
    signature is exactly "loss falls, sRSA never rises"
    (docs/claude_logs/speed-30min-2026-08-23.md, graphed sRSA 0.0238 against an
    eager 0.5158). So loss alone cannot clear this harness, and neither can
    sRSA alone - a low sRSA could equally be a mis-called eval. This measures
    all four things that separate those cases:

      floor         sRSA of the UNTRAINED network, same eval, same probe
      loss curve    TrainingSaver, exact per gradient step
      sRSA          the eval this script uses (fixed probe)
      sRSA, loop    `run_spatial_analysis`, i.e. what training itself calls

    If floor ~= trained sRSA while the loss falls, the representation is not
    forming and the harness is void. If they differ only between the two eval
    calls, the eval call is the bug, not the training.
    """
    from curious_george.evaluation.spatial import evaluate_spatial_representation
    from curious_george.training.loop import run_spatial_analysis
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    cfg = build_cfg(seed=seed, n_updates=updates)
    comps = setup_training(cfg)
    device = get_device()
    RampedLearningRate(comps.algo.optimizer, device=device).set_factor(1.0)
    RampedLearningRate(comps.predictiveNet.optimizer, device=device).set_factor(1.0)

    def score(tag: str, *, probe: int | None) -> float:
        t = time.time()
        m = evaluate_spatial_representation(
            comps.predictiveNet, comps.env, comps.ac_agent,
            n_trajs=int(cfg.exp.eval_trajs), traj_timesteps=int(cfg.predNet.seqdur),
            sleepstd=0.03, probe_seed=probe,
        )
        print(f"  {tag:<34} sRSA={m['sRSA']:+.5f}  SWdist={m['SWdist']:.5f}  "
              f"({time.time() - t:.0f}s)", flush=True)
        return float(m["sRSA"])

    print("\n--- NEGATIVE CONTROL: untrained network ---", flush=True)
    floor = [score(f"untrained, probe {p}", probe=p) for p in (PROBE_SEED, PROBE_SEED + 1)]

    print(f"\n--- training {updates} updates ---", flush=True)
    for update in range(updates):
        exps, _ = comps.algo.collect_experiences()
        comps.algo.update_parameters(exps=exps)
        if update % 25 == 0 or update == updates - 1:
            loss = comps.predictiveNet.TrainingSaver["loss"]
            print(f"  update {update:4d}  wm_steps={len(loss):6d}  "
                  f"loss(last 256)={float(loss.iloc[-256:].mean()):.6f}", flush=True)

    loss = comps.predictiveNet.TrainingSaver["loss"].to_numpy(dtype=np.float64)
    print(f"\n--- world-model loss, mean of the 256 steps ending at each count ---")
    for m in (256, 1000, 2000, 4000, 8000, 16000, 32000, len(loss)):
        if m <= len(loss):
            print(f"  {m:6d} steps : {loss[max(0, m - 256):m].mean():.6f}")

    print(f"\n--- TRAINED, {len(loss)} world-model gradient steps ---", flush=True)
    trained = [score(f"trained, probe {p}", probe=p) for p in (PROBE_SEED, PROBE_SEED + 1)]
    score("trained, no fixed probe", probe=None)
    print("  run_spatial_analysis (what the training loop calls):", flush=True)
    run_spatial_analysis(cfg, comps, wandb_log=False)

    print(f"\nVERDICT INPUTS: untrained sRSA {floor}, trained sRSA {trained}, "
          f"loss {loss[:256].mean():.6f} -> {loss[-256:].mean():.6f}")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _window_mean(loss: np.ndarray, *, step: int, width: int) -> float:
    """Mean world-model loss over the `width` gradient steps ending at `step`.

    A single per-step loss is one minibatch, so a point comparison of two arms
    is a comparison of two minibatches. The window is the smallest honest unit.
    """
    if step > len(loss):
        return np.nan
    return float(np.mean(loss[max(0, step - width):step]))


def report(*, out: Path, width: int) -> None:
    runs: dict[str, list[tuple[int, np.ndarray, dict]]] = {}
    for path in sorted(out.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        runs.setdefault(meta["arm"], []).append((meta["seed"], data["wm_loss"], meta))
    if not runs:
        print(f"no .npz under {out}")
        return

    n_steps = min(len(loss) for arm in runs.values() for _, loss, _ in arm)
    # The two warmup windows are milestones in their own right: they are where
    # each ramp reaches the configured lr, so "did it cost anything, and had it
    # paid back" are questions about those exact counts.
    any_meta = next(iter(runs.values()))[0][2]
    per_update = any_meta["wm_steps_per_update"]
    boundaries = {int(round(f * any_meta["n_updates"])) * per_update
                  for f in (0.05, 0.15)}
    milestones = sorted(
        {m for m in (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000) if m <= n_steps}
        | {b for b in boundaries if 0 < b <= n_steps}
        | {n_steps}
    )

    order = [a for a in ARMS if a in runs]
    print(f"\nWORLD-MODEL LOSS, mean over the {width} gradient steps ending at each count")
    print(f"(matched counts; every arm ran >= {n_steps} world-model gradient steps)\n")
    header = f"{'arm':<10}{'seeds':>6}" + "".join(f"{m:>15,}" for m in milestones)
    print(header)
    print("-" * len(header))
    baseline = {}
    for arm in order:
        seeds = sorted(s for s, _, _ in runs[arm])
        cells = []
        for m in milestones:
            vals = np.array([_window_mean(loss, step=m, width=width) for _, loss, _ in runs[arm]])
            cells.append(vals)
            if arm == "none":
                baseline[m] = vals
        print(f"{arm:<10}{len(seeds):>6}"
              + "".join(f"{v.mean():>15.5f}" for v in cells))
        print(f"{'  +/- sd':<10}{'':>6}" + "".join(f"{v.std(ddof=1):>15.5f}" for v in cells))
        print(f"{'  min-max':<10}{'':>6}"
              + "".join(f"{v.min():.4f}-{v.max():.4f}".rjust(15) for v in cells))

    if "none" in runs:
        print("\nDELTA vs `none` (negative = warmup has LOWER loss = better),"
              " with the baseline's own seed spread for scale\n")
        head = f"{'arm':<10}" + "".join(f"{m:>15,}" for m in milestones)
        print(head)
        print("-" * len(head))
        for arm in order:
            if arm == "none":
                continue
            cells = []
            for m in milestones:
                vals = np.array([_window_mean(loss, step=m, width=width)
                                 for _, loss, _ in runs[arm]])
                delta = vals.mean() - baseline[m].mean()
                spread = max(baseline[m].std(ddof=1), vals.std(ddof=1))
                cells.append(f"{delta:+.5f}{'*' if abs(delta) > spread else ' '}")
            print(f"{arm:<10}" + "".join(c.rjust(15) for c in cells))
        print("\n* = |delta| exceeds the larger of the two arms' seed standard deviations.")
        print(f"baseline seed sd at each count: "
              + "  ".join(f"{m:,}:{baseline[m].std(ddof=1):.5f}" for m in milestones))

    if "none" in runs:
        # PAIRED by seed. Every arm ran the SAME seed set, and most of the
        # run-to-run variation is the seed's own trajectory, which cancels in
        # a per-seed difference. Comparing two independent means throws that
        # cancellation away and is the weaker test.
        base = {s: loss for s, loss, _ in runs["none"]}
        print("\nPAIRED DELTA vs `none`, same seed (mean of the 3 per-seed differences"
              " +/- their sd)\n")
        head = f"{'arm':<10}" + "".join(f"{m:>15,}" for m in milestones)
        print(head)
        print("-" * len(head))
        for arm in order:
            if arm == "none":
                continue
            cells = []
            for m in milestones:
                diffs = np.array([
                    _window_mean(loss, step=m, width=width)
                    - _window_mean(base[seed], step=m, width=width)
                    for seed, loss, _ in runs[arm] if seed in base
                ])
                sd = diffs.std(ddof=1)
                mark = "*" if abs(diffs.mean()) > 2 * sd and (diffs > 0).all() | (diffs < 0).all() else " "
                cells.append(f"{diffs.mean():+.5f}+/-{sd:.5f}{mark}")
            print(f"{arm:<10}" + "".join(c.rjust(15) for c in cells))
        print("\n* = every seed moved the same way AND |mean| > 2 sd of the paired"
              " differences.")

    print("\nPOLICY-SIDE and SPATIAL, mean +/- sd across seeds (whole-run means)\n")
    cols = ("entropy", "policy_loss", "value_loss", "grad_norm", "curious_mean", "loc_entropy")
    print(f"{'arm':<10}" + "".join(f"{c:>15}" for c in cols) + f"{'sRSA':>17}")
    print("-" * (10 + 15 * len(cols) + 17))
    for arm in order:
        rows = []
        for path in sorted(out.glob(f"{arm}_seed*.npz")):
            data = np.load(path, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            rows.append(({c: float(np.nanmean(data[c])) for c in cols},
                         meta["spatial"].get("sRSA", np.nan)))
        cells = []
        for c in cols:
            v = np.array([r[0][c] for r in rows])
            cells.append(f"{v.mean():.4f}+/-{v.std(ddof=1):.4f}")
        srsa = np.array([r[1] for r in rows], dtype=float)
        print(f"{arm:<10}" + "".join(x.rjust(15) for x in cells)
              + f"{srsa.mean():.4f}+/-{np.nanstd(srsa, ddof=1):.4f}".rjust(17))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate")
    sub.add_parser("livegate")
    d = sub.add_parser("diagnose")
    d.add_argument("--updates", type=int, default=200)
    d.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("probe")
    p.add_argument("--updates", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--updates", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--no-spatial", action="store_true")
    a = sub.add_parser("report")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    if args.cmd == "gate":
        raise SystemExit(gate())
    if args.cmd == "livegate":
        raise SystemExit(live_gate())
    if args.cmd == "diagnose":
        diagnose(updates=args.updates, seed=args.seed)
    elif args.cmd == "probe":
        probe(updates=args.updates, seed=args.seed)
    elif args.cmd == "run":
        run_arm(arm_name=args.arm, seed=args.seed, n_updates=args.updates,
                out=args.out, spatial=not args.no_spatial)
    else:
        report(out=args.out, width=args.width)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""THROWAWAY: does CRITIC BURN-IN improve EARLY training?

Critic burn-in = for the first K updates, train only the value head and
suppress the policy-gradient term, so the critic is a decent baseline before
the actor starts following its advantages.

Nothing under curious_george/ is modified. The intervention is a `loss_fn`
handed to the already-loss-agnostic updater: `rl/update/updater.py` resolves
`loss_fn` from `algo.loss_name`, and `algo.update_parameters` hands the SAME
object to `GraphPolicyTrainer`, so one assignment covers both the eager and
the CUDA-graph policy paths.

`policy_coef` is a 0-dim DEVICE TENSOR, not a Python float, and that is
load-bearing: `GraphPolicyTrainer` bakes `loss_kwargs` into a captured graph
once, so a float would freeze at its capture-time value. A tensor is read at
its address on every replay, so `.fill_()` flips burn-in on and off.
The same mechanism is a live BUG elsewhere in the repo, found while writing
this: `algo.py:353` builds `loss_kwargs` as Python floats ONCE, at graph
construction on the first update, and `policy_graph.py:120` bakes them into the
capture - so under `rl.cuda_graph=True` the `rl.entropy_coef_final` ramp
(`training/schedule.py::EntropySchedule`, applied at `loop.py:153`) is silently
dead from update 1 on. `slurm/train_fast.sh` offers both flags together
(lines 312 and 316). Not fixed here: this task is scoped to throwaway/.

Two subcommands, deliberately separate (collection never analyses):
    run     - train ONE arm x ONE seed, write <out>/<arm>_seed<N>.npz
    report  - read every .npz in <out> and print the cross-seed table

Run from the repository root (`Configs/` is resolved against the cwd, as every
other script in this repo does).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# The intervention
# ---------------------------------------------------------------------------


def scaled_ppo_clip_loss(
    dist,
    value,
    sb,
    *,
    clip_eps: float,
    entropy_coef: float,
    value_loss_coef: float,
    policy_coef: torch.Tensor,
):
    """`losses.ppo_clip_loss` with the policy term scaled by a LIVE scalar.

    `policy_coef` is a 0-dim tensor: 1.0 reproduces `ppo_clip_loss` bitwise
    (gated by `gate_loss_identity`), 0.0 removes the policy gradient entirely
    and leaves a pure value-function regression.

    The body is duplicated from `curious_george/rl/update/losses.py` rather
    than wrapped, because the combined loss there gives no differentiable
    handle on its policy term. The duplication is checked, not assumed:
    `gate_loss_identity` fails the run if the two ever disagree.
    """
    from curious_george.rl.update.losses import _LOG2, LossTerms

    policy_entropy = dist.entropy().mean()

    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
    surr1 = ratio * sb.advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * sb.advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    value_clipped = sb.value + torch.clamp(value - sb.value, -clip_eps, clip_eps)
    surr1 = (value - sb.returnn).pow(2)
    surr2 = (value_clipped - sb.returnn).pow(2)
    value_loss = torch.max(surr1, surr2).mean()

    loss = (
        policy_coef * policy_loss
        - entropy_coef * policy_entropy
        + value_loss_coef * value_loss
    )

    terms = LossTerms(
        policy_entropy_bits=policy_entropy.detach() / _LOG2,
        value_mean=value.detach().mean(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
    )
    return loss, terms


def gate_loss_identity(*, device: torch.device) -> None:
    """Fail loudly unless policy_coef=1 reproduces `ppo_clip_loss` bitwise."""
    from torch.distributions.categorical import Categorical
    from torch_ac.utils import DictList

    from curious_george.rl.update.losses import ppo_clip_loss

    g = torch.Generator(device="cpu").manual_seed(0)
    n, a = 64, 4
    logits = torch.randn(n, a, generator=g).to(device).requires_grad_(True)
    dist = Categorical(logits=torch.log_softmax(logits, dim=1))
    value = torch.randn(n, generator=g).to(device).requires_grad_(True)
    sb = DictList({
        "action": torch.randint(0, a, (n,), generator=g).to(device),
        "log_prob": torch.randn(n, generator=g).to(device),
        "advantage": torch.randn(n, generator=g).to(device),
        "returnn": torch.randn(n, generator=g).to(device),
        "value": torch.randn(n, generator=g).to(device),
    })
    kw = dict(clip_eps=0.2, entropy_coef=0.013, value_loss_coef=0.7)

    ref_loss, ref_terms = ppo_clip_loss(dist, value, sb, **kw)
    one = torch.ones((), device=device)
    got_loss, got_terms = scaled_ppo_clip_loss(dist, value, sb, policy_coef=one, **kw)
    assert torch.equal(ref_loss, got_loss), (ref_loss.item(), got_loss.item())
    for f in fields(ref_terms):
        r, o = getattr(ref_terms, f.name), getattr(got_terms, f.name)
        assert torch.equal(r, o), (f.name, r.item(), o.item())

    zero = torch.zeros((), device=device)
    off_loss, _ = scaled_ppo_clip_loss(dist, value, sb, policy_coef=zero, **kw)
    (grad,) = torch.autograd.grad(off_loss, logits, retain_graph=True)
    # entropy_coef != 0 here, so the logits still get the ENTROPY gradient;
    # what must vanish is the difference the advantages make.
    sb2 = DictList(dict(sb))
    sb2.advantage = sb.advantage * 3.0
    off2, _ = scaled_ppo_clip_loss(dist, value, sb2, policy_coef=zero, **kw)
    (grad2,) = torch.autograd.grad(off2, logits)
    assert torch.equal(grad, grad2), "policy_coef=0 still lets advantages reach the actor"
    print("gate: scaled_ppo_clip_loss(policy_coef=1) == ppo_clip_loss, bitwise; "
          "policy_coef=0 blocks the advantage gradient")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

# ⚠️ `exp.device_env=True` makes `logs["return_per_episode"]` a hard-coded 0.0
# (collector.py:231 - "this backend rejects environment rewards"), so extrinsic
# return is NOT a measurement in this config and is deliberately not recorded.
# The agent's actual objective here is the curiosity reward, which is.
OVERRIDES = [
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


@dataclass
class UpdateRecord:
    """One training update. Every field is a scalar; arrays go in the npz."""

    update: int
    env_steps: int
    policy_grad_steps: int
    wm_grad_steps: int
    burn_in_active: int  # 0/1, so it survives the float array round-trip
    value_loss: float
    policy_loss: float
    entropy_bits: float
    grad_norm: float
    wm_loss_mean: float  # mean of the TrainingSaver rows this update appended
    value_mean: float
    # 1 - Var(returnn - value) / Var(returnn), on the ROLLOUT's own values.
    # `exps.returnn = value + advantage` (collector.py:627), so this is exactly
    # the fraction of return variance the critic explained AT THE MOMENT it
    # produced these advantages - the most direct read of the burn-in claim,
    # and unlike value_loss it is scale-free.
    explained_variance: float
    advantage_abs_mean: float
    curious_reward_mean: float
    loc_entropy: float
    loc_entropy_5: float
    actor_max_delta: float  # max |w - w_init| over the actor head
    critic_max_delta: float  # ... over the critic head
    seconds: float


def _flat(module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def gpu_neighbours() -> list[str]:
    """Other processes holding this GPU, recorded so the timings stay honest.

    Sibling experiments share this box; measured 1.17 s/update alone against
    3.76 s/update with three neighbours. Nothing REPORTED here is wall-clock
    indexed - every metric is a function of gradient-step number - but a
    throughput figure without this list would be a lie.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return [line.strip() for line in out]


def build_config(*, seed: int, updates: int):
    from hydra import compose, initialize_config_dir

    cfg_dir = Path("Configs").resolve()
    if not cfg_dir.is_dir():
        raise SystemExit(f"run me from the repository root; no {cfg_dir}")
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="main", overrides=OVERRIDES + [f"exp.seed={seed}"])

    from curious_george.training.schedule import TrainingSchedule

    probe = TrainingSchedule.from_config(cfg)
    cfg.rl.episodes_total = updates * probe.episodes_per_update
    return cfg


def run_arm(*, seed: int, burn_in_updates: int, updates: int, out: Path, srsa: bool) -> Path:
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from curious_george.training.schedule import TrainingSchedule
    from curious_george.training.setup import setup_training
    from curious_george.utils.common import get_device

    device = get_device()
    if device.type != "cuda":
        raise SystemExit(f"this experiment's config is CUDA-only (got {device})")
    gate_loss_identity(device=device)

    cfg = build_config(seed=seed, updates=updates)
    schedule = TrainingSchedule.from_config(cfg)
    print(schedule.summary())
    print(f"burn-in: {burn_in_updates}/{updates} updates "
          f"({burn_in_updates * schedule.policy_steps_per_update} policy grad steps suppressed)")

    t_setup = time.time()
    comps = setup_training(cfg)
    algo, pN = comps.algo, comps.predictiveNet

    # THE INTERVENTION. `algo.loss_name` is what both policy paths resolve.
    policy_coef = torch.ones((), device=device)
    from functools import partial

    algo.loss_name = partial(scaled_ppo_clip_loss, policy_coef=policy_coef)

    actor_init, critic_init = _flat(algo.acmodel.actor), _flat(algo.acmodel.critic)
    setup_seconds = time.time() - t_setup
    print(f"setup+compile: {setup_seconds:.1f} s", flush=True)

    neighbours_start = gpu_neighbours()
    print(f"GPU compute apps at start ({len(neighbours_start)}): {neighbours_start}", flush=True)

    records: list[UpdateRecord] = []
    t_loop = time.time()
    for update in range(updates):
        burn_in = update < burn_in_updates
        policy_coef.fill_(0.0 if burn_in else 1.0)

        t0 = time.time()
        wm_rows_before = len(pN.TrainingSaver)
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps=exps)
        wm_loss = pN.TrainingSaver["loss"].values[wm_rows_before:]

        values, advantages = logs1["values"], logs1["advantages"]
        returnn = values + advantages

        records.append(UpdateRecord(
            update=update,
            env_steps=(update + 1) * schedule.frames_per_update,
            policy_grad_steps=(update + 1) * schedule.policy_steps_per_update,
            wm_grad_steps=len(pN.TrainingSaver),
            burn_in_active=int(burn_in),
            value_loss=logs2["value_loss"],
            policy_loss=logs2["policy_loss"],
            entropy_bits=logs2["entropy"],
            grad_norm=logs2["grad_norm"],
            wm_loss_mean=float(np.mean(wm_loss)) if len(wm_loss) else math.nan,
            value_mean=logs2["value"],
            explained_variance=float(1.0 - advantages.var() / returnn.var()),
            advantage_abs_mean=float(np.abs(advantages).mean()),
            curious_reward_mean=float(logs1["curious_rewards"].mean()),
            loc_entropy=float(logs1["loc_entropy"]),
            loc_entropy_5=float(logs1["loc_entropy_5"]),
            actor_max_delta=float((_flat(algo.acmodel.actor) - actor_init).abs().max()),
            critic_max_delta=float((_flat(algo.acmodel.critic) - critic_init).abs().max()),
            seconds=time.time() - t0,
        ))
        if update % 20 == 0 or update == updates - 1:
            r = records[-1]
            print(f"[{update:4d}] {'BURN-IN' if burn_in else '       '} "
                  f"wm_loss={r.wm_loss_mean:.5f} value_loss={r.value_loss:.5f} "
                  f"expl_var={r.explained_variance:+.4f} "
                  f"actor_delta={r.actor_max_delta:.3e} "
                  f"loc_H={r.loc_entropy:.3f} {r.seconds:.2f}s", flush=True)
    loop_seconds = time.time() - t_loop
    # Update 0 pays the one-off torch.compile + CUDA-graph capture, so the
    # throughput of the run is the rest.
    steady_seconds = sum(r.seconds for r in records[1:])

    # --- sanity: did the burn-in actually happen? --------------------------
    burn_in_rows = [r for r in records if r.burn_in_active]
    sanity = {
        # The actor is a SEPARATE head from the critic in ACModelSR (no shared
        # trunk when exp.with_obs=False), so a suppressed policy term must
        # leave it bitwise untouched under Adam with zero grad and no
        # weight decay. Anything but 0.0 here means the patch did not take.
        "actor_max_delta_at_end_of_burn_in":
            burn_in_rows[-1].actor_max_delta if burn_in_rows else None,
        "critic_max_delta_at_end_of_burn_in":
            burn_in_rows[-1].critic_max_delta if burn_in_rows else None,
        # Positive control that the check CAN fire: the actor moves by the
        # first non-burn-in update in every arm, including the baseline.
        "actor_max_delta_first_free_update":
            next((r.actor_max_delta for r in records if not r.burn_in_active), None),
        "actor_max_delta_final": records[-1].actor_max_delta,
        "critic_max_delta_final": records[-1].critic_max_delta,
    }
    print("sanity:", json.dumps(sanity, indent=2), flush=True)
    if burn_in_rows:
        assert sanity["actor_max_delta_at_end_of_burn_in"] == 0.0, "burn-in did NOT freeze the actor"
        assert sanity["critic_max_delta_at_end_of_burn_in"] > 0.0, "critic did not train during burn-in"
    assert sanity["actor_max_delta_final"] > 0.0, "actor never trained"

    # `training/loop.py::run_spatial_analysis` PRINTS its metrics and returns
    # None, so calling it would leave the number in scrollback instead of in
    # the artifact. This calls the same function it calls on its on-policy
    # branch, with the same arguments read from the same cfg keys.
    spatial = {}
    if srsa:
        from curious_george.evaluation.spatial import evaluate_spatial_representation

        assert cfg.exp.onpolicy_prnn_eval and not cfg.exp.random_action_agent
        t0 = time.time()
        metrics = evaluate_spatial_representation(
            pN, comps.env, comps.ac_agent, sleepstd=0.03, wandb_nameext="_onPolicy",
            n_trajs=cfg.exp.get("eval_trajs", 8),
            traj_timesteps=cfg.predNet.seqdur,
            trainDecoder=cfg.exp.get("eval_decoder", False),
            legacy_timesteps=cfg.exp.get("eval_timesteps", 15000),
            # NOT what run_spatial_analysis does: it leaves the eval rollout's
            # start position free, so every arm's score carries its own rollout
            # noise. Fixing it removes that noise source from an ACROSS-ARM
            # comparison; the trajectories still differ, because the policy does.
            probe_seed=0,
        )
        spatial = {k: float(v) for k, v in metrics.items() if np.isscalar(v)}
        spatial["SI_mean"] = float(np.mean(metrics["SI"]))
        spatial["seconds"] = time.time() - t0
        print("spatial:", json.dumps(spatial, indent=2), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    arm = "baseline" if burn_in_updates == 0 else f"burnin{burn_in_updates}"
    path = out / f"{arm}_seed{seed}.npz"
    meta = dict(
        arm=arm,
        seed=seed,
        burn_in_updates=burn_in_updates,
        updates=updates,
        policy_steps_per_update=schedule.policy_steps_per_update,
        wm_steps_per_update=schedule.world_model_steps_per_update,
        frames_per_update=schedule.frames_per_update,
        overrides=OVERRIDES,
        setup_seconds=setup_seconds,
        loop_seconds=loop_seconds,
        steady_seconds=steady_seconds,
        env_steps_per_second=(updates - 1) * schedule.frames_per_update / steady_seconds,
        gpu_neighbours_start=neighbours_start,
        gpu_neighbours_end=gpu_neighbours(),
        sanity=sanity,
        spatial=spatial,
        git_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
    )
    columns = {
        f.name: np.array([getattr(r, f.name) for r in records], dtype=np.float64)
        for f in fields(UpdateRecord)
    }
    np.savez(
        path,
        meta=json.dumps(meta),
        wm_loss_per_step=pN.TrainingSaver["loss"].values.astype(np.float64),
        **columns,
    )
    print(f"wrote {path}  ({loop_seconds / 60:.1f} min loop, "
          f"{meta['env_steps_per_second']:.0f} env steps/s)")
    return path


# ---------------------------------------------------------------------------
# Analysis (reads npz only; collects nothing)
# ---------------------------------------------------------------------------


@dataclass
class ArmSeries:
    arm: str
    seed: int
    burn_in_updates: int
    meta: dict
    wm_loss_per_step: np.ndarray
    columns: dict[str, np.ndarray]


def load(out: Path) -> list[ArmSeries]:
    series = []
    for path in sorted(out.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        series.append(ArmSeries(
            arm=meta["arm"], seed=meta["seed"], burn_in_updates=meta["burn_in_updates"],
            meta=meta, wm_loss_per_step=z["wm_loss_per_step"],
            columns={k: z[k] for k in z.files if k not in ("meta", "wm_loss_per_step")},
        ))
    return series


def _window_mean(x: np.ndarray, end: int, width: int) -> float:
    return float(np.mean(x[max(0, end - width): end]))


def report(out: Path, checkpoints: list[float], window_updates: int) -> None:
    series = load(out)
    if not series:
        raise SystemExit(f"no .npz under {out}")
    burn_in_of = {s.arm: s.burn_in_updates for s in series}
    arms = sorted(burn_in_of, key=burn_in_of.__getitem__)
    updates = min(len(s.columns["update"]) for s in series)
    wm_per_update = int(series[0].meta["wm_steps_per_update"])
    pol_per_update = int(series[0].meta["policy_steps_per_update"])

    print(f"\narms: {arms}")
    for a in arms:
        ss = [s for s in series if s.arm == a]
        print(f"  {a:12s} n={len(ss)} seeds={sorted(s.seed for s in ss)} "
              f"burn_in_updates={ss[0].burn_in_updates} "
              f"loop_min={np.mean([s.meta['loop_seconds'] for s in ss]) / 60:.1f}")
    print(f"  matched on {updates} updates = {updates * pol_per_update} policy "
          f"and {updates * wm_per_update} world-model gradient steps")
    print(f"  window = trailing {window_updates} updates\n")

    def col(name):
        return lambda s, u: _window_mean(s.columns[name], u, window_updates)

    metrics = [
        ("world_model_loss", lambda s, u: _window_mean(
            s.wm_loss_per_step, u * wm_per_update, window_updates * wm_per_update)),
        ("value_loss", col("value_loss")),
        ("explained_variance", col("explained_variance")),
        ("curious_reward_mean", col("curious_reward_mean")),
        ("policy_loss", col("policy_loss")),
        ("policy_entropy_bits", col("entropy_bits")),
        ("location_entropy", col("loc_entropy")),
        ("advantage_abs_mean", col("advantage_abs_mean")),
    ]
    marks = [max(window_updates, int(round(f * updates))) for f in checkpoints]

    for name, fn in metrics:
        print(f"=== {name} ===")
        for u in marks:
            print(f"  at update {u} ({u * pol_per_update} policy / "
                  f"{u * wm_per_update} world-model gradient steps)")
            per_arm = {a: np.array([fn(s, u) for s in series if s.arm == a]) for a in arms}
            base = per_arm.get("baseline")
            for a in arms:
                v = per_arm[a]
                seeds = "[" + " ".join(f"{x:.5f}" for x in v) + "]"
                line = f"    {a:10s} {v.mean():>10.5f} +/-{v.std():<9.5f} seeds {seeds}"
                if base is not None and a != "baseline" and len(base) > 1:
                    d = v.mean() - base.mean()
                    # "Detectable" = the arms' means are further apart than the
                    # WIDER of the two seed spreads. With n=3 that is the honest
                    # bar; anything smaller is not distinguishable from seed noise.
                    spread = max(base.max() - base.min(), v.max() - v.min())
                    line += (f"  delta {d:+.5f}  "
                             f"{'INSIDE' if abs(d) <= spread else 'OUTSIDE'} "
                             f"seed spread {spread:.5f}")
                print(line)
        print()

    # PAIRED view. A seed fixes the initial weights and every environment
    # stream, so arm-vs-baseline WITHIN a seed removes the variance that the
    # blocks above have to absorb. With n=3 the strongest statement available
    # is "all three seeds agree in sign" (sign test p=0.25 one-tailed), which
    # is suggestive and nothing more - so it is reported as sign agreement,
    # not as significance.
    print("=== paired per-seed delta vs the SAME seed's baseline ===")
    by_seed = {(s.arm, s.seed): s for s in series}
    for name, fn in metrics:
        print(f"  {name}")
        for a in arms:
            if a == "baseline":
                continue
            for u in marks:
                d = [fn(by_seed[(a, sd)], u) - fn(by_seed[("baseline", sd)], u)
                     for sd in sorted(s.seed for s in series if s.arm == a)
                     if ("baseline", sd) in by_seed]
                signs = {np.sign(x) for x in d}
                agree = "all agree" if len(signs) == 1 else "SIGNS DISAGREE"
                print(f"    {a:10s} u={u:<4d} deltas ["
                      + " ".join(f"{x:+.5f}" for x in d)
                      + f"] mean {np.mean(d):+.5f}  {agree}")
        print()

    keys = sorted({k for s in series for k in s.meta.get("spatial", {}) if k != "seconds"})
    if keys:
        print(f"=== end-of-run spatial representation ({updates * wm_per_update} "
              f"world-model gradient steps) ===")
        for k in keys:
            print(f"  {k}")
            for a in arms:
                vals = np.array([s.meta["spatial"][k] for s in series if s.arm == a])
                print(f"    {a:12s} {vals.mean():.5f} +/-{vals.std():.5f}  "
                      f"seeds {np.round(vals, 5).tolist()}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train one arm x one seed")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--burn-in-updates", type=int, required=True,
                   help="K: updates with the policy-gradient term suppressed (0 = baseline)")
    r.add_argument("--updates", type=int, required=True)
    r.add_argument("--out", type=Path, default=Path("throwaway/critic_burn_in_out"))
    r.add_argument("--srsa", action="store_true", help="one spatial analysis at the end")

    p = sub.add_parser("report", help="table across every npz in --out")
    p.add_argument("--out", type=Path, default=Path("throwaway/critic_burn_in_out"))
    p.add_argument("--checkpoints", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--window-updates", type=int, default=20)

    a = ap.parse_args()
    if a.cmd == "run":
        run_arm(seed=a.seed, burn_in_updates=a.burn_in_updates, updates=a.updates,
                out=a.out, srsa=a.srsa)
    else:
        report(a.out, a.checkpoints, a.window_updates)


if __name__ == "__main__":
    main()

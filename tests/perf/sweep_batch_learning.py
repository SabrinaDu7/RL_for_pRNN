"""Does a larger pooled world-model batch reduce pRNN loss faster PER GRADIENT STEP?

`sweep_trainstep_batch.py` measured cost only and found that on GPU a batch of
8-128 costs the same as a batch of 1. That flat region is only useful if a
bigger batch also produces a BETTER gradient. This script measures that.

Design: every configuration takes exactly ONE world-model gradient step per
update (frames = num_envs * seqdur gives one 256-step segment per env, and
batched_wm pools them). So `num_envs` IS the world-model batch size, and update
count IS gradient-step count. Configurations are compared at matched gradient
steps; wall-clock is recorded as the secondary axis.

Confound to keep in mind: num_envs=1 cannot use the device-resident env
(`exp.device_env` requires >1), so its wall-clock is not comparable to the
others. Its loss-per-gradient-step is.

Usage:
    uv run python tests/perf/sweep_batch_learning.py --batches 8,32,128 --updates 1500
    uv run python tests/perf/sweep_batch_learning.py --batches 1,8 --updates 20   # calibration
"""

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]

BASE_OVERRIDES = [
    "logging.wandb_log=false",
    "logging.save_every_steps=0",
    "logging.analysis_every_steps=0",
    "logging.plot_every_steps=0",
    "logging.log_every_steps=1000000000",
]


def build_overrides(
    *, num_envs: int, seqdur: int, device_env: bool, pool: bool, extra: list[str]
) -> list[str]:
    """Config for one arm: one 256-step segment per env.

    pool=True  -> one pooled gradient step of batch num_envs per update.
    pool=False -> num_envs SERIAL batch-1 gradient steps per update (the old
                  dynamics), but still with the fast device-resident rollout.
    """
    pooled = pool and num_envs > 2  # train_world_model_on_episodes needs >2 segments
    return [
        *BASE_OVERRIDES,
        f"exp.num_envs={num_envs}",
        f"rl.frames={num_envs * seqdur}",
        f"exp.device_env={'true' if (device_env and num_envs > 1) else 'false'}",
        f"predNet.batched_wm={'true' if pooled else 'false'}",
        f"predNet.batched_curiosity={'true' if num_envs > 1 else 'false'}",
        # PPO minibatch must not exceed the rollout
        f"rl.ppo_batch_size={min(256 * num_envs, 4096)}",
        *extra,
    ]


def run_arm(
    *, num_envs: int, updates: int, seqdur: int, device_env: bool, pool: bool, extra: list[str]
) -> dict:
    """Train one arm for `updates` updates; return per-gradient-step loss + timing."""
    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=build_overrides(
                num_envs=num_envs, seqdur=seqdur, device_env=device_env, pool=pool, extra=extra
            ),
        )

    from curious_george.training.setup import setup_training

    comps = setup_training(cfg)
    algo, pN = comps.algo, comps.predictiveNet

    n_before = len(pN.TrainingSaver)
    update_times: list[float] = []
    wm_steps: list[int] = []

    start = time.perf_counter()
    for _ in range(updates):
        t0 = time.perf_counter()
        exps, _ = algo.collect_experiences()
        algo.update_parameters(exps=exps, update_params=True)
        update_times.append(time.perf_counter() - t0)
        wm_steps.append(len(pN.TrainingSaver) - n_before)
    total = time.perf_counter() - start

    losses = [float(v) for v in pN.TrainingSaver["loss"].to_numpy()[n_before:]]
    frames_per_update = num_envs * seqdur
    return {
        "arm": f"B{num_envs}" + ("" if pool else "-serial"),
        "num_envs": num_envs,
        "wm_batch": num_envs if pool else 1,
        "pool": pool,
        "updates": updates,
        "grad_steps": len(losses),
        "grad_steps_per_update": wm_steps[-1] / updates if updates else 0,
        "frames_per_update": frames_per_update,
        "total_s": total,
        "s_per_update_median": sorted(update_times)[len(update_times) // 2],
        "grad_steps_per_sec": len(losses) / total,
        "fps": updates * frames_per_update / total,
        "device_env": bool(device_env and num_envs > 1),
        "batched_wm": pool and num_envs > 2,
        "losses": losses,
        "update_times_s": update_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,32,128", help="pooled arms")
    parser.add_argument(
        "--serial-batches",
        default="",
        help="arms with batched_wm=false: num_envs SERIAL batch-1 steps/update "
        "(old dynamics) on top of the device-resident rollout",
    )
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--seqdur", type=int, default=256)
    parser.add_argument("--no-device-env", action="store_true")
    parser.add_argument("--out", default="tests/perf/results/batch_learning_sweep.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    arms_spec = [(int(b), True) for b in args.batches.split(",") if b]
    arms_spec += [(int(b), False) for b in args.serial_batches.split(",") if b]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for B, pool in arms_spec:
        label = f"num_envs={B}" + ("" if pool else " (serial WM)")
        print(f"\n=== {label}  ({args.updates} updates) ===", flush=True)
        arm = run_arm(
            num_envs=B,
            updates=args.updates,
            seqdur=args.seqdur,
            device_env=not args.no_device_env,
            pool=pool,
            extra=args.override,
        )
        results.append(arm)
        print(
            f"  grad_steps={arm['grad_steps']} "
            f"({arm['grad_steps_per_update']:.2f}/update)  "
            f"{arm['s_per_update_median']:.3f}s/update  "
            f"{arm['grad_steps_per_sec']:.2f} steps/s  "
            f"fps={arm['fps']:.0f}",
            flush=True,
        )
        if arm["losses"]:
            first = arm["losses"][0]
            last = sum(arm["losses"][-10:]) / len(arm["losses"][-10:])
            print(f"  loss {first:.5f} -> {last:.5f} (mean of last 10)", flush=True)

        out_path.write_text(
            json.dumps(
                {
                    "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__,
                    "seqdur": args.seqdur,
                    "updates_requested": args.updates,
                    "arms": results,
                },
                indent=2,
            )
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

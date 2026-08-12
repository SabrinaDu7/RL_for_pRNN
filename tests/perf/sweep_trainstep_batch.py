"""Wall-clock cost of ONE pRNN ``trainStep`` as a function of batch size.

Answers: how many world-model gradient steps per second can each device
sustain, and how much does a larger pooled batch actually cost? The 256-step
recurrence is sequential, so per-step time is expected to be launch-bound
(nearly flat) at small batch and only become compute-bound at large batch.

This measures COST ONLY. It says nothing about how much a larger batch
improves the gradient - that needs a learning run.

Usage:
    uv run python tests/perf/sweep_trainstep_batch.py --device cuda
    uv run python tests/perf/sweep_trainstep_batch.py --device cpu --batches 1,8,32
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]


def build_prnn(device: torch.device, overrides: list[str]):
    """Build the project's real pRNN via the ordinary hydra + setup path."""
    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)

    from curious_george.training.setup import setup_env, setup_world_model

    env = setup_env(cfg, seed_offset=0)
    pN = setup_world_model(cfg, env, wandb_log=False)
    pN.pRNN.to(device)  # same placement call PRNNAdapter.to makes
    return cfg, pN


def make_batch(
    *, pN, B: int, L: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic obs/act with the exact dtypes and shapes the batched world-model
    path builds in ``PRNNAdapter.train_on_episodes_batched``.

    obs: (B, L+1, X) float32 in [0,1]; act: (B, L, num_acts+num_hd) int64 one-hot
    over head direction plus a binary forward/speed channel.
    """
    shell = pN.env_shell
    num_acts = shell.action_space.n
    num_hd = shell.numHDs
    obs_dim = pN.obs_size

    obs = torch.rand((B, L + 1, obs_dim), dtype=torch.float32, device=device)
    act = torch.zeros((B, L, num_acts + num_hd), dtype=torch.int64, device=device)
    directions = torch.randint(0, num_hd, (B, L), device=device)
    act.scatter_(2, (num_acts + directions).unsqueeze(-1), 1)
    act[:, :, 2] = (torch.rand((B, L), device=device) < 0.25).to(torch.int64)
    return obs, act


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_batch(
    *, pN, B: int, L: int, device: torch.device, warmup: int, reps: int
) -> dict:
    """Median/mean seconds for one full trainStep (fwd + bwd + RMSprop) at batch B."""
    obs, act = make_batch(pN=pN, B=B, L=L, device=device)

    for _ in range(warmup):
        pN.trainStep(obs, act, batched=True)
    sync(device)

    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        pN.trainStep(obs, act, batched=True)
        sync(device)
        samples.append(time.perf_counter() - start)

    samples.sort()
    mean = sum(samples) / len(samples)
    median = samples[len(samples) // 2]
    peak = (
        torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    )
    return {
        "batch": B,
        "seconds_median": median,
        "seconds_mean": mean,
        "seconds_min": samples[0],
        "seconds_max": samples[-1],
        "steps_per_sec": 1.0 / median,
        "trajs_per_sec": B / median,
        "peak_gib": peak,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", default="1,8,32,128,1024")
    parser.add_argument("--seqdur", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    batches = [int(b) for b in args.batches.split(",")]

    cfg, pN = build_prnn(device, args.override)
    print(
        f"device={device} pRNNtype={cfg.predNet.pRNNtype} "
        f"hidden={cfg.predNet.hiddensize} L={args.seqdur} "
        f"obs_size={pN.obs_size} act_size={pN.act_size} reps={args.reps}"
    )
    print(f"{'B':>6} {'s/step':>10} {'steps/s':>9} {'trajs/s':>10} {'ms/traj':>9} {'GiB':>6}")

    rows = []
    for B in batches:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            row = time_batch(
                pN=pN, B=B, L=args.seqdur, device=device,
                warmup=args.warmup, reps=args.reps,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>6} {'OOM':>10}")
            torch.cuda.empty_cache()
            continue
        rows.append(row)
        gib = f"{row['peak_gib']:.2f}" if row["peak_gib"] is not None else "-"
        print(
            f"{B:>6} {row['seconds_median']:>10.4f} {row['steps_per_sec']:>9.2f} "
            f"{row['trajs_per_sec']:>10.1f} {1000 * row['seconds_median'] / B:>9.3f} {gib:>6}"
        )

    if args.out:
        payload = {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "seqdur": args.seqdur,
            "pRNNtype": cfg.predNet.pRNNtype,
            "hiddensize": cfg.predNet.hiddensize,
            "reps": args.reps,
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

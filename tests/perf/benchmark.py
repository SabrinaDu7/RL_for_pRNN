"""Perf + metric benchmark: run N training updates, record stage timings and
learning metrics, dump JSON. This is the gate for performance refactors
(replaces bitwise golden comparison where RNG order changes - see
docs/perf_baseline.md).

Usage:
    uv run python tests/perf/benchmark.py --updates 5 --out results/baseline.json
    CUDA_VISIBLE_DEVICES="" uv run python tests/perf/benchmark.py ...   # CPU run
    ... --profile results/baseline.prof                                # cProfile
    ... --include-plot --include-analysis        # also time the periodic blocks
    ... --override exp.num_envs=4                # any hydra override

Compare two runs with tests/perf/compare_metrics.py.
"""

import argparse
import cProfile
import datetime
import importlib.metadata
import json
import os
import platform
import pstats
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

BASE_OVERRIDES = [
    "logging.wandb_log=false",
    "logging.save_interval=0",
    "logging.analysis_interval=0",
    "logging.log_interval=1000000",  # skip periodic logging inside run loop
]


def tensor_stats(x) -> dict:
    return {"mean": float(x.mean()), "std": float(x.std()), "absmax": float(x.abs().max())}


def package_git_commit(package: str) -> str | None:
    """Installed direct-url VCS revision, when the package came from git."""
    try:
        direct_url = importlib.metadata.distribution(package).read_text("direct_url.json")
        metadata = json.loads(direct_url or "{}")
        return metadata.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return None


def run_benchmark(
    cfg,
    n_updates: int,
    include_plot: bool,
    include_analysis: bool,
    sync_stages: bool = False,
    warmup_updates: int = 0,
) -> dict:
    import torch

    from curious_george.training.setup import setup_training
    from curious_george.utils.common import DEVICE
    from curious_george.utils.timing import timer

    timer.enabled = True
    # follows the device actually in use, not mere availability
    timer.sync_cuda = sync_stages

    comps = setup_training(cfg)
    algo = comps.algo
    for _ in range(warmup_updates):
        warm_exps, _ = algo.collect_experiences()
        algo.update_parameters(exps=warm_exps)
    timer.reset()
    training_saver_start = len(comps.predictiveNet.TrainingSaver)

    metrics: dict = {
        "policy_loss": [], "value_loss": [], "grad_norm": [], "entropy": [],
        "curious_mean": [], "curious_std": [],
        "values_mean": [], "advantages_mean": [], "advantages_std": [],
        "num_episodes": [], "prnn_loss": [],
    }
    update_times = []

    t_start = time.perf_counter()
    for _ in range(n_updates):
        t0 = time.perf_counter()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps=exps)
        update_times.append(time.perf_counter() - t0)

        logs = {**logs1, **logs2}
        metrics["policy_loss"].append(logs["policy_loss"])
        metrics["value_loss"].append(logs["value_loss"])
        metrics["grad_norm"].append(logs["grad_norm"])
        metrics["entropy"].append(logs["entropy"])
        cr = torch.as_tensor(logs["curious_rewards"])
        metrics["curious_mean"].append(float(cr.mean()))
        metrics["curious_std"].append(float(cr.std()))
        v = torch.as_tensor(logs["values"])
        adv = torch.as_tensor(logs["advantages"])
        metrics["values_mean"].append(float(v.mean()))
        metrics["advantages_mean"].append(float(adv.mean()))
        metrics["advantages_std"].append(float(adv.std()))
        metrics["num_episodes"].append(logs["num_episodes"])
    total_train_s = time.perf_counter() - t_start

    # per-trainStep pRNN losses accumulated by prnn's TrainingSaver
    ts = comps.predictiveNet.TrainingSaver
    if len(ts) and "loss" in ts:
        metrics["prnn_loss"] = [
            float(x) for x in ts["loss"].iloc[training_saver_start:].tolist()
        ]

    if include_plot:
        from curious_george.training.loop import on_device
        with timer("log/sample_trajectory"):
            with on_device([comps.predictiveNet, comps.acmodel], "cpu"):
                comps.predictiveNet.plotSampleTrajectory(env=comps.env, agent=comps.ac_agent)

    if include_analysis:
        from curious_george.training.loop import run_spatial_analysis
        with timer("analysis/spatial"):
            run_spatial_analysis(cfg, comps, wandb_log=False)

    frames = n_updates * cfg.rl.frames
    return {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "git": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                capture_output=True, text=True).stdout.strip(),
            # the device actually used (CG_DEVICE-controlled, bound at import
            # in utils.common), NOT merely what is available - a CPU run on a
            # GPU box used to self-label "cuda" and corrupt the perf record.
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "synchronized_stage_timing": sync_stages,
            "cuda_device": (
                torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None
            ),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            },
            "dependencies": {
                "torch": torch.__version__,
                "prnn_git": package_git_commit("prnn"),
            },
            "n_updates": n_updates,
            "warmup_updates": warmup_updates,
            "frames_per_update": cfg.rl.frames,
            "num_envs": cfg.exp.num_envs,
            "seed": cfg.exp.seed,
        },
        "fps": round(frames / total_train_s, 1),
        "total_train_s": round(total_train_s, 3),
        "update_times_s": [round(t, 3) for t in update_times],
        "timings": timer.report(),
        "metrics": metrics,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--profile", type=str, default=None, help="also write cProfile stats here")
    p.add_argument("--include-plot", action="store_true")
    p.add_argument("--include-analysis", action="store_true")
    p.add_argument(
        "--sync-stages",
        action="store_true",
        help="synchronize CUDA/MPS at timer boundaries for attribution; perturbs FPS",
    )
    p.add_argument(
        "--warmup-updates",
        type=int,
        default=0,
        help="unmeasured updates before timing (useful for accelerator compilation)",
    )
    p.add_argument("--override", action="append", default=[], help="extra hydra overrides")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # plotSampleTrajectory calls plt.show()

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        cfg = compose(config_name="main", overrides=BASE_OVERRIDES + args.override)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        Path(args.profile).parent.mkdir(parents=True, exist_ok=True)
        prof = cProfile.Profile()
        prof.enable()
        result = run_benchmark(
            cfg,
            args.updates,
            args.include_plot,
            args.include_analysis,
            args.sync_stages,
            args.warmup_updates,
        )
        prof.disable()
        prof.dump_stats(args.profile)
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
    else:
        result = run_benchmark(
            cfg,
            args.updates,
            args.include_plot,
            args.include_analysis,
            args.sync_stages,
            args.warmup_updates,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nFPS: {result['fps']}  (updates: {result['update_times_s']})")
    print(json.dumps(result["timings"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Count CUDA runtime API calls per sequential step in one rollout.

The census behind docs/claude_logs/exp_speed_cuda_graph_2026-08-19.md §8: which of
cudaLaunchKernel / cudaMemcpyAsync / cudaStreamSynchronize the rollout actually
spends its CPU on. Pair with find_sync.py, which attributes *data-dependent*
host syncs to a Python line; this one counts the CUDA runtime API instead, and
the two disagreeing is the finding (269 cudaStreamSynchronize, zero of them
data-dependent, 0.2% of CPU).

Run from the repo root, on an idle GPU:
    uv run python tests/perf/profile_api.py env=lroom_multi run=multienv
"""
import os, sys
os.environ["CG_TIMING"] = "0"
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from hydra import compose, initialize_config_dir

overrides = ["logging.wandb_log=False", "logging.save_every_steps=0",
             "logging.analysis_every_steps=0", "logging.plot_every_steps=0",
             "logging.log_every_steps=1000000000"] + sys.argv[1:]
with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
    cfg = compose(config_name="main", overrides=overrides)

from curious_george.training.setup import setup_training
comps = setup_training(cfg)
algo = comps.algo
T = cfg.rl.frames // cfg.exp.num_envs
print(f"config: env={cfg.exp.env_name} device_env={cfg.exp.get('device_env')} "
      f"num_envs={cfg.exp.num_envs} frames={cfg.rl.frames} T={T} "
      f"batched_wm={cfg.predNet.get('batched_wm')} cuda_graph={cfg.predNet.get('cuda_graph')}")

e, _ = algo.collect_experiences(); algo.update_parameters(exps=e)   # warm
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    e, _ = algo.collect_experiences()
    torch.cuda.synchronize()
ev = prof.key_averages()

API = ("cudaStreamSynchronize", "cudaLaunchKernel", "cudaMemcpyAsync",
       "cudaDeviceSynchronize", "cudaMemcpy", "cudaStreamIsCapturing",
       "cudaFuncGetAttributes", "cudaMalloc", "cudaFree", "cudaEventQuery",
       "cudaEventRecord", "cudaGraphLaunch")
print(f"\n=== CUDA runtime API, ONE rollout of T={T} sequential steps ===")
for name in API:
    hits = [x for x in ev if x.key == name]
    if hits:
        n = sum(h.count for h in hits)
        ms = sum(h.self_cpu_time_total for h in hits) / 1e3
        print(f"  {name:<26}{n:>8}  = {n/T:>7.2f}/step   {ms:>8.1f} ms cpu")
tot_cpu = sum(x.self_cpu_time_total for x in ev) / 1e3
tot_gpu = sum(x.self_device_time_total for x in ev) / 1e3
n_ops = sum(x.count for x in ev)
print(f"\n  self CPU {tot_cpu:.1f} ms   self CUDA {tot_gpu:.1f} ms   "
      f"ratio {tot_cpu/max(tot_gpu,1e-9):.1f}x")
print(f"  aten+api invocations {n_ops:,} = {n_ops/T:,.0f}/step")

"""Do `predNet.compile_cell` and `predNet.cuda_graph` compose on the POOLED step?

They attack the same cost by different means - compile fuses the 256-step loop
into fewer, bigger kernels; the graph removes the dispatch and launch of
whatever kernels remain - so neither one's speedup predicts the pair's. This
times `trainStep(batched=True)` in all four combinations.

Steady state is the point. `throwaway/hydra_era/perf/benchmark.py --updates 4` puts
torch.compile's ONE-TIME compilation inside `update/wm_train` and capture
warmup inside the graphed arm, which made the graph look worth 1.16x when the
steady-state stage figure is 4.19x. This script warms up before timing.

Committed rather than left in throwaway/ because
docs/claude_logs/compaction-2026-08-22-speed.md 10 quotes its table.

    uv run python tests/perf/sweep_graph_compile.py

Prints a table; writes nothing. CUDA only.
"""

import time

import numpy as np
import torch

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames, PredictiveNet
from curious_george import AgentInputType, make_env
from curious_george.models.prnn_adapter import _GraphWMTrainer, PRNNAdapter

SEED, L, GROUP, H, REPS = 5, 256, 8, 500, 15
dev = torch.device("cuda")


def build(compile_cell):
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                   act_enc=ActionEncodingsEnum.SpeedHD.value, seed=SEED)
    pN = PredictiveNet(env, hidden_size=H, pRNNtype="thRNN_5win",
                       trainNoiseMeanStd=(0.0, 0.05), dropp=0.1, learningRate=3e-3,
                       weight_decay=3e-3, bptttrunc=int(1e8), wandb_log=False)
    pN.pRNN.to(dev); pN.pRNN.train()
    ad = PRNNAdapter(pN, dev, action_offset=0, cuda_graph=True, compile_cell=compile_cell)
    rows_o, rows_a = [], []
    for k in range(GROUP):
        torch.manual_seed(SEED + 1 + k); np.random.seed(SEED + 1 + k)
        sh = pN.env_shell
        obs_d, acts = [sh.reset()], []
        for _ in range(L):
            a = int(np.random.randint(0, 4)); acts.append(a)
            obs_d.append(sh.step(np.array([a]))[0])
        imgs = torch.stack([torch.tensor(np.asarray(o["image"]), dtype=torch.float) for o in obs_d[:-1]])
        dirs = torch.tensor([o["direction"] for o in obs_d[:-1]])
        o, a = ad._episode_tensors(imgs, dirs, np.asarray(acts), obs_d[L])
        rows_o.append(o); rows_a.append(a)
    return pN, ad, torch.stack(rows_o).to(dev), torch.stack(rows_a).to(dev)


def timeit(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


print(f"pooled step, group={GROUP} L={L} h={H}, {REPS} timed reps\n")
print(f"{'compile_cell':<14}{'graphed':<9}{'ms/step':>10}")
for compile_cell in (False, "layer"):
    for graphed in (False, True):
        pN, ad, obs_b, act_b = build(compile_cell)
        if graphed:
            tr = _GraphWMTrainer(pN, dev)
            fn = lambda: tr.train_batch(obs_b, act_b, batched=True)
        else:
            fn = lambda: pN.trainStep(obs_b, act_b, batched=True, return_stats=False)
        print(f"{str(compile_cell):<14}{str(graphed):<9}{timeit(fn, REPS):>10.2f}")
        del pN, ad, obs_b, act_b
        torch.cuda.empty_cache()

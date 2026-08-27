#!/usr/bin/env bash
# Cluster health check: reproduce local4060-tyro-e0.001_curious_26-08-26-23-45-37
# on a Mila GPU, against today's code.
#
# Every field is copied from that run's wandb config, so a difference in the
# curves is a difference in the CODE or the HARDWARE, not in the setup:
#   collect     device backend, num_envs 256, episodes_per_env 1, seqdur 256,
#               rollout_cuda_graph
#   prnn        43,936 grad steps, 8 episodes each, compile=layer, cuda_graph,
#               batched, batched_curiosity, curiosity_cuda_graph OFF
#   policy      175,744 grad steps, entropy 0.001, cuda_graph
#   eval        analysis every 3,333,328 env steps; plot every 7,499,989
#   run         seed 2, save/archive every 8,388,608
#
# PYTHONUNBUFFERED so the sRSA lines are readable while it runs, not only on exit.
set -eu
cd ~/experiments/RL_for_pRNN
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "head $(git log --oneline -1)"

uv run python main_train.py reference \
  --collect.backend DEVICE --collect.num-envs 256 --collect.episodes-per-env 1 \
  --collect.episode-steps 256 --collect.rollout-cuda-graph \
  --train-prnn.batched --train-prnn.batched-curiosity \
  --train-prnn.episodes-per-grad-step 8 --train-prnn.compile LAYER \
  --train-prnn.cuda-graph --train-prnn.no-curiosity-cuda-graph \
  --train-prnn.total-grad-steps 43936 \
  --train-policy.total-grad-steps 175744 --train-policy.entropy-coef 0.001 \
  --train-policy.cuda-graph \
  --eval.analysis-every-steps 3333328 --eval.plot-every-steps 7499989 \
  --run.seed 2 --run.save-every-steps 8388608 --run.archive-every-steps 8388608 \
  --run.exp-name mila-parity-e0.001

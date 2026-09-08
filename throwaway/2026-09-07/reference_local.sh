#!/bin/bash
# The original L-room MSE run (the `reference` preset = the July 2026 config the
# questions repo's task checkpoint came from), re-trained locally on the dev
# box while the cluster CPU job waits for a sapphire node.
cd /home/sabrina/Documents/experiments/RL_for_pRNN
export PYTHONUNBUFFERED=1 CG_DEVICE=cuda
L=throwaway/2026-09-07/fresh_runs.log
echo "$(date '+%H:%M:%S') start reference-local (preset reference, seed 2, wandb curious-george)" >> $L
uv run python main_train.py reference --run.seed 2 --run.exp-name pRNN-clean-local > throwaway/2026-09-07/reference_local.log 2>&1
echo "reference-local exit $? $(date '+%H:%M:%S')" >> $L

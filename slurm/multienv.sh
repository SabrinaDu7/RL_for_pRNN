#!/bin/bash
#SBATCH --job-name=multienv
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --time=4-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Multi-room pRNN training. Usage:
#   sbatch slurm/multienv.sh rooms      # run 1: the frozen three rooms
#   sbatch slurm/multienv.sh pool       # run 2: the 500-layout seeded pool
#
# GPU on `long`, against train_prnn.sh's CPU/sapphire choice. Two reasons, and
# the second is the practical one:
#
#   1. That verdict (async_bench_10111153: CUDA 558 FPS with the GPU ~91% idle
#      vs 674 FPS on 16 CPUs; sapphire ~1270 FPS) was measured for SERIAL
#      world-model training, where the bill is op COUNT. Pooling cuts trainStep
#      calls 8x per update, which is exactly the quantity that verdict rested
#      on. Locally the pooled path runs at ~2400 FPS on an RTX 4060.
#   2. The CPU partitions are unusable right now - checked 2026-08-13, every
#      sapphire node is reserved, draining or allocated, so a 16-CPU job queues
#      indefinitely. `long` has 108 nodes in mixed state with free GPUs.
#
# So this is the available choice, not a re-measured one. FPS is logged; read it
# from the run rather than trusting either number above.
#
# 4-12:00:00 against long's 7-day limit: 240,000 pooled gradient steps is 491.5M
# environment steps. The run is EXPECTED to be cut off by the wall clock, which
# is why archive_every_steps exists - the step-tagged series under
# <run>/checkpoints/ is the deliverable, not a finished run.

set -eo pipefail   # NOT -u: WANDB_API_KEY is optional, creds come from ~/.netrc
LAYOUTS="${1:-rooms}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')   layouts=$LAYOUTS"

module --force purge
module load python/3.10
export PATH="$HOME/.local/bin:$PATH"

# Cap torch/BLAS threads to the allocation: without this torch spawns one
# thread per PHYSICAL core while the cgroup grants fewer, and update times go
# from 4s to 51s (async_bench_10110503).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${LAYOUTS}_${SLURM_JOB_ID}"
export CG_DEVICE=cuda
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p "$RL_STORAGE"

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv sync

# ~/.netrc already carries the credentials; the env var is an optional override.
wandb login "${WANDB_API_KEY:-}" >/dev/null 2>&1 || true

# Checkpoints live in node-local $SLURM_TMPDIR, which is destroyed when the job
# ends - including when it is killed or times out, which this job is expected
# to be. Sync every 10 min, not 30: at archive_every_steps the series grows
# faster than the old cadence preserved it.
export DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p "$DEST_DIR"
( while true; do sleep 600; rsync -a "$RL_STORAGE/" "$DEST_DIR/"; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; rsync -a "$RL_STORAGE/" "$DEST_DIR/" || true' EXIT

uv run main_train.py \
    env=lroom_multi run=multienv \
    exp.layouts="$LAYOUTS" \
    exp.exp_name="multienv-$LAYOUTS"

kill $SYNC_PID 2>/dev/null || true
rsync -a "$RL_STORAGE/" "$DEST_DIR/"
cp /home/mila/d/dus/scratch/pRNN/logs/$SLURM_JOB_NAME*$SLURM_JOB_ID* "$DEST_DIR/" || true

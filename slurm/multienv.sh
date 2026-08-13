#!/bin/bash
#SBATCH --job-name=multienv
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=4-12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --constraint=sapphire
#SBATCH --exclusive
#
# Multi-room pRNN training. Usage:
#   sbatch slurm/multienv.sh rooms      # run 1: the frozen three rooms
#   sbatch slurm/multienv.sh pool       # run 2: the 500-layout seeded pool
#
# CPU on purpose, inherited from train_prnn.sh: async_bench_10111153 measured
# CUDA at 558 FPS with the GPU ~91% idle against 674 FPS on 16 CPUs, and
# Sapphire Rapids nodes run this at ~1270 FPS. That was measured for SERIAL
# world-model training; pooling cuts trainStep calls 8x per update, which
# changes the dispatch arithmetic that verdict rested on. Locally the pooled
# path reaches ~2360 FPS on an RTX 4060. Treat the CPU choice here as the
# proven default, not as a re-measured one.
#
# 4-12:00:00 against main's 5-day limit: 240,000 pooled gradient steps is
# 491.5M environment steps, ~107 h at 1270 FPS. The run is EXPECTED to be cut
# off by the wall clock, which is why archive_every_steps exists - the
# step-tagged series under <run>/checkpoints/ is the deliverable, not a
# finished run.

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
export CG_DEVICE=cpu
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

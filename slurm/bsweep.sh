#!/bin/bash
# Phase C gate: learning-curve comparison across parallel-env counts.
# Array of 9 CPU jobs: B in {1,4,8} x seeds {2,3,4}, 1500 updates each
# (rl.steps=3072000 frames). Compare with scripts/analysis_bsweep.py; flip
# exp.num_envs default only if the B>1 curves match B=1 across seeds.
#
#SBATCH --job-name=bsweep
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%A_%a.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-8

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

# cap torch/BLAS threads to the allocation - without this torch spawns one
# thread per PHYSICAL core of the node while the cgroup grants fewer,
# causing thrashing (async_bench_10110503: 4-51s update times on 16 cpus)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# task id -> (B, seed)
BS=(1 4 8)
SEEDS=(2 3 4)
B=${BS[$((SLURM_ARRAY_TASK_ID % 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}
echo "num_envs=$B seed=$SEED"

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
export CG_DEVICE=cpu   # same device for every B (and CPU > GPU at this model size)
mkdir -p $RL_STORAGE

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv sync

wandb login $WANDB_API_KEY

export DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR
( while true; do
    sleep 1800
    rsync -a $RL_STORAGE/ $DEST_DIR/
done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

uv run main_train.py \
    exp.exp_name="bsweep-B${B}-s${SEED}" \
    exp.with_obs=False exp.input_type=pRNN \
    exp.num_envs=$B exp.seed=$SEED \
    rl.steps=3072000

kill $SYNC_PID 2>/dev/null
rsync -a $RL_STORAGE/ $DEST_DIR/
mv /home/mila/d/dus/scratch/pRNN/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}* $DEST_DIR/ 2>/dev/null

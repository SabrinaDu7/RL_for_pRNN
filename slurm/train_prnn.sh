#!/bin/bash
#SBATCH --job-name=RL_for_pRNN
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

module --force purge
module load python/3.10

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p $RL_STORAGE

cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

git fetch origin
git switch main

rm -rf .venv
uv venv .venv
source .venv/bin/activate
echo ".venv directory: $PWD"
echo "uv syncing in $PWD"
uv sync

wandb login $WANDB_API_KEY

# Periodically sync tmpdir outputs to scratch so a killed/timed-out job
# doesn't lose checkpoints living only in node-local $SLURM_TMPDIR.
export DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR
( while true; do
    sleep 600
    rsync -a $RL_STORAGE/ $DEST_DIR/
done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

uv run main_train.py exp.exp_name=pRNN exp.with_obs=False exp.input_type=pRNN

# Final sync + move logs to scratch
kill $SYNC_PID 2>/dev/null
rsync -a $RL_STORAGE/ $DEST_DIR/
mv /home/mila/d/dus/scratch/pRNN/logs/$JOB_ID* $DEST_DIR/

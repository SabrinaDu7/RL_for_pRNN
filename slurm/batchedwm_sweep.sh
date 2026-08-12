#!/bin/bash
# Curve gate for predNet.batched_wm: does ONE pooled world-model gradient step
# per update (segments stacked (8, 256)) learn the same as 8 sequential steps?
# 9 CPU jobs: 3 arms x seeds {2,3,4}, 1500 updates each.
#   arm 1 = batched_wm=false            (reference band)
#   arm 2 = batched_wm=true             (pooled step, same lr)
#   arm 3 = batched_wm=true, lr x2      (pooled-step lr compensation probe)
# Run names reuse the bsweep convention (wmsweep-B<arm>-s<seed>) so
# scripts/analysis_bsweep.py works as-is:
#   uv run python scripts/analysis_bsweep.py --prefix wmsweep-B --fig wmsweep.png
# (its "B=1" reference band is arm 1 = the serial baseline).
#
#SBATCH --job-name=wmsweep
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%A_%a.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-8

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CG_DEVICE=cpu

# task id -> (arm, seed)
ARM=$((SLURM_ARRAY_TASK_ID % 3 + 1))
SEEDS=(2 3 4)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}
case $ARM in
    1) WM=false; LR=3e-3 ;;
    2) WM=true;  LR=3e-3 ;;
    3) WM=true;  LR=6e-3 ;;
esac
echo "arm=$ARM batched_wm=$WM predNet.lr=$LR seed=$SEED"

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
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
    exp.exp_name="wmsweep-B${ARM}-s${SEED}" \
    exp.with_obs=False exp.input_type=pRNN \
    exp.seed=$SEED \
    predNet.batched_wm=$WM predNet.lr=$LR \
    rl.episodes_total=12000 # 3,072,000 steps / predNet.seqdur=256

kill $SYNC_PID 2>/dev/null
rsync -a $RL_STORAGE/ $DEST_DIR/
mv /home/mila/d/dus/scratch/pRNN/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}* $DEST_DIR/ 2>/dev/null

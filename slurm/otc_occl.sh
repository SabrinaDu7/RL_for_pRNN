#!/bin/bash
#SBATCH --job-name=OTC_occl
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Occluded object-exposure (scenario C) for ONE object location, swept over seeds.
# The location-control matrix needs one submission per OBJ_LOC; the peak must
# follow the object, otherwise the effect is regional drift rather than coding.
#
#   OBJ_LOC='[12,7]' sbatch --job-name=OTC_occl_12_7 slurm/otc_occl.sh
OBJ_LOC="${OBJ_LOC:-[14,7]}"
SEED_START="${SEED_START:-5300}"; SEED_END="${SEED_END:-5302}"

module --force purge; module load python/3.10
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p $RL_STORAGE

cd $HOME/experiments; cp -r RL_for_pRNN $SLURM_TMPDIR/; cd $SLURM_TMPDIR/RL_for_pRNN
rm -rf .venv; uv venv .venv; source .venv/bin/activate; uv sync
wandb login $WANDB_API_KEY 2>/dev/null

# the occluded baseline this exposure starts from (shipped to scratch)
export CUR_CKPT_DIR="$SCRATCH/pRNN/occl_base"
DEST_DIR="$SCRATCH/pRNN/$JOB_ID"; mkdir -p $DEST_DIR
( while true; do sleep 900; rsync -a $RL_STORAGE/ $DEST_DIR/; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT

for SEED in $(seq $SEED_START $SEED_END); do
  echo "=== obj $OBJ_LOC seed $SEED $(date +%H:%M) ==="
  uv run tasks/otc/main_task.py \
    exp.exp_name=Cmat exp.seed=$SEED exp.see_through_walls=False \
    tasks.new_obj_loc=$OBJ_LOC tasks.otc.presence_prob=1.0 \
    tasks.training.lr_trials=[2,0,8] tasks.training.num_trajs=3000 \
    tasks.training.saving_interval_trajs=3000 \
    logging.wandb_project=curious-george-otc
done
kill $SYNC 2>/dev/null; rsync -a $RL_STORAGE/ $DEST_DIR/
echo "Outputs in $DEST_DIR"

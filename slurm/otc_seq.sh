#!/bin/bash
#SBATCH --job-name=OTC_seq
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Sequential displacement, ONE seed per submission so seeds run in parallel.
# The sequence ends with an empty phase (object REMOVED), so every location is
# scored after the object leaves - the departure test the first attempt lacked.
SEED="${SEED:-5200}"
SEQ="${SEQ:-[[7,11],[7,2],[4,7],[]]}"
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
DEST_DIR="$SCRATCH/pRNN/$JOB_ID"; mkdir -p $DEST_DIR
uv run tasks/otc/main_task.py \
  exp.exp_name=SEQ4 exp.seed=$SEED "tasks.otc.sequence=$SEQ" \
  tasks.training.lr_trials=[2,0,8] tasks.training.num_trajs=4000 \
  tasks.training.saving_interval_trajs=4000 \
  logging.wandb_project=curious-george-otc
rsync -a $RL_STORAGE/ $DEST_DIR/
echo "Outputs in $DEST_DIR"

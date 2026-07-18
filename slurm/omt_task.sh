#!/bin/bash
#SBATCH --job-name=RL_for_pRNN_omt
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# OMT (tasks/omt/main_task.py) runs on GPU (DEVICE hardcoded cuda) and
# historically finishes in ~1h (wandb curious-george-omt); 3h is buffer.

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p $RL_STORAGE

# Input checkpoints: the finished pre-migration training run (07-15).
# Exported env vars take precedence over the repo .env (load_dotenv does
# not override existing environment variables).
CKPT_DIR=$SCRATCH/pRNN/RL_for_pRNN_10129601/pRNN_curious_26-07-15-12-21-12
export PRNN_CUR_CKPT=$CKPT_DIR/predictiveNet_state.pt
export ACMODEL_CUR_CKPT=$CKPT_DIR/status.pt

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

rm -rf .venv
uv venv .venv
source .venv/bin/activate
echo "uv syncing in $PWD"
uv sync

wandb login $WANDB_API_KEY

export DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR

uv run tasks/omt/main_task.py \
    tasks.testing.start_random=False \
    logging.wandb_project=curious-george-omt

# OMT saves nets under ./<run_name>/ (relative save_path) and figures under
# ./results/ - both live in the tmpdir repo copy, so sync them explicitly.
rsync -a $RL_STORAGE/ $DEST_DIR/
for d in *-cur-* *-rand-* results outputs; do
    [ -d "$d" ] && rsync -a "$d" $DEST_DIR/
done
# cp, not mv: SLURM holds --output/--error open while the job runs, so mv
# fails with "Device or resource busy" and its exit code 1 marks the whole
# job FAILED even after a clean run (train_prnn.sh has the same latent bug).
cp /home/mila/d/dus/scratch/pRNN/logs/$JOB_ID* $DEST_DIR/ || true

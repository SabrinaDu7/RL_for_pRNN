#!/bin/bash
#SBATCH --job-name=OMT_cur_start_rand
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Object Memory Task (tasks/omt/main_task.py) - see tasks/README.md.
#
# GPU is REQUIRED: main_task.py hardcodes DEVICE = torch.device("cuda") with
# no CPU fallback. (Opposite of train_prnn.sh, which is CPU-only on purpose
# because there the GPU sat ~91% idle.)
#
# One submission = ONE object location swept over a seed range. The
# three-location experiment is three sbatch calls - see the knobs below.

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------- knobs ----
# Override at submit time, e.g.
#   sbatch --job-name=OMT_cur_obj_14_7 slurm/omt_task.sh
#   sbatch --job-name=OMT_cur_obj_7_2 \
#          --export=ALL,OBJ_LOC='[7,2]',SEED_START=5300,SEED_END=5320 \
#          slurm/omt_task.sh
# Keep the seed ranges disjoint across locations so wandb runs stay separable.
OBJ_LOC="${OBJ_LOC:-[14,7]}"
SEED_START="${SEED_START:-5200}"
SEED_END="${SEED_END:-5220}"
OMT_WANDB_PROJECT="${OMT_WANDB_PROJECT:-curious-george-omt}"

# Input checkpoints: a FINISHED main_train.py run. OMT never trains from
# scratch. CUR_CKPT_DIR is the SAME variable the local .env sets - only the
# value is machine-specific, so no code or config branches on the machine.
# get_ckpt_env_vars derives predictiveNet_state.pt + status.pt from it, and an
# exported var beats the repo .env (load_dotenv does not override).
export CUR_CKPT_DIR="${CUR_CKPT_DIR:-/home/mila/d/dus/scratch/pRNN/RL_for_pRNN_10178850/pRNN_curious_26-07-23-10-06-25}"

echo "OBJ_LOC=$OBJ_LOC  seeds=$SEED_START..$SEED_END"
echo "CUR_CKPT_DIR=$CUR_CKPT_DIR"
for f in predictiveNet_state.pt status.pt; do
    if [ ! -f "$CUR_CKPT_DIR/$f" ]; then
        echo "FATAL: $f missing under $CUR_CKPT_DIR" >&2
        exit 1
    fi
done

# ------------------------------------------------------------ environment --
module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

# Cap torch/BLAS threads to the allocation: without this torch spawns one
# thread per PHYSICAL core of the node while the cgroup grants fewer, which
# thrashes (async_bench_10110503: 4-51s update times on 16 cpus).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p $RL_STORAGE

# pRNN and minigrid are uv git pins (pyproject [tool.uv.sources]) since the
# 2026-07 migration to LevensteinLab/pRNN - no sibling source copies needed.
cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN
pwd
git status --short

rm -rf .venv
uv venv .venv
source .venv/bin/activate
echo "uv syncing in $PWD"
uv sync

wandb login $WANDB_API_KEY

# ------------------------------------------------------------------ sync ---
# OMT writes everything under $RL_STORAGE/omt/<run_name>/<traj_count>/, so one
# root covers it. Sync periodically so a timeout does not lose results living
# only in node-local $SLURM_TMPDIR.
export DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR

sync_outputs() {
    rsync -a $RL_STORAGE/ $DEST_DIR/
}

( while true; do sleep 1800; sync_outputs; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

# ------------------------------------------------------------------- run ---
# saving_interval=1000000 disables intermediate saves; trainNovelObject still
# writes the final checkpoint at (num_batches-1)*trajs_per_batch, which is the
# one the hidden-state analysis consumes.
for SEED in $(seq "$SEED_START" "$SEED_END"); do
    echo "=== seed $SEED  obj_loc $OBJ_LOC  $(date '+%H:%M:%S') ==="
    uv run just omt-start-rand \
        tasks.new_obj_loc=$OBJ_LOC \
        tasks.training.saving_interval=1000000 \
        logging.wandb_project=$OMT_WANDB_PROJECT \
        exp.seed=$SEED
done

# ------------------------------------------------------------ final sync ---
kill $SYNC_PID 2>/dev/null
sync_outputs
# cp, not mv: SLURM holds --output/--error open while the job runs, so mv
# fails with "Device or resource busy" and its exit code 1 marks a clean run
# FAILED (Mila cluster.md known-failure #2).
cp /home/mila/d/dus/scratch/pRNN/logs/$JOB_ID* $DEST_DIR/ || true
echo "Outputs in $DEST_DIR"

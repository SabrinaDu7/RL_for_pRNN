#!/bin/bash
#SBATCH --job-name=RL_for_pRNN
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --constraint=sapphire
#SBATCH --exclusive     # whole node, no neighbours

# CPU-only on purpose: async_bench_10111153 measured CUDA at 558 FPS with the
# GPU ~91% idle vs 674 FPS on 16 CPUs (docs/perf_log.md cluster verdicts).
# sapphire constraint: Sapphire Rapids nodes (cn-m001 class) run this at
# ~1270 FPS -> full 20.5M frames in ~4.5h (run 10129601); Milan (cn-h*) nodes
# measured ~2x slower (~656 FPS, run 10150201) and cn-h001 specifically has
# a chronic noisy-neighbor I/O storm. 15h wall leaves buffer either way.

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

# cap torch/BLAS threads to the allocation - without this torch spawns one
# thread per PHYSICAL core of the node while the cgroup grants fewer,
# causing thrashing (async_bench_10110503: 4-51s update times on 16 cpus)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export CG_DEVICE=cpu
export RL_STORAGE=$SLURM_TMPDIR/outputs/$JOB_ID
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
mkdir -p $RL_STORAGE

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

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
    sleep 1800
    rsync -a $RL_STORAGE/ $DEST_DIR/
done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

uv run main_train.py exp.exp_name=pRNN exp.with_obs=False exp.input_type=pRNN

# Final sync + copy logs to scratch
kill $SYNC_PID 2>/dev/null
rsync -a $RL_STORAGE/ $DEST_DIR/
# cp, not mv: SLURM holds --output/--error open while the job runs, so mv
# fails with "Device or resource busy" -> exit 1 marks a clean run FAILED
# (Mila cluster.md known-failure #2). Fixed in omt_task.sh by ae38bcf.
cp /home/mila/d/dus/scratch/pRNN/logs/$JOB_ID* $DEST_DIR/ || true

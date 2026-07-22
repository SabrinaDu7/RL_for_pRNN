#!/bin/bash
#SBATCH --job-name=RL_for_pRNN
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#
# GPU run with CUDA-graph world-model training (predNet.cuda_graph=true).
#
# WHY THIS FLIPPED FROM CPU-ONLY: the old verdict (CUDA 558 FPS / GPU ~91%
# idle vs 674 FPS on 16 CPUs) was correct for EAGER CUDA - the wm BPTT is
# ~5-10k tiny sequential ops and the per-op kernel-launch tax dominates.
# CUDA graphs replay that whole launch sequence in one submission, which
# removes the tax: locally trainStep 109ms CPU -> 15ms graphed (7.4x), and
# end-to-end 894 -> 1100 FPS. See docs/exp_cuda_graphs_wm.md.
#
# THE CLUSTER NUMBER IS NOT KNOWN YET - that is what this run measures.
# The local box has 8 cores + an RTX 4060; the cluster has 16 cores and a
# different GPU, and its CPU baseline was already ~1270 FPS (run 10129601,
# sapphire). A stronger CPU baseline plus a different launch-latency/compute
# ratio can move the speedup either way. Do not assume 1.23x here.
#
# --constraint=sapphire was DROPPED: it selects Sapphire Rapids CPU nodes and
# no GPU script in this repo combines it with --gres, so keeping it risks an
# unschedulable job. CPU throughput still matters a lot (collect/env_step is
# the largest remaining stage), so RECORD THE NODE - landing on a Milan-class
# node (~2x slower CPU, and cn-h001 has a chronic noisy-neighbor I/O storm)
# confounds the comparison against the sapphire CPU baseline.

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

# cap torch/BLAS threads to the allocation - without this torch spawns one
# thread per PHYSICAL core of the node while the cgroup grants fewer,
# causing thrashing (async_bench_10110503: 4-51s update times on 16 cpus).
# Still needed on GPU: collection and env stepping remain CPU-bound.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

# Device / graph arm, overridable so a matched CPU control is one flag away:
#   sbatch --export=ALL,CG_DEVICE=cpu,CUDA_GRAPH=false slurm/train_prnn.sh
# (the control still allocates a GPU it will not use - acceptable for a
# one-off; drop --gres by hand if that matters).
# CG_DEVICE=cuda with no visible GPU now fails LOUDLY at import rather than
# silently running on CPU (curious_george/utils/common.py).
export CG_DEVICE="${CG_DEVICE:-cuda}"
CUDA_GRAPH="${CUDA_GRAPH:-true}"
echo "CG_DEVICE=$CG_DEVICE  predNet.cuda_graph=$CUDA_GRAPH"

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

# Record the resolved prnn commit: the CUDA-graph path REQUIRES the
# capture-safe masking fix (float-mask multiplies in clip_mask/predict) on
# branch sdu/rl-integration. Without it, capture aborts with
# "operation not permitted when stream is capturing".
uv run python -c "import prnn, torch; from prnn.utils.Architectures import pRNN; \
print('prnn Option-B capture fix present:', hasattr(pRNN, '_tile_mask')); \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

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

# Sample GPU utilization so the speed result is interpretable: graphs make the
# GPU efficient WHILE it works, but the duty cycle across the whole loop is
# low (locally ~12% average), so a small mean here is expected, not a fault.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader -l 30 > $RL_STORAGE/gpu_util.csv 2>/dev/null &
GPU_PID=$!

trap 'kill $SYNC_PID $GPU_PID 2>/dev/null' EXIT

# NOTE on dynamics: cuda_graph keeps ONE optimizer step per segment (same
# step count and math as the serial CPU path), so learning curves should
# MATCH the CPU run - but NOT bitwise. Dropout/noise are drawn inside the
# graph, so the RNG realization differs while the distributions do not.
# Compare curves (pRNN loss, curious reward, SI/sRSA), not exact values.
# NOTE on resume: the graphed path rebuilds the optimizer as capturable and
# asserts empty optimizer state, so it is FRESH RUNS ONLY. Do not add
# logging.load_worldmodel/load_acmodel to this command without addressing that.
uv run main_train.py exp.exp_name=pRNN exp.with_obs=False exp.input_type=pRNN \
    predNet.cuda_graph=$CUDA_GRAPH

# Final sync + copy logs to scratch
kill $SYNC_PID $GPU_PID 2>/dev/null
rsync -a $RL_STORAGE/ $DEST_DIR/
# cp, not mv: SLURM holds --output/--error open while the job runs, so mv
# fails with "Device or resource busy" and its exit code 1 marks the whole
# job FAILED even after a clean run (fixed in omt_task.sh by ae38bcf; this
# script had the same latent bug).
cp /home/mila/d/dus/scratch/pRNN/logs/$JOB_ID* $DEST_DIR/ || true

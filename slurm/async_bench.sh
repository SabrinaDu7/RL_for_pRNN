#!/bin/bash
# Cluster benchmark: async vs sync rollout collection at B=8 on a wide CPU
# node (the 8-core dev box showed only +6-9% for async because workers
# contended with torch threads - docs/perf_log.md Phase C1). 16 cores give
# the 8 env workers their own cores. Compares FPS + per-stage timings; JSONs
# land in $SCRATCH/pRNN/<job>/ - inspect with tests/perf/compare_metrics.py.
#
#SBATCH --job-name=async_bench
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CPUs: $SLURM_CPUS_PER_TASK  Node: $(hostname)"

module --force purge
module load python/3.10

export PATH="$HOME/.local/bin:$PATH"

# cap torch/BLAS threads to the allocation - without this torch spawns one
# thread per PHYSICAL core of the node while the cgroup grants fewer,
# causing thrashing (async_bench_10110503: 4-51s update times on 16 cpus)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
export CG_DEVICE=cpu

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN

rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv sync

OUT=$SLURM_TMPDIR/bench_results
mkdir -p $OUT

# 5 updates each for stabler numbers than the dev box's 3
for MODE in true false; do
    TAG=$([ "$MODE" = "true" ] && echo async || echo sync)
    echo "=== B=8 $TAG ==="
    uv run python tests/perf/benchmark.py --updates 5 \
        --override exp.async_envs=$MODE \
        --out $OUT/b8_${TAG}.json 2>&1 | grep -E "^FPS|collect/env_step" -A2
done

# bonus: B=4 async vs sync (fewer workers, longer per-env streams)
for MODE in true false; do
    TAG=$([ "$MODE" = "true" ] && echo async || echo sync)
    echo "=== B=4 $TAG ==="
    uv run python tests/perf/benchmark.py --updates 5 \
        --override exp.num_envs=4 --override exp.async_envs=$MODE \
        --out $OUT/b4_${TAG}.json 2>&1 | grep -E "^FPS|collect/env_step" -A2
done

echo "=== metric gate: async vs sync must PASS (bitwise-equivalent) ==="
uv run python tests/perf/compare_metrics.py $OUT/b8_sync.json $OUT/b8_async.json | tail -3

echo "=== B=8 GPU (CG_DEVICE=cuda) with utilization sampling ==="
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader -l 1 > $OUT/gpu_util.csv &
GPU_MON=$!
CG_DEVICE=cuda uv run python tests/perf/benchmark.py --updates 5 \
    --out $OUT/b8_gpu.json 2>&1 | grep -E "^FPS|collect/env_step" -A2
kill $GPU_MON 2>/dev/null
echo "GPU utilization (mean/max % over the run):"
awk -F', ' '{gsub(/ %/,"",$2); s+=$2; n++; if ($2>m) m=$2} END {if (n) printf "  mean %.1f%%  max %d%%  (%d samples)\n", s/n, m, n}' $OUT/gpu_util.csv

DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR
rsync -a $OUT/ $DEST_DIR/
mv /home/mila/d/dus/scratch/pRNN/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}* $DEST_DIR/ 2>/dev/null
echo "results in $DEST_DIR"

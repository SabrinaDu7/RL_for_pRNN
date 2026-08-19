#!/bin/bash
# Does the CUDA-graph world-model result transfer off the RTX 4060?
#
# Every number in docs/exp_speed_cuda_graph_2026-08-19.md is dev-box (RTX 4060,
# 8 logical CPUs, 4 torch threads). Two of its conclusions are hardware-shaped
# and MUST be re-measured before the cluster runs adopt them:
#
#   1. The graph win (12.3x on the world-model step) should GROW on a faster
#      GPU: the workload is CPU/launch-bound, so a faster GPU widens the
#      CPU:CUDA gap the graph removes. Prediction, not a measurement.
#   2. Concurrency was capped at 2.47x aggregate by the dev box's 8 CPUs, NOT by
#      the GPU. A node with more cores per GPU should scale further. This is the
#      one number most likely to be wrong off-box.
#
# This is a BENCHMARK job, not a training run: a few minutes per arm, no
# checkpoints, no wandb. Usage:  sbatch slurm/graph_bench.sh
#
#SBATCH --job-name=graph_bench
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -eo pipefail
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CPUs: $SLURM_CPUS_PER_TASK  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module --force purge
module load python/3.10
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
export UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache
export CG_DEVICE=cuda

cd $HOME/experiments
cp -r RL_for_pRNN $SLURM_TMPDIR/
cd $SLURM_TMPDIR/RL_for_pRNN
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

OUT=$SLURM_TMPDIR/graph_results
mkdir -p $OUT
BASE="--override env=lroom_multi --override run=multienv --override exp.layouts=one"

echo "### gate: the graph must be correct before its speed means anything"
uv run python -m pytest tests/test_cuda_graph_wm.py tests/test_cuda_graph_diag_guard.py \
    tests/golden_omt -q 2>&1 | tail -3

echo "### 1. the four world-model regimes (dev box: 1.19 / 3.53 / 10.19 / 1.14 grad/s)"
for arm in "A_pooled_nograph True False" "B_serial_nograph False False" \
           "C_serial_graph False True" "D_pooled_graph True True"; do
    set -- $arm
    echo "=== $1  batched_wm=$2 cuda_graph=$3 ==="
    uv run python tests/perf/benchmark.py --updates 6 --warmup-updates 1 --out $OUT/$1.json \
        $BASE --override predNet.batched_wm=$2 --override predNet.cuda_graph=$3 \
        2>&1 | tail -2
done

echo "### 2. num_envs with serial WM + graph (dev box: 10.88 / 27.21 / 37.50)"
for B in 8 32 64 128; do
    F=$((B*256)); PB=$((F/4))
    echo "=== num_envs=$B ==="
    uv run python tests/perf/benchmark.py --updates 4 --warmup-updates 1 --out $OUT/envs_$B.json \
        $BASE --override predNet.batched_wm=False --override predNet.cuda_graph=True \
        --override exp.num_envs=$B --override rl.frames=$F --override rl.ppo_batch_size=$PB \
        2>&1 | tail -2
done

echo "### 3. trainStep batch scaling - where this GPU's knee is (4060: flat to 128)"
uv run python tests/perf/sweep_trainstep_batch.py --device cuda \
    --batches 1,8,32,128,256,512,1024,2048 --reps 5 --out $OUT/trainstep_sweep.json 2>&1 | tail -12

echo "### 4. concurrency - the number most likely to differ (4060 was CPU-capped at 2.47x)"
for N in 2 4 8; do
    echo "=== $N concurrent ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 2 \
        > $OUT/util_conc$N.csv 2>/dev/null &
    MON=$!
    for i in $(seq 1 $N); do
        uv run python tests/perf/benchmark.py --updates 4 --warmup-updates 1 \
            --out $OUT/conc${N}_$i.json $BASE \
            --override predNet.batched_wm=False --override predNet.cuda_graph=True \
            --override exp.seed=$((i+10)) > $OUT/conc${N}_$i.log 2>&1 &
    done
    wait $(jobs -p | grep -v $MON) 2>/dev/null || true
    wait
    kill $MON 2>/dev/null || true
done

echo "### summary, on the only axis that matters"
uv run python - "$OUT" <<'PY'
import json, glob, sys, os, statistics as st
O = sys.argv[1]
def g(f):
    try: return json.load(open(f))
    except Exception: return None
print(f"{'arm':<22}{'GRAD/s':>9}{'s/upd':>9}{'fps':>10}")
for n in ["A_pooled_nograph","B_serial_nograph","C_serial_graph","D_pooled_graph"]:
    d = g(f"{O}/{n}.json")
    if d: print(f"{n:<22}{d['wm_grad_steps_per_s']:>9.2f}"
                f"{d['total_train_s']/d['meta']['n_updates']:>9.3f}{d['fps']:>10.0f}")
print()
for B in [8,32,64,128]:
    d = g(f"{O}/envs_{B}.json")
    if d: print(f"{'num_envs='+str(B):<22}{d['wm_grad_steps_per_s']:>9.2f}"
                f"{d['total_train_s']/d['meta']['n_updates']:>9.3f}{d['fps']:>10.0f}")
print()
solo = g(f"{O}/envs_8.json")
solo = solo["wm_grad_steps_per_s"] if solo else None
for N in [2,4,8]:
    v = [d["wm_grad_steps_per_s"] for d in (g(f) for f in sorted(glob.glob(f"{O}/conc{N}_*.json"))) if d]
    if v and solo:
        print(f"{str(N)+' concurrent':<22}{sum(v):>9.2f} aggregate   per-proc {st.median(v):>6.2f}"
              f"   x{sum(v)/solo:.2f} vs solo   retained {100*st.median(v)/solo:.0f}%")
PY

DEST_DIR="$SCRATCH/pRNN/$JOB_ID"
mkdir -p $DEST_DIR
rsync -a $OUT/ $DEST_DIR/
echo "results in $DEST_DIR"

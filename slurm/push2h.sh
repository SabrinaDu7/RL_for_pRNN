#!/bin/bash
# Close the gap to "plateau in under 2 hours" on CLUSTER hardware.
#
# Target: 286,720 gradient steps (the 73.4M-env-step sRSA plateau) in < 2 h
# => 40 grad/s. graph_bench_10420332 measured Mila at 30.88 grad/s with
# cuda_graph + num_envs=128, so this needs another ~1.3x. Three candidates,
# none of them measured on this hardware yet:
#   1. more width (num_envs 256, 512) - the dev box kept gaining to 256
#   2. cuda_graph + compile_cell=layer COMPOSED - never tested together
#      anywhere; they may stack, be redundant, or conflict outright
#   3. the concurrency arms 10420332 did not reach (N=4, 8)
#
#SBATCH --job-name=push2h
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

set -eo pipefail
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module --force purge && module load python/3.10
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache CG_DEVICE=cuda

BRANCH="${1:-sdu/speed}"
SRC=$HOME/experiments/RL_for_pRNN
git -C $SRC fetch -q origin
git clone -q --shared $SRC $SLURM_TMPDIR/RL_for_pRNN
cd $SLURM_TMPDIR/RL_for_pRNN
git fetch -q "$SRC" "refs/remotes/origin/$BRANCH"
git checkout -q --detach FETCH_HEAD
echo "benchmarking $(git rev-parse --short HEAD)"
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

OUT=$SLURM_TMPDIR/out; DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p $OUT $DEST
save () { rsync -a "$OUT/" "$DEST/" 2>/dev/null || true; }
trap save EXIT
B="--override env=lroom_multi --override run=multienv --override exp.layouts=one \
--override predNet.batched_wm=False"

bench () { # tag, extra overrides...
  local tag=$1; shift
  echo "=== $tag ==="
  uv run python tests/perf/benchmark.py --updates 4 --warmup-updates 2 \
      --out $OUT/$tag.json $B "$@" 2>&1 | tail -2 || echo "  FAILED"
  save
}

echo "### 1. width beyond 128, graphed"
for N in 128 256 512; do
  F=$((N*256))
  bench "g_envs_$N" --override predNet.cuda_graph=True \
     --override exp.num_envs=$N --override rl.frames=$F --override rl.ppo_batch_size=$((F/4))
done

echo "### 2. DO cuda_graph AND compile_cell COMPOSE? (never tested together)"
for N in 8 64 128; do
  F=$((N*256))
  bench "gc_envs_$N" --override predNet.cuda_graph=True --override predNet.compile_cell=layer \
     --override exp.num_envs=$N --override rl.frames=$F --override rl.ppo_batch_size=$((F/4))
  bench "c_envs_$N" --override predNet.cuda_graph=False --override predNet.compile_cell=layer \
     --override exp.num_envs=$N --override rl.frames=$F --override rl.ppo_batch_size=$((F/4))
done

echo "### 3. concurrency N=4 (10420332 only reached N=2)"
for i in 1 2 3 4; do
  uv run python tests/perf/benchmark.py --updates 4 --warmup-updates 1 --out $OUT/conc4_$i.json \
     $B --override predNet.cuda_graph=True --override exp.seed=$((i+10)) > $OUT/conc4_$i.log 2>&1 &
done
wait; save

echo "### SUMMARY - grad steps/s, and hours to the 286,720-step plateau"
uv run python - "$OUT" <<'PY'
import json, glob, sys, statistics as st
O = sys.argv[1]; PLATEAU = 286_720
def g(f):
    try: return json.load(open(f))
    except Exception: return None
rows = []
for f in sorted(glob.glob(f"{O}/*.json")):
    d = g(f)
    if d and "conc" not in f:
        rows.append((f.split("/")[-1][:-5], d["wm_grad_steps_per_s"],
                     d["total_train_s"]/d["meta"]["n_updates"]))
print(f"{'arm':<16}{'GRAD/s':>9}{'s/upd':>9}{'plateau':>10}{'<2h?':>7}")
for n, gs, spu in sorted(rows, key=lambda r: -r[1]):
    h = PLATEAU/gs/3600
    print(f"{n:<16}{gs:>9.2f}{spu:>9.3f}{h:>9.1f}h{'  YES' if h < 2 else '   no':>7}")
v = [d["wm_grad_steps_per_s"] for d in (g(f) for f in sorted(glob.glob(f"{O}/conc4_*.json"))) if d]
if v:
    print(f"\n4 concurrent: {sum(v):.2f} aggregate, per-proc {st.median(v):.2f}")
PY
save
echo "results in $DEST"

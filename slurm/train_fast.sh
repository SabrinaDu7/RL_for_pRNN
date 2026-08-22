#!/bin/bash
# Production multi-room training, tuned to finish inside 2 hours.
#
#   sbatch slurm/train_fast.sh [layouts] [envcfg] [branch]
#     layouts: one | rooms | pool      (default rooms)
#     envcfg : lroom_multi | squareroom_multi
#
# WHY THIS FITS IN 2 H, and what was traded to get there:
#
# 1. The full 491.5M-step budget CANNOT fit. It is 1,920,000 world-model
#    gradient steps; 2 h needs 267 grad/s against a hard ceiling of ~62 set by
#    the trainStep itself. No engineering closes a 4.3x gap.
# 2. So the run targets the sRSA PLATEAU instead, which the project's own data
#    locates (docs/results/probe-remapping-2026-08-13.md 3): per-room sRSA is
#    0.7382 at 52.4M steps, 0.7870 at 73.4M, 0.7905 at 94.4M. 73.4M is 99.6% of
#    the 94M value. episodes_total below stops there.
#    ⚠️ This is a SCIENTIFIC trade, not a free win: the E1 object-cell fraction
#    kept rising past 94M (0.047 -> 0.067 at 482M), so a run aimed at the
#    OBJECT question still needs the long budget. This preset is for the
#    map-formation question.
# 3. GPU TYPE IS LOAD-BEARING. Same config, measured: L40S 52.45 grad/s vs
#    Quadro RTX 8000 30.88 - a 1.7x spread that decides whether it fits
#    (1.52 h vs 2.58 h). Hence the explicit l40s request; without it the run
#    silently misses the target on a Turing node.
#
#SBATCH --job-name=train_fast
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1

set -eo pipefail
LAYOUTS="${1:-rooms}"; ENVCFG="${2:-lroom_multi}"; BRANCH="${3:-sdu/speed}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)  layouts=$LAYOUTS env=$ENVCFG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module --force purge && module load python/3.10
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache CG_DEVICE=cuda

SRC=$HOME/experiments/RL_for_pRNN
git -C $SRC fetch -q origin
git clone -q --shared $SRC $SLURM_TMPDIR/RL_for_pRNN
cd $SLURM_TMPDIR/RL_for_pRNN
git fetch -q "$SRC" "refs/remotes/origin/$BRANCH"
git checkout -q --detach FETCH_HEAD
echo "training $(git rev-parse --short HEAD)"
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p $DEST
save () { rsync -a outputs/ "$DEST/outputs/" 2>/dev/null || true; }
trap save EXIT

NUM_ENVS=128
FRAMES=$((NUM_ENVS*256))
# 286,720 episodes x 256 steps = 73.4M env steps = the sRSA plateau
uv run python main_train.py env=$ENVCFG run=multienv exp.layouts=$LAYOUTS \
    predNet.batched_wm=False predNet.cuda_graph=True \
    exp.num_envs=$NUM_ENVS rl.frames=$FRAMES rl.ppo_batch_size=$((FRAMES/4)) \
    rl.episodes_total=286720 \
    logging.archive_every_steps=2097152 logging.save_every_steps=8388608 \
    logging.analysis_every_steps=8388608 logging.plot_every_steps=0 \
    exp.exp_name=fast-$LAYOUTS 2>&1 | tail -40
save

# Spatial curves are skipped in-run under cuda_graph (the eval moves the model
# and would invalidate captured graphs), so score the archives offline here -
# same numbers, computed after rather than during.
RUN=$(ls -dt outputs/fast-${LAYOUTS}_* 2>/dev/null | head -1)
if [ -n "$RUN" ]; then
  echo "=== offline spatial scoring of $RUN ==="
  uv run python scripts/multienv/checkpoint_curve.py --run "$RUN" \
      --env $ENVCFG --layouts $LAYOUTS --spatial 2>&1 | tail -25
  save
fi
echo "results in $DEST"

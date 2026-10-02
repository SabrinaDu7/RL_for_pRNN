#!/bin/bash
# Multi-room training with a NOVEL ONE-CELL OBJECT somewhere new in every episode - the
# scattered design of 2026-10-01. The body is placed.sh's; the source is
# `env.source:scattered` (curious_george/envs/layouts.py::Scattered): the eight selected
# rooms, each repeated n_placements times with a dot at a different admissible cell, the
# device pool drawing a layout per stream at every episode boundary.
#
#   sbatch slurm/scattered.sh <n_placements> <exclude> [seed] [branch] [label] [extra flags...]
#
#     n_placements : copies per room (16 -> 128 layouts)
#     exclude      : ONE argument, "x,y" per room, the held-out test spot kept out of the
#                    draw (Q18's: '9,13 13,2 2,10 9,10 9,10 2,9 9,9 2,13'); "" = none
#     seed         : run.seed (default 2); the placements are drawn by the source's own
#                    seed (default 0), recorded in provenance
#     branch       : default sdu/mixed-count-mse
#     label        : appended to the run name
#     extra        : passed VERBATIM, PRESET-LEVEL (before the env.source subcommand).
#
# Every run this launcher submits logs to wandb project `curious-george-multienv`.
#
#SBATCH --job-name=scattered
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
# GPU TYPE IS LOAD-BEARING: the same configuration measures 52.45 grad/s on an
# L40S against 30.88 on a Quadro RTX 8000.

set -eo pipefail
NPL="${1:?n_placements: copies per room}"
EXCLUDE="${2:-}"; SEED="${3:-2}"; BRANCH="${4:-sdu/mixed-count-mse}"; LABEL="${5:-}"
shift $(( $# < 5 ? $# : 5 ))
EXTRA=("$@")
# Unquoted on purpose: the "x,y" words must reach tyro as separate arguments.
EXCLUDEFLAG=${EXCLUDE:+--env.source.exclude $EXCLUDE}
NAME="mx-scattered-n8-p${NPL}-s${SEED}${LABEL:+-$LABEL}"

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "$NAME  (placements per room: $NPL, exclude: ${EXCLUDE:-none}, seed=$SEED, branch=$BRANCH)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module --force purge && module load python/3.10
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 CG_DEVICE=cuda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache

SRC=$HOME/experiments/RL_for_pRNN
mkdir -p "$SCRATCH/pRNN"
flock "$SCRATCH/pRNN/.gitfetch.lock" git -C "$SRC" fetch -q origin \
  || echo "[git] fetch failed (concurrent job?); using $SRC as-is"
git clone -q --shared "$SRC" "$SLURM_TMPDIR/RL_for_pRNN"
cd "$SLURM_TMPDIR/RL_for_pRNN"
git fetch -q "$SRC" "refs/remotes/origin/$BRANCH"
git checkout -q --detach FETCH_HEAD
echo "training $(git rev-parse --short HEAD)"

# .env is COMMITTED and points RL_STORAGE at /home/sabrina; load_dotenv defaults
# to override=False, so exporting wins.
export RL_STORAGE="$SLURM_TMPDIR/RL_for_pRNN/outputs"
mkdir -p "$RL_STORAGE"
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p "$DEST"
save () { rsync -a outputs/ "$DEST/outputs/" 2>/dev/null || true; }
trap save EXIT

# ORDER IS LOAD-BEARING: preset-level flags BEFORE `env.source:scattered`, the
# source's own flags after it. `tests/test_slurm_invocations.py` parses this line.
uv run python main_train.py multienv-fast \
    --run.seed "$SEED" --run.exp-name "$NAME" --run.wandb-project curious-george-multienv \
    "${EXTRA[@]}" \
    env.source:scattered --env.source.impassable --env.source.positions 0 1 2 3 5 6 7 8 --env.source.n-placements "$NPL" $EXCLUDEFLAG \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe through `tail` alone: a job once died with no visible traceback
# because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -30
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

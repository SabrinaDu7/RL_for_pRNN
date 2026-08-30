#!/bin/bash
# Multi-room training on the 5 (or 10) selected rooms, either affordance.
#
#   sbatch slurm/multienv.sh [impassable] [n] [seed] [branch]
#     impassable : "true" | "false"   (default false, i.e. the walkable arm)
#     n          : rooms, 1..10       (default 5)
#     seed       : run.seed           (default 2)
#     branch     : default sdu/multienv
#
# WHY THE SOURCE IS A SUBCOMMAND. `env.source` is a tyro UNION, so a member has
# to be selected before its fields exist:
#
#     main_train.py multienv env.source:selected --env.source.n 5 --env.source.impassable
#
# `--env.source.impassable False` is NOT that - it is an unrecognized option
# followed by a stray positional, and it cost job 10563027. `impassable` is a
# bool, so the flag form is `--env.source.impassable` / `--env.source.no-impassable`.
# The walkable arm needs no override at all: the preset already carries
# `Selected(n=5, impassable=False)`.
#
# THE ROOMS ARE THE SAME IN BOTH ARMS. `Selected` pins ANCHORS and applies the
# affordance on top, because the walkable and impassable admissible pools are
# different sequences and 0 of the 5 indices name the same room in both. See
# `envs/layouts.py::Selected` and `tests/test_selected_rooms.py`.
#
#SBATCH --job-name=multienv
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
IMP="${1:-false}"; N="${2:-5}"; SEED="${3:-2}"; BRANCH="${4:-sdu/multienv}"
case "$IMP" in
  true|True|1)  FLAG=--env.source.impassable;    TAG=impassable ;;
  false|False|0) FLAG=--env.source.no-impassable; TAG=walkable ;;
  *) echo "impassable must be true or false, got $IMP" >&2; exit 1 ;;
esac
NAME="mx-${TAG}-n${N}-s${SEED}"

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "$NAME  ($TAG, n=$N, seed=$SEED, branch=$BRANCH)"
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

uv run python main_train.py multienv \
    env.source:selected --env.source.n "$N" "$FLAG" \
    --run.seed "$SEED" --run.exp-name "$NAME" \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe through `tail` alone: a job once died with no visible traceback
# because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -30
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

#!/bin/bash
# Multi-room training with ONE landmark per room, at the cells named - the
# one-object design of 2026-09-21. The body is multienv.sh's; only the source
# differs, and it is `env.source:placed` (curious_george/envs/layouts.py::Placed).
#
#   sbatch slurm/placed.sh <anchors> [kind] [seed] [branch] [label] [extra flags...]
#
#     anchors : ONE argument, "x,y" per room, space-separated, MiniGrid
#               coordinates. Eight rooms: '2,2 13,2 13,7 9,7 9,13 2,13 7,2 2,7'.
#               Every cell the stencil paints must be floor; the config refuses
#               a placement that is not, before any GPU time is spent.
#     kind    : which of the run's landmark kinds every room gets, by index into
#               EnvContent.kinds: 0 triangle3 (blue), 1 plus (green), 2 block3
#               (red) under the default content. Default 2.
#     seed    : run.seed (default 2)
#     branch  : default sdu/mixed-count-mse
#     label   : appended to the run name
#     extra   : passed VERBATIM to main_train.py, PRESET-LEVEL (placed before
#               the env.source subcommand - tyro applies flags to the directly
#               preceding subcommand). The MSE1024 recipe this launcher was
#               written to repeat, read off that run's provenance.json:
#                 --arch-prnn.loss MSE --train-policy.normalize-reward \
#                 --arch-prnn.hidden-size 1024 \
#                 --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT \
#                 --eval.plot-every-steps 3333328
#
# Every run this launcher submits logs to wandb project `curious-george-multienv`,
# IN the invocation below (see multienv.sh for why that is not a flag).
#
#SBATCH --job-name=placed
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
ANCHORS="${1:?anchors: one argument, 'x,y' per room, space-separated}"
KIND="${2:-2}"; SEED="${3:-2}"; BRANCH="${4:-sdu/mixed-count-mse}"; LABEL="${5:-}"
shift $(( $# < 5 ? $# : 5 ))
EXTRA=("$@")
N=$(wc -w <<< "$ANCHORS")
# Unquoted on purpose: the "x,y" words must reach tyro as separate arguments.
ANCHORFLAG="--env.source.anchors $ANCHORS"
NAME="mx-placed-n${N}-k${KIND}-s${SEED}${LABEL:+-$LABEL}"

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "$NAME  (anchors: $ANCHORS, kind=$KIND, seed=$SEED, branch=$BRANCH)"
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

# ORDER IS LOAD-BEARING: preset-level flags BEFORE `env.source:placed`, the
# source's own flags after it. `tests/test_slurm_invocations.py` parses this line.
uv run python main_train.py multienv-fast \
    --run.seed "$SEED" --run.exp-name "$NAME" --run.wandb-project curious-george-multienv \
    "${EXTRA[@]}" \
    env.source:placed --env.source.impassable --env.source.kind "$KIND" $ANCHORFLAG \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe through `tail` alone: a job once died with no visible traceback
# because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -30
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

#!/bin/bash
# Resume a finished run and keep training, optionally with a set of pRNN hidden units
# held at fixed values - the ablate-and-retrain arm of the 2026-09-24 exploration.
#
#   sbatch slurm/resume_clamp.sh <run_dir> <clamp.npz|none> <label> [wm_extra] [branch] [extra flags...]
#
#     run_dir  : ABSOLUTE directory holding predictiveNet_state.pt and policy.pt (the
#                job runs in $SLURM_TMPDIR, so a run-relative path does not resolve).
#     clamp    : an .npz with `units` and `values` (models/unit_clamp.py), or `none`
#                for the resume-only control. Absolute path, same reason.
#     label    : appended to the run name.
#     wm_extra : world-model gradient steps to ADD to the checkpoint's own count
#                (the budget is a grand total - training/loop.py refuses a total
#                already reached); the policy's total is held at 4x. Multiple of
#                32. Default 8192. The checkpoint's count is `wm_done` below: the
#                MSE1024 policy.pt sits at 83,886,080 frames = 40,960 steps at
#                2,048 frames per step (the run trained on to 43,936, the final
#                archive is the one it saved).
#     branch   : default sdu/mixed-count-mse
#     extra    : passed VERBATIM, PRESET-LEVEL. Repeat the resumed run's own
#                preset flags here (read them off its provenance.json), e.g. for
#                the MSE1024 recipe:
#                  --arch-prnn.loss MSE --train-policy.normalize-reward \
#                  --arch-prnn.hidden-size 1024 \
#                  --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT \
#                  --eval.plot-every-steps 3333328
#
# The CUDA graphs are OFF for both learners (a resumed optimizer state refuses them,
# configs.py::__post_init__) and the layer compile is OFF (the clamp hook sits inside
# what it would trace). Every run logs to wandb project `curious-george-multienv`.
#
#SBATCH --job-name=resume
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1

set -eo pipefail
RUN_DIR="${1:?run_dir: absolute directory with predictiveNet_state.pt and policy.pt}"
CLAMP="${2:?clamp: an .npz of units/values, or none}"; LABEL="${3:?label}"
WM_EXTRA="${4:-8192}"; BRANCH="${5:-sdu/mixed-count-mse}"
SEED=2   # the resumed run's; the RNG stream is new either way
shift $(( $# < 5 ? $# : 5 ))
EXTRA=("$@")
[ "$CLAMP" = "none" ] && CLAMP=""
CLAMPFLAG=${CLAMP:+--arch-prnn.clamp-units $CLAMP}
WM_DONE="${WM_DONE:-40960}"   # world-model steps the checkpoint has behind it (frames / 2048)
WM_TOTAL=$(( WM_DONE + WM_EXTRA )); POL_TOTAL=$(( WM_TOTAL * 4 ))
BUDGET="--train-prnn.total-grad-steps $WM_TOTAL --train-policy.total-grad-steps $POL_TOTAL"
CKPTFLAGS="--run.prnn-ckpt $RUN_DIR/predictiveNet_state.pt --run.policy-ckpt $RUN_DIR/policy.pt"
NAME="mx-impassable-n8-s2-resume-${LABEL}"

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "$NAME  (from $RUN_DIR, clamp=${CLAMP:-none}, +$WM_EXTRA wm steps, branch=$BRANCH)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$RUN_DIR/predictiveNet_state.pt" ] && [ -f "$RUN_DIR/policy.pt" ] || { echo "checkpoints missing in $RUN_DIR"; exit 1; }
[ -z "$CLAMP" ] || [ -f "$CLAMP" ] || { echo "clamp file missing: $CLAMP"; exit 1; }

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

export RL_STORAGE="$SLURM_TMPDIR/RL_for_pRNN/outputs"
mkdir -p "$RL_STORAGE"
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p "$DEST"
save () { rsync -a outputs/ "$DEST/outputs/" 2>/dev/null || true; }
trap save EXIT

# ORDER IS LOAD-BEARING: preset-level flags BEFORE `env.source:selected`.
# `tests/test_slurm_invocations.py` parses this line.
uv run python main_train.py multienv-fast \
    --run.seed "$SEED" --run.exp-name "$NAME" --run.wandb-project curious-george-multienv \
    $CKPTFLAGS --train-prnn.no-cuda-graph --train-policy.no-cuda-graph --train-prnn.compile OFF \
    $BUDGET $CLAMPFLAG "${EXTRA[@]}" \
    env.source:selected --env.source.impassable --env.source.positions 0 1 2 3 5 6 7 8 \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -30
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

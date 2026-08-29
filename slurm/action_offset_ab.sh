#!/bin/bash
# The action_offset A/B, at the mila-parity configuration, one arm per job.
#
#   sbatch slurm/action_offset_ab.sh [offset] [entropy] [seed] [branch]
#     offset  : 0 | 1   (arch_prnn.action_offset - which action shares a row
#                        with obs[t]; see docs/action-offset-ab-2026-08-29.md)
#     entropy : train_policy.entropy_coef        (default 0.01)
#     seed    : run.seed                         (default 2)
#     branch  : branch to check out              (default sdu/predict-next-obs)
#
# WHY THIS EXISTS RATHER THAN A train_fast.sh FLAG. train_fast.sh owns the 2-hour
# multi-room production preset: its own budget, its own regime knobs, its own
# `single` path that is NOT this configuration. This runs the configuration the
# A/B is defined against - `mila-parity-e0.001_curious_26-08-27-14-32-32`,
# verbatim - so the arms stay comparable to a run that already exists. Adding a
# fifteenth positional argument to train_fast.sh would have made two experiments
# share one budget and one set of defaults.
#
# WHAT IT IS TESTING. At entropy_coef=0.001 the offset-1 policy collapses in both
# local seeds: policy_entropy to 0.49/0.39 bits against offset 0's 1.45/1.51, and
# spatial coverage to 2.84/2.78 bits against a flat ~7. Prediction loss is
# IDENTICAL before the first collapse (ratio 0.998 and 1.050) and only diverges
# as coverage falls, so the loss gap looks downstream of exploration rather than
# intrinsic to the circuit. h[t] is a better basis for action selection than
# h[t-1], so the same entropy bonus buys less resistance. Run the 2x2 -
# {offset 0, offset 1} x {0.001, 0.01} - and the prediction is that offset 1 at
# 0.01 holds coverage and closes most of the loss gap. If the gap SURVIVES stable
# coverage, the explanation is wrong and the cost is intrinsic.
#
#SBATCH --job-name=offset_ab
#SBATCH --output=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.out
#SBATCH --error=/home/mila/d/dus/scratch/pRNN/logs/%x_%j.err
#SBATCH --partition=long
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
# GPU TYPE IS LOAD-BEARING: the same configuration measures 52.45 grad/s on an
# L40S against 30.88 on a Quadro RTX 8000. Without the explicit request an arm
# can land on a Turing node and be incomparable to its partner.

set -eo pipefail
OFFSET="${1:-1}"; ENT="${2:-0.01}"; SEED="${3:-2}"; BRANCH="${4:-sdu/predict-next-obs}"
case "$OFFSET" in 0|1) ;; *) echo "offset must be 0 or 1, got $OFFSET" >&2; exit 1 ;; esac

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "action_offset=$OFFSET entropy_coef=$ENT seed=$SEED branch=$BRANCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module --force purge && module load python/3.10
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache CG_DEVICE=cuda

SRC=$HOME/experiments/RL_for_pRNN
# SERIALIZED and NON-FATAL, for the reason train_fast.sh gives: every job fetches
# into the same shared $SRC, and with `set -e` a transient ref race kills the job
# outright. Losing the lock costs a staleness warning, not the run.
mkdir -p "$SCRATCH/pRNN"
flock "$SCRATCH/pRNN/.gitfetch.lock" git -C "$SRC" fetch -q origin \
  || echo "[git] fetch of $SRC failed (concurrent job?); using it as-is"
git clone -q --shared "$SRC" "$SLURM_TMPDIR/RL_for_pRNN"
cd "$SLURM_TMPDIR/RL_for_pRNN"
git fetch -q "$SRC" "refs/remotes/origin/$BRANCH"
git checkout -q --detach FETCH_HEAD
echo "training $(git rev-parse --short HEAD)"

# .env is COMMITTED and carries a machine-specific RL_STORAGE pointing at
# /home/sabrina, so a clean clone anywhere else writes into that home and dies
# with EACCES. load_dotenv() defaults to override=False, so exporting wins.
export RL_STORAGE="$SLURM_TMPDIR/RL_for_pRNN/outputs"
mkdir -p "$RL_STORAGE"
uv run python -c "
from curious_george.log_and_store.storage import get_storage_dir
import os
d = get_storage_dir(); print(f'storage dir -> {d}')
os.makedirs(d, exist_ok=True)
t = os.path.join(d, '.writetest'); open(t, 'w').close(); os.remove(t)
print('storage is writable')
" || { echo 'PREFLIGHT FAILED: storage dir not writable'; exit 1; }
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p "$DEST"
save () { rsync -a outputs/ "$DEST/outputs/" 2>/dev/null || true; }
trap save EXIT

NAME="mila-off${OFFSET}-e${ENT}-s${SEED}"
# `mila-parity-e0.001_curious_26-08-27-14-32-32` verbatim, plus the two knobs
# under test. The gradient-step counts are the ground truth the arms are matched
# on: 43,936 world-model and 175,744 policy steps, 89,980,928 environment steps.
uv run python main_train.py reference \
    --collect.backend DEVICE --collect.num-envs 256 --collect.episodes-per-env 1 \
    --collect.episode-steps 256 --collect.rollout-cuda-graph \
    --train-prnn.batched --train-prnn.batched-curiosity \
    --train-prnn.episodes-per-grad-step 8 --train-prnn.compile LAYER \
    --train-prnn.cuda-graph --train-prnn.no-curiosity-cuda-graph \
    --train-prnn.total-grad-steps 43936 \
    --train-policy.total-grad-steps 175744 --train-policy.cuda-graph \
    --train-policy.entropy-coef "$ENT" \
    --arch-prnn.action-offset "$OFFSET" \
    --run.wandb \
    --eval.analysis-every-steps 3333328 --eval.plot-every-steps 7499989 \
    --run.seed "$SEED" \
    --run.save-every-steps 8388608 --run.archive-every-steps 8388608 \
    --run.exp-name "$NAME" \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe training output through `tail`: job 10444214 died with exit 1 and no
# visible traceback because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -25
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

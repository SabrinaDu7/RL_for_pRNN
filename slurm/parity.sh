#!/bin/bash
# The `parity` preset on the cluster, with optional overrides.
#
#   sbatch slurm/parity.sh [offset] [entropy] [seed] [branch] [ent_final] [label] [extra flags...]
#     every argument is OPTIONAL; omit one and the PRESET's own default is used
#     and no flag is passed at all. `sbatch slurm/parity.sh` alone is therefore
#     a test of the shipped defaults, which is what makes it useful as a
#     "did the refactor break training" check.
#
#     label     : appended to the run name, so arms that differ only in the
#                 extra flags stay distinguishable in wandb
#     extra...  : passed VERBATIM to main_train.py - how a baseline arm rides
#                 this launcher, e.g.
#                   ... '' rndfwd --arch-policy.agent RANDOM
#                   ... '' rnduni --arch-policy.agent RANDOM \
#                       --arch-policy.random-action-probs 0.25 0.25 0.25 0.25
#                   ... '' count  --train-policy.no-curious --train-policy.k-count 0.1
#
#     offset    : arch_prnn.action_offset (0 | 1) - which action shares a row
#                 with obs[t]; see docs/prnn-io-alignment.md
#     entropy   : train_policy.entropy_coef. The preset's 0.003 is the MEASURED
#                 knee (docs/entropy-sweep-and-noise-floor-2026-08-29.md).
#     seed      : run.seed
#     branch    : branch to check out (default main)
#     ent_final : train_policy.entropy_coef_final; ramps LINEARLY in environment
#                 steps. Setting it DISABLES the policy CUDA graph below - a
#                 captured step bakes the coefficient in, so the ramp would
#                 silently never happen, which is how job
#                 mila-off1-e0.001to0.01-s2 was wasted. configs.py refuses the
#                 combination outright.
#
# WHY THE CONFIG IS NOT REPEATED HERE. It used to be: this file carried the
# whole `mila-parity` flag list while configs.py::_parity carried the same
# numbers as a preset, and nothing checked that the two agreed. One fact, one
# home - the preset. The gradient-step counts it fixes (43,936 world-model and
# 175,744 policy over 89,980,928 environment steps) are printed by
# TrainingSchedule.summary() at startup; read them there, not from a comment.
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
# EMPTY means "do not pass the flag", so the preset's own default stands and
# `sbatch slurm/parity.sh` tests exactly what ships.
OFFSET="${1:-}"; ENT="${2:-}"; SEED="${3:-}"; BRANCH="${4:-main}"; ENT_FINAL="${5:-}"; LABEL="${6:-}"
shift $(( $# < 6 ? $# : 6 ))
EXTRA=("$@")
case "${OFFSET:-0}" in 0|1) ;; *) echo "offset must be 0 or 1, got $OFFSET" >&2; exit 1 ;; esac

echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)"
echo "overrides: offset=${OFFSET:-<preset>} entropy=${ENT:-<preset>}${ENT_FINAL:+ -> $ENT_FINAL} seed=${SEED:-<preset>} branch=$BRANCH${LABEL:+ label=$LABEL}${EXTRA[*]:+ extra: ${EXTRA[*]}}"
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

NAME="parity${OFFSET:+-off$OFFSET}${ENT:+-e$ENT}${ENT_FINAL:+to$ENT_FINAL}${SEED:+-s$SEED}${LABEL:+-$LABEL}"
[ "$NAME" = "parity" ] && NAME="parity-defaults"
# Empty expands to NOTHING (unquoted, deliberately) so a constant-coefficient
# run passes no flag at all and `entropy_coef_final` keeps its None default.
RAMP=${ENT_FINAL:+--train-policy.entropy-coef-final $ENT_FINAL}
# A captured policy step bakes entropy_coef in, so a ramp under
# --train-policy.cuda-graph is silently pinned at its start value - which is
# exactly how job mila-off1-e0.001to0.01-s2 was wasted. configs.py now REFUSES
# the combination, so drop the graph rather than let sbatch fail 20 minutes in.
# The PRESET already enables the policy graph, so the default override is
# nothing; a ramp has to turn it OFF explicitly.
POLICY_GRAPH=
if [ -n "$ENT_FINAL" ]; then
  POLICY_GRAPH=--train-policy.no-cuda-graph
  echo "[entropy ramp] policy CUDA graph DISABLED - a captured step cannot see"
  echo "[entropy ramp] a changing coefficient. This arm is NOT throughput-"
  echo "[entropy ramp] comparable to the graphed ones; compare on gradient steps."
fi
# `mila-parity-e0.001_curious_26-08-27-14-32-32` verbatim, plus the two knobs
# under test. The gradient-step counts are the ground truth the arms are matched
# on: 43,936 world-model and 175,744 policy steps, 89,980,928 environment steps.
uv run python main_train.py parity \
    ${OFFSET:+--arch-prnn.action-offset $OFFSET} \
    ${ENT:+--train-policy.entropy-coef $ENT} \
    ${SEED:+--run.seed $SEED} \
    $POLICY_GRAPH $RAMP \
    "${EXTRA[@]}" \
    --run.exp-name "$NAME" \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe training output through `tail`: job 10444214 died with exit 1 and no
# visible traceback because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -25
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

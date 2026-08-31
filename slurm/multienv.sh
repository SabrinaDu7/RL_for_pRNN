#!/bin/bash
# Multi-room training on the 5 (or 10) selected rooms, either affordance.
#
#   sbatch slurm/multienv.sh [impassable] [n] [seed] [branch] [wm_grad_steps] [agent] [norm] [entropy] [positions] [label] [extra flags...]
#     impassable : "true" | "false"   (default false, i.e. the walkable arm)
#     n          : rooms, 1..10       (default 5)
#     seed       : run.seed           (default 2)
#     branch     : default sdu/multienv
#     wm_grad_steps : world-model gradient steps; policy is held at 4x. Empty
#                     uses the preset's 43,936. (Wall-clock notes from before
#                     the 2026-08-31 device-pool fast reset are stale; time a
#                     run, don't trust an old number here.)
#     agent      : "random" for the BASELINE (actions from RAND_ACT_PROBA, no
#                  policy updates), empty for the learned policy.
#     norm       : advantage whitening. Since 2026-08-31 the preset whitens
#                  by DEFAULT (configs.py::_parity), so empty = whitened,
#                  "norm" is the explicit spelling of the same thing, and
#                  "raw" passes --train-policy.no-normalize-advantage.
#     entropy    : train_policy.entropy_coef; empty uses the preset's default
#                  (one home: configs.py::_parity, and its docstring holds the
#                  raw-era vs whitened-era knee history).
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
# World-model gradient steps; the policy's is held at 4x, which is the ratio the
# parity shape fixes and what keeps ppo_batch_size at 2048. Empty = the preset's
# own 43,936. Measured: multi-room runs at ~35,120 env steps/s against
# single-room's ~54,300, so the full budget is ~43 min of training BEFORE the
# analysis events - which is why a half budget exists. Per-room sRSA over the
# walkable run went 0.617 -> 0.713 -> 0.786 -> 0.795 -> 0.804, so half the
# budget buys nearly all of the quality.
WM="${5:-}"
# "random" makes actions come from RAND_ACT_PROBA instead of the policy, through
# the SAME collector - the baseline the learned policy is measured against.
AGENT="${6:-}"
# Whitening is the preset DEFAULT since 2026-08-31; "raw" is the arm that
# needs a flag now. ⚠️ Whitening rescales |adv| ~0.12 -> ~1, so an
# entropy_coef tuned in one era means something ~8x different in the other -
# the preset docstring (configs.py::_parity) is the one home for that history.
NORM="${7:-}"
# positions  : comma-separated ROOMS_SELECTED POSITIONS (not source indices;
#               position 4 is source index 83). Empty keeps "first n". The CE
#               plan's 8-room set is 0,1,2,3,5,6,7,8.
# label       : appended to the run name, so arms differing only in the extra
#               flags stay distinguishable in wandb.
# extra...    : passed VERBATIM to main_train.py, PRESET-LEVEL (they are
#               placed before the env.source subcommand - tyro applies flags
#               to the directly preceding subcommand, so source-level flags
#               cannot ride here; positions has its own argument for that).
#               e.g. ... '' ce --arch-prnn.loss CE --train-policy.normalize-reward
# train_policy.entropy_coef. Empty uses the preset's own default (one home:
# configs.py::_parity - whitened era; its docstring records the raw-era knee,
# the measured 67-70% collapse when 0.003 rode whitened advantages, and the
# ratio-matched ~0.024).
ENT="${8:-}"
POS="${9:-}"; LABEL="${10:-}"
shift $(( $# < 10 ? $# : 10 ))
EXTRA=("$@")
POSFLAG=${POS:+--env.source.positions ${POS//,/ }}
case "$IMP" in
  true|True|1)  FLAG=--env.source.impassable;    TAG=impassable ;;
  false|False|0) FLAG=--env.source.no-impassable; TAG=walkable ;;
  *) echo "impassable must be true or false, got $IMP" >&2; exit 1 ;;
esac
NAME="mx-${TAG}-n${N}-s${SEED}${WM:+-wm$WM}${AGENT:+-$AGENT}${NORM:+-$NORM}${ENT:+-e$ENT}${LABEL:+-$LABEL}"
ENTFLAG=${ENT:+--train-policy.entropy-coef $ENT}
# tyro takes the enum MEMBER NAME, not its value: --arch-policy.agent RANDOM.
case "$NORM" in
  "")   NORMFLAG= ;;
  raw)  NORMFLAG="--train-policy.no-normalize-advantage" ;;
  norm) NORMFLAG="--train-policy.normalize-advantage" ;;
  *) echo "norm must be 'norm' or empty, got $NORM" >&2; exit 1 ;;
esac
case "$AGENT" in
  ""|policy) AGENTFLAG= ;;
  random|RANDOM) AGENTFLAG="--arch-policy.agent RANDOM" ;;
  *) echo "agent must be random or empty, got $AGENT" >&2; exit 1 ;;
esac
BUDGET=${WM:+--train-prnn.total-grad-steps $WM --train-policy.total-grad-steps $((WM*4))}

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

# ORDER IS LOAD-BEARING. tyro applies an option to the directly preceding
# subcommand, so every preset-level flag must come BEFORE `env.source:selected`;
# putting --run.seed after it makes it "unrecognized". This cost jobs 10563252-4.
# `tests/test_slurm_invocations.py` parses this exact line so the next reorder
# fails at gate time instead of after a GPU allocation.
uv run python main_train.py multienv-fast \
    --run.seed "$SEED" --run.exp-name "$NAME" $BUDGET $AGENTFLAG $NORMFLAG $ENTFLAG "${EXTRA[@]}" \
    env.source:selected --env.source.n "$N" "$FLAG" $POSFLAG \
    > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe through `tail` alone: a job once died with no visible traceback
# because the tail showed the config dump instead of the error.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -30
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
save
echo "results in $DEST"

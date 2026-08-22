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
# 2. So it runs as far as 2 h reaches: 600,000 gradient steps = 153.6M env
#    steps, 31% of the full budget. That is comfortably past the sRSA plateau
#    the project's own data locates (per-room sRSA 0.7382 at 52.4M, 0.7870 at
#    73.4M, 0.7905 at 94.4M) and past the 94.4M point too.
#    ⚠️ Still a SCIENTIFIC trade: the E1 object-cell fraction kept rising past
#    94M (0.047 -> 0.067 at 482M), so a run aimed at the OBJECT question needs
#    the long budget. This preset answers the map-formation question.
# wandb is OFF here (job 10444214 died at wandb init on a compute node), and
# BECAUSE it is off, in-run analysis must be off too: with wandb_log=false
# run_behavior_analysis takes the save_analysis_of_agent_behav branch, which
# writes plotly figures through kaleido - a headless browser that does not
# exist on a compute node. That killed job 10444320 twelve minutes in.
# Nothing is lost: spatial curves are already skipped under cuda_graph and are
# recovered from the archives at the end of this script.
#
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
export PYTHONUNBUFFERED=1
export JOB_ID="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" UV_CACHE_DIR=$SLURM_TMPDIR/uv_cache CG_DEVICE=cuda

SRC=$HOME/experiments/RL_for_pRNN
git -C $SRC fetch -q origin
git clone -q --shared $SRC $SLURM_TMPDIR/RL_for_pRNN
cd $SLURM_TMPDIR/RL_for_pRNN
git fetch -q "$SRC" "refs/remotes/origin/$BRANCH"
git checkout -q --detach FETCH_HEAD
echo "training $(git rev-parse --short HEAD)"

# .env is COMMITTED and carries a machine-specific absolute path
# (RL_STORAGE="/home/sabrina/.../outputs"), so a clean clone on any other
# machine tries to write to that user's home. Job 10444304 died on exactly
# this: PermissionError: [Errno 13] Permission denied: '/home/sabrina'.
# The older slurm scripts never hit it because they cp -r'd the cluster
# checkout, whose .env had been edited locally; cloning a named ref for
# traceability lost that override.
# python-dotenv's load_dotenv() defaults to override=False, so an exported
# variable wins over .env - no repo edit needed here.
export RL_STORAGE="$SLURM_TMPDIR/RL_for_pRNN/outputs"
mkdir -p "$RL_STORAGE"
# Preflight: fail LOUDLY and immediately rather than 60 s into a GPU
# allocation, and prove the path the code will actually use.
uv run python -c "
from curious_george.storage import get_storage_dir
import os, sys
d = get_storage_dir()
print(f'storage dir -> {d}')
os.makedirs(d, exist_ok=True)
t = os.path.join(d, '.writetest')
open(t, 'w').close(); os.remove(t)
print('storage is writable')
" || { echo 'PREFLIGHT FAILED: storage dir not writable'; exit 1; }
rm -rf .venv && uv venv .venv && source .venv/bin/activate && uv sync

DEST="$SCRATCH/pRNN/$JOB_ID"; mkdir -p $DEST
save () { rsync -a outputs/ "$DEST/outputs/" 2>/dev/null || true; }
trap save EXIT

NUM_ENVS=128
FRAMES=$((NUM_ENVS*256))
# ppo_batch_size is NOT scaled with frames, deliberately. Scaling it holds
# policy gradient steps per UPDATE fixed at 16, which means one policy step per
# 2048 environment steps against the num_envs=8 baseline's one per 128 - a 16x
# dilution of the policy that generates the behaviour the world model learns
# from. Measured cost of undoing that, num_envs=128 on this box:
#
#   ppo_batch  policy/upd  1 per N env  dilution   grad/s   2h reaches
#        8192          16         2048       16x    89.46      109.9M
#        2048          64          512        4x    80.18       98.5M
#        1024         128          256        2x    70.21       86.2M
#         512         256          128        1x    55.05       67.6M
#
# 1024 is the pick: 2x dilution instead of 16x, and 86.2M env steps still
# clears the 73.4M plateau (per-room sRSA 0.7870). Buying the last 1.27x of
# speed would cost 8x more dilution, and the brief is explicitly to not
# sacrifice the learning dynamics.
PPO_BATCH=1024
# wm_segment_stride=8 trains the world model on every 8th segment: 16 steps per
# update = 1 per 2048 environment steps, EXACTLY the reference series' ratio.
#
# This is a correction, not a tuning knob. Job 10444495 ran stride 1 (1 step per
# 256 env steps, 8x the reference) and its per-room sRSA PEAKED at 0.7732 by
# 25.2M steps - 98% of the reference plateau in a third of the experience - then
# FELL to 0.5581 by 83.9M while prediction loss kept improving. Loss down, place
# code down: over-training the predictor relative to experience. SWdist also ran
# 2.3x the reference throughout.
#
# Striding also makes the run FASTER, because the world model is most of an
# update: 17,815 -> 28,560 environment steps/s measured locally.
# rl.episodes_total IS the world-model gradient-step budget under serial
# training (schedule.py: total_wm_steps = total_steps/seqdur = episodes_total).
# 400,000 episodes = 102.4M environment steps (NOTE: with a stride, episodes
# no longer equal world-model gradient steps - 102.4M/2048 = 50,000 of those).
#
# Sized from the MEASURED PRODUCTION RATE, not the benchmark. tests/perf
# benchmarks reported 106.74 grad/s at 4 updates; a real run sustains
# 19,286 env steps/s = 75.3 grad/s, because a benchmark excludes archiving,
# checkpoint saves and the multi-room layout resampling. Believe the run.
#   setup ~10 min + 102.4M/~45,000 env steps/s = ~38 min training
#   + ~12 min offline scoring = ~1.0 h end to end.
# 102.4M env steps clears BOTH the 73.4M plateau (per-room sRSA 0.7870) and the
# 94.4M point (0.7905).
#
# archive_every_steps=8388608 gives 12 archives. At 2097152 it was 79, and
# scoring 79 checkpoints with --spatial does not fit in the window - the
# archive cadence is part of the time budget, not free.
#
# num_envs=128 rather than 512 DELIBERATELY. Measured on an L40S, gc scales
# 106.74 -> 125.35 -> 138.18 grad/s at 128/256/512, so 512 would buy 1.29x more
# steps - but each doubling also halves the policy's gradient steps per
# ENVIRONMENT step (ppo_epochs fixed, ppo_batch_size scaled with frames), and
# the policy generates the behaviour the world model learns from. 512 is 64x
# diluted against the num_envs=8 baseline. 1.29x more steps is not worth 4x
# more dilution; 128 already fits inside 2 h.
uv run python main_train.py env=$ENVCFG run=multienv exp.layouts=$LAYOUTS \
    predNet.batched_wm=False predNet.cuda_graph=True predNet.compile_cell=layer \
    predNet.wm_segment_stride=8 \
    exp.num_envs=$NUM_ENVS rl.frames=$FRAMES rl.ppo_batch_size=$PPO_BATCH \
    rl.episodes_total=400000 \
    logging.wandb_log=false \
    logging.archive_every_steps=8388608 logging.save_every_steps=8388608 \
    exp.analyze_agent_behav=False \
    logging.analysis_every_steps=0 logging.plot_every_steps=0 \
    exp.exp_name=fast-$LAYOUTS > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
# Never pipe training output through `tail`. Job 10444214 died with exit 1 and
# NO visible traceback because `| tail -40` showed the last 40 lines of the
# hydra config dump instead of the error - the same way job 10416788's gate
# failure was hidden. Full log to $DEST, excerpt printed here.
grep -vE '^Processing|^\s*$' "$DEST/train.log" | tail -25
[ -n "${TRAIN_RC:-}" ] && { echo "TRAINING FAILED rc=$TRAIN_RC"; tail -40 "$DEST/train.log"; exit $TRAIN_RC; }
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

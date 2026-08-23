#!/bin/bash
# Production multi-room training, tuned to finish inside 2 hours.
#
#   sbatch slurm/train_fast.sh [layouts] [envcfg] [branch] [seed]
#     layouts: single | one | rooms | pool   (default rooms)
#     $5 = entropy_coef (default 0.01; the 2026-07 reference used 0)
#              single = the original one-room MiniGrid-LRoom-v0 (env=lroom),
#                       for eyeballing against older single-room runs
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
# wandb is ON, and in-run sRSA/SWdist analysis with it. Both were off in the
# first working version for reasons that no longer hold:
#   - job 10444214 died at wandb init on a compute node. main_train.py now
#     degrades to no-wandb instead of aborting, so that cannot kill a run.
#   - job 10444320 died in kaleido, because wandb_log=false routes behaviour
#     analysis into save_analysis_of_agent_behav -> fig.write_image() -> a
#     headless browser. That path is ONLY reachable with wandb off. With wandb
#     on, plotSampleTrajectory logs wandb.Image(plt.gcf()) - matplotlib, no
#     browser - and log_behavior uses wandb.Plotly, which ships JSON. So
#     plot_every_steps is on: "Observation Sequence" (the prediction images)
#     plus the OPA figures, the same keys the 2026-07 reference logged.
#     Verified locally: all four figure keys reach wandb, exit 0.
#   - the spatial eval moves the pRNN off-device, which was unsafe under a
#     captured graph. The final config does not use cuda_graph, so it is safe.
# The offline scoring at the end still runs: wandb is the live view, the
# archives remain the checkable record.
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
# layouts=single means the ORIGINAL single-room env (env=lroom,
# MiniGrid-LRoom-v0), not a one-element multi-room set. That is the shape older
# runs used, so it is what a human can eyeball against them on wandb. It has no
# exp.layouts key at all, and needs device_env turned on explicitly because
# Configs/env/lroom.yaml defaults it False.
if [ "$LAYOUTS" = "single" ]; then
  ENVCFG=lroom
  LAYOUT_OVERRIDES="exp.device_env=True"
else
  LAYOUT_OVERRIDES="exp.layouts=$LAYOUTS"
fi
# $4 = seed. Replication is the point: every result document in this repo
# says "n = 1 seed per arm. No replication." A preset that cannot express a
# second seed guarantees that stays true.
SEED="${4:-2}"
# $5 = rl.entropy_coef. Configs/run/multienv.yaml sets 0.01 ("nothing
# resisted policy collapse at 0.0"), but the single-room reference run
# pRNN_curious_26-07-23-10-06-25 used 0, and at 0.01 the bonus is 0.0198
# against an advantage scale of 0.0369 - over half the learning signal - so
# policy_entropy pins near maximum and MI_policy collapses to ~0.015 against
# that run's 0.28. Being able to set it is the point.
ENT="${5:-0.01}"
# $6 = episodes (= env steps / seqdur). Default is this preset's own budget;
# pass 80000 to match pRNN_curious_26-07-23-10-06-25, whose config records
# steps: 20480000 / trajs_total: 80000, when comparing against that run.
EPISODES_ARG="${6:-}"
# $7 = wm_pool_group, $8 = ppo_batch_size. Both are regime knobs, not tuning:
# wm_pool_group sets world-model steps per unit of EXPERIENCE (1 per
# group*seqdur env steps) and ppo_batch_size sets policy steps per env step.
# The reference ran 1 wm step per 256 (serial) and 1 policy step per 64
# (ppo_batch_size=256 at frames=2048); this preset defaults to 1 per 2048 and
# 1 per 256, so those are the two axes that separate them.
POOL_ARG="${7:-}"
PB_ARG="${8:-}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)  layouts=$LAYOUTS env=$ENVCFG seed=$SEED ent=$ENT pool=$POOL_GROUP ppo_batch=$PPO_BATCH episodes=$EPISODES"
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
PPO_BATCH=${PB_ARG:-1024}
POOL_GROUP=${POOL_ARG:-8}
EPISODES=${EPISODES_ARG:-400000}                 # = world-model gradient-step budget / pool group
SEQDUR=256                      # predNet.seqdur; episodes are this many env steps
TOTAL_STEPS=$((EPISODES*SEQDUR))

# Cadences are DERIVED from a count of events, not written as raw step numbers.
# The quantity anyone actually reasons about is "how many points do I want on
# the curve"; a literal like 8388608 hides that and silently rescales the
# moment the budget changes.
#
# N_ANALYSIS=100 rather than the 12 an 8388608 literal gave: sRSA is noisy at
# this sample size (untrained 0.062 +/- 0.040; a trained single-room run swung
# 0.45-0.63 between adjacent events), so 12 points cannot separate a trend from
# the estimator. The 2026-07 reference logged ~50 over a 5x shorter run.
N_ANALYSIS=100
N_PLOT=12                       # figures are heavy; far fewer than the scalars
ANALYSIS_EVERY=$((TOTAL_STEPS/N_ANALYSIS))
PLOT_EVERY=$((TOTAL_STEPS/N_PLOT))

# exp.offpolicy_prnn_eval logs sRSA_offPolicy from a RANDOM agent beside the
# on-policy one. Without it a falling sRSA is ambiguous: the map may have
# degraded, or the policy may simply have sharpened and stopped visiting much
# of the room. The random walker's coverage does not change with training, so
# the pair separates those.
# wm_pool_group=8: 16 world-model steps per update, EACH POOLED OVER 8 SEGMENTS.
# That is 1 step per 2048 environment steps - the reference series' step COUNT -
# with the reference's gradient QUALITY. Arrived at by elimination, on measured
# cluster runs:
#
#   serial, every segment    1 step /  256 env steps, 1 segment  job 10444495
#     -> OVER-TRAINS: sRSA peaks 0.7732 @25.2M then falls to 0.5581 @83.9M
#        while loss keeps improving; SWdist 2.3x the reference
#   serial, stride 8         1 step / 2048 env steps, 1 segment  job 10444819
#     -> right COUNT, 8x the gradient variance. sRSA rises monotonically and
#        SWdist is BETTER than reference (0.008-0.019 vs 0.018-0.038), but it
#        reaches only 0.6383 against 0.7905 at the same step count
#   pool all 128             1 step /32768 env steps, 128 segs
#     -> 16x too few steps
#   pool in GROUPS OF 8      1 step / 2048 env steps, 8 segments  <- this
#
# This drops cuda_graph, which never engages on the pooled path
# (PRNNAdapter._use_graph_wm is consulted only in train_on_episode).
# compile_cell=layer still applies - it compiles pRNN.rnn.forward, which both
# paths use. Cost: 28,560 -> 16,029 env steps/s locally, still inside 2 h.
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
uv run python main_train.py env=$ENVCFG run=multienv $LAYOUT_OVERRIDES \
    predNet.batched_wm=True predNet.wm_pool_group=$POOL_GROUP predNet.compile_cell=layer \
    exp.num_envs=$NUM_ENVS rl.frames=$FRAMES rl.ppo_batch_size=$PPO_BATCH \
    rl.episodes_total=$EPISODES rl.entropy_coef=$ENT \
    logging.wandb_log=true \
    logging.archive_every_steps=8388608 logging.save_every_steps=8388608 \
    exp.offpolicy_prnn_eval=True \
    logging.analysis_every_steps=$ANALYSIS_EVERY logging.plot_every_steps=$PLOT_EVERY \
    exp.seed=$SEED exp.exp_name=fast-$LAYOUTS-e$ENT-g$POOL_GROUP-p$PPO_BATCH-s$SEED > "$DEST/train.log" 2>&1 || TRAIN_RC=$?
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
RUN=$(ls -dt outputs/fast-${LAYOUTS}-e${ENT}-g${POOL_GROUP}-p${PPO_BATCH}-s${SEED}_* 2>/dev/null | head -1)
# checkpoint_curve.py is the multi-room scorer and takes --layouts; skip it
# for the single-room shape, where sRSA/SWdist are logged live to wandb by
# the ordinary single-room eval path instead.
if [ -n "$RUN" ] && [ "$LAYOUTS" != "single" ]; then
  echo "=== offline spatial scoring of $RUN ==="
  uv run python scripts/multienv/checkpoint_curve.py --run "$RUN" \
      --env $ENVCFG --layouts $LAYOUTS --spatial 2>&1 | tail -25
  save
fi
echo "results in $DEST"

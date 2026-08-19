#!/usr/bin/env bash
# cuda_graph CORRECTNESS gate on 1-LRoom, SERIAL world-model training.
# The only difference between arms is predNet.cuda_graph.
#
# Arms run CONCURRENTLY on purpose. The claim under test is prediction loss as a
# function of GRADIENT-STEP NUMBER, which is invariant to how fast either arm
# runs, so sharing the GPU cannot bias it. The speed claim is NOT taken from
# here - it comes from tests/perf/benchmark.py on an idle GPU (doc section 3).
#
# In-run analysis and plotting are off in BOTH arms: under cuda_graph the
# model-moving diagnostics are skipped by construction, so leaving them on for
# the eager arm alone would charge it for work the other never does.
set -u
LOG="${2:-./gate_logs}"
mkdir -p "$LOG"
DUR="${1:-2400}"   # seconds per arm; $2 = log dir
COMMON="env=lroom_multi run=multienv exp.layouts=one \
predNet.batched_wm=False exp.seed=2 \
logging.wandb_log=false logging.analysis_every_steps=0 logging.plot_every_steps=0 \
logging.archive_every_steps=524288 logging.save_every_steps=0 \
logging.log_every_steps=204800"

for arm in nograph graph; do
  [ "$arm" = graph ] && CG=True || CG=False
  echo "launching $arm (cuda_graph=$CG) at $(date +%T), ${DUR}s cap"
  timeout "$DUR" uv run python main_train.py $COMMON \
      predNet.cuda_graph=$CG exp.exp_name=gate-$arm \
      > "$LOG/$arm.log" 2>&1 &
done
wait
echo "GATE DONE $(date +%T)"

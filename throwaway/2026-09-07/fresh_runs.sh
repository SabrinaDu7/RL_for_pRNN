#!/bin/bash
# The two fresh runs of the cleanup check, back to back on the local GPU.
cd /home/sabrina/Documents/experiments/RL_for_pRNN
export PYTHONUNBUFFERED=1 CG_DEVICE=cuda
L=throwaway/2026-09-07/fresh_runs.log
echo "$(date '+%H:%M:%S') start parity (reference parity-s2-mseB_curious_26-08-31-01-32-31)" >> $L
uv run python throwaway/2026-09-07/rerun.py "blake-richards/curious-george/parity-s2-mseB_curious_26-08-31-01-32-31" clean-sept-parity-s2-mseB curious-george > throwaway/2026-09-07/parity.log 2>&1
echo "parity exit $? $(date '+%H:%M:%S')" >> $L
uv run python throwaway/2026-09-07/rerun.py outputs/fetched/mx-impassable-n8-s2-focal5mlp_curious_26-08-31-23-18-20 clean-sept-mx-impassable-n8-s2-focal5mlp curious-george-multienv > throwaway/2026-09-07/multienv.log 2>&1
echo "multienv exit $? $(date '+%H:%M:%S')" >> $L
echo "ALL DONE $(date '+%H:%M:%S')" >> $L

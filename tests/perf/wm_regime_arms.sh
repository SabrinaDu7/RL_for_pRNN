#!/usr/bin/env bash
# The four world-model regimes, on the only axis that matters (gradient steps/s).
# Produces the table in docs/exp_speed_cuda_graph_2026-08-19.md section 3.
#
#   bash tests/perf/wm_regime_arms.sh /tmp/wm_arms
#
# Arms run STRICTLY SEQUENTIALLY and the GPU must be otherwise idle - check
# `nvidia-smi --query-compute-apps` first. GPU contention is what made the
# 2026-08-18 session report a 4% change as 1.29x.
#
# D (pooled + graph) is the NEGATIVE CONTROL: predNet.cuda_graph is wired only
# into PRNNAdapter.train_on_episode (serial), never train_on_episodes_batched,
# so D must match A. If it does not, the graph is engaging somewhere unintended.
# Gradient-steps/s across world-model regimes on the production multienv config.
# Strictly sequential: a shared GPU is what invalidated the previous session's
# width measurement.
set -u
OUT="${1:?usage: bash $0 <output-dir>}"
mkdir -p "$OUT"
BASE="env=lroom_multi run=multienv"
run () {  # name, extra overrides...
  local name=$1; shift
  local ov=""
  for o in $BASE "$@"; do ov="$ov --override $o"; done
  echo "=== $name : $* ==="
  timeout 1800 uv run python tests/perf/benchmark.py --updates 6 --warmup-updates 1 \
      --out "$OUT/$name.json" $ov > "$OUT/$name.log" 2>&1
  echo "  exit=$?"
  mkdir -p "$OUT"
  uv run python - "$OUT/$name.json" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("   NO JSON:",e); raise SystemExit
m=d["metrics"]
print(f"   fps={d['fps']:>9}  updates/s={d['updates_per_s']:>7}  "
      f"WMgrad={d['wm_grad_steps']:>4}  GRAD/s={d['wm_grad_steps_per_s']:>7}  "
      f"s/upd={d['total_train_s']/d['meta']['n_updates']:.3f}")
print(f"   prnn_loss[-1]={m['prnn_loss'][-1] if m['prnn_loss'] else float('nan'):.6f}  "
      f"entropy[-1]={m['entropy'][-1]:.4f}")
for k,v in list(d["timings"].items())[:6]:
    print(f"     {k:<28}{v['total_s']:>8.3f}s  {v['calls']:>6} calls")
PYEOF
}
run A_pooled_nograph   predNet.batched_wm=True  predNet.cuda_graph=False
run B_serial_nograph   predNet.batched_wm=False predNet.cuda_graph=False
run C_serial_graph     predNet.batched_wm=False predNet.cuda_graph=True
run D_pooled_graph     predNet.batched_wm=True  predNet.cuda_graph=True

# Branch sdu/gpu-batched-wm: batched world-model training + RNN loop fusion

Goal: make the GPU (or wider CPU) worthwhile by changing the WORK SHAPE, per
the 2026-07-13 analysis: at B=8/hidden=500 the kernels are too small and too
sequential for a GPU (measured 558 FPS @ ~91% idle vs 674 CPU). Two levers,
both behind flags, both curve-gated before becoming defaults.

## Point 2 - batched segment training (the near-term win)

Today `train_world_model_on_episodes` runs 8 SEQUENTIAL trainSteps per update,
each a (1, 256) BPTT - the dominant cost (~1.1 s/update) in the worst possible
shape. Stack the 8 equal-length segments into ONE (8, 256) batched
forward/backward: 8x wider matmuls, 8x fewer Python-loop iterations.

- Flag: `predNet.batched_wm: false` (default off - this is an OPTIMIZATION-
  SEMANTICS change: 8 optimizer steps become 1 step on the pooled gradient,
  like raising batch size 8x; possibly scale predNet.lr when gating).
- Landmine to verify FIRST: adapter.py's docstring claims prnn's
  `predict(batched=True)` has a permutation bug (that note was written about
  the single-step path). Before building on pRNN.forward(batched=True), test
  batched-vs-serial forward equality (dropout off / eval mode, zero noise);
  if the bug is real, fix it in prnn (branch sdu/prnn-perf-optim, repin) or
  drive pRNN.rnn(batched=True) directly like BatchedSRTracker does.
- Prereq check: segments must be equal length (guaranteed by seqdur cuts in
  training; assert + serial fallback for ragged cases).
- Gates: (1) exact - batched forward outputs equal serial forwards
  (unit test, no dropout/noise); (2) speed - benchmark harness; (3) CURVE
  GATE before default flip - bsweep-style multi-seed comparison of
  batched_wm on/off (pRNN loss, curious rewards, SI/sRSA trajectories),
  since the optimizer path genuinely changes.

## Point 3 - RNN loop fusion (compile / CUDA graphs)

Kill per-timestep kernel-launch + Python overhead in thetaRNN's sequential
loop (the reason the GPU idles even when fed). Order of attempts:
1. `torch.compile(mode="reduce-overhead")` on the CELL (not the layer) -
   smallest blast radius, cell is loop-free.
2. Compile/capture the per-step loop body (cell + mask indexing); known
   blockers catalogued in perf_baseline.md: np.mod on Python ints, asserts
   in forward, dynamic segment lengths, single/batched branches. All are
   prnn-repo changes -> commit to sdu/prnn-perf-optim, repin.
3. CUDA graphs only if (1)+(2) leave launch overhead dominant on GPU.
- Gates: bitwise where the math is untouched (compile should be
  numerics-preserving for these ops, verify with the metric harness at tight
  rtol); FPS on both CPU and GPU; fall back flag `predNet.compile: false`.

## Expected outcome

Batched wm at B=8: wm_train 1.1 s -> ~0.2-0.3 s/update (CPU), more on GPU.
Combined with compile, the GPU question gets re-benchmarked (async_bench.sh
GPU section) - if utilization stays single-digit after both, the honest
conclusion is that this workload is CPU-shaped until hidden_size/B grow.

## Baseline (branch point d3ee72b)

Suite 105 passed / 0 failed; cluster B=8 sync 674 FPS; wm_train ~7.4 s per
5 updates (cluster, 16 cpus); full run ~6 h wall (Sabrina's 2026-07-14 run).

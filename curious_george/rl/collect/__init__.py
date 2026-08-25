# Rollout collection: the hot loop, its CUDA graph, and the science bookkeeping
# observed along the way. Re-exports live one level up in `curious_george.rl`,
# so this file exists to make the directory a real package rather than a
# namespace one - `pkgutil.walk_packages` skips namespace packages, which hid
# this whole subtree (collector, rollout_graph, diagnostics, agent, format)
# from `uv run pypatree`.

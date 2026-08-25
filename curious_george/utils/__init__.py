# Shared primitives with no dependency on the training or RL packages: the
# device handle, enums, timing, checkpoint IO, CUDA-graph helpers. Deliberately
# import-free so anything may depend on it.
#
# Exists for the same reason as rl/collect/__init__.py: without it the
# directory is a namespace package and `pkgutil.walk_packages` omits it, so it
# never appeared in `uv run pypatree`.

"""CUDA-graph capture of one PPO minibatch gradient step.

Why this exists: with the world model graphed, `update/policy` is the largest
remaining stage - 1822 ms of a 3229 ms update at `ppo_batch_size=256`, because
one update takes `ppo_epochs * frames / batch_size` = 512 gradient steps on a
small actor-critic. Each is ~3.6 ms of almost pure PyTorch dispatch: the model
is tiny, so the GPU is idle while Python decides what to launch. Measured
3.558 -> 0.275 ms/step captured, **12.9x**.

Two things make this harder than the world model's graph, and both are about
addresses, which a captured graph bakes in:

1. **`exps` is rebuilt every update.** A graph capturing reads straight from it
   would dangle after the first rollout. So the fields the loss touches are
   mirrored into STATIC buffers that `bind()` refills per update; the graph
   only ever reads the mirror.
2. **The minibatch indices change every step.** `inds` is likewise a static
   buffer copied into before each replay, so one capture serves all
   `frames / batch_size` minibatches of every epoch.

And one that is about the host: the policy's `Categorical` would host-sync
during capture. `curious_george.utils.cuda_graph.no_dist_validation` owns that
story.
"""

from __future__ import annotations

import torch
from torch_ac.utils import DictList

from curious_george.utils.cuda_graph import no_dist_validation

# Fields `_index_policy_batch` reads. `obs.direction` is nested and handled
# separately; `obs.image` is deliberately absent, matching that function's
# with_CV=False path.
_FLAT_FIELDS = ("SR", "action", "value", "advantage", "returnn", "log_prob")


class GraphPolicyTrainer:
    """One captured PPO minibatch step, replayed per minibatch.

    Lifetime is one training run: `bind(exps)` once per update, `step(inds)`
    per minibatch. Graphs are keyed by minibatch size so a ragged final batch
    (when `batch_size` does not divide `frames`) captures its own rather than
    replaying one built for the wrong shape.
    """

    def __init__(self, acmodel, optimizer, *, loss_fn, loss_kwargs: dict,
                 max_grad_norm: float) -> None:
        self.acmodel = acmodel
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs
        self.max_grad_norm = float(max_grad_norm)
        self.static: DictList | None = None
        self.graphs: dict[int, dict] = {}
        self._param_ptrs: tuple[int, ...] = ()
        self._ensure_capturable_optimizer()

    # --- parameter-address safety, same contract as _GraphWMTrainer ---------
    def _fingerprint(self) -> tuple[int, ...]:
        return tuple(p.data_ptr() for p in self.acmodel.parameters())

    def _invalidate_if_moved(self) -> None:
        if self.graphs and self._fingerprint() != self._param_ptrs:
            self.graphs.clear()

    def _ensure_capturable_optimizer(self) -> None:
        """Adam must be `capturable=True` to have its step recorded.

        Rebuilt preserving every param group, and only from empty state: a
        populated optimizer means a resumed run, whose accumulated moments
        would be silently dropped by the rebuild.
        """
        opt = self.optimizer
        assert type(opt) is torch.optim.Adam, (
            f"policy graph assumes Adam, got {type(opt).__name__}"
        )
        if opt.param_groups[0].get("capturable", False):
            return
        assert all(len(st) == 0 for st in opt.state.values()), (
            "policy graph supports fresh runs only (optimizer state must be empty)"
        )
        groups = []
        for g in opt.param_groups:
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = list(g["params"])
            ng["capturable"] = True
            groups.append(ng)
        self.optimizer = torch.optim.Adam(groups)

    # --- static mirror of the rollout ---------------------------------------
    def bind(self, exps) -> None:
        """Point the graph at this update's rollout by refilling the mirror.

        Allocated on first call; `rl.frames` is fixed for a run, so every later
        update is a copy into the same storage - which is the whole point, since
        the captured graph reads those addresses.
        """
        # NOTE: attribute access, not `exps[f]`. torch_ac's DictList overrides
        # __getitem__ to index every VALUE by the argument, so `exps["SR"]`
        # means "index each field with the string 'SR'", not "the SR field".
        if self.static is None:
            self.static = DictList({f: getattr(exps, f).clone() for f in _FLAT_FIELDS})
            self.static.obs = DictList()
            if hasattr(exps.obs, "direction"):
                self.static.obs.direction = exps.obs.direction.clone()
            return
        for f in _FLAT_FIELDS:
            getattr(self.static, f).copy_(getattr(exps, f))
        if hasattr(self.static.obs, "direction"):
            self.static.obs.direction.copy_(exps.obs.direction)

    # --- capture and replay --------------------------------------------------
    def _region(self, inds_s):
        from curious_george.rl.update.policy import _index_policy_batch

        sb = _index_policy_batch(self.static, inds_s, self.acmodel)
        dist, value = self.acmodel(sb.obs, SR=sb.SR)
        loss, terms = self.loss_fn(dist, value, sb, **self.loss_kwargs)
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.acmodel.parameters(), self.max_grad_norm
        ).detach()
        self.optimizer.step()
        # One stacked stats row, so a replay costs the caller ONE accumulate
        # kernel instead of five scalar reads.
        return torch.stack([
            terms.policy_entropy_bits, terms.value_mean,
            terms.policy_loss, terms.value_loss, grad_norm,
        ])

    def _capture(self, inds: torch.Tensor) -> None:
        inds_s = inds.detach().clone()

        # Warmup and capture run REAL optimizer steps, so snapshot the weights
        # and Adam's moments and restore them in place afterwards - keeping the
        # same tensors the graph captured, so the next replay continues from
        # exactly the pre-capture state.
        w_snap = [p.detach().clone() for p in self.acmodel.parameters()]
        opt_snap = {
            id(p): {k: v.detach().clone() for k, v in st.items() if isinstance(v, torch.Tensor)}
            for p, st in self.optimizer.state.items()
        }

        with no_dist_validation():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._region(inds_s)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stats_s = self._region(inds_s)

        with torch.no_grad():
            for p, w in zip(self.acmodel.parameters(), w_snap):
                p.copy_(w)
            for p, st in self.optimizer.state.items():
                prior = opt_snap.get(id(p), {})
                for k, v in st.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    v.copy_(prior[k]) if k in prior else v.zero_()

        self.graphs[inds_s.numel()] = dict(graph=graph, inds_s=inds_s, stats_s=stats_s)
        self._param_ptrs = self._fingerprint()

    def step(self, inds: torch.Tensor) -> torch.Tensor:
        """Run one graphed gradient step on `inds`; returns its stats row
        (policy_entropy_bits, value_mean, policy_loss, value_loss, grad_norm)."""
        self._invalidate_if_moved()
        key = inds.numel()
        if key not in self.graphs:
            self._capture(inds)
        g = self.graphs[key]
        g["inds_s"].copy_(inds)
        g["graph"].replay()
        return g["stats_s"]

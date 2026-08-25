"""The seam between the RL code and the pRNN world model.

Everything that touches `prnn` at rollout/training time goes through
`PRNNAdapter`. Nothing else in curious_george should call
`pN.env_shell.env2pred`, `pN.predict*`, or `pN.trainStep` directly.

Temporal conventions (confirmed against ../pRNN source):

- All `*_5win` architectures set `predOffset=0`, so `predict()` returns
  `obs_pred[t]` targeting `obs[t]` (the SAME timestep); the "prediction" comes
  from `inMask` zeroing the observation input on 5 of every 6 steps, not from
  a +1 offset. Docstrings in prnn claiming t+1 describe the base-class default
  (`predOffset=1`) which every 5win subclass overrides.
- `pastSR` (== not a `prevAct` architecture): action index t is the action
  taken AFTER observing obs[t], HD comes from the current step (`SpeedHD`),
  and the hidden state aligns to the current/past position. `prevAct`
  architectures shift actions by one and pair with `SpeedNextHD`
  (HD from the next step); their hidden state aligns to the next step.
"""

import numpy as np
import torch

from prnn.utils import PredictiveNet
from curious_george.utils.timing import timer

FORWARD_IDX = 2  # ActionEncodings.forwardIDX - the action bit SpeedHD keeps


def flat_obs_rows(obs_dicts) -> torch.Tensor:
    """Raw obs dicts -> (N, H*W*C) float32 rows in [0,1].

    Bitwise-equal to env2pred's per-dict get_visual loop + /255, with one
    numpy stack instead of N Python-level reshapes/copies.
    """
    images = [obs["image"] for obs in obs_dicts]
    if torch.is_tensor(images[0]):
        return torch.stack(images).reshape(len(images), -1).to(torch.float32) / 255
    stacked = np.stack(images)
    return (
        torch.from_numpy(stacked.reshape(len(images), -1)).to(torch.float32)
        / 255
    )


def encode_speed_hd_rows(act_np, hd_np, num_acts: int, num_hd: int) -> torch.Tensor:
    """Vectorized SpeedHD rows: [action one-hot, forward bit only | HD one-hot].

    int64 (N, num_acts + num_hd), matching ActionEncodings.SpeedHD per row.
    A negative action leaves its action block zero (OneHot's no-action flag,
    applied per row - the per-env length-1 sequences this replaces had one
    flag per row anyway).
    """
    a = torch.as_tensor(np.asarray(act_np), dtype=torch.int64).reshape(-1)
    hd = torch.as_tensor(np.asarray(hd_np), dtype=torch.int64).reshape(-1)
    out = torch.zeros((len(a), num_acts + num_hd), dtype=torch.int64)
    out[a == FORWARD_IDX, FORWARD_IDX] = 1
    out[torch.arange(len(a)), num_acts + hd] = 1
    return out


def encode_speed_hd_seq(act_np, hd_np, num_acts: int, num_hd: int) -> torch.Tensor:
    """Sequence variant of encode_speed_hd_rows, shape (1, L, num_acts+num_hd).

    Keeps ActionEncodings.OneHot's SEQUENCE-level no-action flag: act[0] < 0
    zeroes the action block for the whole sequence.
    """
    rows = encode_speed_hd_rows(act_np, hd_np, num_acts, num_hd)
    if len(rows) and np.asarray(act_np).reshape(-1)[0] < 0:
        rows[:, :num_acts] = 0
    return rows.unsqueeze(0)


def infer_past_sr(predictive_net: PredictiveNet) -> bool:
    """pastSR is determined by the architecture family (see module docstring).

    Detection is by pRNNtype key, not str(pN.pRNN): upstream prnn builds the
    architectures from partial(MaskedRNN, ...) factories, so the class repr no
    longer contains "prevAct".
    """
    return "prevAct" not in predictive_net.pRNNtype


def validate_action_encoding(predictive_net: PredictiveNet, env, pastSR: bool) -> None:
    """The env's action encoding must match the architecture's convention:
    pastSR=True pairs with SpeedHD (current HD), pastSR=False with
    SpeedNextHD (next HD). Mismatch silently misaligns SRs by one step.
    """
    assert pastSR ^ ("Next" in str(env.encodeAction)), (
        f"Action encoding {env.encodeAction} inconsistent with "
        f"architecture {type(predictive_net.pRNN).__name__} (pastSR={pastSR})"
    )


class _GraphCuriosityForward:
    """CUDA-graph capture of the curiosity forward pass.

    The last ungraphed region of a training update, and the one the pooled
    regime pays for most: `wm_pool_group=8` needs 8x the updates of g=1 for
    the same world-model gradient-step budget, so it pays this per-update cost
    eight times over (0.3 vs 2.3 min/run at 80,000 steps).

    Much simpler than `_GraphWMTrainer` below, because there is nothing to
    undo: one `pN.predict` under `no_grad`, no backward, no optimizer step, so
    warmup and capture leave the weights exactly as they found them. Only the
    RNG advances - dropout and the initial-state noise run INSIDE the graph
    and draw fresh on every replay, as they do in eager.

    Keyed on shape. `target_offset` changes the sequence length (the next_obs
    alignment appends a zero-action step) and `exp.num_envs` changes the
    batch, so one capture per (batch, obs length, action length).

    The returned tensors are the graph's STATIC OUTPUTS, valid until the next
    replay. `PRNNAdapter._curiosity_errors` is the only caller and reduces
    them immediately, which is why they never escape the adapter.
    """

    def __init__(self, pN: PredictiveNet, device: torch.device):
        self.pN = pN
        self.device = device
        self.graphs: dict[tuple[int, int, int], dict] = {}
        self._addresses: tuple[int, ...] = ()

    def _addresses_now(self) -> tuple[int, ...]:
        """Parameter AND buffer storages, for the reason `_GraphWMTrainer.
        _fingerprint` gives: a captured graph reads the addresses it saw at
        capture time, and a device round trip can replace them. Buffers are in
        here because `thRNN_5win` multiplies by `inMask_f`/`actMask_f`/
        `outMask_f` inside the captured region, and a stale mask fails
        silently rather than loudly.
        """
        return tuple(
            x.data_ptr()
            for x in (*self.pN.pRNN.parameters(), *self.pN.pRNN.buffers())
        )

    def _capture(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> None:
        obs_s = obs_b.detach().clone()
        act_s = act_b.detach().clone()

        def region():
            with torch.no_grad():
                obs_pred, obs_next, _ = self.pN.predict(obs_s, act_s, batched=True)
            return obs_pred, obs_next

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                region()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pred_s, next_s = region()

        self.graphs[(obs_s.size(0), obs_s.size(1), act_s.size(1))] = dict(
            graph=graph, obs_s=obs_s, act_s=act_s, pred_s=pred_s, next_s=next_s
        )
        self._addresses = self._addresses_now()

    def predict(
        self, obs_b: torch.Tensor, act_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(obs_pred, obs_next)` for obs_b (N, L+1, X), act_b (N, L, A),
        both already on device. Equal shapes share one graph."""
        if self.graphs and self._addresses_now() != self._addresses:
            self.graphs.clear()  # a device round trip freed the old storages
        key = (obs_b.size(0), obs_b.size(1), act_b.size(1))
        if key not in self.graphs:
            self._capture(obs_b, act_b)
        g = self.graphs[key]
        g["obs_s"].copy_(obs_b)
        g["act_s"].copy_(act_b)
        g["graph"].replay()
        return g["pred_s"], g["next_s"]


class _GraphWMTrainer:
    """CUDA-graph capture of the world-model trainStep, serial or pooled.

    Equivalent to `PredictiveNet.trainStep` for the MaskedRNN / SpeedHD /
    theta==0 path with the RL defaults (with_homeostat=False, no encoder
    training). The captured region is `pN.predict` + MSE loss + backward +
    one RMSprop step, over static raw obs/act buffers refilled per call.

    Serves BOTH training paths. `batched` selects `predict`'s layout and is
    part of the graph key, because at B=1 the batched layout (1, L, X, B) is
    not the serial one (1, L, X) - reusing one graph for both would silently
    feed the wrong shape. `train_on_episode` captures batched=False at B=1;
    `train_on_episodes_batched` captures batched=True at B=wm_pool_group.

    The optimizer step is INSIDE the captured region, so weights advance
    between replays: N sequential replays reproduce N sequential eager
    trainSteps, which is what preserves `wm_pool_group`'s regime (16 steps per
    update, each pooled over 8 segments, each at the previous step's weights).
    Gated bitwise in tests/test_cuda_graph_wm.py with dropout and noise off.

    `predict()` runs WHOLE inside the graph - dropout and internal noise
    included, both of which use CUDA's graph-safe RNG (verified to advance
    per replay, so every replay draws fresh). This requires the prnn
    capture-safe masking fix (float-mask multiplies in clip_mask/predict,
    branch sdu/rl-integration): the previous numpy-bool scatters could not be
    captured, which is why an earlier version hand-rolled predict's tail here.

    ONE optimizer step per segment - identical step COUNT and math to the
    serial eager path, so this needs no curve gate (bitwise with dropout+noise
    off; distributionally identical otherwise, since the in-graph RNG stream
    realizes different-but-equivalent draws than pure-eager order).

    Fresh-run only: the optimizer is rebuilt capturable, which requires empty
    optimizer state (asserted). Resuming a run with populated optimizer state
    onto this path is not yet supported.
    """

    def __init__(self, pN: PredictiveNet, device: torch.device):
        self.pN = pN
        self.device = device
        # (batched, B, L+1) -> static buffers + graph. B is in the key because
        # a ragged final group (when wm_pool_group does not divide the segment
        # count) is a different shape and needs its own capture.
        self.graphs: dict[tuple[bool, int, int], dict] = {}
        self._param_ptrs: tuple[int, ...] = ()
        self._ensure_capturable_optimizer()

    def _fingerprint(self) -> tuple[int, ...]:
        """Storage addresses of the pRNN parameters.

        A captured graph writes to the addresses its parameters had AT CAPTURE
        TIME. `Module.to()` replaces `param.data`, so ANY device round-trip
        leaves every captured graph pointing at freed memory - a use-after-free
        that silently corrupts whatever the allocator hands those blocks to
        next, and only sometimes crashes.

        This bit us for real: `logging.plot_interval`/`analysis_interval`
        (both 200) wrap plotting and the spatial eval in
        `on_device([predictiveNet, acmodel], "cpu")` (training/loop.py,
        evaluation/spatial.py). A 2026-07-22 cluster run died 7 updates after
        the update-200 event with `direction`=184 - the freed parameter block
        had been reused for `exps.obs.direction` and graph replay overwrote it.

        Guarding inside `PRNNAdapter.to()` is NOT sufficient: `on_device` calls
        `world_model.device._move()` straight on the PredictiveNet and never
        goes through the adapter. Checking the addresses before every replay
        covers every path, present and future.
        """
        return tuple(p.data_ptr() for p in self.pN.pRNN.parameters())

    def _invalidate_if_moved(self) -> None:
        if self.graphs and self._fingerprint() != self._param_ptrs:
            self.graphs.clear()

    def _ensure_capturable_optimizer(self) -> None:
        """Rebuild pN.optimizer with capturable=True, PRESERVING every param
        group. PredictiveNet builds RMSprop with per-group lr/weight_decay
        (W/W_out/W_in scaled by rootk, a bias group scaled by bias_lr) and
        non-default alpha/eps - collapsing them changes the learning rates and
        silently breaks equivalence."""
        opt = self.pN.optimizer
        assert type(opt) is torch.optim.RMSprop, (
            f"predNet.cuda_graph assumes plain RMSprop, got {type(opt).__name__} "
            "(RMSpropEG / eg_lr path not supported)"
        )
        if opt.param_groups[0].get("capturable", False):
            return
        assert all(len(st) == 0 for st in opt.state.values()), (
            "predNet.cuda_graph supports fresh runs only "
            "(optimizer state must be empty at capture time)"
        )
        groups = []
        for g in opt.param_groups:
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = list(g["params"])
            ng["capturable"] = True
            groups.append(ng)
        self.pN.optimizer = torch.optim.RMSprop(groups)

    def _capture(self, obs_b: torch.Tensor, act_b: torch.Tensor, *, batched: bool) -> None:
        pN, opt = self.pN, self.pN.optimizer
        obs_s = obs_b.detach().clone()
        act_s = act_b.detach().clone()

        def region():
            obs_pred, obs_next, h = pN.predict(obs_s, act_s, batched=batched)
            if isinstance(obs_pred, tuple):
                obs_pred = obs_pred[0]
            loss = pN.loss_fn(obs_pred, obs_next, h)
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()
            return loss

        # Snapshot to undo the warmup's effect: this region runs REAL optimizer
        # steps during warmup and capture. The optimizer state is snapshotted
        # (not just zeroed) because a re-capture can happen MID-TRAINING - a
        # new segment length, or a device move invalidating the graphs - and
        # zeroing would silently wipe RMSprop's accumulated second moments.
        w_snap = [p.detach().clone() for p in pN.pRNN.parameters()]
        opt_snap = {
            id(p): {k: v.detach().clone() for k, v in st.items() if isinstance(v, torch.Tensor)}
            for p, st in opt.state.items()
        }

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                region()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss_s = region()

        # Restore pristine weights and optimizer state IN PLACE - keeping the
        # SAME tensors the graph captured - so the next replay continues from
        # exactly the pre-capture state. Params with no prior state (the first
        # capture) are zeroed, which is a fresh RMSprop start.
        with torch.no_grad():
            for p, w in zip(pN.pRNN.parameters(), w_snap):
                p.copy_(w)
            for p, st in opt.state.items():
                prior = opt_snap.get(id(p), {})
                for k, v in st.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    v.copy_(prior[k]) if k in prior else v.zero_()

        self.graphs[(batched, obs_s.size(0), obs_s.size(1))] = dict(
            graph=graph, obs_s=obs_s, act_s=act_s, loss_s=loss_s
        )
        self._param_ptrs = self._fingerprint()

    def train_batch(self, obs_b: torch.Tensor, act_b: torch.Tensor, *, batched: bool) -> None:
        """One graphed pRNN gradient step on obs_b (B, L+1, X), act_b (B, L, A),
        both already on device. Equal shapes share one graph."""
        self._invalidate_if_moved()  # a device round-trip freed the old params
        key = (batched, obs_b.size(0), obs_b.size(1))
        if key not in self.graphs:
            self._capture(obs_b, act_b, batched=batched)
        g = self.graphs[key]
        g["obs_s"].copy_(obs_b)
        g["act_s"].copy_(act_b)
        g["graph"].replay()
        self.pN.recordTrainingTrial(g["loss_s"].item())
        self.pN.numTrainingEpochs += 1

    def train_segment(self, obs: torch.Tensor, act: torch.Tensor) -> None:
        """obs (L+1, X_obs) float in [0,1], act (L, A) - the `_episode_tensors`
        contract. The serial path's single segment, at B=1 and batched=False."""
        self.train_batch(
            obs.unsqueeze(0).to(self.device), act.unsqueeze(0).to(self.device),
            batched=False,
        )


class PRNNAdapter:
    def __init__(
        self,
        predictive_net: PredictiveNet,
        device: torch.device,
        pastSR: bool,
        cuda_graph: bool = False,
        batched_curiosity: bool = False,
        compile_cell: bool | str = False,
        curiosity_cuda_graph: bool = False,
    ):
        self.pN = predictive_net
        self.device = device
        self.pastSR = pastSR
        # Theta-cycle nets (thcyc*) roll k+1 windows along dim 0 of predict()'s
        # returns; masked nets (thRNN_5win*) have no .k attribute.
        self.theta = "thcyc" in self.pN.pRNNtype
        if self.theta:
            self.k = self.pN.pRNN.k + 1

        # Vectorized obs/action formatting replicates the SpeedHD encoding
        # only; anything else falls back to env_shell.env2pred per call.
        shell = self.pN.env_shell
        self.fast_speedhd = getattr(shell.encodeAction, "__name__", "") == "SpeedHD"
        self.num_acts = shell.action_space.n
        self.num_hd = shell.numHDs

        # CUDA-graph world-model training (predNet.cuda_graph). Only viable on a
        # CUDA device for the maskless-theta SpeedHD path; trainer is built
        # lazily on first use so weights are already on-device.
        self.cuda_graph = bool(cuda_graph) and self.device.type == "cuda"
        self._graph_trainer: _GraphWMTrainer | None = None
        self.batched_curiosity = bool(batched_curiosity)

        # predNet.curiosity_cuda_graph: replay the batched curiosity forward.
        # Built lazily on first use, and only ever reached by the two batched
        # curiosity paths - a serial-curiosity run never captures anything, so
        # the flag needs no compatibility check of its own.
        self.curiosity_cuda_graph = (
            bool(curiosity_cuda_graph) and self.device.type == "cuda"
        )
        self._graph_curiosity: _GraphCuriosityForward | None = None

        # predNet.compile_cell: fuse the recurrent cell with torch.compile.
        # The 256-step loop is dispatch-bound across many tiny ops - ranked by
        # TIME, `mm` is 26.6% and the LayerNorm chain ~28%, with no single hot
        # spot (docs/exp_speed_cuda_graph_2026-08-19.md 9d). Fusing the cell
        # collapses that chain into one kernel: measured 1.39x on the
        # world-model step on CUDA, 1.01x on CPU, so it is CUDA-only.
        #
        # DEFAULT mode deliberately, not "reduce-overhead" - that mode IS CUDA
        # graphs, and would silently reintroduce the captured-memory-pool
        # failure this flag exists to avoid.
        #
        # NOT semantics-free: fusion may reorder floating-point operations, so
        # this needs the same learning gate as any other change. It carries no
        # captured parameter addresses, which is what separates it from
        # predNet.cuda_graph.
        # "layer" compiles the WHOLE 256-step loop, which dynamo unrolls into
        # one graph - the torch analogue of jax.lax.scan + XLA fusion, and far
        # better than fusing the cell alone: 3.89x vs 1.14x on the isolated
        # layer. "cell" fuses only the LayerNorm chain inside one step.
        #
        # Verified graph-free: torch._inductor.config.triton.cudagraphs is
        # False in DEFAULT mode, so nothing is captured and no memory pool is
        # recorded. That is the whole point of preferring this to cuda_graph.
        #
        # ⚠️ "layer" specialises on sequence length (dynamic=False). pN.predict
        # is called from several sites, and a new length costs a fresh ~89 s
        # compile. Measure recompiles before trusting it in production.
        self.compile_mode = str(compile_cell) if compile_cell else ""
        if compile_cell and self.device.type == "cuda":
            if self.compile_mode == "layer":
                rnn = self.pN.pRNN.rnn
                rnn.forward = torch.compile(rnn.forward, dynamic=False)
            else:
                cell = self.pN.pRNN.rnn.cell
                cell.forward = torch.compile(cell.forward)

    def seq2pred(self, obs_dicts, act_np):
        """env_shell.env2pred equivalent (bitwise, SpeedHD) without per-item
        Python loops. obs_dicts has len(act_np)+1 entries, as env2pred expects."""
        if not self.fast_speedhd:
            return self.pN.env_shell.env2pred(obs_dicts, act_np)
        obs = flat_obs_rows(obs_dicts).unsqueeze(0)
        hd = [o["direction"] for o in obs_dicts[:-1]]
        act = encode_speed_hd_seq(act_np, hd, self.num_acts, self.num_hd)
        return obs, act

    @property
    def hidden_size(self) -> int:
        return self.pN.hidden_size

    def to(self, device) -> None:
        self.pN.pRNN.to(device)

    def reset_state(self) -> None:
        self.pN.reset_state(device=str(self.device))

    def init_sr(self, obs) -> torch.Tensor:
        """SR before the first action of an episode.

        pastSR nets start from zeros (the SR for step 0 is 'from step -1');
        next-step nets bootstrap from a zero-action prediction on the first obs.
        """
        if self.pastSR:
            return torch.zeros((1, self.pN.hidden_size), device=self.device)

        obs_pN, act_pN = self.pN.env_shell.env2pred([obs, obs], np.array([0]))
        act_pN = torch.zeros_like(act_pN)
        obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
        with torch.no_grad():
            SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        return SR

    def next_sr(self, act, obs) -> torch.Tensor:
        """SR for step t based on obs and action from step t-1.

        The caller chooses which obs to pass: the pre-action obs (pastSR) or
        the post-action obs (next-step nets).
        """
        if self.theta:
            obs = [obs] * (self.k + 1)
            act = act.repeat(self.k)

            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                SR = self.pN.predict(obs_pN, act_pN)[2][0]
        elif self.fast_speedhd:
            # single-row conversion; bitwise-equal to env2pred([obs, obs], act)
            # followed by the [:, :-1, :] slice the historical path took
            obs_pN = flat_obs_rows([obs]).unsqueeze(0).to(self.device)
            act_pN = (
                encode_speed_hd_rows(act, [obs["direction"]], self.num_acts, self.num_hd)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                SR = self.pN.predict_single(obs_pN, act_pN).squeeze(dim=0)
        else:
            obs = [obs, obs]

            obs_pN, act_pN = self.pN.env_shell.env2pred(obs, act)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad():
                SR = self.pN.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)
        return SR

    def _curiosity_errors(
        self,
        obs_b: torch.Tensor,
        act_b: torch.Tensor,
        *,
        target_offset: int,
    ) -> torch.Tensor:
        """Per-step observation-prediction MSE for equal-length segments.

        `obs_b` (N, L+1+target_offset, X_obs), `act_b` (N, L+target_offset,
        X_act), both on device. Returns (N, L): one row per segment, one
        column per action, already trimmed so column i is the error for
        action i.

        The one home for the curiosity forward. Both batched callers route
        through it, so `predNet.curiosity_cuda_graph` covers both and the
        error reduction has a single spelling.
        """
        with timer("collect/curious/predict"):
            if self.curiosity_cuda_graph:
                if self._graph_curiosity is None:
                    self._graph_curiosity = _GraphCuriosityForward(
                        self.pN, self.device
                    )
                obs_pred, obs_next = self._graph_curiosity.predict(obs_b, act_b)
            else:
                with torch.no_grad():
                    obs_pred, obs_next, _ = self.pN.predict(
                        obs_b, act_b, batched=True
                    )
        with timer("collect/curious/error"):
            # Batched masked-pRNN layout is (phase=1, L, X, B).
            return ((obs_pred - obs_next) ** 2).mean(dim=2)[0].transpose(0, 1)[
                :, target_offset:
            ]

    def prediction_mses(
        self,
        obss: list,
        actions_np: np.ndarray,
        done_indices: list[int],
        last_observations: list,
        num_frames: int,
        target_offset: int = 0,
    ) -> torch.Tensor:
        """Per-step observation-prediction MSE over the collected rollout,
        computed per episode segment. Used as the curiosity reward.

        ALIGNMENT CONTRACT (see throwaway/ported/docs_legacy/refactor_baseline.md flaw #1): with
        predOffset=0, prediction row t targets obss[t].
        - target_offset=0 (legacy): MSEs[i] is the error reconstructing
          obss[i], the observation BEFORE action i.
        - target_offset=1 (next_obs): MSEs[i] is the error on the observation
          action i PRODUCED. The episode's final observation gets a real
          prediction row too: the pass is extended by one step that feeds
          last_obs with a zeroed action row (the same zero-action convention
          init_sr uses), so every action's reward is computed the same way -
          no boundary special case.
        """
        assert target_offset in (0, 1)
        segment_lengths = np.diff(done_indices)
        can_batch = (
            self.batched_curiosity
            and self.fast_speedhd
            and not self.theta
            and len(segment_lengths) > 1
            and np.all(segment_lengths == segment_lengths[0])
        )
        if can_batch:
            return self._prediction_mses_batched(
                obss,
                actions_np,
                done_indices,
                last_observations,
                num_frames,
                target_offset,
            )

        with torch.no_grad():
            MSEs = torch.zeros(num_frames, device=self.device)

            for idx in range(1, len(done_indices)):
                start_episode, end_episode = done_indices[idx - 1], done_indices[idx]
                last_obs = last_observations[idx - 1]
                _, _, _, errors = self.episode_prediction_rows(
                    obss[start_episode:end_episode],
                    actions_np[start_episode:end_episode],
                    last_obs,
                    target_offset=target_offset,
                )
                MSEs[start_episode:end_episode] = errors

        return MSEs

    def _prediction_mses_batched(
        self,
        obss: list,
        actions_np: np.ndarray,
        done_indices: list[int],
        last_observations: list,
        num_frames: int,
        target_offset: int,
    ) -> torch.Tensor:
        """One full-sequence pRNN forward for all equal-length segments.

        This preserves the per-row computation but changes dropout/noise RNG
        ordering relative to serial segment calls. It is therefore explicitly
        flag-gated for high-throughput runs; with stochasticity disabled the
        result matches serial forwards to normal float32 reduction tolerance.
        """
        with timer("collect/curious/format"):
            obs_rows = []
            act_rows = []
            for idx in range(1, len(done_indices)):
                start, end = done_indices[idx - 1], done_indices[idx]
                acts_ep = actions_np[start:end]
                last_obs = last_observations[idx - 1]
                if target_offset == 0:
                    acts_now = acts_ep
                    obs_now = list(obss[start:end]) + [last_obs]
                else:
                    acts_now = np.append(acts_ep, 0)
                    obs_now = list(obss[start:end]) + [last_obs, last_obs]

                obs_formatted, act_formatted = self.seq2pred(obs_now, acts_now)
                if target_offset == 1:
                    act_formatted[:, -1, :] = 0
                obs_rows.append(obs_formatted[0])
                act_rows.append(act_formatted[0])

            obs_b = torch.stack(obs_rows).to(self.device)
            act_b = torch.stack(act_rows).to(self.device)
        errors = self._curiosity_errors(obs_b, act_b, target_offset=target_offset)

        MSEs = torch.zeros(num_frames, device=self.device)
        for b, (start, end) in enumerate(
            zip(done_indices[:-1], done_indices[1:])
        ):
            MSEs[start:end] = errors[b]
        return MSEs

    def prediction_mses_device(
        self,
        *,
        images_tb: torch.Tensor,
        directions_tb: torch.Tensor,
        actions_tb: torch.Tensor,
        last_batches: list[tuple[torch.Tensor, ...]],
        target_offset: int,
    ) -> torch.Tensor:
        """Batched curiosity directly from device-resident rollout tensors.

        ``images_tb`` and friends are in collector order ``(T,B,...)``;
        recurrent segments are presented to pRNN in env-major order, exactly
        matching the ordinary ``done_indices`` path. No observation/action
        round-trip through NumPy or Python dictionaries is required.
        """
        assert target_offset in (0, 1)
        assert self.fast_speedhd and not self.theta
        T, B = actions_tb.shape
        segments = len(last_batches)
        if segments == 0 or T % segments:
            raise ValueError("device curiosity requires equal closed segments")
        L = T // segments
        N = B * segments

        with timer("collect/curious/format"):
            images = (
                images_tb.permute(1, 0, 2, 3, 4)
                .reshape(N, L, -1)
                .to(torch.float32)
                / 255
            )
            directions = (
                directions_tb.permute(1, 0).reshape(N, L).long()
            )
            actions = actions_tb.permute(1, 0).reshape(N, L).long()
            last_images = (
                torch.stack([batch[0] for batch in last_batches])
                .permute(1, 0, 2, 3, 4)
                .reshape(N, 1, -1)
                .to(torch.float32)
                / 255
            )

            if target_offset == 0:
                obs_b = torch.cat([images, last_images], dim=1)
                act_b = torch.zeros(
                    (N, L, self.num_acts + self.num_hd),
                    dtype=torch.int64,
                    device=images.device,
                )
                act_b[:, :, FORWARD_IDX] = (
                    actions == FORWARD_IDX
                ).to(torch.int64)
                act_b.scatter_(
                    2,
                    (self.num_acts + directions).unsqueeze(-1),
                    1,
                )
            else:
                obs_b = torch.cat(
                    [images, last_images, last_images], dim=1
                )
                act_b = torch.zeros(
                    (N, L + 1, self.num_acts + self.num_hd),
                    dtype=torch.int64,
                    device=images.device,
                )
                act_b[:, :L, FORWARD_IDX] = (
                    actions == FORWARD_IDX
                ).to(torch.int64)
                act_b[:, :L].scatter_(
                    2,
                    (self.num_acts + directions).unsqueeze(-1),
                    1,
                )

        errors = self._curiosity_errors(obs_b, act_b, target_offset=target_offset)
        return errors.reshape(B, segments, L).reshape(B * T)

    def episode_prediction_rows(
        self,
        obss_ep: list,
        acts_ep: np.ndarray,
        last_obs,
        target_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One episode's reward-pass predictions, row i aligned to action i.

        Returns (pred_rows, target_rows, hidden_rows, mses), each with
        len(acts_ep) rows: under target_offset=0 (legacy) row i targets
        obss_ep[i]; under target_offset=1 (next_obs) row i targets the obs
        action i produced (last row's target is last_obs, predicted via the
        appended zero-action step). hidden_rows are the pRNN states at each
        prediction row (theta-window dim 0 for thcyc nets is squeezed away
        for the masked mainline nets).
        """
        assert target_offset in (0, 1)
        with torch.no_grad():
            if target_offset == 0:
                acts_now = acts_ep
                obs_now = list(obss_ep) + [last_obs]
            else:
                # extra step so last_obs is also a prediction target
                acts_now = np.append(acts_ep, 0)
                obs_now = list(obss_ep) + [last_obs, last_obs]

            obs_formatted, act_formatted = self.seq2pred(obs_now, acts_now)
            obs_formatted, act_formatted = obs_formatted.to(self.device), act_formatted.to(self.device)
            if target_offset == 1:
                act_formatted[:, -1, :] = 0  # zero-action step (init_sr convention)

            obs_pred, obs_next, h = self.pN.predict(obs_formatted, act_formatted) # obs_next is reformatted version of obs_formatted
            obs_pred, obs_next = obs_pred.squeeze(0), obs_next.squeeze(0)
            errors = ((obs_pred - obs_next) ** 2).mean(dim=1)

        return (
            obs_pred[target_offset:],
            obs_next[target_offset:],
            h.squeeze(0)[target_offset:],
            errors[target_offset:],
        )

    def make_batched_tracker(self, num_envs: int) -> "BatchedSRTracker":
        return BatchedSRTracker(self.pN, self.device, num_envs)

    def _episode_tensors(self, images_tensor, hd_tensor, act_np: np.ndarray, last_obs):
        """Format one episode segment for the pRNN: (obs (L+1, X) float in
        [0,1], act (L, A) int64 SpeedHD). Shared by the serial and batched
        training paths so their formatting cannot drift. Requires
        fast_speedhd."""
        L = len(images_tensor)
        last_img = flat_obs_rows([last_obs]).to(images_tensor.device)
        obs = torch.cat(
            [images_tensor.detach().reshape(L, -1).to(torch.float32) / 255, last_img]
        )
        hd = hd_tensor.detach().cpu().numpy()
        act = encode_speed_hd_seq(act_np, hd, self.num_acts, self.num_hd)[0]
        return obs, act

    def _use_graph_wm(self) -> bool:
        return self.cuda_graph and self.fast_speedhd and not self.theta

    def train_on_episode(self, images_tensor, hd_tensor, act_np: np.ndarray, last_obs) -> None:
        """One pRNN gradient step on a single episode segment."""
        if self._use_graph_wm():
            obs, act = self._episode_tensors(images_tensor, hd_tensor, act_np, last_obs)
            self._graph_trainer = self._graph_trainer or _GraphWMTrainer(self.pN, self.device)
            self._graph_trainer.train_segment(obs, act)
            return
        if self.fast_speedhd:
            obs, act = self._episode_tensors(images_tensor, hd_tensor, act_np, last_obs)
            obs, act = obs.unsqueeze(0), act.unsqueeze(0)
        else:
            images_np = images_tensor.detach().cpu().numpy()
            hd_np = hd_tensor.detach().cpu().numpy()
            obs_for_pN = [
                {"image": images_np[i], "direction": hd_np[i].item()}
                for i in range(len(images_np))
            ]
            obs, act = self.pN.env_shell.env2pred(obs_for_pN + [last_obs], act_np)

        obs = obs.to(self.device)
        act = act.to(self.device)
        # return_stats=False: trainStep's sparsity/meanrate diagnostics cost a
        # full reduction over h plus two GPU->CPU syncs per gradient step, and
        # they do not enter the loss unless the homeostat is on. Nothing here
        # reads them.
        self.pN.trainStep(obs, act, return_stats=False)
        self.pN.numTrainingEpochs += 1

    def train_on_episodes_batched(
        self, exps, done_indices: list[int], last_observations: list,
        *, group: int = 0,
    ) -> None:
        """Pooled pRNN gradient steps over equal-length episode segments,
        stacked to (B, L). batched trainStep's loss is the mean of the
        per-segment losses.

        `group=0` pools ALL B segments into ONE step (the historical
        predNet.batched_wm behaviour). `group=g` instead takes B/g steps, each
        pooled over g segments - the middle design between one pooled step and
        B serial ones.

        Why the middle exists: the two extremes each get one thing wrong at
        num_envs=128. Serial takes 128 steps per update = 1 per 256 env steps,
        8x the reference's rate, and over-trains (job 10444495: sRSA peaks
        0.7732 then falls to 0.5581). `wm_segment_stride=8` fixes the COUNT but
        each step then sees ONE segment where the reference pools eight, i.e.
        8x the gradient variance - and job 10444819 reached only sRSA 0.6383
        against the reference 0.7905 at the SAME step count. Pooling all 128
        gives one step per 32768 env steps, 16x too few.

        `group=8` at num_envs=128 gives 16 steps per update, each pooled over 8
        segments: the reference's step COUNT and its gradient QUALITY.

        Requires fast_speedhd and equal segment lengths (callers check)."""
        with timer("update/wm/format"):
            B = len(done_indices) - 1
            L = done_indices[1] - done_indices[0]
            images = exps.obs.image.reshape(B, L, -1)
            directions = exps.obs.direction.reshape(B, L).long()
            actions = exps.action.reshape(B, L).long()

            last_images = flat_obs_rows(last_observations).to(
                device=self.device, dtype=torch.float32
            )
            obs_b = torch.cat(
                [images.to(torch.float32) / 255, last_images[:, None, :]], dim=1
            )

            act_b = torch.zeros(
                (B, L, self.num_acts + self.num_hd),
                dtype=torch.int64,
                device=self.device,
            )
            act_b[:, :, FORWARD_IDX] = (actions == FORWARD_IDX).to(torch.int64)
            act_b.scatter_(
                2,
                (self.num_acts + directions).unsqueeze(-1),
                1,
            )
            # The speed channel is already zero when the action is the
            # segment-start sentinel (-1).  The former ``if no_action.any()``
            # converted a device scalar to a Python bool and synchronized the
            # whole GPU for a branch that never changed the tensor.

        with timer("update/wm/train_step"):
            g = int(group) if group and group > 0 else obs_b.size(0)
            graphed = self._use_graph_wm()
            if graphed:
                self._graph_trainer = self._graph_trainer or _GraphWMTrainer(self.pN, self.device)
            for i in range(0, obs_b.size(0), g):
                if graphed:
                    # train_batch bumps numTrainingEpochs itself, as the serial
                    # graphed path does.
                    self._graph_trainer.train_batch(
                        obs_b[i : i + g], act_b[i : i + g], batched=True
                    )
                    continue
                self.pN.trainStep(
                    obs_b[i : i + g], act_b[i : i + g], batched=True, return_stats=False
                )
                self.pN.numTrainingEpochs += 1


class SingleSRTracker:
    """B=1 SR tracker delegating to the adapter's predict_single/reset_state
    path - bitwise-identical to the historical serial rollout (same calls,
    same RNG consumption order, including the randInit noise draw on reset).
    """

    def __init__(self, adapter: "PRNNAdapter", initial_obs):
        self.adapter = adapter
        self._initial = adapter.init_sr(initial_obs)

    def initial_sr(self) -> torch.Tensor:  # (1, H)
        return self._initial

    def step(self, det_np: np.ndarray, pre_obss: list, post_obss: list) -> torch.Tensor:
        obs = pre_obss[0] if self.adapter.pastSR else post_obss[0]
        return self.adapter.next_sr(det_np, obs)

    def reset_env(self, b: int, current_obs) -> torch.Tensor:
        assert b == 0
        self.adapter.reset_state()  # randInit noise draw, same order as before
        return self.adapter.init_sr(current_obs)

    def end_rollout(self) -> None:
        self.adapter.reset_state()


class NullSRTracker:
    """No world model: SRs are empty tensors (shape (B, 0))."""

    def __init__(self, device: torch.device, num_envs: int):
        self.device = device
        self.B = num_envs

    def initial_sr(self) -> torch.Tensor:
        return torch.zeros((self.B, 0), device=self.device)

    def step(self, det_np, pre_obss, post_obss) -> torch.Tensor:
        return torch.zeros((self.B, 0), device=self.device)

    def reset_env(self, b: int, current_obs) -> torch.Tensor:
        return torch.zeros((1, 0), device=self.device)

    def end_rollout(self) -> None:
        pass


class BatchedSRTrackerShim:
    """Adapts BatchedSRTracker to the shared tracker interface (B > 1).

    Resets are to zero state/phase (not the serial path's randInit noise) -
    a documented Phase 5 semantic; B>1 runs are not bit-comparable to B=1.
    """

    def __init__(self, adapter: "PRNNAdapter", num_envs: int):
        assert adapter.pastSR, "batched mode currently supports pastSR nets only"
        self.adapter = adapter
        self.tracker = adapter.make_batched_tracker(num_envs)

    def initial_sr(self) -> torch.Tensor:
        return self.tracker.sr().clone()

    def step(self, det_np: np.ndarray, pre_obss: list, post_obss: list) -> torch.Tensor:
        obs_src = pre_obss if self.adapter.pastSR else post_obss
        if self.adapter.fast_speedhd:
            # one batched conversion (bitwise-equal to the per-env env2pred loop)
            with timer("collect/sr/format_and_transfer"):
                obs_x = flat_obs_rows(obs_src).to(self.adapter.device)
                act_x = encode_speed_hd_rows(
                    det_np, [o["direction"] for o in obs_src],
                    self.adapter.num_acts, self.adapter.num_hd,
                ).to(self.adapter.device)
        else:
            obs_rows, act_rows = [], []
            for b, obs in enumerate(obs_src):
                o_x, a_x = self.adapter.pN.env_shell.env2pred([obs, obs], det_np[b:b + 1])
                obs_rows.append(o_x[:, 0, :])
                act_rows.append(a_x[:, 0, :])
            obs_x = torch.cat(obs_rows, dim=0).to(self.adapter.device)
            act_x = torch.cat(act_rows, dim=0).to(self.adapter.device)
        with timer("collect/sr/recurrent"):
            return self.tracker.step(obs_x, act_x).clone()

    def step_device(
        self,
        *,
        actions: torch.Tensor,
        images: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Device-native SpeedHD formatting and one batched recurrent step."""
        if not self.adapter.fast_speedhd:
            raise ValueError("device_env currently requires SpeedHD encoding")
        with timer("collect/sr/format_and_transfer"):
            obs_x = images.reshape(images.shape[0], -1).to(torch.float32) / 255
            act_x = torch.zeros(
                (images.shape[0], self.adapter.num_acts + self.adapter.num_hd),
                dtype=torch.int64,
                device=images.device,
            )
            act_x[:, FORWARD_IDX] = (actions == FORWARD_IDX).to(torch.int64)
            act_x.scatter_(
                1,
                (self.adapter.num_acts + directions.long()).unsqueeze(1),
                1,
            )
        with timer("collect/sr/recurrent"):
            return self.tracker.step_synchronized(
                obs_x=obs_x, act_x=act_x
            ).clone()

    def reset_env(self, b: int, current_obs) -> torch.Tensor:
        self.tracker.reset_env(b)
        return torch.zeros((1, self.adapter.pN.hidden_size), device=self.adapter.device)

    def reset_all_envs(self) -> torch.Tensor:
        self.tracker.reset_all()
        return self.tracker.sr().clone()

    def end_rollout(self) -> None:
        self.tracker.reset_all()


def make_sr_tracker(adapter, device: torch.device, envs_obs: list):
    """Pick the tracker for B environments: exact serial path at B=1,
    batched stepping at B>1, empty SRs without a world model."""
    B = len(envs_obs)
    if adapter is None:
        return NullSRTracker(device, B)
    if B == 1:
        return SingleSRTracker(adapter, envs_obs[0])
    return BatchedSRTrackerShim(adapter, B)


class BatchedSRTracker:
    """Stateful batched equivalent of PredictiveNet.predict_single.

    Steps B independent pRNN streams in one forward pass by calling the RNN
    layer directly with its `batched=True` 4-D layout (input (k+1, T=1, D, B),
    trailing batch dim) - `pRNN.forward(single=True)` does not forward the
    `batched` flag, so the direct rnn() call remains the single-step seam.
    (predict(batched=True) itself is fixed on the LevensteinLab
    sdu/rl-integration branch and used for full-sequence prediction.)

    Per-env phase counters and hidden states allow envs to reset at different
    times. With trainNoiseMeanStd=(0,0) a batched step is exactly equal to B
    serial predict_single calls (see tests/test_batched_tracker.py); with
    noise the streams are distributionally identical but consume the RNG in a
    different order than serial stepping.

    Only used for num_envs > 1; the B=1 path keeps using predict_single so
    the golden fixture is untouched.
    """

    def __init__(self, pN: PredictiveNet, device: torch.device, num_envs: int):
        assert not hasattr(pN.pRNN, "k") or pN.pRNN.k == 0, (
            "BatchedSRTracker supports the masked (thRNN_*win) nets; "
            "theta-cycle nets need the k+1 window rollout."
        )
        self.pN = pN
        self.device = device
        self.B = num_envs
        self.hidden_size = pN.pRNN.rnn.cell.hidden_size
        self.in_mask = np.asarray(pN.pRNN.inMask, dtype=np.float32)
        self.act_mask = np.asarray(pN.pRNN.actMask, dtype=np.float32)
        self.phase_k = pN.phase_k
        self.reset_all()

    def reset_all(self) -> None:
        # Allocated ONCE and zeroed in place thereafter. The hidden state is an
        # input to (and output of) the per-step cell call, so anything that
        # captures that call bakes in its address; rebinding it to a fresh
        # allocation would leave the capture reading stranded memory - the same
        # failure mode as world_model/device.py's buffers, which fails silently.
        if getattr(self, "state", None) is None:
            self.state = torch.zeros((self.B, 1, self.hidden_size), device=self.device)
        else:
            self.state.zero_()
        self.phases = np.zeros(self.B, dtype=np.int64)

    def reset_env(self, i: int) -> None:
        self.state[i].zero_()
        self.phases[i] = 0

    def sr(self) -> torch.Tensor:
        """Current SRs, shape (B, hidden_size) (trimmed to pN.hidden_size)."""
        return self.state[:, 0, : self.pN.hidden_size]

    def _input_and_noise(self, obs_x, act_x):
        x = torch.cat((obs_x, act_x), dim=1).permute(1, 0)[None, None]
        noise = self.pN.pRNN.generate_noise(
            self.pN.trainNoiseMeanStd,
            (1, 1, self.hidden_size, self.B),
        ).to(self.device)
        return x, noise

    def _run_cell(self, x, noise) -> torch.Tensor:
        with timer("collect/sr/cell"):
            with torch.no_grad():
                _, state = self.pN.pRNN.rnn(
                    x, internal=noise, state=self.state, batched=True
                )
        # copy_, not rebind - see reset_all.
        self.state.copy_(state[0].unsqueeze(1))
        return self.sr()

    def step(self, obs_x: torch.Tensor, act_x: torch.Tensor) -> torch.Tensor:
        """One batched step. obs_x (B, obs_size), act_x (B, act_size) -
        the single-timestep rows produced by env2pred per env.
        Returns the new SRs, shape (B, pN.hidden_size).
        """
        with timer("collect/sr/mask_noise"):
            obs_m = torch.as_tensor(
                self.in_mask[self.phases], dtype=obs_x.dtype, device=self.device
            )
            act_m = torch.as_tensor(
                self.act_mask[self.phases], dtype=act_x.dtype, device=self.device
            )
            obs_x = obs_x * obs_m[:, None]
            act_x = act_x * act_m[:, None]
            self.phases = (self.phases + 1) % self.phase_k
            x, noise = self._input_and_noise(obs_x, act_x)
        return self._run_cell(x, noise)

    def step_synchronized(
        self, *, obs_x: torch.Tensor, act_x: torch.Tensor
    ) -> torch.Tensor:
        """Faster step when every stream has the same phase/reset schedule.

        DeviceTableShellPool only permits synchronized seqdur cuts, so its
        phase mask is one scalar. The generic tracker constructs and uploads
        a length-B mask because independently terminating CPU envs can have
        different phases.
        """
        phase = int(self.phases[0])
        with timer("collect/sr/mask_noise"):
            if self.in_mask[phase] == 0:
                obs_x = torch.zeros_like(obs_x)
            elif self.in_mask[phase] != 1:
                obs_x = obs_x * float(self.in_mask[phase])
            if self.act_mask[phase] == 0:
                act_x = torch.zeros_like(act_x)
            elif self.act_mask[phase] != 1:
                act_x = act_x * float(self.act_mask[phase])
            self.phases.fill((phase + 1) % self.phase_k)
            x, noise = self._input_and_noise(obs_x, act_x)
        return self._run_cell(x, noise)

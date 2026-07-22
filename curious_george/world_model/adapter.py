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

FORWARD_IDX = 2  # ActionEncodings.forwardIDX - the action bit SpeedHD keeps


def flat_obs_rows(obs_dicts) -> torch.Tensor:
    """Raw obs dicts -> (N, H*W*C) float32 rows in [0,1].

    Bitwise-equal to env2pred's per-dict get_visual loop + /255, with one
    numpy stack instead of N Python-level reshapes/copies.
    """
    imgs = np.stack([o["image"] for o in obs_dicts])
    return torch.from_numpy(imgs.reshape(len(obs_dicts), -1)).to(torch.float32) / 255


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


class _GraphWMSegmentTrainer:
    """CUDA-graph capture of the single-segment world-model trainStep.

    Reproduces `PredictiveNet.trainStep` for the MaskedRNN / SpeedHD /
    theta==0 path with the RL defaults (with_homeostat=False, no encoder
    training), splitting it into an eager prologue and a captured region:

    - Eager prologue (per update): `restructure_inputs` runs dropout and the
      numpy-mask input scatter - the ops that CANNOT be captured (host-side
      numpy indexing). Its outputs are copied into static buffers.
    - Captured region (once per segment length L): `generate_noise` (in-graph
      RNG, verified to advance per replay), the RNN timestep loop, outlayer,
      output masking as a float multiply (capture-safe; the bool-scatter in
      `predict` calls nonzero() and cannot be captured), MSE loss, backward,
      and one RMSprop step.

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
        self.H = pN.hidden_size
        self.noise_std = float(pN.trainNoiseMeanStd[1])
        self.graphs: dict[int, dict] = {}  # L -> static buffers + graph
        self._ensure_capturable_optimizer()

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

    def _capture(self, L: int, outmask_np: np.ndarray) -> None:
        prnn, opt = self.pN.pRNN, self.pN.optimizer
        x_s = torch.zeros(1, L, prnn.W_in.shape[1], device=self.device)
        tgt_s = torch.zeros(1, L, self.pN.obs_size, device=self.device)
        mask_f = torch.as_tensor(
            np.resize(np.asarray(outmask_np), L), device=self.device
        ).float().view(1, L, 1)
        state_z = torch.zeros(1, 1, self.H, device=self.device)

        def region():
            noise = prnn.generate_noise((0.0, self.noise_std), (1, L, self.H))
            h, _ = prnn.rnn(x_s, internal=noise, state=state_z, theta=0)
            h = h[:, :, : self.H]
            y = prnn.outlayer(h) * mask_f
            loss = self.pN.loss_fn(y, tgt_s, h)
            opt.zero_grad(set_to_none=False)
            loss.backward()
            opt.step()
            return loss

        # snapshot to undo the warmup steps' effect on weights + opt state
        w_snap = [p.detach().clone() for p in prnn.parameters()]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                region()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss_s = region()

        # restore pristine weights and zero the (now-allocated) optimizer state
        # tensors in place - keeping the SAME tensors the graph captured - so
        # the first real replay is the first real step from a fresh optimizer.
        with torch.no_grad():
            for p, w in zip(prnn.parameters(), w_snap):
                p.copy_(w)
            for st in opt.state.values():
                for v in st.values():
                    if isinstance(v, torch.Tensor):
                        v.zero_()

        self.graphs[L] = dict(graph=graph, x_s=x_s, tgt_s=tgt_s, loss_s=loss_s)

    def train_segment(self, obs: torch.Tensor, act: torch.Tensor) -> None:
        """obs (L+1, X_obs) float in [0,1], act (L, A) - the `_episode_tensors`
        contract. Runs one graphed pRNN gradient step on the segment."""
        prnn = self.pN.pRNN
        obs_b = obs.unsqueeze(0).to(self.device)
        act_b = act.unsqueeze(0).to(self.device)
        x_t, obs_target, outmask = prnn.restructure_inputs(
            obs_in=obs_b, act=act_b, obs_target=None
        )
        L = x_t.size(1)
        if L not in self.graphs:
            self._capture(L, np.asarray(outmask))
        g = self.graphs[L]
        g["x_s"].copy_(x_t.detach())
        g["tgt_s"].copy_(obs_target.detach())
        g["graph"].replay()
        self.pN.recordTrainingTrial(g["loss_s"].item())
        self.pN.numTrainingEpochs += 1


class PRNNAdapter:
    def __init__(
        self,
        predictive_net: PredictiveNet,
        device: torch.device,
        pastSR: bool,
        cuda_graph: bool = False,
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
        self._graph_trainer: _GraphWMSegmentTrainer | None = None

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

        ALIGNMENT CONTRACT (see docs/refactor_baseline.md flaw #1): with
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
            if self._graph_trainer is None:
                self._graph_trainer = _GraphWMSegmentTrainer(self.pN, self.device)
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
        _, _, _ = self.pN.trainStep(obs, act)
        self.pN.numTrainingEpochs += 1

    def train_on_episodes_batched(self, exps, done_indices: list[int], last_observations: list) -> None:
        """ONE pooled pRNN gradient step over all equal-length episode
        segments, stacked to (B, L): batched trainStep's loss is the mean of
        the per-segment losses, so this replaces B sequential optimizer steps
        with one step on the pooled gradient (optimization-semantics change -
        flag-gated via predNet.batched_wm, curve-gate before defaulting on).
        Requires fast_speedhd and equal segment lengths (callers check)."""
        obs_rows, act_rows = [], []
        for idx in range(1, len(done_indices)):
            s, e = done_indices[idx - 1], done_indices[idx]
            obs, act = self._episode_tensors(
                exps.obs.image[s:e],
                exps.obs.direction[s:e],
                exps.action[s:e].detach().cpu().numpy(),
                last_observations[idx - 1],
            )
            obs_rows.append(obs)
            act_rows.append(act)

        obs_b = torch.stack(obs_rows).to(self.device)  # (B, L+1, X)
        act_b = torch.stack(act_rows).to(self.device)  # (B, L, A)
        _, _, _ = self.pN.trainStep(obs_b, act_b, batched=True)
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
        return self.tracker.step(obs_x, act_x).clone()

    def reset_env(self, b: int, current_obs) -> torch.Tensor:
        self.tracker.reset_env(b)
        return torch.zeros((1, self.adapter.pN.hidden_size), device=self.adapter.device)

    def end_rollout(self) -> None:
        self.adapter.reset_state()
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
        self.state = torch.zeros((self.B, 1, self.hidden_size), device=self.device)
        self.phases = np.zeros(self.B, dtype=np.int64)

    def reset_env(self, i: int) -> None:
        self.state[i].zero_()
        self.phases[i] = 0

    def sr(self) -> torch.Tensor:
        """Current SRs, shape (B, hidden_size) (trimmed to pN.hidden_size)."""
        return self.state[:, 0, : self.pN.hidden_size]

    def step(self, obs_x: torch.Tensor, act_x: torch.Tensor) -> torch.Tensor:
        """One batched step. obs_x (B, obs_size), act_x (B, act_size) -
        the single-timestep rows produced by env2pred per env.
        Returns the new SRs, shape (B, pN.hidden_size).
        """
        pRNN = self.pN.pRNN
        obs_m = torch.as_tensor(self.in_mask[self.phases], dtype=obs_x.dtype, device=self.device)
        act_m = torch.as_tensor(self.act_mask[self.phases], dtype=act_x.dtype, device=self.device)
        obs_x = obs_x * obs_m[:, None]
        act_x = act_x * act_m[:, None]
        self.phases = (self.phases + 1) % self.phase_k

        x = torch.cat((obs_x, act_x), dim=1)          # (B, D)
        x = x.permute(1, 0)[None, None]                # (1, 1, D, B)
        noise = pRNN.generate_noise(
            self.pN.trainNoiseMeanStd, (1, 1, self.hidden_size, self.B)
        ).to(self.device)

        with torch.no_grad():
            _, state = pRNN.rnn(x, internal=noise, state=self.state, batched=True)
        self.state = state[0].unsqueeze(1)             # (B, 1, H)
        return self.sr()


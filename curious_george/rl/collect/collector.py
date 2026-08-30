"""Unified rollout collection for B >= 1 environments.

One loop serves both cases: at B=1 the SR tracker delegates to the exact
predict_single/reset_state path and every RNG-consuming call happens in the
historical order (sample -> env.step -> SR noise -> [reset noise -> env.reset]),
so the collected rollout is bitwise-identical to the pre-unification serial
collector (gated by tests/golden/golden_v0.pt). At B>1 the tracker steps all
streams in one batched forward.

Flat layout is env-major: index = b*T + t with T = num_frames // B. Episode
segments never span environment boundaries; GAE runs per env stream.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch_ac.utils import DictList

from curious_george.envs.access import get_subroom_id
from curious_george.envs.vector import AsyncShellPool, DeviceTableShellPool
from curious_george.rl.collect.diagnostics import (
    LocationStats,
    check_large_jump,
    new_joint_probabilities,
)
from curious_george.rl.collect.rollout_graph import RolloutBuffers
from curious_george.rl.update.advantage import compute_gae
from curious_george.rl.update.rewards import (
    REWARD_ALIGNMENTS,
    compute_curious_rewards,
)
from curious_george.utils.timing import timer


def _agent_pos(env):
    return env.get_agent_pos() if hasattr(env, "get_agent_pos") else env.agent_pos


def get_dist_travelled(start_locs, end_locs):
    """L1 distance between start and end locations (grid moves are axis-aligned)."""
    return torch.abs(end_locs - start_locs).sum(dim=1)


def _preprocess_policy_obss(obss, acmodel, preprocess_obss, device):
    """Only materialize fields consumed by the policy.

    The default MiniGrid preprocessor also stacks the RGB image and tokenizes
    the mission string.  ACModelSR with ``with_CV=False`` never reads either
    field, so doing that work in every rollout step only creates CPU work and
    a larger host-to-device transfer.
    """
    # Keep the B=1 golden path byte-for-byte on the historical preprocessor;
    # high-throughput collection is the B>1 path.
    if len(obss) == 1 or getattr(acmodel, "with_CV", True):
        return preprocess_obss(obss, device=device)
    directions = np.asarray([obs["direction"] for obs in obss], dtype=np.uint8)
    return DictList({
        "direction": torch.tensor(directions, device=device, dtype=torch.uint8)
    })


def _device_policy_obss(images, directions, acmodel):
    """Build only the device fields consumed by the configured policy."""
    data = {"direction": directions.to(torch.uint8)}
    if getattr(acmodel, "with_CV", True):
        data["image"] = images.to(torch.float32)
    return DictList(data)


#: The project's random-action distribution over (left, right, forward, pickup).
#: Forward-weighted: a uniform walker mostly spins on the spot and covers little.
#: ⚠️ This constant has FOUR spellings in the tree - `storage.RAND_ACT_PROBA`,
#: `training.setup.RAND_ACT_PROBA`, `circuit_diagnostics.PROBE_ACTION_P` and a
#: bare literal in `checkpoint_series` - which is a defect, not a convention.
#: Imported here rather than copied so this is not a fifth.
def _random_actions(n: int, device: torch.device) -> torch.Tensor:
    from curious_george.log_and_store.storage import RAND_ACT_PROBA

    probs = torch.as_tensor(RAND_ACT_PROBA, dtype=torch.float32, device=device)
    return torch.multinomial(probs.expand(n, -1), num_samples=1).squeeze(1)


@dataclass
class RolloutConfig:
    num_frames: int
    device: torch.device
    prnn_seqdur: int = 0
    action_offset: int = 0
    random_actions: bool = False
    """Draw actions from `RANDOM_ACTION_PROBS` instead of the policy.

    The BASELINE, and it goes through this same function on purpose: the point
    of "how would a random walker do" is that everything except action selection
    is held fixed - same backend, same batch, same rooms, same world-model
    training. A separate serial routine (the retired
    `randomAgent_collect_exp_and_update`) answered a different question and
    forced `num_envs == 1` for reasons that were about that routine, not about
    random actions."""
    curious_agent: bool = False
    reward_alignment: str = "legacy"
    intrinsic: bool = False
    discount: float = 0.99
    gae_lambda: float = 0.95
    k_int: float = 1.0
    k_curious: float = 1.0


@dataclass
class CollectorState:
    """State that persists across collect calls (episodes span rollouts)."""

    obs_b: list
    loc_b: list
    mask_b: np.ndarray
    sr: torch.Tensor  # (B, H)
    # episode logging (cumulative done counter never resets - historical)
    ep_return: list
    ep_reshaped: list
    ep_frames: list
    done_counter: int = 0
    finished_returns: list = field(default_factory=list)
    finished_reshaped: list = field(default_factory=list)
    finished_frames: list = field(default_factory=list)


@dataclass
class CollectResult:
    exps: DictList
    logs: dict
    # flat (B*T) views kept for analysis code that reads algo attributes
    directions: np.ndarray
    locs: list
    subroom_ids: list
    actions: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    SRs: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    curious_rewards: torch.Tensor
    int_rewards: torch.Tensor
    done_indices: list
    last_observations: list
    joint_dist: np.ndarray


def collect_rollout(
    *,
    envs: list,
    acmodel,
    tracker,
    adapter,
    preprocess_obss,
    state: CollectorState,
    cfg: RolloutConfig,
    loc_stats: LocationStats,
    subroom_size_: int | None,
    intrinsic_ref=None,  # IntrinsicReference (B=1 only) or None
    rollout_graph=None,  # GraphRolloutStepper (exp.rollout_cuda_graph) or None
) -> CollectResult:
    pool = envs if isinstance(envs, AsyncShellPool) else None
    device_pool = envs if isinstance(envs, DeviceTableShellPool) else None
    B = len(envs)
    T = cfg.num_frames // B
    device = cfg.device

    if rollout_graph is not None and device_pool is None:
        raise ValueError("exp.rollout_cuda_graph requires the device env table")

    if device_pool is not None:
        if cfg.prnn_seqdur <= 0 or T % cfg.prnn_seqdur:
            raise ValueError(
                "device_env requires positive synchronized prnn_seqdur cuts "
                "that divide frames/num_envs"
            )
        if intrinsic_ref is not None:
            raise ValueError(
                "device_env does not carry the single-env intrinsic reference"
            )

    if pool is not None and cfg.action_offset:
        raise ValueError(
            "action_offset=1 builds h[0] from the observation the tracker is "
            "handed, and AsyncShellPool resets its environments only after the "
            "timestep loop - so that observation would be the finished "
            "episode's. Use the device or serial backend."
        )

    diagnostics_env = envs if pool is not None or device_pool is not None else envs[0]
    joint = new_joint_probabilities(diagnostics_env, getattr(acmodel, "act_dim"))

    obss = None if device_pool is not None else [[None] * T for _ in range(B)]
    locs = None if device_pool is not None else [[None] * T for _ in range(B)]
    subroom_ids: list = []
    done_indices_b = (
        [list(range(0, T + 1, cfg.prnn_seqdur)) for _ in range(B)]
        if device_pool is not None
        else [[0] for _ in range(B)]
    )
    last_obs_b: list[list] = [[] for _ in range(B)]

    # log_probs is DERIVED after the loop from the stored logits, not
    # accumulated per step - see the record block below.
    #
    # A graphed rollout REUSES its stepper's buffers, because the captured
    # replays write to the addresses they saw at capture time. Everything
    # downstream copies out (`flat` permutes and so cannot alias, the rest
    # goes through .cpu()), so the reuse is invisible past this function.
    buffers = (
        rollout_graph.buffers
        if rollout_graph is not None
        else RolloutBuffers.allocate(
            num_steps=T,
            num_envs=B,
            hidden_size=state.sr.shape[1],
            act_dim=getattr(acmodel, "act_dim"),
            device=device,
            image_shape=device_pool.image_shape if device_pool is not None else None,
        )
    )
    actions, values, rewards = buffers.actions, buffers.values, buffers.rewards
    masks, SRs, policy_logits = buffers.masks, buffers.srs, buffers.policy_logits

    if device_pool is not None:
        device_images = buffers.images
        device_directions = buffers.directions
        device_positions = buffers.positions
        device_obs = device_pool.observation_device()
        device_reset_index = 0
        device_pool.prepare_resets(count=T // cfg.prnn_seqdur)
        # Each item is (post image, post direction, post position, initial
        # position) for one synchronized segment, all still on-device.
        device_last_batches: list[tuple[torch.Tensor, ...]] = []

    last_post_obs = None  # final pre-reset obs (intrinsic tail, action_offset=1)

    def _close_device_segment(post_images, post_directions) -> None:
        """End a synchronized device segment: bank its tail, reset, re-observe.

        `post_images`/`post_directions` are the observation AFTER the step that
        closed the segment - what `step_device` returned in eager, what a
        re-gather from the pool returns after a replay. Same values either way,
        because the pool's position/direction tensors are the authority.
        """
        nonlocal device_reset_index, device_obs
        device_last_batches.append(
            (
                post_images.clone(),
                post_directions.clone(),
                device_pool.positions.clone(),
                device_segment_initial.clone(),
            )
        )
        # This backend rejects environment rewards/terminations, so
        # synchronized segment statistics are known without B Python
        # bookkeeping iterations.
        #
        # EXTRINSIC RETURN IS NOT MEASURED HERE, and it is deliberately NOT
        # recorded as 0.0. A zero reaching wandb as `return_mean` cannot be told
        # apart from "the agent earned nothing" - harmless in a goal-less L-room
        # where the true return is 0 anyway, and wrong the first time this
        # backend runs a goal env (MiniGrid-LRoom_Goal-v0 exists). Omitting the
        # series is honest; a flat zero is not. The curious agent's actual
        # learning signal is intrinsic and logged separately.
        #
        # NOTE the segment/episode conflation this backend also introduces:
        # there are no environment terminations at all, so a "finished episode"
        # here is a `prnn_seqdur`-step SEGMENT. `done_counter` and
        # `finished_frames` count segments. The names are left alone because
        # `num_episodes` is a wandb key shared with every historical run.
        state.done_counter += B
        state.finished_frames.extend([cfg.prnn_seqdur] * B)

        # ORDER IS THE CIRCUIT. `reset_all_envs` builds h[0] from the
        # observation it is given, so under action_offset=1 the environment has
        # to have moved first - otherwise h[0] encodes the view the finished
        # episode ended on, from a position the agent has already left.
        # Under offset 0 nothing reads the observation, and the historical order
        # is kept because both calls consume RNG.
        if cfg.action_offset == 0:
            state.sr = tracker.reset_all_envs()
            device_pool.apply_prepared_reset(index=device_reset_index)
            device_reset_index += 1
            device_obs = device_pool.observation_device()
        else:
            device_pool.apply_prepared_reset(index=device_reset_index)
            device_reset_index += 1
            device_obs = device_pool.observation_device()
            state.sr = tracker.reset_all_envs(
                images=device_obs[0], directions=device_obs[1]
            )

    if rollout_graph is not None:
        rollout_graph.prepare(sr=state.sr)

    for t in range(T):
        if rollout_graph is not None:
            # ONE replay covers the policy forward, the per-timestep records,
            # the environment step and the pRNN step. Only what a graph cannot
            # express stays here: the Python-known segment boundaries, and the
            # episode mask, whose value the replay reads from a static row.
            if t % cfg.prnn_seqdur == 0:
                device_segment_initial = device_pool.positions.clone()
            with timer("collect/graph_step"):
                rollout_graph.step(mask=float(state.mask_b[0]))
            seq_done = (t + 1) % cfg.prnn_seqdur == 0
            state.mask_b.fill(1 - seq_done)
            if seq_done:
                _close_device_segment(*device_pool.observation_device())
            continue

        # --- action selection (one batched forward) ----------------------
        with timer("collect/policy_fwd"):
            with timer("collect/policy/preprocess"):
                if device_pool is not None:
                    preprocessed = _device_policy_obss(
                        device_obs[0], device_obs[1], acmodel
                    )
                else:
                    preprocessed = _preprocess_policy_obss(
                        state.obs_b, acmodel, preprocess_obss, device
                    )
            with timer("collect/policy/network"):
                with torch.no_grad():
                    dist, value = acmodel(preprocessed, SR=state.sr)
            with timer("collect/policy/sample"):
                action = (
                    _random_actions(dist.probs.shape[0], device)
                    if cfg.random_actions
                    else dist.sample()  # based on SR from step t-1
                )
            with timer("collect/policy/action_to_host"):
                # The device table consumes the sampled tensor directly.
                # CPU/async environments still require this synchronizing D2H.
                det_np = None if device_pool is not None else action.cpu().numpy()

        # The device path still needs the episode's first positions; the serial
        # and async paths recorded theirs only for `dist_travelled`, which is no
        # longer logged.
        if cfg.prnn_seqdur > 0 and t % cfg.prnn_seqdur == 0 and device_pool is not None:
            device_segment_initial = device_pool.positions.clone()

        # record buffers indexed by pre-step state
        with timer("collect/policy/record"):
            if device_pool is not None:
                # Device-env cuts are synchronized, so every stream has the
                # same boundary mask and no per-step H2D tensor is needed.
                masks[t].fill_(float(state.mask_b[0]))
            else:
                masks[t] = torch.as_tensor(state.mask_b, device=device)
            SRs[t] = state.sr
            actions[t] = action
            values[t] = value
            # Store the distribution's OWN normalised logits and derive
            # log_prob/probs once, batched, after the loop. Both are pure
            # functions of (logits, action) and neither feeds the environment,
            # so nothing in the rollout depends on them being computed here -
            # but doing so costs a gather and a softmax on every one of the T
            # sequential steps. Measured: Categorical construction + sample +
            # log_prob + probs is 159.2 ms per 256-step rollout, of which
            # construction + sample alone is 75.5 ms.
            #
            # BIT-EXACT, not merely equivalent: `dist.logits` is exactly the
            # tensor `Categorical.log_prob` gathers from and `Categorical.probs`
            # softmaxes, so deriving them later reproduces the same floats. The
            # naive `log_softmax(logits).gather(...)` does NOT - verified - and
            # would break the bitwise oracle in tests/golden_omt/, which
            # models.py's redundant log_softmax exists to protect.
            policy_logits[t] = dist.logits.detach()

        # --- environment stepping ----------------------------------------
        pre_obs_b = None if device_pool is not None else list(state.obs_b)
        seq_done = cfg.prnn_seqdur > 0 and (t + 1) % cfg.prnn_seqdur == 0
        if device_pool is not None:
            with timer("collect/env_step"):
                pre_images, pre_directions = device_obs
                device_images[t].copy_(pre_images)
                device_directions[t].copy_(pre_directions)
                device_positions[t].copy_(device_pool.positions)
                post_images, post_directions, step_rewards = (
                    device_pool.step_device(actions=action)
                )
                device_obs = (post_images, post_directions)
                rewards[t].copy_(step_rewards)
                state.mask_b.fill(1 - seq_done)
        elif pool is not None:
            with timer("collect/env_step"):
                # one parallel step for all B envs (positions ride the infos)
                obs_next_b, step_rewards, _, loc_next_b = pool.step(det_np)
                done_b = [seq_done] * B
                for b in range(B):
                    if check_large_jump(state.loc_b[b], loc_next_b[b]) and t % cfg.prnn_seqdur != 0:
                        print("====== DEBUG START ======")
                        print(f"Large jump detected at step {t} (env {b}): from {state.loc_b[b]} to {loc_next_b[b]}")
                        print("====== DEBUG END ======")
                    obss[b][t] = pre_obs_b[b]
                    locs[b][t] = state.loc_b[b]
                    rewards[t, b] = step_rewards[b]

                    state.ep_return[b] += float(step_rewards[b])
                    state.ep_reshaped[b] += float(step_rewards[b])
                    state.ep_frames[b] += 1

                    state.obs_b[b] = obs_next_b[b]
                    state.loc_b[b] = loc_next_b[b]
                    state.mask_b[b] = 1 - seq_done
                    last_post_obs = obs_next_b[b]
        else:
            done_b = [False] * B
            with timer("collect/env_step"):
                for b, env in enumerate(envs):
                    # CAREFUL: obs_next is post-action; pre_obs_b[b] is pre-action
                    obs_next, reward, terminated, truncated, _ = env.step(
                        det_np[b : b + 1]
                    )
                    loc = _agent_pos(env)

                    done = terminated or truncated
                    if cfg.prnn_seqdur > 0 and (t + 1) % cfg.prnn_seqdur == 0:
                        done = True
                    done_b[b] = done

                    # DEBUG (historical: modulo only evaluated when a jump is seen)
                    if (
                        check_large_jump(state.loc_b[b], loc)
                        and t % cfg.prnn_seqdur != 0
                    ):
                        print("====== DEBUG START ======")
                        print(
                            f"Large jump detected at step {t} (env {b}): "
                            f"from {state.loc_b[b]} to {loc}"
                        )
                        print("====== DEBUG END ======")

                    obss[b][t] = pre_obs_b[b]
                    locs[b][t] = state.loc_b[b]
                    rewards[t, b] = reward

                    state.ep_return[b] += reward
                    state.ep_reshaped[b] += float(reward)
                    state.ep_frames[b] += 1

                    state.obs_b[b] = obs_next
                    state.loc_b[b] = loc
                    state.mask_b[b] = 1 - done
                    last_post_obs = obs_next

        # --- SR step (batched; before any reset, matching serial order) ---
        with timer("collect/sr_step"):
            if device_pool is not None:
                # Which observation the pRNN ingests IS the circuit: the one the
                # action was chosen from (offset 0), or the one it produced
                # (offset 1). `step_device` already returned the post-step
                # tensors; the pre-step ones are the rows just banked.
                sr_images, sr_directions = (
                    (device_images[t], device_directions[t])
                    if cfg.action_offset == 0
                    else (post_images, post_directions)
                )
                state.sr = tracker.step_device(
                    actions=action, images=sr_images, directions=sr_directions,
                )
            else:
                state.sr = tracker.step(det_np, pre_obs_b, state.obs_b)

        # Device episodes can only end at synchronized, Python-known seqdur
        # boundaries. No tensor value is inspected and no D2H occurs here.
        if device_pool is not None:
            if seq_done:
                _close_device_segment(post_images, post_directions)
            continue

        # --- per-env episode termination -----------------------------------
        # The old placement cloned the full (B, H) state once for every done
        # environment.  Clone once, then replace individual reset rows.
        if B > 1 and any(done_b):
            state.sr = state.sr.clone()
        for b in range(B):
            if not done_b[b]:
                continue
            if intrinsic_ref is not None and rewards[t, b].item() > 1e-5:
                intrinsic_ref.update_on_done(state, det_np)
            state.done_counter += 1
            state.finished_returns.append(state.ep_return[b])
            state.finished_reshaped.append(state.ep_reshaped[b])
            state.finished_frames.append(state.ep_frames[b])

            # The finished episode's last observation, banked before anything
            # resets it - it is the world model's final prediction target.
            last_obs_b[b].append(state.obs_b[b])

            def restart_env() -> None:
                if pool is None:
                    state.obs_b[b] = envs[b].reset()  # completely new position
                    state.loc_b[b] = _agent_pos(envs[b])

            # ORDER IS THE CIRCUIT, not a style choice. `reset_env` builds h[0]
            # from the observation it is handed, so under action_offset=1 the
            # environment has to have moved FIRST - otherwise h[0] encodes the
            # finished episode's last view, from a position the agent has
            # already left, and nothing says so.
            # Under action_offset=0 `init_sr` returns zeros and never reads the
            # observation, so the historical order is kept: both calls consume
            # RNG and tests/golden pins the sequence bitwise.
            if cfg.action_offset:
                restart_env()
            new_row = tracker.reset_env(b, state.obs_b[b])
            if B == 1:
                state.sr = new_row
            else:
                state.sr[b] = new_row[0]
            if cfg.action_offset == 0:
                restart_env()
            state.ep_return[b] = 0.0
            state.ep_reshaped[b] = 0.0
            state.ep_frames[b] = 0
            done_indices_b[b].append(t + 1)

        # pool resets are synchronized (seqdur cuts fire for every env at the
        # same t); one round-trip after the per-env bookkeeping above
        if pool is not None and any(done_b):
            assert all(done_b), "pool episode cuts must be synchronized"
            obs_reset_b, loc_reset_b = pool.reset_all()
            for b in range(B):
                state.obs_b[b] = obs_reset_b[b]
                state.loc_b[b] = loc_reset_b[b]

    if rollout_graph is not None:
        # Replays write the SR into the tracker's state buffer and nowhere
        # else; the carried copy is what the bootstrap value below and the
        # next rollout read.
        state.sr = rollout_graph.current_sr()

    if device_pool is not None:
        # Only directions and positions are needed by CPU diagnostics. Images
        # and terminal observations stay on-device for PPO/world-model work.
        meta_tb = torch.cat(
            (device_directions.unsqueeze(-1), device_positions), dim=-1
        ).cpu().numpy()
        last_observations = [
            {
                "mission": device_pool.mission,
                "image": batch[0][b],
                "direction": batch[1][b],
            }
            for b in range(B)
            for batch in device_last_batches
        ]
    else:
        meta_tb = np.asarray(
            [
                [
                    (
                        obss[b][t]["direction"],
                        locs[b][t][0],
                        locs[b][t][1],
                    )
                    for b in range(B)
                ]
                for t in range(T)
            ]
        )

    # Policy probabilities are diagnostics-only. Keep them on the training
    # device during the recurrent rollout and transfer once here; the old path
    # performed a second device->host copy in every timestep. np.add.at uses
    # the same historical (t, b) accumulation order without a Python loop.
    # `policy_logits` holds the distribution's own normalised logits (see
    # above); turn them into log-probs and probabilities now, in one batched
    # op each. Neither writes back, so a graphed rollout's reused buffer is
    # left holding logits, as its name says.
    log_probs = policy_logits.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

    probs_np = policy_logits.softmax(dim=-1).cpu().numpy()
    np.add.at(
        joint,
        (
            meta_tb[:, :, 0].reshape(-1),
            meta_tb[:, :, 1].reshape(-1),
            meta_tb[:, :, 2].reshape(-1),
        ),
        probs_np.reshape(B * T, -1),
    )

    # make sure last obs is included in done indices (per env stream)
    if device_pool is None:
        for b in range(B):
            if done_indices_b[b][-1] != T:
                done_indices_b[b].append(T)
                last_obs_b[b].append(state.obs_b[b])

    # --- flatten env-major: index = b*T + t --------------------------------
    flat_obss = (
        None
        if device_pool is not None
        else [obss[b][t] for b in range(B) for t in range(T)]
    )
    directions = meta_tb[:, :, 0].transpose(1, 0).reshape(B * T)
    positions = meta_tb[:, :, 1:].transpose(1, 0, 2).reshape(B * T, 2)
    flat_locs = [tuple(map(int, position)) for position in positions]

    if subroom_size_ is not None:
        # one batched call; (t, b) order matches the historical per-step appends
        step_locs = meta_tb[:, :, 1:].reshape(B * T, 2)
        subroom_ids = get_subroom_id(torch.from_numpy(step_locs), subroom_size_).tolist()

    def flat(x):  # (T, B, ...) -> (B*T, ...)
        return x.permute(1, 0, *range(2, x.dim())).reshape(B * T, *x.shape[2:])

    f_actions = flat(actions)
    f_values = flat(values)
    f_rewards = flat(rewards)
    f_log_probs = flat(log_probs)
    f_masks = flat(masks)
    f_SRs = flat(SRs)
    curious_rewards = torch.zeros(B * T, device=device)
    int_rewards = torch.zeros(B * T, device=device)

    done_indices: list[int] = []
    if device_pool is None:
        last_observations = []
    for b in range(B):
        done_indices.extend(
            b * T + d for d in done_indices_b[b] if not (b > 0 and d == 0)
        )
        if device_pool is None:
            last_observations.extend(last_obs_b[b])

    # --- curiosity reward ---------------------------------------------------
    actions_np = None
    logs: dict = {}
    if cfg.curious_agent:
        with timer("collect/curious_rewards"):
            if device_pool is not None:
                curious_rewards = adapter.prediction_mses_device(
                    images_tb=device_images,
                    directions_tb=device_directions,
                    actions_tb=actions,
                    last_batches=device_last_batches,
                    target_offset=REWARD_ALIGNMENTS[cfg.reward_alignment],
                )
            else:
                actions_np = f_actions.cpu().numpy()
                curious_rewards = compute_curious_rewards(
                    adapter,
                    obss=flat_obss,
                    actions_np=actions_np,
                    done_indices=done_indices,
                    last_observations=last_observations,
                    num_frames=B * T,
                    alignment=cfg.reward_alignment,
                )

    # Diagnostics and the serial world-model fallback consume NumPy actions;
    # one bulk transfer after any device-native curiosity pass.
    if actions_np is None:
        actions_np = f_actions.cpu().numpy()

    # --- intrinsic tail (B=1 only, historical code path) --------------------
    if intrinsic_ref is not None:
        int_rewards = intrinsic_ref.tail(state, f_SRs, last_post_obs)

    # --- bootstrap value + GAE per env stream -------------------------------
    if device_pool is not None:
        preprocessed = _device_policy_obss(device_obs[0], device_obs[1], acmodel)
    else:
        preprocessed = _preprocess_policy_obss(
            state.obs_b, acmodel, preprocess_obss, device
        )
    with torch.no_grad():
        _, next_values = acmodel(preprocessed, SR=state.sr)
    with timer("collect/gae"):
        advantages_tb = compute_gae(
            rewards=rewards,
            int_rewards=int_rewards.reshape(B, T).transpose(0, 1),
            curious_rewards=curious_rewards.reshape(B, T).transpose(0, 1),
            values=values,
            masks=masks,
            final_next_values=next_values,
            final_masks=state.mask_b,
            discount=cfg.discount,
            gae_lambda=cfg.gae_lambda,
            k_int=cfg.k_int,
            k_curious=cfg.k_curious,
        )
    advantages = flat(advantages_tb)

    exps = DictList()
    exps.obs = flat_obss
    exps.SR = f_SRs
    exps.action = f_actions
    exps.value = f_values
    exps.reward = f_rewards
    exps.advantage = advantages
    exps.returnn = (
        exps.value + exps.advantage
    )  # approximates current and discounted future returns
    exps.log_prob = f_log_probs
    exps.done_indices = done_indices
    exps.last_observations = last_observations

    loc_entropy, loc_entropy_5 = loc_stats.update(flat_locs)

    if device_pool is not None:
        # Reuse the already resident rollout bank. Values intentionally match
        # preprocess_images (float32 in [0,255]); the pRNN formatter divides
        # by 255 later.
        exps.obs = DictList({
            "image": flat(device_images).to(torch.float32),
            "direction": flat(device_directions).to(torch.uint8),
        })
    else:
        exps.obs = preprocess_obss(exps.obs, device=device)

    tracker.end_rollout()

    from curious_george.utils.common import mean_by_action  # local import: avoids cycle

    with timer("collect/log_prep"):
        curious_np = curious_rewards.cpu().numpy()
        adv_np = advantages.cpu().numpy()
        if cfg.curious_agent:
            curious_by_action = mean_by_action(curious_np, actions_np)
            logs = {f"curious_reward_{k}": v for k, v in curious_by_action.items()}
        adv_by_action = mean_by_action(adv_np, actions_np)

        # The two return keys are ABSENT, not zero, on a backend that cannot
        # measure extrinsic reward (see _close_device_segment). A missing series
        # cannot be misread; a flat zero can.
        if device_pool is None:
            logs["return_per_episode"] = list(state.finished_returns)
            logs["reshaped_return_per_episode"] = list(state.finished_reshaped)
        logs.update({
            "num_frames_per_episode": list(state.finished_frames),
            "num_frames": B * T,
            "num_episodes": state.done_counter,
            # numpy arrays, not Python lists: consumers only run synthesize()
            # (mean/std) on these, and .tolist() on 2048-long GPU tensors was
            # a measurable per-update sync cost.
            "intrinsic_rewards": int_rewards.cpu().numpy(),
            "curious_rewards": curious_np,
            "values": f_values.cpu().numpy(),
            "advantages": adv_np,
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
            "joint_dist": joint,
            "locs": flat_locs,
            "subroom_ids": subroom_ids,
            **{f"avg_adv_{k}": v for k, v in adv_by_action.items()},
        })

    state.finished_returns = []
    state.finished_reshaped = []
    state.finished_frames = []

    return CollectResult(
        exps=exps,
        logs=logs,
        directions=directions,
        locs=flat_locs,
        subroom_ids=subroom_ids,
        actions=f_actions,
        values=f_values,
        rewards=f_rewards,
        masks=f_masks,
        SRs=f_SRs,
        log_probs=f_log_probs,
        advantages=advantages,
        curious_rewards=curious_rewards,
        int_rewards=int_rewards,
        done_indices=done_indices,
        last_observations=last_observations,
        joint_dist=joint,
    )

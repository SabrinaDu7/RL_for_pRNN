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
from curious_george.rl.update.advantage import compute_gae_batched
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


@dataclass
class RolloutConfig:
    num_frames: int
    device: torch.device
    prnn_seqdur: int = 0
    pastSR: bool = True
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
    init_loc_b: list
    # episode logging (cumulative done counter never resets - historical)
    ep_return: list
    ep_reshaped: list
    ep_frames: list
    done_counter: int = 0
    finished_returns: list = field(default_factory=list)
    finished_reshaped: list = field(default_factory=list)
    finished_frames: list = field(default_factory=list)


def init_collector_state(envs, tracker) -> CollectorState:
    obs_b = [env.reset() for env in envs]
    loc_b = [_agent_pos(env) for env in envs]
    B = len(envs)
    return CollectorState(
        obs_b=obs_b,
        loc_b=loc_b,
        mask_b=np.ones(B, dtype=np.float32),
        sr=tracker.initial_sr(),
        init_loc_b=[torch.tensor(loc) for loc in loc_b],
        ep_return=[0.0] * B,
        ep_reshaped=[0.0] * B,
        ep_frames=[0] * B,
    )


@dataclass
class CollectResult:
    exps: DictList
    logs: dict
    # flat (B*T) views kept for analysis code that reads algo attributes
    obss: list
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
) -> CollectResult:
    pool = envs if isinstance(envs, AsyncShellPool) else None
    device_pool = envs if isinstance(envs, DeviceTableShellPool) else None
    B = len(envs)
    T = cfg.num_frames // B
    device = cfg.device

    if device_pool is not None:
        if cfg.prnn_seqdur <= 0 or T % cfg.prnn_seqdur:
            raise ValueError(
                "device_env requires positive synchronized prnn_seqdur cuts "
                "that divide frames/num_envs"
            )
        if not cfg.pastSR or intrinsic_ref is not None:
            raise ValueError(
                "device_env currently supports batched pastSR collection "
                "without the single-env intrinsic reference"
            )

    diagnostics_env = (
        pool
        if pool is not None
        else device_pool
        if device_pool is not None
        else envs[0]
    )
    joint = new_joint_probabilities(diagnostics_env, getattr(acmodel, "act_dim"))

    obss = [[None] * T for _ in range(B)]
    locs = [[None] * T for _ in range(B)]
    subroom_ids: list = []
    done_indices_b = [[0] for _ in range(B)]
    last_obs_b: list[list] = [[] for _ in range(B)]

    actions = torch.zeros((T, B), device=device, dtype=torch.int)
    values = torch.zeros((T, B), device=device)
    rewards = torch.zeros((T, B), device=device)
    log_probs = torch.zeros((T, B), device=device)
    masks = torch.zeros((T, B), device=device)
    SRs = torch.zeros((T, B, state.sr.shape[1]), device=device)
    policy_probs = torch.zeros(
        (T, B, getattr(acmodel, "act_dim")), device=device
    )

    if device_pool is not None:
        obs_shape = tuple(device_pool.obs_bank.shape[3:])
        device_images = torch.empty(
            (T, B, *obs_shape), dtype=torch.uint8, device=device
        )
        device_directions = torch.empty((T, B), dtype=torch.long, device=device)
        device_positions = torch.empty((T, B, 2), dtype=torch.long, device=device)
        device_obs = device_pool.observation_device()
        device_segment_initial = device_pool.positions.clone()
        device_segment_steps = 0
        device_reset_index = 0
        device_pool.prepare_resets(count=T // cfg.prnn_seqdur)
        # Each item is (post image, post direction, post position, initial
        # position) for one synchronized segment, all still on-device.
        device_last_batches: list[tuple[torch.Tensor, ...]] = []

    dist_travelled = 0
    last_post_obs = None  # final pre-reset obs (intrinsic tail, non-pastSR)

    for t in range(T):
        # --- action selection (one batched forward) ----------------------
        with timer("collect/policy_fwd"):
            with timer("collect/policy/preprocess"):
                if device_pool is not None:
                    preprocessed = DictList({"direction": device_obs[1]})
                else:
                    preprocessed = _preprocess_policy_obss(
                        state.obs_b, acmodel, preprocess_obss, device
                    )
            with timer("collect/policy/network"):
                with torch.no_grad():
                    dist, value = acmodel(preprocessed, SR=state.sr)
            with timer("collect/policy/sample"):
                action = dist.sample()  # based on SR from step t-1
            with timer("collect/policy/action_to_host"):
                # The device table consumes the sampled tensor directly.
                # CPU/async environments still require this synchronizing D2H.
                det_np = None if device_pool is not None else action.cpu().numpy()

        if cfg.prnn_seqdur > 0 and t % cfg.prnn_seqdur == 0:  # First loc of traj
            if device_pool is not None:
                device_segment_initial = device_pool.positions.clone()
            elif pool is not None:
                state.init_loc_b = [torch.tensor(loc) for loc in state.loc_b]
            else:
                state.init_loc_b = [torch.tensor(_agent_pos(env)) for env in envs]

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
            log_probs[t] = dist.log_prob(action)
            policy_probs[t] = dist.probs.detach()

        # --- environment stepping ----------------------------------------
        pre_obs_b = None if device_pool is not None else list(state.obs_b)
        done_b = [False] * B
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
                seq_done = (
                    cfg.prnn_seqdur > 0
                    and (t + 1) % cfg.prnn_seqdur == 0
                )
                done_b = [seq_done] * B
                state.mask_b.fill(1 - seq_done)
                device_segment_steps += 1
        elif pool is not None:
            with timer("collect/env_step"):
                # one parallel step for all B envs (positions ride the infos)
                obs_next_b, step_rewards, _, loc_next_b = pool.step(det_np)
                seq_done = cfg.prnn_seqdur > 0 and (t + 1) % cfg.prnn_seqdur == 0
                for b in range(B):
                    done_b[b] = seq_done
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
          with timer("collect/env_step"):
            for b, env in enumerate(envs):
                # CAREFUL: obs_next is post-action; pre_obs_b[b] is pre-action
                obs_next, reward, terminated, truncated, _ = env.step(det_np[b:b + 1])
                loc = _agent_pos(env)

                done = terminated or truncated
                if cfg.prnn_seqdur > 0 and (t + 1) % cfg.prnn_seqdur == 0:
                    done = True
                done_b[b] = done

                # DEBUG (historical: modulo only evaluated when a jump is seen)
                if check_large_jump(state.loc_b[b], loc) and t % cfg.prnn_seqdur != 0:
                    print("====== DEBUG START ======")
                    print(f"Large jump detected at step {t} (env {b}): from {state.loc_b[b]} to {loc}")
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
                state.sr = tracker.step_device(
                    actions=action,
                    images=device_images[t],
                    directions=device_directions[t],
                )
            else:
                state.sr = tracker.step(det_np, pre_obs_b, state.obs_b)

        # Device episodes can only end at synchronized, Python-known seqdur
        # boundaries. No tensor value is inspected and no D2H occurs here.
        if device_pool is not None:
            if done_b[0]:
                device_last_batches.append(
                    (
                        post_images.clone(),
                        post_directions.clone(),
                        device_pool.positions.clone(),
                        device_segment_initial.clone(),
                    )
                )
                for b in range(B):
                    state.done_counter += 1
                    state.ep_frames[b] += device_segment_steps
                    state.finished_returns.append(state.ep_return[b])
                    state.finished_reshaped.append(state.ep_reshaped[b])
                    state.finished_frames.append(state.ep_frames[b])
                    state.ep_return[b] = 0.0
                    state.ep_reshaped[b] = 0.0
                    state.ep_frames[b] = 0
                    done_indices_b[b].append(t + 1)

                # Preserve reset order: recurrent state first, environment
                # RNG streams second. Both resets are batched.
                state.sr = tracker.reset_all_envs()
                obs_reset_b, loc_reset_b = device_pool.apply_prepared_reset(
                    index=device_reset_index
                )
                device_reset_index += 1
                state.obs_b = obs_reset_b
                state.loc_b = [tuple(loc) for loc in loc_reset_b]
                device_obs = device_pool.observation_device()
                device_segment_steps = 0
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

            # reset order preserved: tracker/pN state first, then env
            new_row = tracker.reset_env(b, state.obs_b[b])
            if B == 1:
                state.sr = new_row
            else:
                state.sr[b] = new_row[0]
            last_obs_b[b].append(state.obs_b[b])

            dist_travelled = get_dist_travelled(
                state.init_loc_b[b].unsqueeze(0),
                torch.tensor(state.loc_b[b]).unsqueeze(0),
            ).item()
            if pool is None:
                state.obs_b[b] = envs[b].reset()  # completely new position
                state.loc_b[b] = _agent_pos(envs[b])
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

    if device_pool is not None:
        # Materialize science/analysis views only after the rollout. These
        # bulk transfers synchronize, but there is no barrier between any two
        # environment transitions.
        images_bt = device_images.permute(1, 0, 2, 3, 4).cpu().numpy()
        meta_bt = torch.cat(
            [
                device_directions.unsqueeze(-1),
                device_positions,
            ],
            dim=-1,
        ).permute(1, 0, 2).cpu().numpy()
        last_images = torch.stack(
            [batch[0] for batch in device_last_batches]
        ).permute(1, 0, 2, 3, 4).cpu().numpy()
        last_meta = torch.stack(
            [
                torch.cat(
                    [
                        batch[1].unsqueeze(-1),
                        batch[2],
                        batch[3],
                    ],
                    dim=-1,
                )
                for batch in device_last_batches
            ]
        ).permute(1, 0, 2).cpu().numpy()

        for b in range(B):
            for t in range(T):
                direction, x, y = meta_bt[b, t]
                obss[b][t] = {
                    "mission": device_pool.mission,
                    "image": images_bt[b, t],
                    "direction": int(direction),
                }
                locs[b][t] = (int(x), int(y))
            for segment in range(last_images.shape[1]):
                direction, x, y, init_x, init_y = last_meta[b, segment]
                last_obs_b[b].append(
                    {
                        "mission": device_pool.mission,
                        "image": last_images[b, segment],
                        "direction": int(direction),
                    }
                )
                if b == B - 1 and segment == last_images.shape[1] - 1:
                    dist_travelled = (
                        abs(int(x) - int(init_x))
                        + abs(int(y) - int(init_y))
                    )

    # Policy probabilities are diagnostics-only. Keep them on the training
    # device during the recurrent rollout and transfer once here; the old path
    # performed a second device->host copy in every timestep. Accumulation
    # remains in historical (t, b) order for exact float32 results.
    probs_np = policy_probs.cpu().numpy()
    for t in range(T):
        for b in range(B):
            hd = obss[b][t]["direction"]
            x, y = locs[b][t]
            joint[hd, x, y, :] += probs_np[t, b]

    # make sure last obs is included in done indices (per env stream)
    for b in range(B):
        if done_indices_b[b][-1] != T:
            done_indices_b[b].append(T)
            last_obs_b[b].append(state.obs_b[b])

    # --- flatten env-major: index = b*T + t --------------------------------
    flat_obss = [obss[b][t] for b in range(B) for t in range(T)]
    flat_locs = [locs[b][t] for b in range(B) for t in range(T)]

    if subroom_size_ is not None:
        # one batched call; (t, b) order matches the historical per-step appends
        step_locs = np.asarray([locs[b][t] for t in range(T) for b in range(B)])
        subroom_ids = get_subroom_id(torch.from_numpy(step_locs), subroom_size_).tolist()

    def flat(x):  # (T, B, ...) -> (B*T, ...)
        return x.permute(1, 0, *range(2, x.dim())).reshape(B * T, *x.shape[2:])

    f_actions = flat(actions)
    f_values = flat(values)
    f_rewards = flat(rewards)
    f_log_probs = flat(log_probs)
    f_masks = flat(masks)
    f_SRs = flat(SRs)
    advantages_tb = torch.zeros((T, B), device=device)
    curious_rewards = torch.zeros(B * T, device=device)
    int_rewards = torch.zeros(B * T, device=device)

    done_indices: list[int] = []
    last_observations: list = []
    for b in range(B):
        done_indices.extend(b * T + d for d in done_indices_b[b] if not (b > 0 and d == 0))
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
        preprocessed = DictList({"direction": device_obs[1]})
    else:
        preprocessed = _preprocess_policy_obss(
            state.obs_b, acmodel, preprocess_obss, device
        )
    with torch.no_grad():
        _, next_values = acmodel(preprocessed, SR=state.sr)
    with timer("collect/gae"):
        compute_gae_batched(
            advantages=advantages_tb,
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

        logs.update({
            "return_per_episode": list(state.finished_returns),
            "reshaped_return_per_episode": list(state.finished_reshaped),
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
            "dist_travelled": dist_travelled,
            **{f"avg_adv_{k}": v for k, v in adv_by_action.items()},
        })

    state.finished_returns = []
    state.finished_reshaped = []
    state.finished_frames = []

    return CollectResult(
        exps=exps,
        logs=logs,
        obss=flat_obss,
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

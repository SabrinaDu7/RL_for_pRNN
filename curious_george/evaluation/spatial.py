"""Spatial-representation metrics: SI, sRSA and sleep-wake distance (SWdist).

The three headline correctness metrics for the pRNN (see throwaway/ported/docs_legacy/refactor_baseline.md):
- sRSA should be HIGH: representational similarity tracks spatial proximity.
- SWdist should be LOW: spontaneous ("sleep") activity stays on the wake manifold.
- SI: per-unit spatial information of the place fields.

Collection is multi-trajectory (OMT-style): n_trajs rollouts of traj_timesteps
steps each - the SAME length as training trajectories (predNet.seqdur), so the
evaluated hidden-state distribution matches what training produces (a single
long rollout never resets the pRNN state; training does, every seqdur steps).
All trajectories are collected and pooled under ONE CPU device move, and each
metric is computed once on the pooled activity.

Logging: these are pRNN metrics and are logged by prnn, not by this repo
(Sabrina, 2026-07-09). The pooled path computes them here TEMPORARILY and
returns them without logging; moving computation+logging into prnn (on
precomputed activity) is the next refactor phase. The legacy path
(trainDecoder=True -> prnn's calculateSpatialRepresentation) still logs
internally as always.
"""

import numpy as np
import torch
from jaxtyping import Float

from prnn.utils import PredictiveNet
from prnn.analysis.representationalGeometryAnalysis import (
    representationalGeometryAnalysis as RGA,
)

from curious_george.models.device import eval_mode, on_device


def compute_sleep_wake_dist(
    pN: PredictiveNet,
    env,
    agent,
    *,
    sleepstd: float = 0.03,
    wake_timesteps: int = 5000,
    sleep_timesteps: int = 500,
) -> float:
    """Median cosine distance from each sleep frame to its nearest wake frame.

    Standalone variant that collects its own wake rollout; the training loop
    uses evaluate_spatial_representation, which derives SWdist from the shared
    pooled rollouts instead. Expects pN on CPU (numpy interop).
    """
    obs, act, state, _ = pN.collectObservationSequence(env, agent, wake_timesteps)
    with torch.no_grad():
        _, _, h = pN.predict(obs, act)
    wake_h = torch.mean(h, dim=0, keepdims=True)[0]
    return _sleep_wake_dist(
        pN, wake_h.detach().numpy(), sleepstd=sleepstd, sleep_timesteps=sleep_timesteps
    )


def _sleep_wake_dist(
    pN: PredictiveNet,
    wake_h: Float[np.ndarray, "N H"],
    *,
    sleepstd: float,
    sleep_timesteps: int,
) -> float:
    """SWdist from precomputed wake activity (mirrors prnn predictiveNet.py's
    internal computation: noise-driven spontaneous rollout + RGA distance)."""
    with torch.no_grad():
        _, sleep_h, _ = pN.spontaneous(sleep_timesteps, 0, sleepstd)
    sleep_h = torch.mean(sleep_h, dim=0, keepdims=True)[0]
    swdist, _, _ = RGA.calculateSleepWakeDist(
        wake_h, sleep_h.detach().numpy(), metric="cosine"
    )
    return float(swdist)


def collect_pooled_activity(
    pN: PredictiveNet,
    env,
    agent,
    *,
    n_trajs: int,
    traj_timesteps: int,
    onset_transient: int = 20,
) -> tuple[Float[np.ndarray, "N H"], Float[np.ndarray, "N 2"]]:
    """Theta-mean hidden activity and matching positions over `n_trajs` rollouts.

    Shared by the single-room and multi-room evaluations so that a pooled
    measurement across rooms is the SAME statistic as the per-room one, only
    over a bigger row set. Expects `pN` already on CPU and in eval mode.
    """
    h_rows: list = []
    pos_rows: list = []
    for _ in range(n_trajs):
        obs, act, state, _ = pN.collectObservationSequence(
            env, agent, traj_timesteps, discretize=True
        )
        with torch.no_grad():
            _, _, h = pN.predict(obs, act)
        h_mean: Float[torch.Tensor, "T H"] = torch.mean(h, dim=0)  # theta mean
        # h[t] pairs with agent_pos[t] (RGA convention: pos[:-1] vs h)
        h_rows.append(h_mean[onset_transient:])
        pos_rows.append(state["agent_pos"][onset_transient:-1, :])

    return (
        torch.cat(h_rows).detach().numpy(),
        np.concatenate(pos_rows).astype(np.float64, copy=False),
    )


def evaluate_spatial_representation(
    pN: PredictiveNet,
    env,
    agent,
    *,
    n_trajs: int = 8,
    traj_timesteps: int = 256,
    trainDecoder: bool = False,
    legacy_timesteps: int = 15000,
    sleepstd: float = 0.03,
    sleep_timesteps: int = 500,
    onset_transient: int = 20,
    active_time_threshold: int = 200,
    rng: np.random.Generator | None = None,
    probe_seed: int | None = None,
    wandb_nameext: str = "",
) -> dict:
    """Run the spatial eval on CPU and return {"sRSA", "SWdist", "SI"}.

    Default path: n_trajs trajectories of traj_timesteps steps (match
    predNet.seqdur so eval statistics match training), pooled, one pass per
    metric. trainDecoder=True selects the legacy prnn path (own rollout of
    legacy_timesteps steps, decoder fit, prnn-internal wandb logging) plus a
    second rollout for the returned SWdist.

    Moves pN (and the agent's AC model, if any) to CPU for the duration and
    restores placement after.
    """
    modules = [pN]
    if hasattr(agent, "acmodel"):
        modules.append(agent.acmodel)

    if probe_seed is not None:
        # The env owns its OWN Generator (gymnasium `np_random`), which
        # np.random.seed does not touch - so seeding globally left the start
        # position free and the eval unreproducible: identical action
        # sequences, different trajectories. Seeding the env too makes this a
        # FIXED probe, i.e. checkpoints become comparable to each other rather
        # than each carrying its own rollout noise.
        torch.manual_seed(probe_seed)
        np.random.seed(probe_seed)
        env.env.reset(seed=probe_seed)

    # eval_mode disables the input dropout (p=predNet.dropp, 0.15). That
    # dropout implements a DENOISING objective during world-model training -
    # it corrupts obs_out but never obs_target_out (Architectures.clip_mask) -
    # and torch implements it inverted, scaling survivors by 1/(1-p) so that
    # train-mode expectation matches eval mode. Measuring through it therefore
    # reports activations inflated 17.6% and then randomly zeroed: neither the
    # training nor the inference distribution. The agent never sees it either -
    # predict_single skips clip_mask entirely. Every other eval path in the
    # repo already does this (evaluation/task.py, evaluation/probe.py, and the
    # OMT task, now in the questions repo); this one was the outlier.
    #
    # The injected noise (predNet.trainNoiseMeanStd) is deliberately KEPT: it
    # is the model's dynamics, and it is what generates the "sleep" activity
    # SWdist compares against.
    with eval_mode(modules), on_device(modules, "cpu"):
        if trainDecoder:
            # legacy path: prnn does its own rollout, figures, decoder fit and
            # wandb logging; SWdist needs the second rollout
            _, SI, _, sRSA = pN.calculateSpatialRepresentation(
                env,
                agent,
                timesteps=legacy_timesteps,
                trainDecoder=True,
                trainHDDecoder=False,
                saveTrainingData=False,
                bitsec=False,
                calculatesRSA=True,
                sleepstd=sleepstd,
                wandb_nameext=wandb_nameext,
            )
            swdist = compute_sleep_wake_dist(
                pN, env, agent, sleepstd=sleepstd, wake_timesteps=legacy_timesteps
            )
            return {"sRSA": sRSA, "SWdist": swdist, "SI": SI}

        # ---- pooled multi-trajectory path ----
        h_pool, pos_pool = collect_pooled_activity(
            pN, env, agent,
            n_trajs=n_trajs,
            traj_timesteps=traj_timesteps,
            onset_transient=onset_transient,
        )

        # prnn owns metric computation AND wandb logging (Sabrina's rule);
        # requires the prnn pin at sdu/prnn-perf-optim or later (pyproject)
        metrics = pN.calculateSpatialMetrics(
            h_pool,
            pos_pool,
            env,
            sleepstd=sleepstd,
            sleep_timesteps=sleep_timesteps,
            active_time_threshold=active_time_threshold,
            # None -> prnn's fixed-seed default, so repeated scoring of one
            # checkpoint agrees; pass a generator to measure subsample noise.
            rng=rng,
            wandb_nameext=wandb_nameext,
        )
    return {
        "sRSA": metrics["sRSA"],
        "SWdist": metrics["SWdist"],
        "SI": metrics["SI"],
        **si_coverage(h_pool, active_time_threshold=active_time_threshold, SI=metrics["SI"]),
    }


def si_coverage(
    h: Float[np.ndarray, "N H"],
    *,
    active_time_threshold: int,
    SI,
) -> dict[str, float]:
    """How much of `mean SI` is a measurement, and how much is a structural zero.

    `prnn.utils.predictiveNet.calculateSpatialMetrics` does

        num_active = (h > 0).sum(axis=0)
        SI.iloc[num_active < active_time_threshold] = 0

    and wandb logs `SI.mean()` over ALL units. So a unit that barely fired is
    averaged in as **0, not excluded**, and `mean SI` conflates "carries no
    spatial information" with "hardly active". That matters with a direction:
    training sparsifies the representation, so more units fall below the
    threshold as a run proceeds and `mean SI` is pushed DOWN for a reason that
    has nothing to do with spatial tuning. It also makes SI incomparable across
    runs with different activity levels - e.g. the historical curves logged
    with dropout on.

    `mean SI` is kept as-is because it is the historical series and pN owns the
    metric (refactor plan, §7.2). These three numbers alongside it are what make
    it readable: the same threshold recomputed on the same activity, so they
    describe exactly the units prnn zeroed.
    """
    num_active = (np.asarray(h) > 0).sum(axis=0)
    active = num_active >= active_time_threshold
    values = np.asarray(SI["SI"] if hasattr(SI, "columns") else SI, dtype=float)
    return {
        "SI_units_total": float(active.size),
        "SI_units_zeroed": float((~active).sum()),
        "SI_mean_active_only": float(values[active].mean()) if active.any() else float("nan"),
    }


def evaluate_multi_room_representation(
    pN: PredictiveNet,
    env,
    agent,
    *,
    layouts: list,
    n_trajs: int = 8,
    traj_timesteps: int = 256,
    sleepstd: float = 0.03,
    sleep_timesteps: int = 500,
    onset_transient: int = 20,
    active_time_threshold: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-room and pooled spatial metrics, and the remapping index between them.

    WHAT THE REMAPPING INDEX MEASURES
    `mean(per-room sRSA) - pooled sRSA`. sRSA asks whether representational
    similarity tracks spatial proximity. Computed WITHIN one room it says the
    map is good there. Computed over rows POOLED across rooms it additionally
    requires that the same (x, y) in different rooms be represented alike,
    because pooled rows at the same position but from different rooms are
    treated as the same location.

    So the two hypotheses separate cleanly:
      - the network carries ONE position-only map (dead reckoning still wins):
        pooled ~= per-room, index ~= 0.
      - the network builds ROOM-SPECIFIC maps (it has bound what it sees to
        where it is): per-room stays high while pooled collapses, index > 0.

    An index near zero with high per-room sRSA is the null this experiment is
    trying to break; an index near zero with LOW per-room sRSA means the map
    itself degraded and the run is uninformative rather than negative. Report
    both numbers, never the index alone.

    Rooms are visited by swapping the eval shell's landmarks, so every room is
    measured through one network and one agent.
    """
    modules = [pN]
    if hasattr(agent, "acmodel"):
        modules.append(agent.acmodel)

    per_room: list[dict] = []
    h_rows: list = []
    pos_rows: list = []
    with eval_mode(modules), on_device(modules, "cpu"):
        for k, layout in enumerate(layouts):
            env.env.unwrapped.landmarks = list(layout.landmarks)
            env.reset()
            h, pos = collect_pooled_activity(
                pN, env, agent,
                n_trajs=n_trajs,
                traj_timesteps=traj_timesteps,
                onset_transient=onset_transient,
            )
            h_rows.append(h)
            pos_rows.append(pos)
            m = pN.calculateSpatialMetrics(
                h, pos, env,
                sleepstd=sleepstd,
                sleep_timesteps=sleep_timesteps,
                active_time_threshold=active_time_threshold,
                rng=rng,
                wandb_nameext=f"_room{k}",
            )
            per_room.append(
                {"key": layout.key, "sRSA": float(m["sRSA"]),
                 "SWdist": float(m["SWdist"]), "SI": float(np.nanmean(m["SI"]))}
            )

        pooled_metrics = pN.calculateSpatialMetrics(
            np.concatenate(h_rows), np.concatenate(pos_rows), env,
            sleepstd=sleepstd,
            sleep_timesteps=sleep_timesteps,
            active_time_threshold=active_time_threshold,
            rng=rng,
            wandb_nameext="_pooled",
        )

    mean_room_srsa = float(np.mean([r["sRSA"] for r in per_room]))
    return {
        "per_room": per_room,
        "pooled": {"sRSA": float(pooled_metrics["sRSA"]),
                   "SWdist": float(pooled_metrics["SWdist"]),
                   "SI": float(np.nanmean(pooled_metrics["SI"]))},
        "mean_room_sRSA": mean_room_srsa,
        "remapping_index": mean_room_srsa - float(pooled_metrics["sRSA"]),
    }

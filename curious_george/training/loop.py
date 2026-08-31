"""The training loop, with interval-triggered sections as plain functions."""

import time
from pathlib import Path

import torch

from tqdm import tqdm

from prnn.utils import save_pN

from curious_george.utils.common import synthesize
from curious_george.evaluation.on_policy import OnPolicyAnalysis, mutual_info_policy
from curious_george.evaluation.spatial import (
    evaluate_multi_room_representation,
    evaluate_spatial_representation,
)
from curious_george.log_and_store.storage import save_analysis_of_agent_behav, save_policy
from curious_george.training import logging as train_log
from curious_george.configs import EvalKind, SpatialEvalPath
from curious_george.utils.enums import AgentType
from curious_george.training.schedule import (
    EntropySchedule,
    TrainingCadence,
    TrainingSchedule,
)
from curious_george.training.setup import RunContext, TrainingComponents
from curious_george.models.device import on_device
from curious_george.utils.checkpoints import StatusCkptKeys
from curious_george.utils.timing import timer


#: Every eval `run_spatial_analysis` can run. Named once, so "is any spatial
#: eval requested" cannot drift from what the driver actually does.
_SPATIAL_EVALS = frozenset(
    {EvalKind.SPATIAL_ONPOLICY, EvalKind.SPATIAL_OFFPOLICY, EvalKind.SPATIAL_MULTIROOM}
)


def run_spatial_analysis(cfg, comps: TrainingComponents, wandb_log: bool) -> None:
    """sRSA + SWdist on- and/or off-policy (CPU; placement restored by contexts).

    Multi-room runs take a different path: each room is measured separately AND
    the rows are pooled across rooms, because the DIFFERENCE between those two
    is the remapping index this experiment turns on.
    """
    random_run = cfg.arch_policy.agent is AgentType.RANDOM
    layouts = getattr(comps.envs, "layouts", None)

    if EvalKind.SPATIAL_MULTIROOM in cfg.eval.evals:
        agent = comps.random_agent if random_run else comps.ac_agent
        # Evaluate a CAPPED, FIXED prefix of the layout set, not all of it. The
        # eval costs one CPU rollout set per room, so a 500-room pool spends more
        # wall-clock measuring than training - measured on the first pool run:
        # 7,167 gradient steps against the 3-room run's 22,542 in the same 5h40m.
        # The prefix is fixed rather than sampled so the series stays comparable
        # across checkpoints, which is the whole point of tracking it over time.
        scored = layouts[: cfg.eval.rooms_max]
        result = evaluate_multi_room_representation(
            comps.predictiveNet, comps.env, agent,
            layouts=scored,
            n_trajs=cfg.eval.n_trajs,
            traj_timesteps=cfg.collect.episode_steps,
            sleepstd=cfg.eval.sleep_std,
        )
        rooms = " ".join(f"{r['sRSA']:.3f}" for r in result["per_room"])
        # flush=True on every eval print, here and below. tqdm writes the
        # progress bar to STDERR, which is unbuffered, while these go to stdout
        # - block-buffered the moment a run is redirected to a file. A 4-hour
        # run therefore shows steady progress and NONE of its measurements: the
        # 2026-08-28 multi-room run completed four analysis events with no
        # `rooms sRSA` line in its log, which is the one series it exists to
        # produce.
        print(
            f"rooms sRSA [{rooms}] mean={result['mean_room_sRSA']:.4f} "
            f"pooled={result['pooled']['sRSA']:.4f} "
            f"remapping={result['remapping_index']:+.4f} "
            f"SWdist={result['pooled']['SWdist']:.4f} "
            f"episodes/room {list(map(int, comps.envs.layout_episodes))[:8]}"
            f"{'' if len(scored) == len(layouts) else f' [{len(scored)}/{len(layouts)} rooms scored]'}",
            flush=True,
        )
        if wandb_log:
            train_log.log_multi_room(result, comps.envs.layout_episodes)

    # Requested evaluations RUN. The old code returned after the multi-room
    # branch, so asking for the on-policy eval and silently getting the
    # multi-room one instead was invisible from the config.
    for kind, on_policy, nameext in [
        (EvalKind.SPATIAL_ONPOLICY, True, "_onPolicy"),
        (EvalKind.SPATIAL_OFFPOLICY, False, "_offPolicy"),
    ]:
        if kind not in cfg.eval.evals:
            continue
        use_ac = on_policy != random_run
        agent = comps.ac_agent if use_ac else comps.random_agent
        metrics = evaluate_spatial_representation(
            comps.predictiveNet, comps.env, agent,
            sleepstd=cfg.eval.sleep_std, wandb_nameext=nameext,
            n_trajs=cfg.eval.n_trajs,
            traj_timesteps=cfg.collect.episode_steps,  # eval trajs match training trajs
            trainDecoder=cfg.eval.spatial_path is SpatialEvalPath.LEGACY_DECODER,
            legacy_timesteps=cfg.eval.legacy_decoder_timesteps,
        )
        print(
            f"{nameext[1:]} sRSA={metrics['sRSA']:.4f} SWdist={metrics['SWdist']:.4f}",
            flush=True,
        )
        if wandb_log:
            train_log.log_spatial(metrics, nameext)


def run_behavior_analysis(
    cfg, comps: TrainingComponents, run_ctx: RunContext, update: int, with_figures: bool = True
) -> None:
    # Reuse the training rollout (free) for EVERY agent: since 22e7c0d the
    # random baseline collects through the same path and fills the same
    # buffers. The old RANDOM special case built a fresh algo clone that
    # dropped reward_alignment (an assertion error under action_offset=1) and
    # random_actions (the "random" analysis rollout sampled the POLICY) - it
    # cost four cluster jobs before this ran for the first time.
    opa = OnPolicyAnalysis(
        comps.algo,
        timesteps=cfg.eval.behaviour_timesteps,
        reuse_last_rollout=True,
    )
    if run_ctx.wandb_log:
        train_log.log_behavior(opa, with_figures=with_figures)
    else:
        save_analysis_of_agent_behav(opa, run_ctx.model_dir, update)


def save_checkpoint(
    cfg,
    comps: TrainingComponents,
    run_ctx: RunContext,
    num_frames: int,
    update: int,
    archive: bool = False,
) -> None:
    """Write the rolling checkpoints, and on `archive` step-tagged copies too.

    The rolling files are overwritten every time, so on their own a run leaves
    exactly ONE of each and the question "when did this representation appear"
    cannot be asked afterwards. The archive is a developmental series: 3.2 MB
    per pRNN, so a few hundred over a run is free.

    BOTH MODELS ARE ARCHIVED, and that is the point of this shape. Until
    2026-08-28 only the pRNN was, while the policy lived in a single rolling
    file - so an archived world model at step N could only ever be paired with
    the policy from the LAST write. Any evaluation that needs the agent that
    produced a checkpoint (an on-policy readout, a behavioural replay) was
    therefore impossible for every step but the last, silently: the files
    existed and loaded, they just were not contemporaries.
    """
    policy_save = {
        StatusCkptKeys.NUM_FRAMES.value: num_frames,
        StatusCkptKeys.UPDATE.value: update,
    }
    if cfg.arch_policy.agent is not AgentType.RANDOM:
        policy_save[StatusCkptKeys.MODEL_STATE.value] = comps.acmodel.state_dict()
        policy_save[StatusCkptKeys.OPTIMIZER_STATE.value] = comps.algo.optimizer.state_dict()
    if getattr(comps.algo, "count_bonus", None) is not None:
        policy_save[StatusCkptKeys.COUNT_VISITS.value] = comps.algo.count_bonus.state_dict()

    save_policy(policy_save, run_ctx.model_dir)

    trains_prnn = comps.predictiveNet is not None and cfg.train_prnn.train
    if trains_prnn:
        save_pN(comps.predictiveNet, run_ctx.model_dir + "predictiveNet_state.pt")

    if archive:
        archive_dir = Path(run_ctx.model_dir) / "checkpoints"
        archive_dir.mkdir(parents=True, exist_ok=True)
        # Named by ENVIRONMENT STEP, the one counter that means the same thing
        # across rollout shapes (see training/schedule.py), and the SAME step
        # in both filenames so a pair is found by matching the number.
        if trains_prnn:
            save_pN(comps.predictiveNet, str(archive_dir / f"predictiveNet_state_step{num_frames:010d}.pt"))
        torch.save(policy_save, archive_dir / f"policy_state_step{num_frames:010d}.pt")
        print(f"archived checkpoint at step {num_frames}", flush=True)

    print(f"pN and policy saved at {run_ctx.model_dir}")


def run_training(cfg, run_ctx: RunContext, comps: TrainingComponents) -> None:
    algo = comps.algo
    num_frames = comps.status[StatusCkptKeys.NUM_FRAMES.value]
    update = comps.status[StatusCkptKeys.UPDATE.value]
    start_time = time.time()
    n_performance = 0
    prnn_eval = bool(cfg.eval.evals & _SPATIAL_EVALS)

    schedule = TrainingSchedule.from_config(cfg)
    cadence = TrainingCadence.from_config(cfg, start_step=num_frames)
    entropy = EntropySchedule.from_config(cfg)
    print(schedule.summary())
    print(entropy.summary())
    # The ceiling `loc_entropy` is measured against. Printed once because it is
    # a property of the world, not a series - and stated at all because an
    # impassable-object arm and a walkable control have DIFFERENT ceilings, so
    # their raw entropies are not comparable without it.
    reachable = cfg.env.reachable_cells
    print(
        f"  reachable cells: {len(reachable)} "
        f"-> loc_entropy ceiling {cfg.env.loc_entropy_ceiling:.4f} bits"
    )
    if run_ctx.wandb_log:
        train_log.log_run_constants(cfg)
    if num_frames:
        remaining = schedule.total_env_steps - num_frames
        if remaining <= 0:
            # A resumed run's budget is the GRAND TOTAL across both phases, not
            # the size of this phase - `num_frames` comes out of the checkpoint.
            # Setting a phase-2-sized budget used to fall straight through the
            # loop and exit 0, so a two-phase experiment produced a run
            # directory, a provenance file and a wandb run for a phase that
            # never trained. Refuse instead: the only thing worse than a failed
            # run is one that looks finished.
            raise ValueError(
                f"nothing to train: resuming from a checkpoint at {num_frames:,} "
                f"environment steps, but the budget is {schedule.total_env_steps:,}. "
                "A resumed run's total_grad_steps is the TOTAL across all phases; "
                f"to add {abs(remaining):,} more steps, raise it past "
                f"{num_frames:,}."
            )
        print(f"  resuming at {num_frames:,} steps -> {remaining:,} to go")

    with tqdm(total=schedule.total_env_steps, desc="Processing") as pbar:
        while num_frames < schedule.total_env_steps:
            update_start = time.time()
            # Read per update: the loss reads algo.entropy_coef fresh each time
            # (rl/algo.py), so a schedule needs no plumbing beyond this.
            algo.entropy_coef = entropy.at(num_frames)

            # ONE collection path for both agents. A random agent is the
            # policy's BASELINE, so it has to differ from it in exactly one
            # thing - action selection - and share the backend, the batch, the
            # rooms and the world-model training. The retired
            # `randomAgent_collect_exp_and_update` was a separate serial routine
            # that forced `num_envs == 1`; that constraint was about the routine,
            # never about random actions.
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(
                exps=exps,
                update_params=(
                    not cfg.arch_policy.freeze_params
                    and cfg.arch_policy.agent is not AgentType.RANDOM
                ),
                # The BASELINE trains its world model on random-walk data - that
                # is the whole measurement. Only the POLICY stands still.
                update_world_model=not cfg.arch_policy.freeze_params,
            )
            logs = {**logs1, **logs2}

            update_duration = time.time() - update_start
            num_frames += logs["num_frames"]
            update += 1
            pbar.update(logs["num_frames"])

            # Each cadence fires at most once per update, so claim them all
            # here: `plot_due` gates both the trajectory figure and the
            # behaviour figures below.
            plot_due = cadence.plot.fire(num_frames)
            log_due = cadence.log.fire(num_frames)
            analysis_due = cadence.analysis.fire(num_frames)
            save_due = cadence.save.fire(num_frames)
            archive_due = cadence.archive.fire(num_frames)

            # --- periodic plotting (expensive: figure + GPU<->CPU model swap)
            if plot_due:
                # plotSampleTrajectory runs predict on CPU tensors; pin the
                # models to CPU for the call (placement restored on exit).
                with timer("log/sample_trajectory"):
                    with on_device([comps.predictiveNet, comps.acmodel], "cpu"):
                        comps.predictiveNet.plotSampleTrajectory(env=comps.env, agent=comps.ac_agent)

            # --- periodic logging -----------------------------------------
            if log_due:
                if run_ctx.wandb_log:
                    stats = train_log.UpdateStats(
                        num_frames=num_frames,
                        fps=logs["num_frames"] / update_duration,
                        duration=int(time.time() - start_time),
                        random_agent=cfg.arch_policy.agent is AgentType.RANDOM,
                        **schedule.gradient_steps_at(update),
                    )
                    mi = (
                        None
                        if cfg.arch_policy.agent is AgentType.RANDOM
                        else mutual_info_policy(logs["joint_dist"])
                    )
                    train_log.log_update(logs, stats, mi)

            # --- periodic analysis ----------------------------------------
            if analysis_due:
                if prnn_eval:
                    with timer("analysis/spatial"):
                        run_spatial_analysis(cfg, comps, run_ctx.wandb_log)
                if EvalKind.BEHAVIOUR in cfg.eval.evals:
                    with timer("analysis/behavior"):
                        # figures (3 Plotly builds over the full rollout) only
                        # on the plot cadence; the MI scalar every analysis event
                        run_behavior_analysis(
                            cfg, comps, run_ctx, update, with_figures=plot_due,
                        )

            # --- early stop -----------------------------------------------
            if cfg.run.early_stop and cfg.arch_policy.agent is not AgentType.RANDOM:
                if "return_per_episode" not in logs:
                    # Loudly, rather than never firing. Under exp.device_env the
                    # backend cannot measure extrinsic return, so this criterion
                    # has nothing to read - and it USED to read a fabricated 0.0
                    # and silently never trigger.
                    raise ValueError(
                        "logging.early_stop needs extrinsic return, which "
                        "exp.device_env does not measure. Turn one of them off."
                    )
                returns = synthesize(logs["return_per_episode"], signs=True)
                if returns["mean"] > 0.9 and returns["std"] < 0.05:
                    n_performance += 1
                    if n_performance == 25:
                        break

            # --- checkpointing ---------------------------------------------
            if save_due or archive_due:
                save_checkpoint(
                    cfg, comps, run_ctx, num_frames, update, archive=archive_due
                )

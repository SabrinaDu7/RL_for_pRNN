"""The training loop, with interval-triggered sections as plain functions."""

import time

import numpy as np
from tqdm import tqdm

from prnn.utils import save_pN

import wandb
from curious_george.utils.common import synthesize
from curious_george.evaluation.on_policy import OnPolicyAnalysis, mutual_info_policy
from curious_george.evaluation.spatial import evaluate_spatial_representation
from curious_george.storage import save_analysis_of_agent_behav, save_status
from curious_george.training import logging as train_log
from curious_george.training.schedule import TrainingCadence, TrainingSchedule
from curious_george.training.setup import RunContext, TrainingComponents
from curious_george.world_model.device import on_device
from curious_george.utils.checkpoints import StatusCkptKeys
from curious_george.utils.timing import timer


def run_spatial_analysis(cfg, comps: TrainingComponents, wandb_log: bool) -> None:
    """sRSA + SWdist on- and/or off-policy (CPU; placement restored by contexts)."""
    for enabled, on_policy, nameext in [
        (cfg.exp.onpolicy_prnn_eval, True, "_onPolicy"),
        (cfg.exp.offpolicy_prnn_eval, False, "_offPolicy"),
    ]:
        if not enabled:
            continue
        use_ac = on_policy != cfg.exp.random_action_agent
        agent = comps.ac_agent if use_ac else comps.random_agent
        metrics = evaluate_spatial_representation(
            comps.predictiveNet, comps.env, agent, sleepstd=0.03, wandb_nameext=nameext,
            n_trajs=cfg.exp.get("eval_trajs", 8),
            traj_timesteps=cfg.predNet.seqdur,  # eval trajs match training trajs
            trainDecoder=cfg.exp.get("eval_decoder", False),
            legacy_timesteps=cfg.exp.get("eval_timesteps", 15000),
        )
        print(f"{nameext[1:]} sRSA={metrics['sRSA']:.4f} SWdist={metrics['SWdist']:.4f}")
        if wandb_log:
            train_log.log_spatial(metrics, nameext)


def run_behavior_analysis(
    cfg, comps: TrainingComponents, run_ctx: RunContext, update: int, with_figures: bool = True
) -> None:
    # Reuse the training rollout (free) unless the random-action path is
    # active, which never fills the algo's buffers.
    opa = OnPolicyAnalysis(
        comps.algo,
        timesteps=25000,
        reuse_last_rollout=not cfg.exp.random_action_agent,
    )
    if run_ctx.wandb_log:
        train_log.log_behavior(opa, with_figures=with_figures)
    else:
        save_analysis_of_agent_behav(opa, run_ctx.model_dir, update)


def save_checkpoint(cfg, comps: TrainingComponents, run_ctx: RunContext, num_frames: int, update: int) -> None:
    status_save = {
        StatusCkptKeys.NUM_FRAMES.value: num_frames,
        StatusCkptKeys.UPDATE.value: update,
    }
    if not cfg.exp.random_action_agent:
        status_save[StatusCkptKeys.MODEL_STATE.value] = comps.acmodel.state_dict()
        status_save[StatusCkptKeys.OPTIMIZER_STATE.value] = comps.algo.optimizer.state_dict()

    save_status(status_save, run_ctx.model_dir)

    if comps.predictiveNet is not None and cfg.predNet.train:
        save_pN(comps.predictiveNet, run_ctx.model_dir + "predictiveNet_state.pt")

    print(f"pN and ACmodel status saved at {run_ctx.model_dir}")


def run_training(cfg, run_ctx: RunContext, comps: TrainingComponents) -> None:
    algo = comps.algo
    num_frames = comps.status[StatusCkptKeys.NUM_FRAMES.value]
    update = comps.status[StatusCkptKeys.UPDATE.value]
    start_time = time.time()
    n_performance = 0
    prnn_eval = cfg.exp.offpolicy_prnn_eval or cfg.exp.onpolicy_prnn_eval

    schedule = TrainingSchedule.from_config(cfg)
    cadence = TrainingCadence.from_config(cfg)
    print(schedule.summary())

    with tqdm(total=schedule.total_steps, desc="Processing") as pbar:
        while num_frames < schedule.total_steps:
            update_start = time.time()

            if cfg.exp.random_action_agent:
                logs = algo.randomAgent_collect_exp_and_update(comps.random_agent)
            else:
                exps, logs1 = algo.collect_experiences()
                logs2 = algo.update_parameters(
                    exps=exps, update_params=(not cfg.exp.random_init_control)
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
                        update=update,
                        num_frames=num_frames,
                        fps=logs["num_frames"] / update_duration,
                        duration=int(time.time() - start_time),
                        random_agent=cfg.exp.random_action_agent,
                    )
                    mi = (
                        None
                        if cfg.exp.random_action_agent
                        else mutual_info_policy(logs["joint_dist"])
                    )
                    train_log.log_update(logs, stats, mi)

            # --- periodic analysis ----------------------------------------
            if analysis_due:
                if prnn_eval:
                    with timer("analysis/spatial"):
                        run_spatial_analysis(cfg, comps, run_ctx.wandb_log)
                if cfg.exp.analyze_agent_behav:
                    with timer("analysis/behavior"):
                        # figures (3 Plotly builds over the full rollout) only
                        # on the plot cadence; the MI scalar every analysis event
                        run_behavior_analysis(
                            cfg, comps, run_ctx, update, with_figures=plot_due,
                        )

            # --- early stop -----------------------------------------------
            if cfg.logging.early_stop and not cfg.exp.random_action_agent:
                returns = synthesize(logs["return_per_episode"], signs=True)
                if returns["mean"] > 0.9 and returns["std"] < 0.05:
                    n_performance += 1
                    if n_performance == 25:
                        break

            # --- checkpointing ---------------------------------------------
            if save_due:
                save_checkpoint(cfg, comps, run_ctx, num_frames, update)

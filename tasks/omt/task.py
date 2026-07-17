"""Object Memory Task, built on the generic task template.

Train: the agent explores env_novel (novel object PRESENT), with both the AC
model and the pRNN learning. Eval: everything frozen, rollouts run in
env_orig (object ABSENT) - the probe is whether the trained pRNN still
predicts the object where it used to be, relative to the untrained control
copy. Do not swap the two envs.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

import wandb
from prnn.utils.Shell import FaramaMinigridShell

from curious_george import (
    AgentType,
    ActorCriticAgent,
    OnPolicyAnalysis,
    save_pN_and_acmodel,
)
from curious_george.envs.access import base_env
from curious_george.evaluation.task import (
    FreezeSpec,
    collect_eval_rollouts,
    collect_eval_rollouts_batched,
    setup_task,
    train_phase,
)
from curious_george.world_model.device import eval_mode, on_device
from tasks.omt.figure import figure_object_learning
from tasks.omt.metrics import State, quantify_object_learning  # noqa: F401 (State is re-exported API)


class ObjectMemoryTask:
    def __init__(
        self,
        args: DictConfig,
        agent_type: AgentType,
        env_orig: FaramaMinigridShell,
        env_novel: FaramaMinigridShell,
        device: torch.device,
        save_path: str,
        acmodel_status_ckpt: str,
        prnn_ckpt: str,
        new_obj_loc_ctrl: list[int] = [7, 11],
        env_eval_builder=None,  # () -> fresh env_orig copy; required for batched eval
    ):
        env_novel_first = env_novel[0] if isinstance(env_novel, (list, tuple)) else env_novel
        self.new_obj_pos = new_obj_loc_ctrl if env_novel_first.get_new_obj_pos() is None else env_novel_first.get_new_obj_pos()
        print(f"New object position in novel environment: {self.new_obj_pos}")
        with open_dict(args):
            args.tasks.new_obj_pos = self.new_obj_pos
        if not args.tasks.control:
            assert env_novel_first.get_new_obj_pos() is not None, (
                "env_novel must contain the novel object (env_train); "
                "env_orig is the eval env without it - do not swap them"
            )

        self.comps = setup_task(
            args,
            env_train=env_novel,   # novel object present: exploration/training
            env_eval=env_orig,     # object absent: memory-probe rollouts
            prnn_ckpt=prnn_ckpt,
            acmodel_ckpt=acmodel_status_ckpt,
            agent_type=agent_type,
            device=device,
            freeze=FreezeSpec(world_model=False, agent=False),
            control_copy=True,
        )

        self.args = args
        self.save_path = save_path
        self.wandb_log = args.logging.wandb_log

        # Analysis knobs
        self.oL_fig = args.tasks.analysis.objLearning_fig
        self.traj_fig = args.tasks.analysis.traj_fig
        self.start_up_bound = args.tasks.testing.start_up_bound
        self.start_low_bound = args.tasks.testing.start_low_bound

        self.seqdur = args.predNet.seqdur  # steps per trajectory
        self.trajs_per_batch = args.rl.trajs_per_batch
        self.trajs_test = args.tasks.testing.trajs

        self.env_eval_builder = env_eval_builder
        self.testTrial = None  # Updated after getTestTrial() called
        self.objectLearning = None  # Updated after quantifyObjectLearning() called

    # legacy attribute names, kept for external readers (figures, harnesses)
    @property
    def env_orig(self):
        return self.comps.env_eval

    @property
    def env_novel(self):
        return self.comps.env_train

    @property
    def pN_post(self):
        return self.comps.pN

    @property
    def pN_control(self):
        return self.comps.pN_control

    @property
    def ac_model(self):
        return self.comps.acmodel

    @property
    def algo(self):
        return self.comps.algo

    @property
    def agent(self):
        return self.comps.agent

    @property
    def agent_type(self):
        return self.comps.agent_type

    def _agent_modules(self):
        """Modules whose device placement tracks the agent: the trained pRNN
        plus the AC model when the agent is an ActorCriticAgent."""
        modules = [self.pN_post]
        if hasattr(self.agent, "acmodel"):
            modules.append(self.agent.acmodel)
        return modules

    def set_start_pos(self):
        self._set_start_pos_on(self.env_orig)

    def _set_start_pos_on(self, env):
        if not self.args.tasks.testing.start_random:
            base_env(env).agent_start_pos = np.random.randint(self.start_low_bound, self.start_up_bound)
            base_env(env).agent_start_dir = np.random.randint(0, 4)

    # ------------------------------------------------------------------ #

    def trainNovelObject(
        self,
        lr_trials: int,
        lrgroups: list,  # [0, 1, 2]
        num_trajs: int,  # num of trajectories to train on in novel env
        saving_interval: int,  # num batches between saves
        analysis_interval: int,  # num batches between analysis events
        device,
    ):
        with torch.no_grad():
            if self.wandb_log:
                self._run_object_analysis(traj_count=0)

        # Scale the pRNN learning rate for the novel-object exposure
        oldlr = [0.0 for _ in lrgroups]
        if isinstance(lr_trials, int):
            lr_trials = [lr_trials for _ in lrgroups]  # type: ignore
        for lidx, lgroup in enumerate(lrgroups):
            oldlr[lidx] = self.pN_post.optimizer.param_groups[lgroup]["lr"]
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx] * lr_trials[lidx]  # type: ignore

        num_batches = num_trajs // self.trajs_per_batch
        # e.g., 1000 trajs / 8 trajs per batch = 125 batches; each batch is one
        # collect_experiences of trajs_per_batch * seqdur steps (= rl.frames)

        def on_save(index: int) -> None:
            save_pN_and_acmodel(self.pN_post, self.algo.acmodel, self.save_path, index * self.trajs_per_batch)

        def on_analysis(index: int, traj_count: int) -> None:
            if self.traj_fig:
                # plotSampleTrajectory runs predict on CPU tensors;
                # placement is restored on exit.
                with on_device(self._agent_modules(), "cpu"):
                    self.pN_post.plotSampleTrajectory(
                        env=self.env_novel,
                        agent=self.agent,
                    )  # Logs to wandb inside if predictiveNet.wandb_log is True
            self._run_object_analysis(traj_count=traj_count)

        train_phase(
            self.comps,
            num_batches=num_batches,
            trajs_per_batch=self.trajs_per_batch,
            seqdur=self.seqdur,
            saving_interval=saving_interval,
            analysis_interval=analysis_interval,
            wandb_log=self.wandb_log,
            on_save=on_save,
            on_analysis=on_analysis,
        )

        save_pN_and_acmodel(self.pN_post, self.algo.acmodel, self.save_path, (num_batches - 1) * self.trajs_per_batch)
        print(f"Saved trained net to {self.save_path}")

        # Restore the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx]

    def _run_object_analysis(self, traj_count: int) -> None:
        """Test trial + metric + (wandb) figure - shared by the pre-training
        snapshot and the analysis-interval callback."""
        testTrial = self.getTestTrial(n_trajs=self.trajs_test)
        objectLearning = self.quantifyObjectLearning(
            ctrl_locs=self.args.tasks.testing.ctrl_locs,
            whichPhase=self.args.tasks.testing.whichPhase,
            traj_count=traj_count,
        )
        if objectLearning is not None and testTrial is not None and self.wandb_log:
            obj_learn_fig = figure_object_learning(
                env_name=self.env_orig,
                run_name=self.save_path,
                traj_num=traj_count,
                save_folder=self.save_path,
                objectLearning=objectLearning,
                testTrial=testTrial,
                rl_storage=".",
                config=self.args,
                pN=self.pN_post,
                show=False,
                save=False,
            )
            wandb.log({"Analysis/ObjectLearning": wandb.Image(obj_learn_fig)})
            wandb.log({"Analysis/Novel Object In-view Times": objectLearning["inviewtimes"]})
            wandb.log({"Analysis/Goal Modulation Vs. Step Count": objectLearning["goalmodulation"]})
            wandb.log({"Analysis/Goal Minus Ctrl Vs. Step Count": objectLearning["goalmodulation"] - objectLearning["ctlmodulation_diffloc"]})
            wandb.log({"Analysis/Avg Distance Travelled": objectLearning["avg_dist"]})
            plt.close()

    # ------------------------------------------------------------------ #

    def _dual_net_predictions(self, obs, act, state, render) -> dict:
        """OMT eval stat: predictions of BOTH nets from the same fixed noise
        state (randInit=False), so trained-vs-control differences are the
        signal. (1,1,500) mirrors the historical hidden-size assumption."""
        h_state = self.pN_post.pRNN.generate_noise((0, 0.03), (1, 1, 500))
        h_state = self.pN_post.pRNN.rnn.cell.actfun(h_state)

        obs_pred, _, _ = self.pN_post.predict(obs, act, state=h_state, randInit=False)
        obs_pred_notrain, _, _ = self.pN_control.predict(obs, act, state=h_state, randInit=False)
        return {"obs_pred": obs_pred, "obs_pred_control": obs_pred_notrain}

    def getTestTrial(self, n_trajs: int):
        eval_modules = [self.pN_post, self.pN_control]
        if hasattr(self.agent, "acmodel"):
            assert isinstance(self.agent, ActorCriticAgent)
            eval_modules.append(self.agent.acmodel)

        with eval_mode(eval_modules, agent=self.agent):
            self.set_start_pos()
            opa = OnPolicyAnalysis(self.algo, timesteps=int(self.seqdur * 30))
            if self.wandb_log:
                wandb.log({"Eval/MI_policy": opa.mi})
                wandb.log({"Eval/OPA_Advantages": wandb.Plotly(opa.plot_advantages())})
                wandb.log({"Eval/OPA_Policy_Heatmaps": wandb.Plotly(opa.plot_policy_heatmaps())})
                if self.args.tasks.analysis.occupancy:
                    wandb.log({"Eval/OPA_Occupancy": wandb.Plotly(opa.plot_occupancy())})

            if self.args.tasks.testing.get("batched", False):
                assert self.env_eval_builder is not None, (
                    "batched eval needs env_eval_builder (fresh env_orig copies)"
                )
                envs_eval = [self.env_eval_builder() for _ in range(n_trajs)]
                for e in envs_eval:
                    assert e.get_new_obj_pos() is None, "eval env copies must NOT contain the novel object"
                rollouts = collect_eval_rollouts_batched(
                    envs_eval=envs_eval,
                    agent=self.agent,
                    pN=self.pN_post,
                    T=self.seqdur,
                    eval_modules=eval_modules,
                    include_render=self.oL_fig,
                    before_each=self._set_start_pos_on,
                    traj_stats_fn=self._dual_net_predictions,
                )
            else:
                rollouts = collect_eval_rollouts(
                    env_eval=self.comps.env_eval,  # CRITICAL: the object-ABSENT env
                    agent=self.agent,
                    pN=self.pN_post,
                    n_trajs=n_trajs,
                    T=self.seqdur,
                    eval_modules=eval_modules,
                    include_render=self.oL_fig,
                    before_each=self.set_start_pos,
                    traj_stats_fn=self._dual_net_predictions,
                )

        objectTest = {
            "obs": rollouts.obs,
            "obs_pred": rollouts.extras["obs_pred"],
            "obs_pred_control": rollouts.extras["obs_pred_control"],
            "agent_pos": rollouts.agent_pos,
            "agent_dir": rollouts.agent_dir,
            "render": rollouts.renders if self.oL_fig else None,
        }

        self.testTrial = objectTest
        return objectTest

    def quantifyObjectLearning(self, ctrl_locs: list[list[int]], whichPhase: int, traj_count: int) -> dict | None:
        assert self.testTrial is not None, (
            "You need to run trainNovelObject and getTestTrial first."
        )
        self.objectLearning = quantify_object_learning(
            self.testTrial,
            env_shell=self.pN_post.env_shell,
            new_obj_pos=self.new_obj_pos,
            ctrl_locs=ctrl_locs,
            whichPhase=whichPhase,
            traj_count=traj_count,
        )
        return self.objectLearning

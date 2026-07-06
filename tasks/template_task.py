"""Skeleton for writing a new evaluation task on the shared template.

Copy this file to tasks/<your_task>/task.py (+ a main_task.py entry like
tasks/omt/main_task.py) and fill in the three task-specific pieces:

  1. which envs to use (env_train vs env_eval - they MAY differ on purpose,
     as in the OMT memory probe; never mix them up),
  2. what stays frozen during the exploration phase (FreezeSpec),
  3. which stats to collect from the frozen eval rollouts (traj_stats_fn).

Everything else (checkpoint loading, algo construction, the training loop,
serial/batched eval collection) comes from curious_george.evaluation.task.
Config: reuse Configs/main.yaml groups and add a Configs/tasks/<name>.yaml
for the task-specific keys (see Configs/tasks/omt.yaml).
"""

import torch

from curious_george import AgentType
from curious_george.evaluation.task import (
    FreezeSpec,
    collect_eval_rollouts,
    setup_task,
    train_phase,
)
from curious_george.world_model.device import eval_mode


class MyTask:
    def __init__(self, args, agent_type: AgentType, env_train, env_eval,
                 prnn_ckpt: str, acmodel_ckpt: str, device: torch.device):
        self.args = args
        self.comps = setup_task(
            args,
            env_train=env_train,          # may be a list for exp.num_envs > 1
            env_eval=env_eval,
            prnn_ckpt=prnn_ckpt,
            acmodel_ckpt=acmodel_ckpt,
            agent_type=agent_type,
            device=device,
            # e.g. frozen agent, learning world model:
            freeze=FreezeSpec(world_model=False, agent=True),
            control_copy=True,            # keep a frozen pre-task pRNN copy
        )

    def explore(self, num_batches: int):
        """Exploration/training phase (whatever FreezeSpec allows learns)."""
        train_phase(
            self.comps,
            num_batches=num_batches,
            trajs_per_batch=self.args.rl.trajs_per_batch,
            seqdur=self.args.predNet.seqdur,
            saving_interval=999,
            analysis_interval=999,
            wandb_log=self.args.logging.wandb_log,
            # on_save=..., on_analysis=...   # task-specific hooks
        )

    def _my_stats(self, obs, act, state, render) -> dict:
        """Per-trajectory eval stats. obs (1,T+1,X) pred-format, act (1,T,A);
        state has agent_pos/agent_dir. Return tensors; they are stacked to
        (B, ...) in EvalRollouts.extras. Example: hidden states from a fresh
        zero-state prediction pass."""
        with torch.no_grad():
            _, _, h = self.comps.pN.predict(obs, act, randInit=False,
                                            state=torch.zeros((1, 1, self.comps.pN.hidden_size)))
        return {"hidden": h}

    def evaluate(self, n_trajs: int):
        """Everything frozen; collect rollouts + stats in env_eval."""
        eval_modules = [self.comps.pN, self.comps.acmodel]
        if self.comps.pN_control is not None:
            eval_modules.append(self.comps.pN_control)

        with eval_mode(eval_modules, agent=self.comps.agent):
            rollouts = collect_eval_rollouts(   # or collect_eval_rollouts_batched
                env_eval=self.comps.env_eval,
                agent=self.comps.agent,
                pN=self.comps.pN,
                n_trajs=n_trajs,
                T=self.args.predNet.seqdur,
                eval_modules=eval_modules,
                traj_stats_fn=self._my_stats,
            )
        # rollouts.obs / .actions / .agent_pos / .agent_dir / .extras["hidden"]
        return rollouts

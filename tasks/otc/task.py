"""Object-Trace Cells task: exposure with STOCHASTIC object presence.

The OMT exposure phase puts the object at a fixed place in every episode, so
the object is a deterministic function of the agent's position. The pRNN's
hidden state already encodes position, which makes the object redundant with
what `h` carries - and gradient descent then absorbs it into the linear readout
rather than the recurrent dynamics (docs/exp_object_trace_cells_2026-07-30.md:
readout +0.0625 vs dynamics +0.0157 gain-corrected, 9/9 runs).

This task breaks the redundancy. Each training env independently gets the
object with probability `presence_prob`, re-randomised every batch. Position no
longer predicts the object, so predicting it requires integrating evidence from
the observation history - which is a job only the recurrent state can do.

Nothing in the prnn package changes: the object is toggled on the environment
(`new_obj_pos = None` and back), and the pRNN sees a different observation
stream as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

import wandb
from curious_george import AgentType, save_pN_and_acmodel
from curious_george.envs.access import base_env
from curious_george.evaluation.task import FreezeSpec, setup_task, train_phase


class ObjectTraceTask:
    """Exposure with per-episode stochastic object presence."""

    def __init__(
        self,
        args: DictConfig,
        agent_type: AgentType,
        envs_train: list,
        env_eval,
        device: torch.device,
        save_path: str,
        prnn_ckpt: str,
        acmodel_status_ckpt: str,
        obj_pos: list[int],
        random_position: bool = False,
        colors: list[str] | None = None,
    ):
        self.obj_pos = list(obj_pos)
        self.random_position = random_position
        # Scenario A: object identity varies per episode at a FIXED location.
        # Position then tells the net "something is here" but not "which" - the
        # only route to predicting the colour is remembering having seen it.
        self.colors = list(colors) if colors else ["green"]
        self.args = args
        self.save_path = str(save_path)
        self.wandb_log = args.logging.wandb_log
        self.seqdur = args.predNet.seqdur
        self.trajs_per_batch = args.rl.trajs_per_batch

        if random_position:
            from scripts.analysis_OMT import (
                get_walkable_mask, get_walkable_minigrid_positions,
            )
            self._walkable = [
                tuple(p.tolist())
                for p in get_walkable_minigrid_positions(get_walkable_mask(env_eval))
            ]
            print(f"random object position over {len(self._walkable)} walkable cells")

        self.comps = setup_task(
            args,
            env_train=envs_train,
            env_eval=env_eval,
            prnn_ckpt=prnn_ckpt,
            acmodel_ckpt=acmodel_status_ckpt,
            agent_type=agent_type,
            device=device,
            freeze=FreezeSpec(world_model=False, agent=False),
            control_copy=True,
        )

    # ------------------------------------------------------------------ #

    def _sample_position(self, rng: np.random.Generator):
        """A walkable cell for the object, or the fixed one when not randomising."""
        if not self.random_position:
            return tuple(self.obj_pos)
        return tuple(self._walkable[rng.integers(len(self._walkable))])

    def _sample_color(self, rng: np.random.Generator) -> str:
        """Object colour for one episode (Scenario A); constant if one colour."""
        return self.colors[int(rng.integers(len(self.colors)))]

    def _set_object(self, env, present: bool, pos=None, color: str | None = None) -> None:
        """Toggle the object on one env; the next reset regenerates the grid.

        Safe because the object is a non-blocking floor tile: the walkable mask
        and all algo geometry are unchanged either way. The banked observation
        wrapper re-keys off the live grid fingerprint inside reset(), so no
        stale object-present observations survive the toggle
        (curious_george/envs/obs_bank.py).
        """
        u = base_env(env)
        u.new_obj_pos = tuple(pos if pos is not None else self.obj_pos) if present else None
        u.new_obj_color = (color or self.colors[0]) if present else None
        env.reset()

    def _randomise_objects(self, rng: np.random.Generator, presence_prob: float) -> int:
        """Independently toggle each training env; return how many got the object."""
        flags = rng.random(len(self.comps.envs_train)) < presence_prob
        for env, present in zip(self.comps.envs_train, flags):
            self._set_object(env, bool(present), pos=self._sample_position(rng),
                             color=self._sample_color(rng))
        return int(flags.sum())

    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_trajs: int,
        presence_prob: float,
        saving_interval_trajs: int,
        lr_trials,
        lrgroups: list,
        wd_trials=None,
        seed: int = 0,
    ) -> None:
        """Exposure phase with the object present in a random subset of envs.

        presence_prob=1.0 reproduces the OMT exposure (object always present).
        """
        rng = np.random.default_rng(seed)
        num_batches = num_trajs // self.trajs_per_batch
        save_every = max(1, saving_interval_trajs // self.trajs_per_batch)

        oldlr, oldwd = [], []
        if isinstance(lr_trials, int):
            lr_trials = [lr_trials for _ in lrgroups]
        # Scenario D: per-group weight-decay scaling. `predNet.sparsity` (f) is
        # init-only (norm.ppf(f) sets the bias at construction), so it cannot be
        # changed on a loaded checkpoint; weight decay is the available lever for
        # pressuring the exposure-phase weight change to be small/localised
        # rather than spread over the whole recurrent matrix.
        if wd_trials is not None and isinstance(wd_trials, (int, float)):
            wd_trials = [wd_trials for _ in lrgroups]
        for lidx, g in enumerate(lrgroups):
            oldlr.append(self.comps.pN.optimizer.param_groups[g]["lr"])
            self.comps.pN.optimizer.param_groups[g]["lr"] = oldlr[lidx] * lr_trials[lidx]
            oldwd.append(self.comps.pN.optimizer.param_groups[g].get("weight_decay", 0.0))
            if wd_trials is not None:
                self.comps.pN.optimizer.param_groups[g]["weight_decay"] = (
                    oldwd[lidx] * wd_trials[lidx])
        if wd_trials is not None:
            print("weight_decay per group: "
                  + ", ".join(f"g{g}={self.comps.pN.optimizer.param_groups[g]['weight_decay']:.2e}"
                              for g in lrgroups))
        print(
            f"presence_prob={presence_prob}  batches={num_batches}  "
            f"save every {save_every} batches (= {save_every * self.trajs_per_batch} trajs)"
        )

        for index in range(num_batches):
            n_present = self._randomise_objects(rng, presence_prob)
            if self.wandb_log:
                wandb.log({"Train/envs_with_object": n_present})
            # one batch at a time so presence can be re-randomised between them
            train_phase(
                self.comps,
                num_batches=1,
                trajs_per_batch=self.trajs_per_batch,
                seqdur=self.seqdur,
                saving_interval=10**9,
                analysis_interval=10**9,
                wandb_log=self.wandb_log,
            )
            if index % save_every == 0:
                self._save(index * self.trajs_per_batch)

        self._save((num_batches - 1) * self.trajs_per_batch)
        for lidx, g in enumerate(lrgroups):
            self.comps.pN.optimizer.param_groups[g]["lr"] = oldlr[lidx]
            self.comps.pN.optimizer.param_groups[g]["weight_decay"] = oldwd[lidx]

    def _save(self, count: int) -> None:
        save_pN_and_acmodel(
            self.comps.pN,
            self.comps.algo.acmodel,
            self.save_path,
            count,
            ac_optimizer=self.comps.algo.optimizer,
        )

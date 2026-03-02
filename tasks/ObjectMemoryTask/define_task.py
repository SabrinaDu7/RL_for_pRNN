import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import wandb
from jaxtyping import Float
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import TypedDict, Optional, Iterable

from RLutils import (
    save_status,
    get_pN,
    get_SR_acmodel,
    get_obss_preprocessor,
    get_algo,
    seed,
    PredictivePPOAlgo,
    get_occupancy_fig,
    get_agent,
    synthesize,
    get_dist_travelled,
    save_pN_and_acmodel,
    OnPolicyAnalysis,
    ActorCriticAgent,
)

from utils import StatusCkptKeys
from prnn.utils import save_pN
from prnn.utils.Shell import FaramaMinigridShell
from tasks.ObjectMemoryTask.figure import figure_object_learning, figure_goal_modulation_vs_trajectories
from utils import AgentType

class State(TypedDict):
    agent_pos: Float[np.ndarray, "T+1 2"]
    agent_dir: Float[np.ndarray, "T+1"]
    SRs: Optional[Float[torch.Tensor, "(T+1) * hidden_size"]] # Not present with RandomActionAgent


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
    ):
        seed(args.exp.seed)

        self.env_orig = env_orig
        self.env_novel = env_novel

        self.new_obj_pos = new_obj_loc_ctrl if self.env_novel.get_new_obj_pos() is None else self.env_novel.get_new_obj_pos()
        print(f"New object position in novel environment: {self.new_obj_pos}")
        with open_dict(args):
            args.tasks.new_obj_pos = self.new_obj_pos

        self.args = args
        self.save_path = save_path
        self.wandb_log = args.logging.wandb_log

        # Analysis
        self.oL_fig = self.args.tasks.analysis.objLearning_fig
        self.traj_fig = self.args.tasks.analysis.traj_fig
        self.start_up_bound = self.args.tasks.testing.start_up_bound
        self.start_low_bound = self.args.tasks.testing.start_low_bound

        self.seqdur = args.predNet.seqdur # steps
        self.trajs_per_batch = args.rl.trajs_per_batch
        self.trajs_test = args.tasks.testing.trajs

        self.pN_control = get_pN(args=args, env=self.env_orig, device=device, pRNN_ckpt=prnn_ckpt)
        self.pN_post = self.pN_control.copy()
        self.pN_post.pRNN.train()
        self.pN_control.pRNN.eval()

        self.pN_control.wandb_log = self.wandb_log
        self.pN_post.wandb_log = self.wandb_log

        obs_space, preprocess_obss = get_obss_preprocessor(
            self.env_orig.observation_space
        )
        self.ac_model = get_SR_acmodel(
            args,
            env_act_space=self.env_orig.action_space,
            obs_space=obs_space,
            acmodel_status_ckpt=acmodel_status_ckpt,
            device=device,
        )
        self.algo = get_algo(
            args,
            env=self.env_novel, # CRITICAL: Novel object env
            predictiveNet=self.pN_post,
            acmodel=self.ac_model,
            preprocess_obss=preprocess_obss,
            AlgoClass=PredictivePPOAlgo,
            acmodel_status_ckpt=acmodel_status_ckpt,
            device=device,
        )

        self.agent_type = agent_type
        self.agent = get_agent(
            env=self.env_novel, 
            agent_Type=self.agent_type, 
            ac_model=self.ac_model, 
            prnn=self.pN_post, 
            device=device
        )

        # Train the decoder
        # if decoder is 'train':
        #    self.decoder = self.trainDecoder()
        # else:
        #    self.decoder = decoder

        # Train the network in the object room #Note, copy ExperienceReplayAnalysis
        # approach for multiple learning rates (consider rec and FF learning rates)

        self.testTrial = None  # Updated after getTestTrial() called
        self.objectLearning = None  # Updated after quantifyObjectLearning() called

    def trainDecoder(self):
        env = self.pN_control.EnvLibrary[0]
        _, _, decoder, sRSA = self.pN_control.calculateSpatialRepresentation(
            env, self.agent, numBatches=10000, trainDecoder=True
        )
        return decoder

    def set_start_pos(self):
        if not self.args.tasks.testing.start_random:
            self.env_orig.env.unwrapped.agent_start_pos = np.random.randint(self.start_low_bound, self.start_up_bound)
            self.env_orig.env.unwrapped.agent_start_dir = np.random.randint(0, 4)

    def trainNovelObject(
        self,
        lr_trials: int,
        lrgroups: list, # [0, 1, 2]
        num_trajs: int, # num of trajectories to train on in novel env
        saving_interval: int, # num batches between logging
        analysis_interval: int, # num batches between logging
        device,
    ):
        with torch.no_grad():
            if self.wandb_log:
                testTrial = self.getTestTrial(n_trajs=self.trajs_test)
                objectLearning = self.quantifyObjectLearning(
                    ctrl_locs=self.args.tasks.testing.ctrl_locs,
                    whichPhase=self.args.tasks.testing.whichPhase,
                    traj_count=0,
                )

                # Initial Object Learning Figure
                obj_learn_fig = figure_object_learning(env_name=self.env_orig, 
                                                                        run_name=self.save_path, 
                                                                        traj_num=0, 
                                                                        save_folder=self.save_path,
                                                                        objectLearning=objectLearning,
                                                                        testTrial=testTrial,
                                                                        rl_storage=".",
                                                                        config=self.args,
                                                                        pN=self.pN_post,
                                                                        show=False,
                                                                        save=False)
                wandb.log({"Analysis/ObjectLearning": wandb.Image(obj_learn_fig)})
                plt.close()

        # Update the learning rate
        oldlr = [0.0 for i in lrgroups]
        if isinstance(lr_trials, int):
            lr_trials = [lr_trials for i in lrgroups] # type: ignore
        for lidx, lgroup in enumerate(lrgroups):
            oldlr[lidx] = self.pN_post.optimizer.param_groups[lgroup]["lr"]
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx] * lr_trials[lidx] # type: ignore

        num_batches = num_trajs // self.trajs_per_batch
        # e.g., 1000 trajs / 8 trajs per batch = 125 batches
        # Each loop/batch will call collect_experiences, which collects trajs_per_batch trajectories = trajs_per_batch * seqdur steps
        # trajs_per_batch * seqdur = 8 * 256 = 2048 steps (defined as frames in Conf1_Adel.yaml)

        for index in tqdm(range(num_batches)): 

            traj_count = (index + 1) * self.trajs_per_batch
            step_count = traj_count * self.seqdur

            if self.wandb_log:
                wandb.log({"step_count": step_count})

            # 125 batches = 125 * 4 gradient steps,
            # 8 trajs per batch = 2048 steps per batch

            if self.agent_type == AgentType.RANDOM:
                exp_logs = self.algo.randomAgent_collect_exp_and_update(self.agent)
            else:
                exps, exp_logs = self.algo.collect_experiences()
                logs2 = self.algo.update_parameters(exps=exps, update_params=True)
                
                if self.wandb_log:
                    cur_rewards = synthesize(exp_logs["curious_rewards"], abs=True)
                    wandb.log({f"Train/cur_rewards": cur_rewards["mean"]})
                    wandb.log({f"Train/cur_rewards": cur_rewards["std"]})
                    for key, val in logs2.items():
                        wandb.log({f"Train/{key}": val})
                    
                    for key, val in exp_logs.items():                                                                                                                        
                        if key.startswith("avg_adv") or key.startswith("curious_reward"):                                                                                                                       
                            wandb.log({f"Train/{key}": val})  

            if index % saving_interval == 0:
                save_pN_and_acmodel(self.pN_post, self.algo.acmodel, self.save_path, index * self.trajs_per_batch)
            
            if index % analysis_interval == 0:
                with torch.no_grad():
                    # Plotting Trajectories
                    if self.traj_fig:
                        self.pN_post.pRNN.to("cpu") # LANDMINE: pRNN's hidden state must be on cpu for plotSampleTrajectory
                        self.pN_post.plotSampleTrajectory(
                                env=self.env_novel,
                                agent=self.agent,
                            ) # Logs to wandb inside the function if predictiveNet.wandb_log is True

                        self.pN_post.pRNN.to(device) # LANDMINE: plotSampleTraj calls agent's getObs, which for some reason sets pRNN to cpu!!!

                    # Object Learning Analysis
                    testTrial = self.getTestTrial(n_trajs=self.trajs_test)
                    objectLearning = self.quantifyObjectLearning(
                        ctrl_locs=self.args.tasks.testing.ctrl_locs,
                        whichPhase=self.args.tasks.testing.whichPhase,
                        traj_count=traj_count,
                    )
                    if objectLearning is not None and testTrial is not None and self.wandb_log:
                        obj_learn_fig = figure_object_learning(env_name=self.env_orig, 
                                                            run_name=self.save_path, 
                                                            traj_num=traj_count, 
                                                            save_folder=self.save_path,
                                                            objectLearning=objectLearning,
                                                            testTrial=testTrial,
                                                            rl_storage=".",
                                                            config=self.args,
                                                            pN=self.pN_post,
                                                            show=False,
                                                            save=False)
                        
                        wandb.log({"Analysis/ObjectLearning": wandb.Image(obj_learn_fig)})
                        wandb.log({"Analysis/Novel Object In-view Times": objectLearning["inviewtimes"]})
                        wandb.log({"Analysis/Goal Modulation Vs. Step Count": objectLearning["goalmodulation"]})
                        wandb.log({"Analysis/Goal Minus Ctrl Vs. Step Count": objectLearning["goalmodulation"] - objectLearning["ctlmodulation_diffloc"]})
                        wandb.log({"Analysis/Avg Distance Travelled": objectLearning["avg_dist"]})

        save_pN_and_acmodel(self.pN_post, self.algo.acmodel, self.save_path, index * self.trajs_per_batch)
        print(f"Saved trained net to {self.save_path}")

        # Return the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx]


    def getTestTrial(
        self,
        n_trajs: int,
    ):
        # Type checking
        render_size = self.env_orig.render_size
        _, view_size, C = self.env_orig.obs_shape
        B = n_trajs
        T = self.seqdur
        A = self.env_orig.getActSize()

        obs: Float[torch.Tensor, "T+1 X"] # NOTE: To check with Alex, what is X here? obs.shape=torch.Size([1, 257, 147])
        act: Float[torch.Tensor, "T A"] # Ex: act.shape=torch.Size([1, 256, 8]) for SpeedHD
        state: State
 
        # Store original device before moving to CPU
        original_device = next(self.pN_post.pRNN.parameters()).device

        # Setting to eval mode
        self.pN_post.pRNN.eval()
        self.pN_control.pRNN.eval()
        
        if hasattr(self.agent, "acmodel"):
            assert isinstance(self.agent, ActorCriticAgent)
            self.agent.acmodel.eval()
            self.agent.argmax = True
        
        self.set_start_pos()
        opa = OnPolicyAnalysis(self.algo, timesteps=int(T * 30))
        if self.wandb_log:
            wandb.log({"Eval/MI_policy": opa.mi})
            wandb.log({"Eval/OPA_Advantages": wandb.Plotly(opa.plot_advantages())})
            wandb.log({"Eval/OPA_Policy_Heatmaps": wandb.Plotly(opa.plot_policy_heatmaps())})

            if self.args.tasks.analysis.occupancy :
                wandb.log({"Eval/OPA_Occupancy": wandb.Plotly(opa.plot_occupancy())})
        
        # Ensure that pN's are on cpu since using numpy
        self.pN_post.pRNN.to(torch.device("cpu"))
        self.pN_control.pRNN.to(torch.device("cpu"))
        
        # Collect observation sequence in the environment
        all_obs = torch.zeros((B, T + 1, view_size * view_size * C), dtype=torch.float32)
        all_obs_pred = torch.zeros((B, T, view_size * view_size * C), dtype=torch.float32)
        all_obs_pred_no_train = torch.zeros((B, T, view_size * view_size * C), dtype=torch.float32)

        all_agent_pos = np.zeros((B, T + 1, 2), dtype=np.float32)
        all_agent_dir = np.zeros((B, T + 1), dtype=np.int32)
        all_renders = np.zeros((B, T + 1, render_size, render_size, C), dtype=np.uint8)

        for n in range(B):

            self.set_start_pos()
            obs, act, state, render = self.pN_post.collectObservationSequence(
                env=self.env_orig, agent=self.agent, tsteps= T, includeRender=self.oL_fig # Critical self.env_orig
            )

            # TODO: Ask if pN should start from the same random state for both trained and control nets. This is what happens internally if randInit=True
            # Experimental decision
            h_state = self.pN_post.pRNN.generate_noise((0, 0.03), (1, 1, 500))
            h_state = self.pN_post.pRNN.rnn.cell.actfun(h_state)

            obs_pred, obs_next, _ = self.pN_post.predict(obs, act, state=h_state, randInit=False) # TODO: Ask if we should have state as an input??
            obs_pred_notrain, _, _ = self.pN_control.predict(obs, act, state=h_state, randInit=False)

            all_obs[n, :, :] = obs
            all_obs_pred[n, :, :] = obs_pred
            all_obs_pred_no_train[n, :, :] = obs_pred_notrain
            all_agent_pos[n, :, :] = state["agent_pos"]
            all_agent_dir[n, :] = state["agent_dir"]
            if self.oL_fig:
                all_renders[n, :, :, :, :] = np.stack(render, axis=0)

        objectTest = {
            "obs": all_obs,
            "obs_pred": all_obs_pred,
            "obs_pred_control": all_obs_pred_no_train,
            "agent_pos": all_agent_pos,
            "agent_dir": all_agent_dir,
            "render": all_renders if self.oL_fig else None,
        }

        self.pN_post.pRNN.to(original_device)
        self.pN_control.pRNN.to(original_device)
        self.pN_post.pRNN.train()

        if hasattr(self.agent, "acmodel"):
            self.agent.acmodel.train()  # type: ignore[attr-defined]
            self.agent.argmax = False # type: ignore

        self.testTrial = objectTest
        return objectTest


    def quantifyObjectLearning(self, ctrl_locs: list[list[int]], whichPhase: int, traj_count: int) -> dict[str, np.ndarray | np.float32 | int] | None:
        assert self.testTrial is not None, (
            "You need to run trainNovelObject and getTestTrial first."
        )

        start_pos = self.testTrial["agent_pos"][:, 0, :]
        end_pos = self.testTrial["agent_pos"][:, -1, :]
        avg_dist = torch.mean(get_dist_travelled(
            start_locs=torch.tensor(start_pos, dtype=torch.float32),
            end_locs=torch.tensor(end_pos, dtype=torch.float32)
        )) # Average distance travelled across all test trajectories

        pos = self.testTrial["agent_pos"][:, whichPhase:, :]
        HD = self.testTrial["agent_dir"][:, whichPhase:]
        obs_pred = self.testTrial["obs_pred"][:, whichPhase:, :]
        obs_pred_notrain = self.testTrial["obs_pred_control"][:, whichPhase:, :]

        B, T, X = obs_pred.shape
        obs_pred = obs_pred.reshape(B * T, X).unsqueeze(dim=0)
        obs_pred_notrain = obs_pred_notrain.reshape(B * T, X).unsqueeze(dim=0)
        pos = pos.reshape(B * (T + 1), 2)
        HD = HD.reshape(B * (T + 1))

        # Get predicted pixel values at the object/control location for trained and control pRNNs
        obs_np = self.pN_post.env_shell.pred2np(obs_pred)
        obs_notrain_np = self.pN_post.env_shell.pred2np(obs_pred_notrain)
        
        locobs, inviewtimes, viewcoords = get_obs_at_loc(obs_np, self.new_obj_pos, pos, HD)
        locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc(obs_notrain_np, self.new_obj_pos, pos, HD)

        if locobs is None:
            print("No views of the goal or control location were found during the test trial.")
            return None
        
        objectloc_deltaobs = locobs - locobs_notrain
        goalmodulation = np.mean(objectloc_deltaobs[:, 1])
        ctlmodulation_diffcolor = np.mean(
            np.concatenate((objectloc_deltaobs[:, 0], objectloc_deltaobs[:, 2]))
        )

        ctl_mods = []
        for control_location in ctrl_locs:
            conobs, _, _ = get_obs_at_loc(obs_np, control_location, pos, HD)
            conobs_notrain, _, _ = get_obs_at_loc(obs_notrain_np, control_location, pos, HD)

            if conobs is not None:
                controlloc_deltaobs = conobs - conobs_notrain
                ctlmodulation_diffloc = np.mean(controlloc_deltaobs[:, 1]) # TODO: Do we just want to control for change in green at cntrl locs?
                ctl_mods.append(ctlmodulation_diffloc)

        ctlmodulation_diffloc = np.mean(ctl_mods)

        objectLearning = {
            "inviewtimes": inviewtimes,
            "viewcoords": viewcoords,
            "objectloc_obs": locobs,
            "controlloc_obs": conobs,
            "objectloc_obs_controlNet": locobs_notrain,
            "controlloc_obs_controlNet": conobs_notrain,
            "objectloc_deltaobs": objectloc_deltaobs,
            "controlloc_deltaobs": controlloc_deltaobs,
            "goalmodulation": goalmodulation,
            "ctlmodulation_diffcolor": ctlmodulation_diffcolor,
            "ctlmodulation_diffloc": ctlmodulation_diffloc,
            "traj_count": traj_count,
            "avg_dist": avg_dist.item(),
        }
        self.objectLearning = objectLearning
        return objectLearning


# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


def get_view_coords(i, j, pos, HD, agent_view_size=7):
    ax, ay = pos
    dx, dy = DIR_TO_VEC[HD]
    rx, ry = -dy, dx

    # Compute the absolute coordinates of the top-left view corner
    sz = agent_view_size
    hs = agent_view_size // 2
    tx = ax + (dx * (sz - 1)) - (rx * hs)
    ty = ay + (dy * (sz - 1)) - (ry * hs)

    lx = i - tx
    ly = j - ty

    # Project the coordinates of the object relative to the top-left
    # corner onto the agent's own coordinate system
    vx = rx * lx + ry * ly
    vy = -(dx * lx + dy * ly)

    return int(vx), int(vy)


def get_obs_at_loc(obs, goal_loc, pos, HD):
    i, j = goal_loc

    locobs = []
    viewtimes = []
    viewcoords = []
    for tt in range(obs.shape[0]):
        # Get egocentric coordinates of the goal/control loc
        # Check that these coordinates are in the agent's 7x7 view
        vx, vy = get_view_coords(i, j, pos[tt, :], HD[tt])
        if (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7):
            locobs.append(obs[tt, vy, vx, :])
            viewtimes.append(tt)
            viewcoords.append([vx, vy])

    if locobs == []:
        # No views of the location were found
        return None, None, None

    locobs = np.stack(locobs, axis=0)
    viewtimes = np.stack(viewtimes, axis=0)
    viewcoords = np.stack(viewcoords, axis=0)
    return locobs, viewtimes, viewcoords

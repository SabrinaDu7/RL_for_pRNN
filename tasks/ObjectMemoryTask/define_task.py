import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import wandb
from jaxtyping import Float
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict

from RLutils import (
    get_pN,
    get_SR_acmodel,
    get_obss_preprocessor,
    get_algo,
    seed,
    PredictivePPOAlgo,
    get_occupancy_fig,
    get_agent,
)

from prnn.utils import save_pN
from prnn.utils.Shell import FaramaMinigridShell
from tasks.ObjectMemoryTask.figure import plot_k_trajectories
from utils import AgentType


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
        decoder="train",
    ):
        seed(args.exp.seed)

        self.env_orig = env_orig
        self.env_novel = env_novel

        self.new_obj_pos = self.env_novel.get_new_obj_pos()
        with open_dict(args):
            args.tasks.new_obj_pos = self.new_obj_pos

        # CRITICAL: These rooms have no goal. So object learning figures are MEANINGLESS.
        # TODO: Just show predictions instead of object learning.

        self.args = args
        self.save_path = save_path
        self.wandb_log = args.logging.wandb_log

        self.seqdur = args.predNet.seqdur # steps
        self.trajs_per_batch = args.rl.trajs_per_batch
        self.trajs_test = args.tasks.testing.trajs

        # Create save directory and save config
        os.makedirs(self.save_path, exist_ok=True)
        OmegaConf.save(args, f"{self.save_path}/config.yaml")

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

    def trainNovelObject(
        self,
        lr_trials: int,
        lrgroups: list, # [0, 1, 2]
        num_trajs: int, # num of trajectories to train on in novel env
        saving_interval: int, # num batches between logging
        analysis_interval: int, # num batches between logging
        resetOptimizer: bool,
        continueTraining: bool,
        device,
    ):

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

            traj_count = index * self.trajs_per_batch
            step_count = traj_count * self.seqdur

            # 125 batches = 125 * 4 gradient steps,
            # 8 trajs per batch = 2048 steps per batch

            if self.agent_type == AgentType.RANDOM:
                exp_logs = self.algo.randomAgent_collect_exp_and_update(self.agent)
            else:
                exps, exp_logs = self.algo.collect_experiences()
                logs2 = self.algo.update_parameters(exps=exps, update_params=True)
                locs = exp_logs["locs"] # Locations visited during last training batch (ie last 8 trajs)
                
                if self.wandb_log:
                    for key, val in logs2.items():
                        wandb.log({f"Train/{key}": val}, step=step_count)

            if index % saving_interval == 0:
                print(f"Completed {index * self.trajs_per_batch} trajectories = {index * self.trajs_per_batch * self.seqdur} steps")
                save_pN(self.pN_post, f"{self.save_path}/pN-{traj_count}.pt")
            
            if index % analysis_interval == 0:
                testing_tsteps = self.trajs_test * self.seqdur

                # Plotting Trajectories
                self.pN_post.pRNN.to("cpu") # LANDMINE: pRNN's hidden state must be on cpu for plotSampleTrajectory
                self.pN_post.plotSampleTrajectory(
                        env=self.env_novel,
                        agent=self.agent,
                    ) # Logs to wandb inside the function if predictiveNet.wandb_log is True
                self.pN_post.pRNN.to(device) # LANDMINE: plotSampleTraj calls agent's getObs, which for some reason sets pRNN to cpu!!!

                if self.args.tasks.analysis.objectLearning:
                    # Object Learning Analysis
                    testTrial = self.getTestTrial(n_trajs=self.trajs_test)
                    objectLearning = self.quantifyObjectLearning(
                        control_location=self.args.tasks.testing.control_location,
                        whichPhase=self.args.tasks.testing.whichPhase,
                        traj_count=traj_count,
                    )
                    if objectLearning is not None and testTrial is not None:
                        torch.save(objectLearning, f"{self.save_path}/objectLearning_{traj_count}.pt")
                        torch.save(testTrial, f"{self.save_path}/testTrial_{traj_count}.pt")
                        print(f"Saved object learning results (traj {traj_count}): {self.save_path}/objectLearning_{traj_count}.pt.")
                
                if self.agent_type == AgentType.AC and self.args.tasks.analysis.occupancy:
                    # Occupancy Analysis
                    occ_fig = get_occupancy_fig(self.algo, timesteps=testing_tsteps)
                    if self.wandb_log:
                        wandb.log({"Analysis/Occupancy": wandb.Plotly(occ_fig)}, step=step_count)


        save_pN(self.pN_post, f"{self.save_path}/pN-{num_trajs}.pt")
        print(f"Saved trained net to {self.save_path}/pN-{num_trajs}.pt")

        # Return the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx]


    def getTestTrial(
        self,
        n_trajs: int,
    ):
        # Store original device before moving to CPU
        original_device = next(self.pN_post.pRNN.parameters()).device

        # Ensure that pN's are on cpu since using numpy
        self.pN_post.pRNN.to(torch.device("cpu"))
        self.pN_control.pRNN.to(torch.device("cpu"))

        self.pN_post.pRNN.eval()
        self.pN_control.pRNN.eval()

        if hasattr(self.agent, "acmodel"):
            self.agent.acmodel.eval() # type: ignore
            self.agent.argmax = True # type: ignore

        # Initialize storage lists for concatenation
        obs_list = []
        act_list = []
        state_pos_list = []
        state_dir_list = []
        state_srs_list = []
        render_list = []
        obs_pred_list = []
        obs_pred_notrain_list = []

        # Get whichPhase from config
        whichPhase = self.args.tasks.testing.whichPhase

        # Collect n_trajs trajectories
        for traj_idx in range(n_trajs):
            # Collect one trajectory of seqdur timesteps (with random initialization)
            obs, act, state, render = self.pN_post.collectObservationSequence(
                env=self.env_orig,
                agent=self.agent,
                tsteps=self.seqdur,
                includeRender=True  # Critical self.env_orig
            )

            # Apply whichPhase slicing to this trajectory
            obs_sliced = obs[whichPhase:]  # List slicing
            act_sliced = act[whichPhase:]  # Array slicing
            state_pos_sliced = state["agent_pos"][whichPhase:, :]
            state_dir_sliced = state["agent_dir"][whichPhase:]
            state_srs_sliced = state["SRs"][whichPhase:, :]
            render_sliced = render[whichPhase:]

            # Get predictions for this trajectory using sliced data
            obs_pred, obs_next, _ = self.pN_post.predict(obs_sliced, act_sliced)
            obs_pred_notrain, _, _ = self.pN_control.predict(obs_sliced, act_sliced)

            # Store sliced data for concatenation
            obs_list.append(obs_sliced)
            act_list.append(act_sliced)
            state_pos_list.append(state_pos_sliced)
            state_dir_list.append(state_dir_sliced)
            state_srs_list.append(state_srs_sliced)
            render_list.append(render_sliced)
            obs_pred_list.append(obs_pred)
            obs_pred_notrain_list.append(obs_pred_notrain)

        # Concatenate all trajectories
        # Observations (lists) - concatenate by extending
        obs_concat = []
        for obs_traj in obs_list:
            obs_concat.extend(obs_traj)

        # Actions (numpy arrays)
        act_concat = np.concatenate(act_list, axis=0)

        # State arrays
        state_concat = {
            "agent_pos": np.concatenate(state_pos_list, axis=0),
            "agent_dir": np.concatenate(state_dir_list, axis=0),
            "SRs": np.concatenate(state_srs_list, axis=0),
        }

        # Renders (lists)
        render_concat = []
        for render_traj in render_list:
            render_concat.extend(render_traj)

        # Predictions (torch tensors along time dimension)
        obs_pred_concat = torch.cat(obs_pred_list, dim=1)
        obs_pred_notrain_concat = torch.cat(obs_pred_notrain_list, dim=1)

        # Create final objectTest dictionary
        objectTest = {
            "obs": obs_concat,
            "obs_pred": obs_pred_concat,
            "obs_pred_control": obs_pred_notrain_concat,
            "state": state_concat,
            "render": render_concat,
        }

        self.pN_post.pRNN.to(original_device)
        self.pN_control.pRNN.to(original_device)
        self.pN_post.pRNN.train()
        if hasattr(self.agent, "acmodel"):
            self.agent.acmodel.train() # type: ignore
            self.agent.argmax = False # type: ignore

        self.testTrial = objectTest
        return objectTest


    def quantifyObjectLearning(self, control_location: list[list[int]], whichPhase: int, traj_count: int) -> dict[str, np.ndarray | np.float32 | int] | None:
        assert self.testTrial is not None, (
            "You need to run trainNovelObject and getTestTrial first."
        )

        # Note: whichPhase slicing is now applied during getTestTrial, so use data as-is
        pos = self.testTrial["state"]["agent_pos"]
        HD = self.testTrial["state"]["agent_dir"]
        obs_pred = self.testTrial["obs_pred"]
        obs_pred_notrain = self.testTrial["obs_pred_control"]

        # Get predicted pixel values at the object/control location for trained and control pRNNs
        obs_np = self.pN_post.env_shell.pred2np(obs_pred, whichPhase=0)
        obs_notrain_np = self.pN_post.env_shell.pred2np(obs_pred_notrain, whichPhase=0)

        locobs, inviewtimes, viewcoords = get_obs_at_loc(obs_np, self.new_obj_pos, pos, HD)
        locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc(obs_notrain_np, self.new_obj_pos, pos, HD)

        # Collect observations from all control locations
        conobs_list = []
        conobs_notrain_list = []

        for ctrl_loc in control_location:
            conobs, _, _ = get_obs_at_loc(obs_np, ctrl_loc, pos, HD)
            conobs_notrain, _, _ = get_obs_at_loc(obs_notrain_np, ctrl_loc, pos, HD)

            if conobs is not None and conobs_notrain is not None:
                conobs_list.append(conobs)
                conobs_notrain_list.append(conobs_notrain)

        # If no valid control locations were found, return None
        if len(conobs_list) == 0:
            print("No views of any control location were found during the test trial.")
            return None

        if locobs is None:
            print("No views of the goal location were found during the test trial.")
            return None

        # Concatenate all observations from all valid control locations
        # This treats each observation equally regardless of which control location it came from
        conobs = np.concatenate(conobs_list, axis=0)
        conobs_notrain = np.concatenate(conobs_notrain_list, axis=0)

        objectloc_deltaobs = locobs - locobs_notrain
        controlloc_deltaobs = conobs - conobs_notrain

        goalmodulation = np.mean(objectloc_deltaobs[:, 1]) # GREEN channel
        ctlmodulation_diffcolor = np.mean(
            np.concatenate((objectloc_deltaobs[:, 0], objectloc_deltaobs[:, 2])) # RED & BLUE
        )
        ctlmodulation_diffloc = np.mean(controlloc_deltaobs[:, 1]) # GREEN at control location

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

    return vx, vy


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
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict

from RLutils import (
    get_pN, 
    make_env, 
    get_SR_acmodel, 
    get_obss_preprocessor,
    get_algo,
    seed,
    PredictivePPOAlgo,
    ActorCriticAgent,
    get_occupancy_fig,
)

from prnn.utils.general import saveFig
from prnn.utils.Shell import FaramaMinigridShell
from prnn.utils import (
    MinigridEnvNames,
    RandomActionAgent,
    ActionEncodingsEnum,
    save_pN,
)

from utils import AgentInputType, AgentType

RAND_ACT_PROBA = np.array([0.15, 0.15, 0.6, 0.1])

def _get_goal_loc(env: FaramaMinigridShell) -> list[int]:
    env_farama_shell = env
    env_rgb_wrapper = env_farama_shell.env
    env_order_enforcing = env_rgb_wrapper.env
    env_passive_checker = env_order_enforcing.env
    env_LEnv_goal = env_passive_checker.env

    goal_loc = env_LEnv_goal.goal_pos
    return goal_loc

def _get_agent(env: FaramaMinigridShell, agent_Type: AgentType, ac_model, pN_post, device) -> ActorCriticAgent | RandomActionAgent:
    if agent_Type == AgentType.RANDOM:
        agent = RandomActionAgent(env.action_space, RAND_ACT_PROBA)
    elif agent_Type == AgentType.AC:
        agent = ActorCriticAgent(env.action_space, ac_model, pN_post, device)
    else:
        raise ValueError(f"Unsupported agent type: {agent_Type}")
    return agent


class ObjectMemoryTask:
    def __init__(
        self,
        args: DictConfig,
        agent_type: AgentType,
        env_novel_name: MinigridEnvNames,
        env_orig_name: MinigridEnvNames,
        device: torch.device,
        save_path: str,
        acmodel_status_ckpt: str,
        prnn_ckpt: str,
        decoder="train",
    ):
        seed(args.exp.seed)

        self.env_novel_name = env_novel_name.value
        self.env_novel = make_env(env_key=env_novel_name.value, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
        self.env_orig = make_env(env_key=env_orig_name.value, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
        self.goal_loc = _get_goal_loc(self.env_novel)

        with open_dict(args):
            args.tasks.goal_loc = self.goal_loc

        self.args = args
        self.save_path = save_path
        self.wandb_log = args.logging.wandb_log

        self.seqdur = args.predNet.seqdur # steps
        self.trajs_per_batch = args.rl.trajs_per_batch

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
        self.agent = _get_agent(
            self.env_novel, 
            self.agent_type, 
            self.ac_model, 
            self.pN_post, 
            device
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
                self.algo.randomAgent_collect_exp_and_update(self.agent)
            else:
                exps, logs1 = self.algo.collect_experiences()
                logs2 = self.algo.update_parameters(exps=exps, update_params=True)
                
                if self.wandb_log:
                    for key, val in logs2.items():
                        wandb.log({f"Train/{key}": val}, step=step_count)

            if index % saving_interval == 0:
                print(f"Completed {index * self.trajs_per_batch} trajectories = {index * self.trajs_per_batch * self.seqdur} steps")
                save_pN(self.pN_post, f"{self.save_path}/pN-{traj_count}.pt")
            
            if index % analysis_interval == 0:
                # Object Learning Analysis
                self.getTestTrial(timesteps=self.args.tasks.testing.timesteps)
                objectLearning = self.quantifyObjectLearning(
                    control_location=self.args.tasks.testing.control_location,
                    whichPhase=self.args.tasks.testing.whichPhase,
                    traj_count=traj_count,
                )
                if objectLearning is not None:
                    torch.save(objectLearning, f"{self.save_path}/objectLearning_{traj_count}.pt")
                    print(f"Saved object learning results (traj {traj_count}): {self.save_path}/objectLearning_{traj_count}.pt.")
                
                # Occupancy Analysis
                if self.agent_type == AgentType.AC:
                    occ_fig = get_occupancy_fig(self.algo, timesteps=self.args.tasks.testing.timesteps)
                    occ_fig.write_image(f"{self.save_path}/Occupancy_{traj_count}.png")

        save_pN(self.pN_post, f"{self.save_path}/pN-{num_trajs}.pt")
        print(f"Saved trained net to {self.save_path}/pN-{num_trajs}.pt")

        # Return the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            self.pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx]


    def getTestTrial(
        self,
        timesteps: int,
    ):
        # Store original device before moving to CPU
        original_device = next(self.pN_post.pRNN.parameters()).device

        # Ensure that pN's are on cpu since using numpy
        self.pN_post.pRNN.to(torch.device("cpu"))
        self.pN_control.pRNN.to(torch.device("cpu"))
        
        self.pN_post.pRNN.eval()
        self.pN_control.pRNN.eval()

        # Collect observation sequence in the environment
        obs, act, state, render = self.pN_post.collectObservationSequence(
            self.env_orig, self.agent, timesteps, includeRender=True # Critical self.env_orig
        )

        obs_pred, obs_next, _ = self.pN_post.predict(obs, act)
        obs_pred_notrain, _, _ = self.pN_control.predict(obs, act)

        objectTest = {
            "obs": obs,
            "obs_pred": obs_pred,
            "obs_pred_control": obs_pred_notrain,
            "state": state,
            "render": render,
        }

        self.pN_post.pRNN.to(original_device)
        self.pN_control.pRNN.to(original_device)
        self.pN_post.pRNN.train()

        self.testTrial = objectTest
        return objectTest


    def quantifyObjectLearning(self, control_location: list[int], whichPhase: int, traj_count: int) -> dict[str, np.ndarray | np.float32 | int] | None:
        assert self.testTrial is not None, (
            "You need to run trainNovelObject and getTestTrial first."
        )

        pos = self.testTrial["state"]["agent_pos"][whichPhase:, :]
        HD = self.testTrial["state"]["agent_dir"][whichPhase:]
        obs_pred = self.testTrial["obs_pred"]
        obs_pred_notrain = self.testTrial["obs_pred_control"]

        # Get predicted pixel values at the object/control location for trained and control pRNNs
        obs_np = self.pN_post.env_shell.pred2np(obs_pred, whichPhase=whichPhase)
        obs_notrain_np = self.pN_post.env_shell.pred2np(obs_pred_notrain, whichPhase=whichPhase)
        
        locobs, inviewtimes, viewcoords = get_obs_at_loc(obs_np, self.goal_loc, pos, HD)
        conobs, _, _ = get_obs_at_loc(obs_np, control_location, pos, HD)

        locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc(obs_notrain_np, self.goal_loc, pos, HD)
        conobs_notrain, _, _ = get_obs_at_loc(obs_notrain_np, control_location, pos, HD)

        if locobs is None or conobs is None:
            print("No views of the goal or control location were found during the test trial.")
            return None

        objectloc_deltaobs = locobs - locobs_notrain
        controlloc_deltaobs = conobs - conobs_notrain

        goalmodulation = np.mean(objectloc_deltaobs[:, 1])
        ctlmodulation_diffcolor = np.mean(
            np.concatenate((objectloc_deltaobs[:, 0], objectloc_deltaobs[:, 2]))
        )
        ctlmodulation_diffloc = np.mean(controlloc_deltaobs[:, 1])

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
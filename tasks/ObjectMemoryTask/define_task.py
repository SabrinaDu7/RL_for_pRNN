import numpy as np
import matplotlib.pyplot as plt
import torch

from RLutils import get_pN, make_env, get_SR_acmodel, get_obss_preprocessor

from prnn.utils.general import saveFig
from prnn.utils import (
    MinigridEnvNames,
    RandomActionAgent,
    ActionEncodingsEnum,
    save_pN,
)

from utils import get_ckpt_env_vars, AgentInputType

RAND_ACT_PROBA = np.array([0.15, 0.15, 0.6, 0.1])
PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()


def get_env_name(env):
    """Helper method to get the actual environment class name through the wrapper hierarchy."""
    temp_env = env
    names = []
    while temp_env is not None:
        class_name = type(temp_env).__name__
        names.append(class_name)
        if hasattr(temp_env, "env"):
            temp_env = temp_env.env
        else:
            break
    return " -> ".join(names)


class ObjectMemoryTask:
    def __init__(
        self,
        args,
        env_novel_name: MinigridEnvNames,
        env_orig_name: MinigridEnvNames,
        device: torch.device,
        decoder="train",
    ):
        self.args = args

        self.env_novel_name = env_novel_name.value
        self.env_novel = make_env(env_key=env_novel_name.value, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)
        self.env_orig = make_env(env_key=env_orig_name.value, input_type=AgentInputType.H_PO.value, act_enc=ActionEncodingsEnum.SpeedHD.value)

        self.pN_control = get_pN(args=args, env=self.env_orig, device=device)
        self.wandb_log = args.logging.wandb_log
        self.pN_control.wandb_log = self.wandb_log
        
        self.agent_random = RandomActionAgent(self.env_novel.action_space, RAND_ACT_PROBA)

        # For clarity
        env_farama_shell = self.env_novel
        env_rgb_wrapper = env_farama_shell.env
        env_order_enforcing = env_rgb_wrapper.env
        env_passive_checker = env_order_enforcing.env
        env_LEnv_16_goal = env_passive_checker.env

        self.goal_loc = env_LEnv_16_goal.goal_pos

        # Train the decoder
        # if decoder is 'train':
        #    self.decoder = self.trainDecoder()
        # else:
        #    self.decoder = decoder

        # Train the network in the object room #Note, copy ExperienceReplayAnalysis
        # approach for multiple learning rates (consider rec and FF learning rates)

        self.pN_post = None  # Updated after trainNovelObject() called
        self.testTrial = None  # Updated after getTestTrial() called
        self.objectLearning = None  # Updated after quantifyObjectLearning() called

    def trainDecoder(self):
        env = self.pN_control.EnvLibrary[0]
        agent = RandomActionAgent(env.action_space, RAND_ACT_PROBA)
        _, _, decoder, sRSA = self.pN_control.calculateSpatialRepresentation(
            env, agent, numBatches=10000, trainDecoder=True
        )
        return decoder

    def trainNovelObject(
        self,
        epochs: int,
        num_trials: int,
        sequence_duration: int,
        lr_trials: int,
        lrgroups: list, # [0, 1, 2]
        resetOptimizer: bool,
        continueTraining: bool,
        device,
        full_filename: str,
    ):
        if continueTraining:
            pN_post = self.pN_control
        else:
            pN_post = self.pN_control.copy()

        agent = RandomActionAgent(self.env_novel.action_space, RAND_ACT_PROBA)

        # print(lr_trials)
        # Update the learning rate
        oldlr = [0.0 for i in lrgroups]
        if isinstance(lr_trials, int):
            lr_trials = [lr_trials for i in lrgroups] # [2, 2, 2]
        for lidx, lgroup in enumerate(lrgroups):
            # (0, 0), (1, 1), (2, 2)
            oldlr[lidx] = pN_post.optimizer.param_groups[lgroup]["lr"]
            pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx] * lr_trials[lidx]

        for _ in range(epochs):
            pN_post.trainingEpoch(
                self.env_novel, # CRITICAL
                agent,
                num_trials=num_trials,
                sequence_duration=sequence_duration,
                learningRate=None,
                forceDevice=device,
            )

        save_pN(pN_post, full_filename)
        print(f"Saved trained net to {full_filename}")

        # Return the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            pN_post.optimizer.param_groups[lgroup]["lr"] = oldlr[lidx]

        self.pN_post = pN_post

        return pN_post


    def getTestTrial(
        self,
        timesteps: int,
    ):
        assert self.pN_post is not None, (
            "You need to train the network in the novel object room first."
        )

        # Ensure that pN's are on cpu since using numpy
        self.pN_post.pRNN.to(torch.device("cpu"))
        self.pN_control.pRNN.to(torch.device("cpu"))
        
        self.pN_post.pRNN.eval()
        self.pN_control.pRNN.eval()

        # NOTE: self.pN acts as a control
        agent = RandomActionAgent(self.env_orig.action_space, RAND_ACT_PROBA)

        # Collect observation sequence in the environment
        obs, act, state, render = self.pN_post.collectObservationSequence(
            self.env_orig, agent, timesteps, includeRender=True
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

        self.testTrial = objectTest
        return objectTest

    def quantifyObjectLearning(self, control_location: list[int], whichPhase: int):
        assert self.pN_post is not None, (
            "You need to train the network in the novel object room first."
        )
        assert self.testTrial is not None, (
            "You need to run trainNovelObject and getTestTrial first."
        )

        pos = self.testTrial["state"]["agent_pos"][whichPhase:, :]
        HD = self.testTrial["state"]["agent_dir"][whichPhase:]
        obs_pred = self.testTrial["obs_pred"]
        obs_pred_notrain = self.testTrial["obs_pred_control"]
        goal_loc = self.goal_loc

        # Get the predicted pixel values at the object/control location
        obs_np = self.pN_post.env_shell.pred2np(obs_pred, whichPhase=whichPhase)
        locobs, inviewtimes, viewcoords = get_obs_at_loc(obs_np, goal_loc, pos, HD)
        conobs, _, _ = get_obs_at_loc(obs_np, control_location, pos, HD)

        # Get the predicted pixel values in the control networks
        obs_np = self.pN_post.env_shell.pred2np(obs_pred_notrain, whichPhase=whichPhase)
        locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc(
            obs_np, goal_loc, pos, HD
        )
        conobs_notrain, _, _ = get_obs_at_loc(obs_np, control_location, pos, HD)

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
        }
        self.objectLearning = objectLearning
        return objectLearning

    def ObjectLearningFigure(self, netname=None, savefolder=None, whichview=1):
        plt.subplot(4, 3, 1)
        self.objPixelChangePanel(self.objectLearning)

        self.exampleObsSequencePanel(whichview=whichview)

        plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(), "ObjectLearning_" + netname, savefolder, filetype="png")

    def objPixelChangePanel(self, objectLearning):
        deltaobs_goal = objectLearning["objectloc_deltaobs"]
        deltaobs_ctl = objectLearning["controlloc_deltaobs"]

        plt.boxplot(
            deltaobs_goal,
            showfliers=False,
            positions=[1.8, 1, 2.2],
            label=["R", "G", "B"],
        )
        plt.boxplot(
            deltaobs_ctl,
            showfliers=False,
            positions=[4.3, 3.5, 4.7],
            label=["R", "G", "B"],
        )
        plt.plot(plt.xlim(), [0, 0], "k--")
        plt.plot([3, 3], plt.ylim(), "k:")
        plt.ylabel("Change in Predicted Observation")

    def exampleObsSequencePanel(self, firstrow=3, whichview=1):
        assert self.objectLearning is not None, (
            "You need to run quantifyObjectLearning first."
        )
        assert self.testTrial is not None, "You need to run getTestTrial first."

        render = self.testTrial["render"]
        obs = self.testTrial["obs"]
        obs_pred = self.testTrial["obs_pred"]
        obs_pred_notrain = self.testTrial["obs_pred_control"]
        pN = self.pN_control

        inviewtimes = self.objectLearning["inviewtimes"]
        extimes = range(inviewtimes[whichview] - 2, inviewtimes[whichview] + 5)

        pN.plotSequence(render, extimes, firstrow, label="State")

        pN.plotSequence(pN.env_shell.pred2np(obs), extimes, firstrow + 1, label="Obs")

        pN.plotSequence(
            pN.env_shell.pred2np(obs_pred_notrain),
            extimes,
            firstrow + 2,
            label="Pred_CTL",
        )

        pN.plotSequence(
            pN.env_shell.pred2np(obs_pred), extimes, firstrow + 3, label="Pred"
        )


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
        vx, vy = get_view_coords(i, j, pos[tt, :], HD[tt])
        if (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7):
            locobs.append(obs[tt, vy, vx, :])
            viewtimes.append(tt)
            viewcoords.append([vx, vy])

    locobs = np.stack(locobs, axis=0)
    viewtimes = np.stack(viewtimes, axis=0)
    viewcoords = np.stack(viewcoords, axis=0)
    return locobs, viewtimes, viewcoords
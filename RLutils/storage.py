import os
import torch
import numpy as np
from typing import Callable, Type

from prnn.utils.Shell import FaramaMinigridShell
from prnn.utils import (
    PredictiveNet,
    pRNNtypes,
    RandomActionAgent,
    load_pN,
)

import RLutils
from RLutils import DEVICE, ACModelSR, PredictivePPOAlgo, ActorCriticAgent
from utils import (
    get_ckpt_env_vars, 
    load_statedict_from_acmodel_status, 
    StatusCkptKeys,
    AgentType,
)

PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()
RAND_ACT_PROBA = np.array([0.15, 0.15, 0.6, 0.1])

def create_folders_if_necessary(path):
    if path == "":
        return
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    elif "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "RLstorage")
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_video_dir(model_name):
    return os.path.join(os.environ["HOME"], "pRNN-RL/RLvideos", model_name)


def get_tmp_dir():
    if "TMPDIR" in os.environ:
        return os.environ["TMPDIR"]
    return "tmp"


def get_tmp_model_dir(model_name):
    return os.path.join(get_tmp_dir(), model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=DEVICE, weights_only=False)


def get_pN(args, env: FaramaMinigridShell, device: torch.device | str, pRNN_ckpt: str = PRNN_CKPT) -> PredictiveNet:
    predictiveNet = PredictiveNet(
        env,
        hidden_size=args.predNet.hiddensize,
        pRNNtype=args.predNet.pRNNtype,
        learningRate=args.predNet.lr,
        bptttrunc=args.predNet.bptttrunc,
        weight_decay=args.predNet.weight_decay,
        neuralTimescale=args.predNet.ntimescale,
        dropp=args.predNet.dropout,
        trainNoiseMeanStd=(args.predNet.noisemean, args.predNet.noisestd),
        f=args.predNet.sparsity,
        wandb_log=args.logging.wandb_log,
    )
    load_pN(model_ckpt_filepath=pRNN_ckpt, device=device, pRNNtype=args.predNet.pRNNtype, predictive_net=predictiveNet)
    return predictiveNet


def get_SR_acmodel(args, 
                   env_act_space, 
                   obs_space: dict,
                   device: torch.device,
                   acmodel_status_ckpt: str = "") -> ACModelSR:

    """
    Loads ACModel and Optimizer State Dictionaries from checkpoint.
    """

    acmodel = ACModelSR(
        obs_space=obs_space,
        action_space=env_act_space,
        SR_size=args.predNet.hiddensize,
        with_CV=args.exp.with_obs,
        rgb=args.exp.rgb,
        with_HD=args.exp.with_HD,
    )

    if acmodel_status_ckpt == "":
        return acmodel.to(device)
    else:
        status = torch.load(
            acmodel_status_ckpt,
            map_location=device,
            weights_only=False
        )

    if StatusCkptKeys.MODEL_STATE.value in status:
        acmodel.load_state_dict(status[StatusCkptKeys.MODEL_STATE.value])
    else:
        print("Warning: model_state not found in acmodel status ckpt. Random agent was used. Returning fresh ACModel")
    
    return acmodel.to(device)


def get_algo(args, 
             env: FaramaMinigridShell, 
             predictiveNet: PredictiveNet, 
             acmodel: ACModelSR,
             preprocess_obss: Callable,
             AlgoClass: Type[PredictivePPOAlgo],
             acmodel_status_ckpt: str,
             device: torch.device) -> PredictivePPOAlgo:
    
    status = torch.load(
        acmodel_status_ckpt,
        map_location=device,
        weights_only=False
    )
    
    pastSR = not ("prevAct" in str(predictiveNet.pRNN))
    algo = AlgoClass(
        env=env,
        acmodel=acmodel,
        predictiveNet=predictiveNet,
        device=device,
        num_frames=args.rl.frames,
        discount=args.rl.discount,
        lr=args.rl.lr,
        gae_lambda=args.rl.gae_lambda,
        entropy_coef=args.rl.entropy_coef,
        value_loss_coef=args.rl.value_loss_coef,
        max_grad_norm=args.rl.max_grad_norm,
        recurrence=args.exp.recurrence,
        adam_eps=args.rl.optim_eps,
        clip_eps=args.rl.ppo_clip_eps,
        epochs=args.rl.ppo_epochs,
        batch_size=args.rl.ppo_batch_size,
        preprocess_obss=preprocess_obss,
        place_cells=args.exp.PC,
        cann=args.exp.CANN,
        train_pN=args.predNet.train,
        noise_mu=args.predNet.noisemean,
        noise_std=args.predNet.noisestd,
        prnn_seqdur=args.predNet.seqdur,
        intrinsic=args.exp.intrinsic,
        k_int=args.rl.k_int,
        pastSR=pastSR,
        curious_agent=args.exp.curious_agent,
        k_curious=args.rl.k_curious,
    )

    if StatusCkptKeys.OPTIMIZER_STATE.value in status:
        algo.optimizer.load_state_dict(status[StatusCkptKeys.OPTIMIZER_STATE.value])

    return algo


def get_goal_loc(env: FaramaMinigridShell) -> list[int]:
    env_farama_shell = env
    env_rgb_wrapper = env_farama_shell.env
    env_order_enforcing = env_rgb_wrapper.env
    env_passive_checker = env_order_enforcing.env
    env_LEnv_goal = env_passive_checker.env

    goal_loc = env_LEnv_goal.goal_pos
    return goal_loc


def get_agent(
        env: FaramaMinigridShell, 
        agent_Type: AgentType, 
        ac_model: ACModelSR, 
        pN_post: PredictiveNet, 
        device: torch.device,
        rand_act_prob: np.ndarray = RAND_ACT_PROBA
    ) -> ActorCriticAgent | RandomActionAgent:

    if agent_Type == AgentType.RANDOM:
        agent = RandomActionAgent(env.action_space, rand_act_prob)
    elif agent_Type == AgentType.AC:
        agent = ActorCriticAgent(env.action_space, ac_model, pN_post, device)

    return agent


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    RLutils.create_folders_if_necessary(path)
    torch.save(status, path)


def save_analysis_of_agent_behav(onpolicyAnalysis, model_dir, update_step):
    figs = {
        "advantages.png": onpolicyAnalysis.plot_advantages(),
        "policy_heatmaps.png": onpolicyAnalysis.plot_policy_heatmaps(),
        "occupancy.png": onpolicyAnalysis.plot_occupancy(),
        "values.png": onpolicyAnalysis.plot_values(),
    }

    outdir = os.path.join(model_dir, "onpolicy_analysis", str(update_step))
    os.makedirs(outdir, exist_ok=True)

    for fname, fig in figs.items():
        savename = os.path.join(outdir, fname)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.write_image(savename)

# def get_vocab(model_dir):
#     return get_status(model_dir)["vocab"]

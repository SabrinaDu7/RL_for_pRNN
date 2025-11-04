import os
import torch
from typing import Callable, Type

from prnn.utils.Shell import FaramaMinigridShell
from prnn.utils import (
    PredictiveNet,
    pRNNtypes,
    load_pN,
)

import RLutils
from RLutils import DEVICE, ACModelSR, PredictivePPOAlgo
from utils import (
    get_ckpt_env_vars, 
    load_statedict_from_acmodel_status, 
    StatusCkptKeys,
)

PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()


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
                   acmodel_status_ckpt: str,
                   device: torch.device) -> ACModelSR:

    """
    Loads ACModel and Optimizer State Dictionaries from checkpoint.
    """
    status = torch.load(
        acmodel_status_ckpt,
        map_location=device,
        weights_only=False
    )

    assert StatusCkptKeys.MODEL_STATE.value in status
    assert StatusCkptKeys.NUM_FRAMES.value in status

    acmodel = ACModelSR(
        obs_space=obs_space,
        action_space=env_act_space,
        SR_size=args.predNet.hiddensize,
        with_CV=args.exp.with_obs,
        rgb=args.exp.rgb,
        with_HD=args.exp.with_HD,
    )
    
    acmodel.load_state_dict(status[StatusCkptKeys.MODEL_STATE.value])
    acmodel.to(device)

    return acmodel


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
        env,
        acmodel,
        predictiveNet,
        device,
        args.rl.frames,
        args.rl.discount,
        args.rl.lr,
        args.rl.gae_lambda,
        args.rl.entropy_coef,
        args.rl.value_loss_coef,
        args.rl.max_grad_norm,
        args.exp.recurrence,
        args.rl.optim_eps,
        args.rl.ppo_clip_eps,
        args.rl.ppo_epochs,
        args.rl.ppo_batch_size,
        preprocess_obss,
        args.exp.PC,
        args.exp.CANN,
        args.predNet.train,
        args.predNet.noisemean,
        args.predNet.noisestd,
        args.predNet.seqdur,
        args.exp.intrinsic,
        args.rl.k_int,
        pastSR,
        args.exp.curious_agent,
        args.rl.k_curious,
    )

    if StatusCkptKeys.OPTIMIZER_STATE.value in status:
        algo.optimizer.load_state_dict(status[StatusCkptKeys.OPTIMIZER_STATE.value])

    return algo


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

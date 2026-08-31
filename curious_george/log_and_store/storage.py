import os
from pathlib import Path
from typing import Callable, Type

import numpy as np
import torch
from prnn.utils import (
    PredictiveNet,
    RandomActionAgent,
    load_pN,
    save_pN,
)
from prnn.utils.Shell import FaramaMinigridShell

from curious_george.envs.access import get_new_obj_pos as access_get_goal_loc
from curious_george.models.policy import ACModel, ACModelSR
from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.agent import ActorCriticAgent
from curious_george.utils.checkpoints import (
    StatusCkptKeys,
)
from curious_george.utils.common import get_device
from curious_george.utils.dev_env import PRNN_CKPT_FILENAME, get_env_var
from curious_george.utils.enums import AgentType

# The value's ONE home is `configs.RAND_ACT_PROBA`; this is the same constant
# as the ndarray `get_agent` consumes, not a second spelling.
from curious_george.configs import RAND_ACT_PROBA as _RAND_ACT_PROBA_CFG

RAND_ACT_PROBA = np.asarray(_RAND_ACT_PROBA_CFG)


def create_folders_if_necessary(path: str):
    if path == "":
        return
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    """Run-output root. Single source of truth: RL_STORAGE in .env
    (get_env_var loads .env lazily, so this works regardless of import order).
    Fallbacks: $SCRATCH/RLstorage, then ./storage."""
    try:
        return get_env_var("RL_STORAGE")
    except EnvironmentError:
        pass
    if "SCRATCH" in os.environ:
        return os.path.join(os.environ["SCRATCH"], "RLstorage")
    return "storage"


def get_model_dir(model_name: str | Path):
    return os.path.join(get_storage_dir(), model_name)


def get_video_dir(model_name: str):
    """Videos beside the run that produced them, under RL_STORAGE.

    Was $HOME/pRNN-RL/RLvideos/<run>, which is not the storage root and so was
    invisible to every rsync in slurm/ - a cluster run's videos never left the
    node. Inert until logging.video_log_freq > 0, wrong either way.
    """
    return os.path.join(get_model_dir(model_name), "videos")


def get_tmp_dir():
    if "TMPDIR" in os.environ:
        return os.environ["TMPDIR"]
    return "tmp"


def get_tmp_model_dir(model_name: str | Path):
    return os.path.join(get_tmp_dir(), model_name)


#: What the policy checkpoint is called. Renamed from `status.pt` on 2026-08-28:
#: the file holds the actor-critic's weights and its optimizer, and "status"
#: named none of that. The KEYS inside it are unchanged - `StatusCkptKeys` is
#: the on-disk schema of the dict, and renaming its VALUES would make every
#: existing checkpoint unreadable.
POLICY_CKPT_FILENAME = "policy.pt"
LEGACY_POLICY_CKPT_FILENAME = "status.pt"


def policy_path(model_dir: str | Path) -> str:
    """Where a policy checkpoint is WRITTEN. Always the current name."""
    return os.path.join(model_dir, POLICY_CKPT_FILENAME)


def find_policy(model_dir: str | Path) -> str:
    """Where to READ a policy checkpoint from, preferring the current name.

    Separate from `policy_path` on purpose: a single function that resolves to
    an old name would eventually write one. Runs finished before the rename
    have `status.pt` and are still readable; new ones never produce it.
    """
    current = policy_path(model_dir)
    if os.path.isfile(current):
        return current
    legacy = os.path.join(model_dir, LEGACY_POLICY_CKPT_FILENAME)
    if os.path.isfile(legacy):
        return legacy
    raise FileNotFoundError(
        f"{model_dir} holds neither {POLICY_CKPT_FILENAME} nor "
        f"{LEGACY_POLICY_CKPT_FILENAME}"
    )


def load_policy(model_dir: str | Path):
    return torch.load(find_policy(model_dir), map_location=get_device(), weights_only=False)


def get_pN(
    args, env: FaramaMinigridShell, device: torch.device | str, pRNN_ckpt: str
) -> PredictiveNet:
    predictiveNet = PredictiveNet(
        env,
        hidden_size=args.arch_prnn.hidden_size,
        pRNNtype=args.arch_prnn.prnn_type.value,
        learningRate=args.train_prnn.lr,
        bptttrunc=args.train_prnn.bptt_trunc,
        weight_decay=args.train_prnn.weight_decay,
        neuralTimescale=args.arch_prnn.n_timescale,
        dropp=args.arch_prnn.dropout,
        trainNoiseMeanStd=(args.arch_prnn.noise_mean, args.arch_prnn.noise_std),
        f=args.arch_prnn.sparsity,
        wandb_log=args.run.wandb,
    )
    load_pN(
        model_ckpt_filepath=pRNN_ckpt,
        device=device,
        pRNNtype=args.arch_prnn.prnn_type.value,
        predictive_net=predictiveNet,
    )
    return predictiveNet


def get_SR_acmodel(
    args,
    env_act_space,
    obs_space: dict,
    device: torch.device,
    acmodel_status_ckpt: str | None = None,
) -> ACModelSR:
    """
    Loads ACModel and Optimizer State Dictionaries from checkpoint.
    """

    acmodel = ACModelSR(
        obs_space=obs_space,
        action_space=env_act_space,
        SR_size=args.arch_prnn.hidden_size,
        with_CV=args.arch_policy.with_obs,
        rgb=args.arch_policy.rgb,
        with_HD=args.arch_policy.with_head_direction,
    )

    if acmodel_status_ckpt == "" or acmodel_status_ckpt is None:
        return acmodel.to(device)
    else:
        status = torch.load(
            acmodel_status_ckpt, map_location=device, weights_only=False
        )

    if StatusCkptKeys.MODEL_STATE.value in status:
        acmodel.load_state_dict(status[StatusCkptKeys.MODEL_STATE.value])
    else:
        print(
            "Warning: model_state not found in acmodel status ckpt. Random agent was used. Returning fresh ACModel"
        )

    return acmodel.to(device)


def get_goal_loc(env: FaramaMinigridShell) -> list[int]:
    return access_get_goal_loc(env)


def get_agent(
    env: FaramaMinigridShell,
    agent_Type: AgentType,
    rand_act_prob: np.ndarray = RAND_ACT_PROBA,
    prnn: PredictiveNet | None = None,
    device: torch.device | None = None,
    ac_model: ACModel | None = None,
    argmax=False,
    action_offset: int = 0,
) -> ActorCriticAgent | RandomActionAgent:
    if agent_Type == AgentType.RANDOM:
        agent = RandomActionAgent(env.action_space, rand_act_prob)
    elif agent_Type == AgentType.AC:
        assert ac_model is not None, "ACModel must be provided for ActorCriticAgent"
        assert prnn is not None, "PredictiveNet must be provided for ActorCriticAgent"
        assert device is not None, "Device must be provided for ActorCriticAgent"

        agent = ActorCriticAgent(
            action_space=env.action_space,
            acmodel=ac_model,
            prnn=prnn,
            device=device,
            argmax=argmax,
            action_offset=action_offset,
        )

    return agent


def save_policy(policy, model_dir):
    path = policy_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(policy, path)


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


def save_pN_and_acmodel(
    pN: PredictiveNet,
    ac_model: ACModel,
    save_path: str,
    count: int,
    *,
    ac_optimizer=None,
):
    """Write one checkpoint step under `save_path/count/`.

    Uses the SAME two filenames and the SAME status keys as main_train.py
    (training/loop.py::save_checkpoint), so a step directory can be handed
    straight to get_ckpt_env_vars / setup_task. `count` lives in the directory
    name, not the filename.

    ac_optimizer: the AC model's optimizer. Passing it makes the step fully
    resumable; omitting it leaves OPTIMIZER_STATE out rather than filling it
    with the pRNN's optimizer, which is what the pre-2026-07-30 layout did.
    """
    step_dir = f"{save_path}/{count}"
    save_pN(pN, f"{step_dir}/{PRNN_CKPT_FILENAME}")
    print(f"Saved trained net to {step_dir}/{PRNN_CKPT_FILENAME}")
    # A step directory is handed straight to CUR_CKPT_DIR to seed another run,
    # so it is an artifact in its own right and needs its own provenance - not
    # just the run directory's one level up.
    from curious_george.log_and_store import provenance

    provenance.write(
        step_dir,
        kind="checkpoint",
        params={"count": count, "run_dir": str(save_path)},
    )
    status_save = {
        StatusCkptKeys.MODEL_STATE.value: ac_model.state_dict(),
        StatusCkptKeys.PRNN_OPTIMIZER_STATE.value: pN.optimizer.state_dict(),
    }
    if ac_optimizer is not None:
        status_save[StatusCkptKeys.OPTIMIZER_STATE.value] = ac_optimizer.state_dict()
    save_policy(status_save, step_dir)


# def get_vocab(model_dir):
#     return load_policy(model_dir)["vocab"]

"""Old-vs-new OMT comparison: getTestTrial + quantifyObjectLearning.

Run from inside either tree: .venv/bin/python compare_omt.py <out.pt>
"""

import sys
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from prnn.utils import MinigridEnvNames, ActionEncodingsEnum
from utils import get_ckpt_env_vars, AgentType, AgentInputType
from RLutils import make_env
from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

OUT = sys.argv[1]
DEVICE = torch.device("cpu")

with initialize_config_dir(config_dir=str(Path.cwd() / "Configs"), version_base=None):
    args = compose(config_name="Conf1_Adel", overrides=[
        "logging.wandb_log=false",
        "predNet.seqdur=32",
        "tasks.testing.trajs=2",
    ])

prnn_ckpt, ac_ckpt = get_ckpt_env_vars(AgentType.AC)

env_orig = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                    act_enc=ActionEncodingsEnum.SpeedHD.value)
env_novel = make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=[7, 2],
                     input_type=AgentInputType.H_PO.value,
                     act_enc=ActionEncodingsEnum.SpeedHD.value)

omt = ObjectMemoryTask(
    args=args, agent_type=AgentType.AC, env_orig=env_orig, env_novel=env_novel,
    save_path="cmp_omt", prnn_ckpt=prnn_ckpt, acmodel_status_ckpt=ac_ckpt,
    device=DEVICE,
)

test = omt.getTestTrial(n_trajs=2)
learn = omt.quantifyObjectLearning(
    ctrl_locs=args.tasks.testing.ctrl_locs,
    whichPhase=args.tasks.testing.whichPhase,
    traj_count=0,
)

results = {
    "obs": test["obs"].clone(),
    "obs_pred": test["obs_pred"].clone(),
    "obs_pred_control": test["obs_pred_control"].clone(),
    "agent_pos": np.asarray(test["agent_pos"]).copy(),
    "agent_dir": np.asarray(test["agent_dir"]).copy(),
    "learn": {k: (np.asarray(v).copy() if not np.isscalar(v) else v) for k, v in learn.items()},
}
torch.save(results, OUT)
print(f"saved {OUT}")
print(f"goalmodulation={learn['goalmodulation']:.6f} inviewtimes={learn['inviewtimes']}")
print(f"agent_pos[0,:3]={results['agent_pos'][0,:3].tolist()}")

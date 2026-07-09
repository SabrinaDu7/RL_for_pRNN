"""Golden fixture for the Object Memory Task (pre-refactor oracle).

Runs the REAL task code end-to-end with the real .env checkpoints, CPU,
pinned seed, wandb off, figures disabled via config (not code):
construction -> trainNovelObject (2 batches, incl. lr_trials scaling and the
analysis-interval eval that fires at batch 0) -> getTestTrial(2) ->
quantifyObjectLearning.

Run:  uv run python tests/golden_omt/capture_golden_omt.py
Writes: tests/golden_omt/golden_omt_v0.pt

The refactored task must reproduce every tensor exactly (same seed => same
RNG consumption order). Also encodes the env wiring guard: training uses
env_novel (object present), eval rollouts use env_orig (object absent).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from prnn.utils import MinigridEnvNames, ActionEncodingsEnum
from curious_george import AgentInputType, AgentType, get_ckpt_env_vars, make_env

try:  # refactored layout
    from tasks.omt.task import ObjectMemoryTask
except ImportError:  # pre-refactor layout
    from tasks.ObjectMemoryTask.define_task import ObjectMemoryTask

DEVICE = torch.device("cpu")
REPO = Path(__file__).resolve().parents[2]
OUT = str(REPO / "tests" / "golden_omt" / "golden_omt_v0.pt")

OVERRIDES = [
    # golden_omt_v0.pt was captured under legacy alignment; pin it so the
    # bitwise gate is immune to changes of the main.yaml default rewards
    # group (a default flip to curious_next_obs broke it on 2026-07-08)
    "rewards=curious",
    "logging.wandb_log=false",
    "predNet.seqdur=32",
    "rl.frames=64",
    "rl.trajs_per_batch=2",
    "tasks.testing.trajs=2",
    "tasks.training.num_trajs=4",       # -> 2 train batches
    "tasks.training.saving_interval=999",
    "tasks.training.analysis_interval=999",
    "tasks.analysis.traj_fig=false",
    "tasks.analysis.objLearning_fig=false",
    "tasks.analysis.occupancy=false",
]


def _param_summary(module) -> dict:
    """Small but discriminative snapshot of a large module's weights."""
    sums = {n: p.detach().double().sum() for n, p in module.named_parameters()}
    first = next(iter(module.parameters())).detach().flatten()[:100].clone()
    return {"sums": sums, "first100": first}


def run_omt_capture() -> dict:
    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        args = compose(config_name="main", overrides=OVERRIDES)

    prnn_ckpt, ac_ckpt = get_ckpt_env_vars(AgentType.AC)

    env_orig = make_env(env_key=MinigridEnvNames.LRoom, input_type=AgentInputType.H_PO.value,
                        act_enc=ActionEncodingsEnum.SpeedHD.value)
    env_novel = make_env(env_key=MinigridEnvNames.LRoom, new_obj_pos=[7, 2],
                         input_type=AgentInputType.H_PO.value,
                         act_enc=ActionEncodingsEnum.SpeedHD.value)

    # env wiring guard: novel env has the object, orig env does not
    assert env_novel.get_new_obj_pos() == [7, 2]
    assert env_orig.get_new_obj_pos() is None

    with tempfile.TemporaryDirectory() as tmp:
        omt = ObjectMemoryTask(
            args=args, agent_type=AgentType.AC, env_orig=env_orig, env_novel=env_novel,
            save_path=tmp, prnn_ckpt=prnn_ckpt, acmodel_status_ckpt=ac_ckpt,
            device=DEVICE,
        )

        post_construction = {
            "acmodel_state": {k: v.clone() for k, v in omt.ac_model.state_dict().items()},
            "pN_post": _param_summary(omt.pN_post.pRNN),
            "pN_control": _param_summary(omt.pN_control.pRNN),
        }

        # capture per-batch training tensors via a wrapper (real method still runs)
        batches = []
        orig_update = omt.algo.update_parameters

        def spy_update(*a, **k):
            batches.append({
                "curious_rewards": omt.algo.curious_rewards.clone(),
                "advantages": omt.algo.advantages.clone(),
                "actions": omt.algo.actions.clone(),
                "locs": list(omt.algo.locs),
            })
            return orig_update(*a, **k)

        omt.algo.update_parameters = spy_update
        omt.trainNovelObject(
            num_trajs=args.tasks.training.num_trajs,
            saving_interval=args.tasks.training.saving_interval,
            analysis_interval=args.tasks.training.analysis_interval,
            lr_trials=args.tasks.training.lr_trials,
            lrgroups=list(args.tasks.training.lrgroups),
            device=DEVICE,
        )
        omt.algo.update_parameters = orig_update

        post_train = {
            "acmodel_state": {k: v.clone() for k, v in omt.ac_model.state_dict().items()},
            "pN_post": _param_summary(omt.pN_post.pRNN),
            "pN_control": _param_summary(omt.pN_control.pRNN),  # must be UNCHANGED
        }

        test = omt.getTestTrial(n_trajs=int(args.tasks.testing.trajs))
        learn = omt.quantifyObjectLearning(
            ctrl_locs=args.tasks.testing.ctrl_locs,
            whichPhase=args.tasks.testing.whichPhase,
            traj_count=0,
        )

    return {
        "post_construction": post_construction,
        "batches": batches,
        "post_train": post_train,
        "test_trial": {
            "obs": test["obs"].clone(),
            "obs_pred": test["obs_pred"].clone(),
            "obs_pred_control": test["obs_pred_control"].clone(),
            "agent_pos": np.asarray(test["agent_pos"]).copy(),
            "agent_dir": np.asarray(test["agent_dir"]).copy(),
        },
        "object_learning": {
            k: (np.asarray(v).copy() if not np.isscalar(v) else v)
            for k, v in learn.items()
        },
        "env_guard": {
            "novel_obj_pos": env_novel.get_new_obj_pos(),
            "orig_obj_pos": env_orig.get_new_obj_pos(),
        },
    }


if __name__ == "__main__":
    fixture = run_omt_capture()
    torch.save(fixture, OUT)
    print(f"saved {OUT}")
    print(f"batches: {len(fixture['batches'])}")
    print(f"batch0 curious mean={fixture['batches'][0]['curious_rewards'].mean():.6e}")
    print(f"goalmodulation={fixture['object_learning']['goalmodulation']:.6f}")
    print(f"agent_pos[0,:3]={fixture['test_trial']['agent_pos'][0, :3].tolist()}")

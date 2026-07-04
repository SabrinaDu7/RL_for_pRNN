"""Old-vs-new I/O comparison harness.

Run from inside either tree (pre-refactor worktree or refactored repo):
    .venv/bin/python compare_io.py <out.pt>
Uses only the shared API surface (flat RLutils imports + prnn) and the real
.env checkpoints. Each section re-seeds so sections are independent.
"""

import sys
import inspect
import numpy as np
import torch

try:
    import curious_george as RLutils
except ImportError:  # pre-refactor tree
    import RLutils
try:
    from curious_george import ACModelSR, PredictivePPOAlgo, ActorCriticAgent
except ImportError:  # pre-refactor tree
    from RLutils import ACModelSR, PredictivePPOAlgo, ActorCriticAgent
from prnn.utils import PredictiveNet, load_pN
from utils import get_ckpt_env_vars, StatusCkptKeys

OUT = sys.argv[1]
DEVICE = torch.device("cpu")
FRAMES, SEQDUR = 64, 32

PRNN_CKPT, AC_CKPT = get_ckpt_env_vars()


def build_env(seed):
    return RLutils.make_env(
        env_key="MiniGrid-LRoom-v0", input_type="pRNN", seed=seed, act_enc="SpeedHD",
    )


def build_models(env):
    pN = PredictiveNet(
        env, hidden_size=500, pRNNtype="thRNN_5win", learningRate=3e-3,
        bptttrunc=1e8, weight_decay=3e-3, neuralTimescale=2, dropp=0.15,
        trainNoiseMeanStd=(0, 0.05), f=0.5, wandb_log=False,
    )
    pN.env_shell.hd_trans = np.array([-1, 1, 0, 0])
    load_pN(model_ckpt_filepath=PRNN_CKPT, device=DEVICE, pRNNtype="thRNN_5win", predictive_net=pN)

    obs_space, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)
    acmodel = ACModelSR(obs_space, env.action_space, 500, False, True, True)  # noObs config
    status = torch.load(AC_CKPT, map_location=DEVICE, weights_only=False)
    acmodel.load_state_dict(status[StatusCkptKeys.MODEL_STATE.value])
    acmodel.to(DEVICE)
    return pN, acmodel, preprocess_obss


results = {}

# --- Section A: ckpt-loaded algo, one collect+update round ---------------
RLutils.seed(11)
env = build_env(11 + 10000)
pN, acmodel, preprocess_obss = build_models(env)

kwargs = dict(
    env=env, acmodel=acmodel, predictiveNet=pN, device=DEVICE,
    num_frames=FRAMES, discount=0.98, lr=3e-4, gae_lambda=0.95,
    entropy_coef=0.0, value_loss_coef=1, max_grad_norm=0.5, recurrence=1,
    adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=16,
    preprocess_obss=preprocess_obss, train_pN=True, noise_mu=0,
    noise_std=0.05, prnn_seqdur=SEQDUR, intrinsic=False, k_int=1,
    pastSR=True, curious_agent=True, k_curious=1,
)
sig = inspect.signature(PredictivePPOAlgo.__init__).parameters
if "place_cells" in sig:
    kwargs["place_cells"] = None
    kwargs["cann"] = None

algo = PredictivePPOAlgo(**kwargs)
exps, logs1 = algo.collect_experiences()
logs2 = algo.update_parameters(exps=exps, update_params=True)
results["A"] = {
    "curious_rewards": algo.curious_rewards.clone(),
    "advantages": algo.advantages.clone(),
    "actions": algo.actions.clone(),
    "log_probs": algo.log_probs.clone(),
    "SRs": exps.SR.clone(),
    "locs": list(logs1["locs"]),
    "policy_loss": logs2["policy_loss"],
    "acmodel_sum": sum(p.double().sum().item() for p in acmodel.parameters()),
    "prnn_sum": sum(p.double().sum().item() for p in pN.pRNN.parameters()),
}

# --- Section B: agent rollout (getObservations) ---------------------------
RLutils.seed(12)
env_b = build_env(12 + 10000)
pN_b, acmodel_b, _ = build_models(env_b)
pN_b.pRNN.to(DEVICE)
pN_b.pRNN.eval()
acmodel_b.eval()
agent = ActorCriticAgent(env_b.action_space, acmodel_b, pN_b, DEVICE, argmax=False, pastSR=True)
obs_l, act, state, _ = agent.getObservations(env_b, 50)
results["B"] = {
    "act": np.asarray(act).copy(),
    "agent_pos": np.asarray(state["agent_pos"]).copy(),
    "agent_dir": np.asarray(state["agent_dir"]).copy(),
    "SRs": np.asarray(state["SRs"]).copy(),
    "obs_imgs": np.stack([o["image"] for o in obs_l]),
}

# --- Section C: on-policy sRSA through calculateSpatialRepresentation -----
RLutils.seed(13)
env_c = build_env(13 + 10000)
pN_c, acmodel_c, _ = build_models(env_c)
pN_c.pRNN.to("cpu")
agent_c = ActorCriticAgent(env_c.action_space, acmodel_c, pN_c, DEVICE, argmax=False, pastSR=True)
_, _, _, sRSA = pN_c.calculateSpatialRepresentation(
    env_c, agent_c, trainDecoder=True, trainHDDecoder=False,
    saveTrainingData=False, bitsec=False, calculatesRSA=True,
    sleepstd=0.03, wandb_nameext="_cmp",
)
results["C"] = {"sRSA": float(sRSA)}

torch.save(results, OUT)
print(f"saved {OUT}")
print(f"A: cur_mean={results['A']['curious_rewards'].mean():.6e} adv_mean={results['A']['advantages'].mean():.6e}")
print(f"B: act[:10]={results['B']['act'][:10]}")
print(f"C: sRSA={results['C']['sRSA']:.6f}")

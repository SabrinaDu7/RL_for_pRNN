import torch
from prnn.utils import (
    make_env,
    PredictiveNet,
    ActionEncodingsEnum,
    MinigridEnvNames,
    pRNNtypes,
    load_pN,
)
from RLutils import ACModelSR, get_obss_preprocessor, DEVICE
from utils import get_ckpt_env_vars, load_statedict_from_acmodel_status, load_acmodel_status, StatusCkptKeys, get_minigrid_env, AgentInputType

SIZE = 16
ENV_NAME = MinigridEnvNames.LRoom18 if SIZE == 18 else MinigridEnvNames.LRoom16
PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()

def _get_env():
    env = get_minigrid_env(env_name=ENV_NAME, input_type = AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    return env
    
def _get_pRNN(env = None, pRNN_ckpt_path = PRNN_CKPT):
    if env is None:
        env = _get_env()

    predictive_net: PredictiveNet = load_pN(model_ckpt_filepath=pRNN_ckpt_path, env=env, pRNNtype=pRNNtypes.masked)
    return predictive_net


def test_load_pRNN():
    """Test loading a pre-trained network."""

    predictive_net = _get_pRNN()

    assert predictive_net is not None
    assert hasattr(predictive_net, "pRNN")
    assert hasattr(predictive_net, "EnvLibrary")
    assert hasattr(predictive_net, "env_shell")
    assert hasattr(predictive_net, "predict")
    assert hasattr(predictive_net, "addEnvironment")


def test_pRNN_with_environment():
    """Test that a loaded network can work with the LRoom environment."""

    env = make_env(env_key=ENV_NAME, act_enc="SpeedHD")
    env.reset()

    predictive_net = _get_pRNN()
    predictive_net.addEnvironment(env)

    # Basic functionality test
    assert len(predictive_net.EnvLibrary) >= 2  # Original env + new env


def test_load_acmodel_from_checkpoint():
    """Test loading a pre-trained ACModel from checkpoint."""
    # Setup
    env = _get_env()
    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
    predNet = _get_pRNN(env)

    # Load checkpoint
    status = load_acmodel_status(acmodel_status_ckpt=ACMODEL_STATUS_CKPT)
    
    # Verify checkpoint has required keys
    assert StatusCkptKeys.MODEL_STATE.value in status
    assert StatusCkptKeys.NUM_FRAMES.value in status
    
    # Create ACModel
    acmodel = ACModelSR(
        obs_space=obs_space,
        action_space=env.action_space,
        SR_size=predNet.hidden_size,
        with_CV=True,
        rgb=True,
        with_HD=True,
    )
    
    # Load weights into acmodel
    load_statedict_from_acmodel_status(
        receiver=acmodel,
        status=status,
        status_key=StatusCkptKeys.MODEL_STATE,
    )
    
    # Verify model works
    acmodel.eval()
    acmodel.to(DEVICE)
    
    with torch.no_grad():
        obs = env.reset()
        preprocessed_obs = preprocess_obss([obs], device=DEVICE)
        SR = torch.zeros(1, predNet.hidden_size, device=DEVICE)
        
        dist, value = acmodel(preprocessed_obs, SR=SR)
        
        # Verify output shapes
        assert dist.probs.shape == (1, env.action_space.n)
        assert value.shape == (1,)
        
        # Verify outputs are valid
        assert torch.all(dist.probs >= 0) and torch.all(dist.probs <= 1)
        assert torch.allclose(dist.probs.sum(dim=1), torch.ones(1).to(DEVICE))
        assert not torch.isnan(value).any()


def test_load_actor_critic_agent():
    """Test creating an ActorCriticAgent with loaded models."""
    env = _get_env()
    predNet = _get_pRNN(env)
    
    # Load and setup ACModel
    status = load_acmodel_status(acmodel_status_ckpt=ACMODEL_STATUS_CKPT)
    obs_space, _ = get_obss_preprocessor(env.observation_space)
    
    acmodel = ACModelSR(
        obs_space=obs_space,
        action_space=env.action_space,
        SR_size=predNet.hidden_size,
        with_CV=True,
        rgb=True,
        with_HD=True,
    )
    load_statedict_from_acmodel_status(
        receiver=acmodel,
        status=status,
        status_key=StatusCkptKeys.MODEL_STATE,
    )
    acmodel.to(DEVICE)
    
    # Create ActorCriticAgent
    from RLutils import ActorCriticAgent
    
    agent = ActorCriticAgent(
        action_space=env.action_space,
        acmodel=acmodel,
        prnn=predNet,
        device=DEVICE
    )
    
    # Test agent can generate observations
    obs, acts, state, render = agent.getObservations(env, tsteps=10, reset=True)
    
    assert len(obs) == 11  # tsteps + 1
    assert len(acts) == 10
    assert "agent_pos" in state
    assert "SRs" in state

"""
def test_pRNN_sRSA()
def test_pRNN_loss()
def test acmodel 

you want to run these values to make sure that you didn't break anything when changing the architecture
"""
    

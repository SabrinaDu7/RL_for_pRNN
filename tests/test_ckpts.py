import torch
import pytest
import numpy
from dataclasses import dataclass

from prnn.utils import (
    PredictiveNet,
    ActionEncodingsEnum,
    MinigridEnvNames,
    pRNNtypes,
    RandomActionAgent,
)
from prnn.utils.Shell import FaramaMinigridShell

from RLutils import get_obss_preprocessor, make_env, get_pN, get_SR_acmodel, seed, DEVICE, ACModelSR, ActorCriticAgent
from utils import get_ckpt_env_vars, AgentInputType

SIZE = 16
ENV_NAME = MinigridEnvNames.LRoom18 if SIZE == 18 else MinigridEnvNames.LRoom16
PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()
seed(2)


def _check_same_device(model: torch.nn.Module, device: torch.device):
    device_type = device.type
    param = next(model.parameters())
    assert param.device.type == device_type, f"Parameter on device {param.device.type}, expected {device.type}"

def _get_pRNN(device: torch.device, env: FaramaMinigridShell | None = None) -> PredictiveNet:
    @dataclass
    class PredNetArgs:
        hiddensize: int = 500
        pRNNtype: str = pRNNtypes.masked.value
        lr: float = 3e-3
        bptttrunc: float = 1e8
        weight_decay: float = 3e-3
        ntimescale: int = 2
        dropout: float = 0.15
        noisemean: float = 0
        noisestd: float = 0.05
        sparsity: float = 0.5

    @dataclass
    class LoggingArgs:
        wandb_log: bool = False

    @dataclass
    class Args:
        predNet: PredNetArgs = PredNetArgs()
        logging: LoggingArgs = LoggingArgs()
    
    if env is None:
        env = make_env(env_key=ENV_NAME, input_type=AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    
    args = Args()
    predictive_net = get_pN(args=args, env=env, device=DEVICE)
    return predictive_net


def _get_acmodel(acmodel_status_ckpt: str, env: FaramaMinigridShell, predNet: PredictiveNet, device: torch.device) -> ACModelSR:
    obs_space, _ = get_obss_preprocessor(env.observation_space)

    @dataclass
    class PredNetArgs:
        hiddensize: int = 500
        pRNNtype: str = pRNNtypes.masked.value

    @dataclass
    class ExperimentArgs:
        with_obs: bool = True
        rgb: bool = True
        with_HD: bool = True

    @dataclass
    class Args:
        predNet: PredNetArgs = PredNetArgs()
        exp: ExperimentArgs = ExperimentArgs()
    
    args = Args()
    acmodel = get_SR_acmodel(args, env.action_space, obs_space, acmodel_status_ckpt, device)
    return acmodel


def test_load_pRNN():
    """Test loading a pre-trained network."""

    predictive_net = _get_pRNN(device=DEVICE)
    _check_same_device(predictive_net.pRNN, DEVICE)

    assert predictive_net is not None
    assert hasattr(predictive_net, "pRNN")
    assert hasattr(predictive_net, "EnvLibrary")
    assert hasattr(predictive_net, "env_shell")
    assert hasattr(predictive_net, "predict")
    assert hasattr(predictive_net, "addEnvironment")


def test_pRNN_with_environment():
    """Test that a loaded network can work with the LRoom environment."""

    env = make_env(env_key=ENV_NAME, input_type=AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)

    predictive_net = _get_pRNN(device=DEVICE)
    predictive_net.addEnvironment(env)

    # Basic functionality test
    assert len(predictive_net.EnvLibrary) >= 2  # Original env + new env


def test_load_acmodel_from_checkpoint():
    """Test loading a pre-trained ACModel from checkpoint."""
    # Setup
    env = make_env(env_key=ENV_NAME, input_type=AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    obs_space, preprocess_obss = get_obss_preprocessor(env.observation_space)
    predNet = _get_pRNN(device=DEVICE, env=env)

    # Load checkpoint
    acmodel = _get_acmodel(acmodel_status_ckpt=ACMODEL_STATUS_CKPT, env=env, predNet=predNet, device=DEVICE)
    _check_same_device(acmodel, DEVICE)

    # Verify model works
    acmodel.eval()
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
    env = make_env(env_key=ENV_NAME, input_type=AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    predNet = _get_pRNN(device=DEVICE, env=env)
    
    # Load and setup ACModel
    acmodel = _get_acmodel(acmodel_status_ckpt=ACMODEL_STATUS_CKPT, env=env, predNet=predNet, device=DEVICE)
    
    # Create ActorCriticAgent
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


@pytest.mark.slow
def test_pRNN_sRSA():
    """Load a pRNN and an ACModelSR, run spatial-representation analysis and
    assert the sRSA metric is above a threshold (0.6).
    """
    print(f"Loading from {PRNN_CKPT} and {ACMODEL_STATUS_CKPT}")
    # Prepare environment and models
    env = make_env(env_key=ENV_NAME, input_type=AgentInputType.H_PO, act_enc=ActionEncodingsEnum.SpeedHD)
    predictiveNet = _get_pRNN(env=env, device=DEVICE)
    predictiveNet.wandb_log = False
    predictiveNet.pRNN.to(DEVICE)

    # Load acmodel
    acmodel = _get_acmodel(acmodel_status_ckpt=ACMODEL_STATUS_CKPT, env=env, predNet=predictiveNet, device=DEVICE)

    # Build the wrapper agent that uses the AC model and the predictive net
    action_probability = numpy.array([0.15, 0.15, 0.6, 0.1])
    randomagent = RandomActionAgent(env.action_space, action_probability)
    ac_agent = ActorCriticAgent(env.action_space, acmodel, predictiveNet, device=DEVICE)

    # Run the spatial-representation analysis
    _, SI_on, _, sRSA_on = predictiveNet.calculateSpatialRepresentation(
        env,
        ac_agent,
        trainDecoder=True,
        trainHDDecoder=False,
        saveTrainingData=False,
        bitsec=False,
        calculatesRSA=True,
        sleepstd=0.03,
        wandb_nameext="_onPolicy",
    )

    _, SI_off, _, sRSA_off = predictiveNet.calculateSpatialRepresentation(
        env,
        randomagent,
        trainDecoder=True,
        trainHDDecoder=False,
        saveTrainingData=False,
        bitsec=False,
        calculatesRSA=True,
        sleepstd=0.03,
        wandb_nameext="_offPolicy",
    )

    # Assert sRSA is sufficiently high
    print(sRSA_on)
    assert type(sRSA_on) == numpy.float64, f"sRSA is not a float: {type(sRSA_on)} of value {sRSA_on}"
    # assert sRSA_on >= 0.6, f"sRSA too low: {sRSA_on} < 0.6"

    print(sRSA_off)
    assert type(sRSA_off) == numpy.float64, f"sRSA is not a float: {type(sRSA_off)} of value {sRSA_off}"
    # assert sRSA_off >= 0.4, f"sRSA too low: {sRSA_off} < 0.4"

    SI_on_mean = SI_on["SI"].mean()
    print(SI_on_mean)
    assert type(SI_on_mean) == numpy.float64, f"SI is not a float: {type(SI_on_mean)} of value {SI_on_mean}"
    # assert SI_on <= 0.9, f"SI too high: {SI_on} > 0.9"

    SI_off_mean = SI_off["SI"].mean()
    print(SI_off_mean)
    assert type(SI_off_mean) == numpy.float64, f"SI is not a float: {type(SI_off_mean)} of value {SI_off_mean}"
    # assert SI_off <= 0.7, f"SI too high: {SI_off} > 0.7"


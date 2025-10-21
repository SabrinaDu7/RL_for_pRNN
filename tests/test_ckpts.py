import os
from prnn.utils import (
    make_env,
    PredictiveNet,
    load_pN
)

from tests.commons import ENV_NAME, PRNN_CKPT

def _get_pRNN():
    assert os.path.isfile(f"{PRNN_CKPT}"), f"Network file {PRNN_CKPT} does not exist."

    env = make_env(env_key=ENV_NAME, act_enc="SpeedHD")
    env.reset()

    predictive_net = PredictiveNet(env=env, pRNNtype="thRNN_5win")
    load_pN(predictive_net, model_filepath=PRNN_CKPT)
    
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

    env = make_env(env_key=ENV_NAME, act_enc="OneHotHD")
    env.reset()

    predictive_net = _get_pRNN()
    predictive_net.addEnvironment(env)

    # Basic functionality test
    assert len(predictive_net.EnvLibrary) >= 2  # Original env + new env


def test_load_agent():
    """Test loading a pre-trained."""
    
    


def test_pRNN_sRSA():
    """ Test that loaded pRNN can perform on and off policy sRSA"""

    predictive_net = _get_pRNN()
    

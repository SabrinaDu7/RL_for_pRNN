import os
import torch

import RLutils
from .other import device


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
    return torch.load(path, map_location=device, weights_only=False)


def get_pN(model_dir):
    return os.path.join(model_dir, "pN.pkl")


def get_predictive_net_state_path(model_dir):
    return os.path.join(model_dir, "predictiveNet_state.pt")


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    RLutils.create_folders_if_necessary(path)
    torch.save(status, path)


def save_predictive_net_state(predictive_net, model_dir):
    """
    Save PredictiveNet state dictionaries to the model directory.
    
    Args:
        predictive_net: PredictiveNet instance to save
        model_dir: Model directory where to save the state
    """
    filepath = get_predictive_net_state_path(model_dir)
    RLutils.create_folders_if_necessary(filepath)
    
    state_dict = {
        'pRNN_state_dict': predictive_net.pRNN.state_dict(),
        'optimizer_state_dict': predictive_net.optimizer.state_dict(),
        'hidden_size': predictive_net.hidden_size,
        'obs_size': predictive_net.obs_size,
        'act_size': predictive_net.act_size,
        'num_training_trials': predictive_net.numTrainingTrials,
        'num_training_epochs': predictive_net.numTrainingEpochs,
        'learning_rate': predictive_net.learningRate,
        'weight_decay': predictive_net.weight_decay,
        'train_noise_mean_std': predictive_net.trainNoiseMeanStd,
    }
    
    # Save encoder if it exists and is trainable
    if hasattr(predictive_net.env_shell, 'encoder') and predictive_net.train_encoder:
        state_dict['encoder_state_dict'] = predictive_net.env_shell.encoder.state_dict()
        if hasattr(predictive_net.env_shell.encoder, 'optimizer'):
            state_dict['encoder_optimizer_state_dict'] = predictive_net.env_shell.encoder.optimizer.state_dict()
    
    torch.save(state_dict, filepath)


def load_predictive_net_state(predictive_net, model_dir):
    """
    Load PredictiveNet state dictionaries from the model directory into an existing instance.
    
    Args:
        predictive_net: PredictiveNet instance to load state into
        model_dir: Model directory where the state is saved
    """
    filepath = get_predictive_net_state_path(model_dir)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Load main network and optimizer
    predictive_net.pRNN.load_state_dict(checkpoint['pRNN_state_dict'])
    predictive_net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load training statistics
    predictive_net.numTrainingTrials = checkpoint.get('num_training_trials', -1)
    predictive_net.numTrainingEpochs = checkpoint.get('num_training_epochs', -1)
    
    # Load encoder if present
    if 'encoder_state_dict' in checkpoint and hasattr(predictive_net.env_shell, 'encoder'):
        predictive_net.env_shell.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if 'encoder_optimizer_state_dict' in checkpoint and hasattr(predictive_net.env_shell.encoder, 'optimizer'):
            predictive_net.env_shell.encoder.optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])


def load_predictive_net_state_dict_only(model_dir):
    """
    Load only the pRNN state dictionary from the model directory.
    Useful when you only need the trained weights.
    
    Args:
        model_dir: Model directory where the state is saved
        
    Returns:
        dict: The pRNN state dictionary
    """
    filepath = get_predictive_net_state_path(model_dir)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    return checkpoint['pRNN_state_dict']


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

import csv
import os
import torch
import logging
import sys

import RLutils
from .other import device


def create_folders_if_necessary(path):
    if path == '':
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
    return os.path.join(os.environ["HOME"], 'pRNN-RL/RLvideos', model_name)


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
    return torch.load(path, map_location=device)


def get_pN(model_dir):
    return os.path.join(model_dir, "pN.pkl")


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    RLutils.create_folders_if_necessary(path)
    torch.save(status, path)

def save_analysis_of_agent_behav(onpolicyAnalysis, model_dir, update_step):
    figs = {
    "advantages.png":            onpolicyAnalysis.plot_advantages(),
    "policy_heatmaps.png":       onpolicyAnalysis.plot_policy_heatmaps(),
    "occupancy.png":             onpolicyAnalysis.plot_occupancy(),
    "values.png":                onpolicyAnalysis.plot_values()}

    outdir = os.path.join(model_dir, "onpolicy_analysis", str(update_step))
    os.makedirs(outdir, exist_ok=True)

    for fname, fig in figs.items():
        savename = os.path.join(outdir, fname)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
        fig.write_image(savename)

# def get_vocab(model_dir):
#     return get_status(model_dir)["vocab"]
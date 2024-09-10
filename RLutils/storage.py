import csv
import os
import torch
import logging
import sys

import RLutils
from .other import device


def create_folders_if_necessary(path):
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


# def get_vocab(model_dir):
#     return get_status(model_dir)["vocab"]
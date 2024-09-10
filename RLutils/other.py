import random
import numpy as np
import torch
import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array, signs=False, abs=False):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    if signs:
        array = np.array(array)
        d["pos"] = np.sum(np.sign(array)[array>0])
        d["neg"] = np.sum(np.abs(np.sign(array)[array<0]))
    if abs:
        d["abs_mean"] = np.mean(np.abs(array))
    return d

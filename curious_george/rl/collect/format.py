"""Observation preprocessing for the policy: MiniGrid dicts -> device tensors.

The policy reads `direction` (and `image` when `arch_policy.with_obs`); the
mission string used to be tokenized on every step for a `text` field no
network ever consumed, and was dropped 2026-09-06.
"""

import numpy
import torch
import torch_ac
import gymnasium as gym


def get_obss_preprocessor(obs_space):
    """(obs_space dict, preprocess_obss) for a MiniGrid observation space with
    an `image` and a `direction`."""
    if not (isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces):
        raise ValueError("Unknown observation space: " + str(obs_space))

    obs_space = {"image": obs_space.spaces["image"].shape, "direction": 1}

    def preprocess_obss(obss, device=None):
        return torch_ac.DictList(
            {
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "direction": preprocess_int([obs["direction"] for obs in obss], device=device),
            }
        )

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_int(integer, device=None):
    integer = numpy.array(integer)
    return torch.tensor(integer, device=device, dtype=torch.uint8)

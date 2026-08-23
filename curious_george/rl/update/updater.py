"""Loss-agnostic policy update driver.

Owns the epoch / minibatch / optimizer / grad-clip machinery (extracted from
the old rl/ppo.py). Which objective is optimized is a `loss_fn` argument -
see update/losses.py for available losses and their shared signature.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Int
from torch_ac.utils import DictList

from curious_george.rl.update.losses import LOSSES, ppo_clip_loss
from curious_george.utils.timing import timer


@dataclass
class UpdateLogs:
    entropy: float
    value: float
    policy_loss: float
    value_loss: float
    grad_norm: float

    def as_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "value": self.value,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "grad_norm": self.grad_norm,
        }


def shuffled_minibatches(*, num_frames: int, batch_size: int) -> list[Int[np.ndarray, " b"]]:
    """The rollout's transition indexes, shuffled and cut into minibatches.

    One PPO gradient step per minibatch. Transitions are independent by this
    point - the advantages were already computed in episode order by
    `compute_gae` - so the shuffle is free to cut across episodes, which is
    what makes a minibatch `batch_size` transitions rather than a whole episode.
    """
    indexes = np.random.permutation(num_frames)
    return [indexes[i : i + batch_size] for i in range(0, num_frames, batch_size)]


def _index_policy_batch(exps, indexes, acmodel):
    """Index only fields consumed by actor/critic and the configured losses.

    The recurrent world model needs ``exps.obs.image`` after PPO, but the
    default SR actor has ``with_CV=False``. Indexing the 147-float RGB row for
    every PPO sample/epoch was therefore pure accelerator traffic.
    """
    if getattr(acmodel, "with_CV", True):
        return exps[indexes]

    sb = DictList({
        "SR": exps.SR[indexes],
        "action": exps.action[indexes],
        "value": exps.value[indexes],
        "advantage": exps.advantage[indexes],
        "returnn": exps.returnn[indexes],
        "log_prob": exps.log_prob[indexes],
    })
    if getattr(acmodel, "with_HD", False):
        sb.obs = DictList({"direction": exps.obs.direction[indexes]})
    else:
        sb.obs = DictList()
    return sb


def update_policy(
    acmodel,
    optimizer,
    exps,
    *,
    loss_fn=ppo_clip_loss,
    loss_kwargs: dict,
    epochs: int,
    batch_size: int,
    num_frames: int,
    max_grad_norm: float,
    update_params: bool = True,
) -> UpdateLogs:
    """Runs `epochs` reshuffled passes over `exps`, one gradient step per minibatch."""
    if isinstance(loss_fn, str):
        loss_fn = LOSSES[loss_fn]

    with timer("update/policy"):
        return _update_policy_epochs(
            acmodel, optimizer, exps, loss_fn, loss_kwargs, epochs, batch_size,
            num_frames, max_grad_norm, update_params,
        )


def _update_policy_epochs(
    acmodel, optimizer, exps, loss_fn, loss_kwargs, epochs, batch_size,
    num_frames, max_grad_norm, update_params,
) -> UpdateLogs:
    for _ in range(epochs):
        # Initialize log values

        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        with timer("update/policy/batch_indexes"):
            batches = shuffled_minibatches(num_frames=num_frames, batch_size=batch_size)

        for inds in batches:
            with timer("update/policy/index"):
                sb = _index_policy_batch(exps, inds, acmodel)

            with timer("update/policy/forward"):
                dist, value = acmodel(sb.obs, SR=sb.SR)
            with timer("update/policy/loss"):
                batch_loss, terms = loss_fn(dist, value, sb, **loss_kwargs)

            # Update actor-critic

            if update_params:
                with timer("update/policy/zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with timer("update/policy/backward"):
                    batch_loss.backward()
                # clip_grad_norm_ returns the pre-clip total L2 norm - the
                # same quantity the old per-parameter .item() sum computed.
                with timer("update/policy/grad_clip"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        acmodel.parameters(), max_grad_norm
                    ).detach()
                with timer("update/policy/adam"):
                    optimizer.step()
            else:
                grad_norm = batch_loss.new_zeros(())

            # Update log values

            log_entropies.append(terms.policy_entropy_bits)
            log_values.append(terms.value_mean)
            log_policy_losses.append(terms.policy_loss)
            log_value_losses.append(terms.value_loss)
            log_grad_norms.append(grad_norm)

    # Aggregate every scalar on-device and perform one host transfer/sync.
    with timer("update/policy/log_sync"):
        summary = torch.stack(
            [
                sum(log_entropies) / len(log_entropies),
                sum(log_values) / len(log_values),
                sum(log_policy_losses) / len(log_policy_losses),
                sum(log_value_losses) / len(log_value_losses),
                sum(log_grad_norms) / len(log_grad_norms),
            ]
        ).detach().cpu().tolist()

    logs = UpdateLogs(
        entropy=summary[0],
        value=summary[1],
        policy_loss=summary[2],
        value_loss=summary[3],
        grad_norm=summary[4],
    )

    return logs

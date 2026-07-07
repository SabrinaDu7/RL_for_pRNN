"""Loss-agnostic policy update driver.

Owns the epoch / minibatch / optimizer / grad-clip machinery (extracted from
the old rl/ppo.py). Which objective is optimized is a `loss_fn` argument -
see update/losses.py for available losses and their shared signature.
"""

from dataclasses import dataclass

import numpy as np
import torch

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


def get_batches_starting_indexes(num_frames: int, recurrence: int, batch_size: int, batch_num: int):
    """Gives, for each batch, the indexes of the observations given to
    the model and the experiences used to compute the loss at first.

    First, the indexes are the integers from 0 to `num_frames` with a step of
    `recurrence`, shifted by `recurrence//2` one time in two for having
    more diverse batches. Then, the indexes are split into the different batches.
    """

    indexes = np.arange(0, num_frames, recurrence)
    indexes = np.random.permutation(indexes)

    # Shift starting indexes by recurrence//2 half the time
    if batch_num % 2 == 1:
        indexes = indexes[(indexes + recurrence) % num_frames != 0]
        indexes += recurrence // 2

    num_indexes = batch_size // recurrence
    batches_starting_indexes = [
        indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
    ]

    return batches_starting_indexes


def update_policy(
    acmodel,
    optimizer,
    exps,
    *,
    loss_fn=ppo_clip_loss,
    loss_kwargs: dict,
    epochs: int,
    batch_size: int,
    recurrence: int,
    num_frames: int,
    max_grad_norm: float,
    batch_num: int,
    update_params: bool = True,
) -> tuple[UpdateLogs, int]:
    """Runs the update epochs over `exps`; returns (logs, new_batch_num)."""
    if isinstance(loss_fn, str):
        loss_fn = LOSSES[loss_fn]

    with timer("update/policy"):
        return _update_policy_epochs(
            acmodel, optimizer, exps, loss_fn, loss_kwargs, epochs, batch_size,
            recurrence, num_frames, max_grad_norm, batch_num, update_params,
        )


def _update_policy_epochs(
    acmodel, optimizer, exps, loss_fn, loss_kwargs, epochs, batch_size,
    recurrence, num_frames, max_grad_norm, batch_num, update_params,
) -> tuple[UpdateLogs, int]:
    for _ in range(epochs):
        # Initialize log values

        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        batches = get_batches_starting_indexes(num_frames, recurrence, batch_size, batch_num)
        batch_num += 1

        for inds in batches: # inds should be multiples of ppo_batch_size
            # Initialize batch values

            batch_entropy = 0
            batch_value = 0
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_loss = 0

            for i in range(recurrence): # only loops once
                # Create a sub-batch of experience

                sb = exps[inds + i]

                # Compute loss

                dist, value = acmodel(sb.obs, SR=sb.SR)
                loss, terms = loss_fn(dist, value, sb, **loss_kwargs)

                # Update batch values

                batch_entropy += terms.policy_entropy_bits
                batch_value += terms.value_mean
                batch_policy_loss += terms.policy_loss
                batch_value_loss += terms.value_loss
                batch_loss += loss

            # Update batch values

            batch_entropy /= recurrence
            batch_value /= recurrence
            batch_policy_loss /= recurrence
            batch_value_loss /= recurrence
            batch_loss /= recurrence

            # Update actor-critic

            if update_params:
                optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in acmodel.parameters()
                    )
                    ** 0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    acmodel.parameters(), max_grad_norm
                )
                optimizer.step()
            else:
                grad_norm = 0.0

            # Update log values

            log_entropies.append(batch_entropy)
            log_values.append(batch_value)
            log_policy_losses.append(batch_policy_loss)
            log_value_losses.append(batch_value_loss)
            log_grad_norms.append(grad_norm)

    logs = UpdateLogs(
        entropy=float(np.mean(log_entropies)),
        value=float(np.mean(log_values)),
        policy_loss=float(np.mean(log_policy_losses)),
        value_loss=float(np.mean(log_value_losses)),
        grad_norm=float(np.mean(log_grad_norms)),
    )

    return logs, batch_num

"""PPO update, extracted from PredictivePPOAlgo.update_parameters.

Pure function of (model, optimizer, experiences, hyperparams) plus the
`batch_num` counter that alternates the half-shifted minibatch indexing.
"""

import numpy as np
import torch


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


def ppo_update(
    acmodel,
    optimizer,
    exps,
    *,
    epochs: int,
    batch_size: int,
    recurrence: int,
    num_frames: int,
    clip_eps: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    batch_num: int,
    update_params: bool = True,
) -> tuple[dict, int]:
    """Runs the PPO epochs over `exps` and returns (logs, new_batch_num)."""

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

                policy_entropy = dist.entropy().mean()

                ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                surr1 = ratio * sb.advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    * sb.advantage
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = sb.value + torch.clamp(
                    value - sb.value, -clip_eps, clip_eps
                )
                surr1 = (value - sb.returnn).pow(2) # AC_model value estimate - target return
                surr2 = (value_clipped - sb.returnn).pow(2) # ppo clipping
                value_loss = torch.max(surr1, surr2).mean()

                loss = (
                    policy_loss
                    - entropy_coef * policy_entropy
                    + value_loss_coef * value_loss
                )

                # Update batch values

                batch_entropy += policy_entropy.item() / torch.log(
                    torch.tensor(2.0)
                )  # convert nats to bits
                batch_value += value.mean().item()
                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
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

    logs = {
        "entropy": np.mean(log_entropies),
        "value": np.mean(log_values),
        "policy_loss": np.mean(log_policy_losses),
        "value_loss": np.mean(log_value_losses),
        "grad_norm": np.mean(log_grad_norms),
    }

    return logs, batch_num

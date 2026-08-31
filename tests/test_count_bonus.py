"""The count-based novelty bonus: curiosity's model-free control.

Analytic gates first - the 1/sqrt schedule has exact answers - then the
wiring: same seed and same trajectories with and without the bonus, differing
only in advantages, which is the entire design (one knob, one term).
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from curious_george.configs import EnvBackend, EnvCfg, EvalKind
from curious_george.envs.layouts import (
    EnvContent,
    LandmarkKind,
    RoomSetRules,
    Uniform,
    Vary,
)
from curious_george.rl.update.rewards import CountBonus, occurrence_index
from curious_george.utils.checkpoints import StatusCkptKeys
from tests.small_config import small_config


def _cfg(k_count: float = 0.0, backend: EnvBackend = EnvBackend.DEVICE):
    cfg = small_config(
        backend=backend,
        num_envs=4,
        episodes_per_env=1,
        episode_steps=16,
        env=EnvCfg(
            content=EnvContent(
                kinds=tuple(LandmarkKind(s, impassable=True) for s in ("triangle3", "plus", "block3"))
            ),
            source=Uniform(n=3, seed=7),
            set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
        ),
        evals=frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )
    return dataclasses.replace(
        cfg, train_policy=dataclasses.replace(cfg.train_policy, k_count=k_count)
    )


# --- the schedule, exactly ---------------------------------------------------


def test_occurrence_index_counts_earlier_equals():
    states = torch.tensor([[5, 5, 7, 5, 7], [1, 2, 3, 4, 5]])
    expected = torch.tensor([[0, 1, 0, 2, 1], [0, 0, 0, 0, 0]])
    assert torch.equal(occurrence_index(states), expected)


def test_first_visit_pays_full_and_repeats_decay_within_the_rollout():
    """m-th visit in a rollout to a fresh state pays 1/sqrt(m) - the term that
    gives a fresh table a gradient at all (without it every step of the first
    rollout pays the identical maximum and the advantage is zero)."""
    bonus = CountBonus.create(n_layouts=1, width=4, height=4, device=torch.device("cpu"))
    layouts = torch.zeros((3, 1), dtype=torch.long)
    directions = torch.zeros((3, 1), dtype=torch.long)
    positions = torch.tensor([[[1, 1]], [[1, 1]], [[2, 1]]])  # stay, stay, move
    r = bonus.rewards(layouts_tb=layouts, positions_tb=positions, directions_tb=directions)
    assert r[:, 0].tolist() == pytest.approx([1.0, 1 / 2 ** 0.5, 1.0])
    assert bonus.counts[0, 1, 1, 0].item() == 2 and bonus.counts[0, 2, 1, 0].item() == 1


def test_counts_are_lifetime_across_rollouts():
    bonus = CountBonus.create(n_layouts=1, width=4, height=4, device=torch.device("cpu"))
    one = dict(
        layouts_tb=torch.zeros((1, 1), dtype=torch.long),
        positions_tb=torch.tensor([[[1, 1]]]),
        directions_tb=torch.zeros((1, 1), dtype=torch.long),
    )
    assert bonus.rewards(**one)[0, 0].item() == pytest.approx(1.0)
    assert bonus.rewards(**one)[0, 0].item() == pytest.approx(1 / 2 ** 0.5)
    assert bonus.rewards(**one)[0, 0].item() == pytest.approx(1 / 3 ** 0.5)


def test_state_distinguishes_direction_and_room():
    """The count key is (room, x, y, head direction) - the pRNN's own input
    granularity; a turn on the spot IS a novel state."""
    bonus = CountBonus.create(n_layouts=2, width=4, height=4, device=torch.device("cpu"))
    layouts = torch.tensor([[0], [0], [1]])
    positions = torch.tensor([[[1, 1]], [[1, 1]], [[1, 1]]])
    directions = torch.tensor([[0], [1], [0]])  # turn, then same pose in room 1
    r = bonus.rewards(layouts_tb=layouts, positions_tb=positions, directions_tb=directions)
    assert r.squeeze(1).tolist() == pytest.approx([1.0, 1.0, 1.0])


# --- the wiring --------------------------------------------------------------


def test_bonus_changes_advantages_and_nothing_else_in_the_rollout():
    """Same seed, first rollout: identical trajectories (the bonus is computed
    AFTER the timestep loop, so it cannot touch action sampling), different
    advantages (the term reached GAE), and the visit table absorbed exactly
    the rollout's steps."""
    from curious_george.training.setup import setup_training

    # setup -> collect per arm: setup_training reseeds everything, so the two
    # arms sample identical action streams; interleaving the collects instead
    # would compare two different draws of the SAME policy.
    base = setup_training(_cfg(k_count=0.0)).algo
    exps_base, logs_base = base.collect_experiences()
    counted = setup_training(_cfg(k_count=1.0)).algo
    exps_counted, logs_counted = counted.collect_experiences()

    assert torch.equal(exps_base.action, exps_counted.action)
    assert not torch.equal(exps_base.advantage, exps_counted.advantage)
    assert "count_bonus_mean" not in logs_base
    assert 0 < logs_counted["count_bonus_mean"] <= 1.0
    assert counted.count_bonus.counts.sum().item() == counted.num_frames
    assert base.count_bonus is None


def test_k_count_requires_the_device_backend():
    from curious_george.training.setup import setup_training

    with pytest.raises(ValueError, match="DEVICE backend"):
        setup_training(_cfg(k_count=1.0, backend=EnvBackend.SERIAL_TABLE))


def test_counts_ride_the_policy_checkpoint(tmp_path):
    """Lifetime counts are training state: a resume that dropped them would
    restart novelty from zero and pay a spurious bonus burst."""
    from curious_george.log_and_store.storage import POLICY_CKPT_FILENAME
    from curious_george.training.loop import save_checkpoint
    from curious_george.training.setup import setup_training

    cfg = _cfg(k_count=1.0)
    comps = setup_training(cfg)
    comps.algo.collect_experiences()
    saved = comps.algo.count_bonus.counts.clone()
    save_checkpoint(cfg, comps, SimpleNamespace(model_dir=str(tmp_path) + "/"),
                    num_frames=64, update=1, archive=False)
    comps.envs.close()

    status = torch.load(tmp_path / POLICY_CKPT_FILENAME, weights_only=False)
    assert StatusCkptKeys.COUNT_VISITS.value in status

    resumed_cfg = dataclasses.replace(
        cfg, run=dataclasses.replace(cfg.run, policy_ckpt=tmp_path / POLICY_CKPT_FILENAME)
    )
    resumed = setup_training(resumed_cfg).algo
    assert torch.equal(resumed.count_bonus.counts.cpu(), saved.cpu())

"""The policy checkpoint: its name, and that it is archived as a series.

Until 2026-08-28 the actor-critic lived in ONE rolling `status.pt` while the
pRNN was archived per step, so an archived world model could only ever be paired
with the last policy the run wrote. Nothing failed - the files existed and
loaded - they simply were not contemporaries, which quietly rules out every
on-policy readout except at the final step.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from curious_george.evaluation.checkpoint_series import archived, archived_policies
from curious_george.log_and_store.storage import (
    LEGACY_POLICY_CKPT_FILENAME,
    POLICY_CKPT_FILENAME,
    find_policy,
    load_policy,
    policy_path,
    save_policy,
)
from curious_george.training.loop import save_checkpoint
from curious_george.utils.checkpoints import StatusCkptKeys
from curious_george.utils.enums import AgentType


def test_policy_path_never_resolves_to_the_old_name(tmp_path: Path) -> None:
    """A writer must not be able to produce `status.pt` again."""
    (tmp_path / LEGACY_POLICY_CKPT_FILENAME).write_text("x")
    assert Path(policy_path(tmp_path)).name == POLICY_CKPT_FILENAME


def test_find_policy_prefers_the_current_name(tmp_path: Path) -> None:
    (tmp_path / LEGACY_POLICY_CKPT_FILENAME).write_text("x")
    (tmp_path / POLICY_CKPT_FILENAME).write_text("x")
    assert Path(find_policy(tmp_path)).name == POLICY_CKPT_FILENAME


def test_find_policy_falls_back_for_runs_older_than_the_rename(tmp_path: Path) -> None:
    (tmp_path / LEGACY_POLICY_CKPT_FILENAME).write_text("x")
    assert Path(find_policy(tmp_path)).name == LEGACY_POLICY_CKPT_FILENAME


def test_find_policy_raises_when_there_is_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_policy(tmp_path)


def test_save_policy_round_trips(tmp_path: Path) -> None:
    save_policy({StatusCkptKeys.NUM_FRAMES.value: 7}, str(tmp_path) + "/")
    assert (tmp_path / POLICY_CKPT_FILENAME).is_file()
    assert load_policy(tmp_path)[StatusCkptKeys.NUM_FRAMES.value] == 7


def _components(*, with_prnn: bool):
    """The narrow surface `save_checkpoint` actually touches, duck-typed."""
    acmodel = nn.Linear(2, 2)
    algo = SimpleNamespace(optimizer=torch.optim.Adam(acmodel.parameters()))
    prnn = None
    if with_prnn:
        inner = nn.Linear(2, 2)
        # Exactly the surface prnn.utils.checkpoints.save_pN reads - no more,
        # so the day it reads something else this fails loudly rather than
        # silently testing a stale contract.
        prnn = SimpleNamespace(
            pRNN=inner,
            optimizer=torch.optim.Adam(inner.parameters()),
            pRNNtype="thRNN_5win",
            hidden_size=2,
            obs_size=2,
            act_size=2,
            numTrainingTrials=0,
            numTrainingEpochs=0,
            learningRate=1e-3,
            weight_decay=0.0,
            trainNoiseMeanStd=(0.0, 0.0),
            env_shell=SimpleNamespace(),
            train_encoder=False,
        )
    return SimpleNamespace(acmodel=acmodel, algo=algo, predictiveNet=prnn)


def _cfg():
    return SimpleNamespace(
        arch_policy=SimpleNamespace(agent=AgentType.AC),
        train_prnn=SimpleNamespace(train=True),
    )


def test_archive_writes_both_models_at_the_same_step(tmp_path: Path) -> None:
    """The regression this file exists for: a pRNN step with no policy step."""
    model_dir = str(tmp_path) + "/"
    save_checkpoint(
        _cfg(), _components(with_prnn=True),
        SimpleNamespace(model_dir=model_dir),
        num_frames=4096, update=3, archive=True,
    )
    prnn_steps = [step for step, _ in archived(tmp_path)]
    policy_steps = sorted(archived_policies(tmp_path))
    assert prnn_steps == [4096]
    assert policy_steps == prnn_steps, "an archived pRNN step has no policy beside it"


def test_rolling_policy_is_still_written_without_archiving(tmp_path: Path) -> None:
    model_dir = str(tmp_path) + "/"
    save_checkpoint(
        _cfg(), _components(with_prnn=False),
        SimpleNamespace(model_dir=model_dir),
        num_frames=512, update=1, archive=False,
    )
    assert (tmp_path / POLICY_CKPT_FILENAME).is_file()
    assert not (tmp_path / "checkpoints").exists()
    assert load_policy(tmp_path)[StatusCkptKeys.UPDATE.value] == 1


def test_archived_policies_is_empty_for_a_pre_rename_run(tmp_path: Path) -> None:
    """An older run has the pRNN series and no policy series, and says so."""
    archive = tmp_path / "checkpoints"
    archive.mkdir()
    (archive / "predictiveNet_state_step0000004096.pt").write_text("x")
    assert [step for step, _ in archived(tmp_path)] == [4096]
    assert archived_policies(tmp_path) == {}

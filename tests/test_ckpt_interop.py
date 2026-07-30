"""Task checkpoints must be loadable by the same code that loads main_train ones.

Before 2026-07-30 they were not: task steps used a `pN-<count>.pt` filename that
`get_ckpt_env_vars` could not find, and stored the pRNN's 4-group RMSprop state
under `optimizer_state` - the key `setup_algo` loads into the AC model's
1-group Adam.
"""

import os

import pytest
import torch

from curious_george.utils.checkpoints import (
    StatusCkptKeys,
    load_prnn_optimizer_state,
    status_optimizer_matches,
)
from curious_george.utils.dev_env import (
    ACMODEL_CKPT_FILENAME,
    PRNN_CKPT_FILENAME,
    resolve_prnn_ckpt,
)


def _optimizer(n_groups: int) -> torch.optim.Optimizer:
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(n_groups)]
    return torch.optim.SGD([{"params": [p]} for p in params], lr=0.1)


# ---------------------------------------------------------------- filenames --

def test_resolve_prefers_canonical_filename(tmp_path):
    (tmp_path / PRNN_CKPT_FILENAME).write_bytes(b"")
    (tmp_path / "pN-200.pt").write_bytes(b"")
    assert resolve_prnn_ckpt(str(tmp_path)) == str(tmp_path / PRNN_CKPT_FILENAME)


def test_resolve_falls_back_to_legacy_filename(tmp_path):
    (tmp_path / "pN-200.pt").write_bytes(b"")
    assert resolve_prnn_ckpt(str(tmp_path)) == str(tmp_path / "pN-200.pt")


def test_resolve_rejects_ambiguous_legacy_dir(tmp_path):
    (tmp_path / "pN-200.pt").write_bytes(b"")
    (tmp_path / "pN-400.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="several"):
        resolve_prnn_ckpt(str(tmp_path))


def test_resolve_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match=PRNN_CKPT_FILENAME):
        resolve_prnn_ckpt(str(tmp_path))


def test_get_ckpt_env_vars_accepts_a_task_step_dir(tmp_path, monkeypatch):
    """The regression this whole module is about: CUR_CKPT_DIR pointed at a
    task checkpoint-step directory used to raise FileNotFoundError."""
    from curious_george.utils.dev_env import get_ckpt_env_vars
    from curious_george.utils.enums import AgentType

    (tmp_path / "pN-200.pt").write_bytes(b"")
    (tmp_path / ACMODEL_CKPT_FILENAME).write_bytes(b"")
    monkeypatch.setenv("CUR_CKPT_DIR", str(tmp_path))

    prnn_ckpt, status_ckpt = get_ckpt_env_vars(AgentType.AC)
    assert os.path.basename(prnn_ckpt) == "pN-200.pt"
    assert os.path.basename(status_ckpt) == ACMODEL_CKPT_FILENAME


# --------------------------------------------------------------- optimizers --

def test_status_optimizer_matches_on_group_count():
    ac, prnn = _optimizer(1), _optimizer(4)
    assert status_optimizer_matches(ac, ac.state_dict())
    assert not status_optimizer_matches(ac, prnn.state_dict())


def test_prnn_optimizer_loads_from_its_own_key():
    saved = _optimizer(4)
    status = {StatusCkptKeys.PRNN_OPTIMIZER_STATE.value: saved.state_dict()}
    assert load_prnn_optimizer_state(_optimizer(4), status)


def test_prnn_optimizer_loads_from_legacy_key():
    """Pre-2026-07-30 task checkpoints put it under OPTIMIZER_STATE."""
    saved = _optimizer(4)
    status = {StatusCkptKeys.OPTIMIZER_STATE.value: saved.state_dict()}
    assert load_prnn_optimizer_state(_optimizer(4), status)


def test_prnn_optimizer_ignores_an_ac_optimizer_under_the_legacy_key():
    """A main_train status.pt has the AC optimizer there - must not be taken."""
    status = {StatusCkptKeys.OPTIMIZER_STATE.value: _optimizer(1).state_dict()}
    assert not load_prnn_optimizer_state(_optimizer(4), status)
    with pytest.raises(KeyError):
        load_prnn_optimizer_state(_optimizer(4), status, strict=True)

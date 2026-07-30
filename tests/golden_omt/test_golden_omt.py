"""Bitwise gate for the OMT refactor: re-run the task end-to-end and compare
every tensor against the pre-refactor fixture golden_omt_v0.pt.

Scope: this answers "given the SAME checkpoint and seed, is the math still
bit-identical after a refactor?" - it is NOT a check that some new checkpoint
works well on OMT. Hence capture_golden_omt.py pins its checkpoint rather than
reading CUR_CKPT_DIR. (test_env_wiring_guard and
test_control_net_untouched_by_training are the two checkpoint-agnostic
invariants here and hold for any checkpoint.)

TODO: add a separate bound-based test (finite outputs, pN_post diverges from
pN_control, non-degenerate curious_rewards) that DOES run on CUR_CKPT_DIR.

Runtime ~1 min.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.golden_omt.capture_golden_omt import run_omt_capture

FIXTURE = Path(__file__).parent / "golden_omt_v0.pt"


def _assert_equal(name, a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{name}: max|diff|={(a - b).abs().max().item():.3e}"
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), name
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{name}: keys differ"
        for k in a:
            _assert_equal(f"{name}.{k}", a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b), f"{name}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(f"{name}[{i}]", x, y)
    else:
        assert a == b, f"{name}: {a!r} != {b!r}"


@pytest.fixture(scope="module")
def fixture_and_rerun():
    assert FIXTURE.exists(), "run capture_golden_omt.py first (pre-refactor tree)"
    golden = torch.load(FIXTURE, weights_only=False)
    fresh = run_omt_capture()
    return golden, fresh


def test_env_wiring_guard(fixture_and_rerun):
    """Training env has the novel object; eval env does not (memory probe)."""
    golden, fresh = fixture_and_rerun
    assert fresh["env_guard"]["novel_obj_pos"] == [7, 2]
    assert fresh["env_guard"]["orig_obj_pos"] is None
    assert golden["env_guard"] == fresh["env_guard"]


def test_construction_bitwise(fixture_and_rerun):
    golden, fresh = fixture_and_rerun
    _assert_equal("post_construction", golden["post_construction"], fresh["post_construction"])


def test_training_batches_bitwise(fixture_and_rerun):
    golden, fresh = fixture_and_rerun
    _assert_equal("batches", golden["batches"], fresh["batches"])
    _assert_equal("post_train", golden["post_train"], fresh["post_train"])


def test_control_net_untouched_by_training(fixture_and_rerun):
    """pN_control must never train during the novel-object exposure."""
    golden, fresh = fixture_and_rerun
    _assert_equal(
        "pN_control frozen",
        fresh["post_construction"]["pN_control"],
        fresh["post_train"]["pN_control"],
    )


def test_eval_trial_and_metrics_bitwise(fixture_and_rerun):
    golden, fresh = fixture_and_rerun
    _assert_equal("test_trial", golden["test_trial"], fresh["test_trial"])
    _assert_equal("object_learning", golden["object_learning"], fresh["object_learning"])

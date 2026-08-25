"""Bitwise gate for the TRAINING path, and the counterpart of
`../curious-george-questions/tests/golden_omt/test_golden_omt.py`.

Scope: given the same seed, does two updates of collect + policy update +
world-model step move every tensor exactly as the pinned fixture says? It is
not a check that training works well - only that a refactor changed no
numerics.

`capture_golden.py` has compared-by-default since `45d3afd`, but it is not a
`test_*.py`, so pytest never collected it and nothing ran it. It went stale at
`d275149` and stayed stale for three days; that commit's dynamics change is
recorded in the FIXTURE VERSIONS block there. This file is what makes the gate
fire.

Runtime ~6 s, so it stays in the default gate rather than behind `slow`.
"""

import pytest
import torch

from tests.golden.capture_golden import OUT, build_fixture, compare_fixtures

FIXTURE = OUT


@pytest.fixture(scope="module")
def reference_and_fresh():
    from pathlib import Path

    path = Path(FIXTURE)
    assert path.exists(), (
        f"missing {path}. Capture it from a tree whose dynamics are reviewed:\n"
        f"  uv run python tests/golden/capture_golden.py --out {path}"
    )
    return torch.load(path, weights_only=False), build_fixture()


@pytest.mark.parametrize("section", ["rounds", "acmodel_state", "prnn_state"])
def test_training_path_bitwise(reference_and_fresh, section):
    """Every leaf under `section` is bit-identical to the fixture."""
    reference, fresh = reference_and_fresh
    bad = compare_fixtures(reference[section], fresh[section], section)
    assert not bad, f"{len(bad)} diverging leaves:\n" + "\n".join(f"  {b}" for b in bad[:20])


def test_fixture_matches_this_torch(reference_and_fresh):
    """A torch bump can move float64 reductions, so a mismatch under one is
    not evidence of a code change. Fail loudly rather than let a red gate above
    be misread."""
    reference, fresh = reference_and_fresh
    assert reference["meta"]["torch"] == fresh["meta"]["torch"], (
        f"fixture captured under torch {reference['meta']['torch']}, running "
        f"{fresh['meta']['torch']} - re-capture before trusting a diff above"
    )


def test_rollout_consumes_rng_identically(reference_and_fresh):
    """The rollout is upstream of the update, so it is the half that isolates
    'the policy update changed' from 'the environment interaction changed'.
    `d275149` moved the latter and not the former; a future change that moves
    this is a much bigger deal."""
    reference, fresh = reference_and_fresh
    keys = ("actions", "locs", "SRs", "curious_rewards", "rewards")
    bad = [
        b
        for key in keys
        for b in compare_fixtures(
            reference["rounds"][0][key], fresh["rounds"][0][key], f"rounds[0].{key}"
        )
    ]
    assert not bad, "round 0 rollout diverged:\n" + "\n".join(f"  {b}" for b in bad[:20])

"""One evaluation from reference weights, pinned bitwise.

The counterpart to `test_golden.py`. That one builds from a seed and trains;
this one LOADS a trained checkpoint and evaluates. Together they separate two
failure modes a cosmetic refactor can cause:

    test_golden       "same seed => same trajectory"  - catches moved RNG
    test_golden_eval  "same weights => same metrics"  - catches a metric whose
                      MEANING changed, with training reproducibility irrelevant

`capture_golden_eval.py` owns the fixture and the reference weights; see its
docstring for how to re-pin against a different checkpoint.

Runtime ~1 min.
"""

import numpy as np
import pytest
import torch

from tests.golden.capture_golden import compare_fixtures
from tests.golden.capture_golden_eval import CKPT_DIR, OUT, build_fixture

#: Every metric the fixture pins, so a failure names which one moved rather
#: than reporting "the fixture differs".
METRICS = (
    "prnn_loss",
    "mi_policy",
    "sRSA",
    "SWdist",
    "SI_per_unit",
    "SI_units_total",
    "SI_units_zeroed",
    "SI_mean_active_only",
)


@pytest.fixture(scope="module")
def reference_and_fresh():
    assert CKPT_DIR.is_dir(), (
        f"reference weights missing at {CKPT_DIR}. They are TRACKED in this repo; "
        "a clean clone should have them."
    )
    assert OUT.exists(), (
        f"missing {OUT}. Capture it from a tree whose behaviour is reviewed:\n"
        f"  uv run python tests/golden/capture_golden_eval.py --recapture"
    )
    return torch.load(OUT, weights_only=False), build_fixture()


@pytest.mark.parametrize("metric", METRICS)
def test_metric_is_bitwise(reference_and_fresh, metric):
    """Each of the five reported quantities reproduces exactly."""
    reference, fresh = reference_and_fresh
    bad = compare_fixtures(
        reference["metrics"][metric], fresh["metrics"][metric], f"metrics.{metric}"
    )
    assert not bad, "\n".join(f"  {b}" for b in bad[:10])


def test_the_rollout_behind_the_metrics_is_bitwise(reference_and_fresh):
    """Pinned separately so a failure says WHERE it diverged: a metric moving
    with an identical rollout is a metric bug; the rollout moving is a
    collection bug. Without this the two are indistinguishable."""
    reference, fresh = reference_and_fresh
    bad = compare_fixtures(reference["rollout"], fresh["rollout"], "rollout")
    assert not bad, "\n".join(f"  {b}" for b in bad[:10])


def test_no_metric_is_nan(reference_and_fresh):
    """A NaN fixture is worse than none: `torch.equal` is False for NaN, so it
    would fail every run and look like a regression.

    This is not hypothetical. Reusing the rollout's hidden states straight out
    of the tracker gave NaN sRSA and SWdist, because a reset zeroes the SR and
    `calculateRSA_space` uses cosine distance - undefined on a zero vector. The
    onset trim in `capture_golden_eval.py` is what fixes it."""
    _, fresh = reference_and_fresh
    for name in ("prnn_loss", "mi_policy", "sRSA", "SWdist", "SI_mean_active_only"):
        value = float(np.asarray(fresh["metrics"][name]))
        assert np.isfinite(value), f"{name} is {value}"


def test_the_weights_are_the_pinned_ones(reference_and_fresh):
    """A fixture captured against different weights is a different experiment.
    Fail loudly rather than reporting every metric as a regression."""
    reference, fresh = reference_and_fresh
    assert reference["meta"]["ckpt"] == fresh["meta"]["ckpt"], (
        f"fixture pinned against {reference['meta']['ckpt']}, running "
        f"{fresh['meta']['ckpt']} - re-capture before trusting any diff"
    )


def test_fixture_matches_this_torch(reference_and_fresh):
    """A torch bump can move float reductions, so a mismatch under one is not
    evidence of a code change."""
    reference, fresh = reference_and_fresh
    assert reference["meta"]["torch"] == fresh["meta"]["torch"], (
        f"fixture captured under torch {reference['meta']['torch']}, running "
        f"{fresh['meta']['torch']}"
    )

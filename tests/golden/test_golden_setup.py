"""Every live composition builds the same objects it did before the config rewrite.

TEMPORARY, for the Hydra -> dataclass migration only; delete it with
`capture_golden_setup.py` once that lands.

The other two golden gates build their objects from literal kwargs, so they are
a valid oracle across a config rewrite - and cover none of it. This one is the
opposite: it exercises `training/setup.py`, where the translation from config
to constructor actually happens, and where a rewrite breaks things.

Proven to fail: changing `adam_eps=cfg.rl.optim_eps` to `cfg.rl.lr` - a
plausible wrong-key slip - is reported per composition, with `ultra` correctly
showing its own lr rather than the default's.

Runtime ~9 s.
"""

import pytest
import torch

from tests.golden.capture_golden_setup import COMPOSITIONS, OUT, build_fixture, compare


@pytest.fixture(scope="module")
def reference_and_fresh():
    assert OUT.exists(), (
        f"missing {OUT}. It MUST be captured on the pre-migration tree:\n"
        f"  uv run python tests/golden/capture_golden_setup.py --recapture"
    )
    return torch.load(OUT, weights_only=False), build_fixture()


@pytest.mark.parametrize("composition", sorted(COMPOSITIONS))
def test_composition_builds_identically(reference_and_fresh, composition):
    """Constructor kwargs, module weights and derived budget, per composition."""
    reference, fresh = reference_and_fresh
    bad = compare(
        {composition: reference[composition]}, {composition: fresh[composition]}
    )
    assert not bad, "\n".join(f"  {b}" for b in bad[:20])


def test_the_fixture_covers_every_live_composition():
    """A composition dropped from COMPOSITIONS would silently stop being gated,
    and this gate exists precisely for the compositions `slurm/` launches."""
    reference = torch.load(OUT, weights_only=False)
    missing = set(COMPOSITIONS) - set(reference)
    assert not missing, f"fixture has no entry for: {sorted(missing)}"
    assert {"default", "multienv", "ultra"} <= set(reference)


def test_fixture_matches_this_torch(reference_and_fresh):
    """Weight hashes are over raw bytes, so a torch bump can move them without
    any code change; a mismatch under one is not evidence of a regression."""
    reference, fresh = reference_and_fresh
    assert reference["meta"]["torch"] == fresh["meta"]["torch"], (
        f"fixture captured under torch {reference['meta']['torch']}, running "
        f"{fresh['meta']['torch']}"
    )

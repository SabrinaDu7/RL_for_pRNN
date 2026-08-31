"""The CE loss at the repo boundary: config, constructor kwargs, adapter math.

The fork owns `predCE` and gates its own math (pRNN_new/tests/test_ce_loss.py).
What lives HERE is everything this repo wraps around it, each with its own way
to fail silently: the config validation, `prediction_loss_kwargs` (the ONE
constructor home), the adapter's surprisal reduction in both tensor layouts
that reach it (serial `(T, X)` at feature_dim=1 and batched `(1, L, X, B)` at
feature_dim=2 - a transposed reshape gives plausible wrong numbers, not a
crash), and the argmax render round-trip.
"""

import dataclasses

import pytest
import torch

from curious_george.configs import ArchPrnnCfg, PredLoss
from curious_george.envs.palette import TILE_CLASS_NAMES, TILE_VOCABULARY, vocab_tensor
from curious_george.log_and_store.storage import prediction_loss_kwargs

N_TILES = 49
C = len(TILE_VOCABULARY)


class _ObsSize:
    def __init__(self, size: int):
        self._size = size

    def getObsSize(self) -> int:
        return self._size


# --- config validation ------------------------------------------------------


def test_focal_gamma_is_refused_under_mse():
    with pytest.raises(ValueError, match="focal_gamma"):
        ArchPrnnCfg(loss=PredLoss.MSE, focal_gamma=2.0)


@pytest.mark.parametrize("gamma", [0.0, -1.0])
def test_non_positive_focal_gamma_is_refused(gamma):
    with pytest.raises(ValueError, match="focal_gamma"):
        ArchPrnnCfg(loss=PredLoss.CE, focal_gamma=gamma)


def test_valid_focal_gamma_constructs():
    assert ArchPrnnCfg(loss=PredLoss.CE, focal_gamma=2.0).focal_gamma == 2.0


# --- prediction_loss_kwargs, the one constructor home -----------------------


def test_mse_kwargs_are_empty():
    """{} keeps the constructor call byte-identical to pre-CE (goldens)."""
    assert prediction_loss_kwargs(ArchPrnnCfg(), _ObsSize(147)) == {}


def test_ce_kwargs_carry_the_full_contract():
    kw = prediction_loss_kwargs(ArchPrnnCfg(loss=PredLoss.CE), _ObsSize(147))
    assert kw["losstype"] == "predCE"
    assert kw["readout"] == "logits"
    assert kw["output_size"] == N_TILES * C
    assert torch.equal(kw["loss_kwargs"]["vocab"], vocab_tensor())
    assert "focal_gamma" not in kw["loss_kwargs"]


def test_ce_focal_gamma_is_threaded():
    kw = prediction_loss_kwargs(
        ArchPrnnCfg(loss=PredLoss.CE, focal_gamma=2.0), _ObsSize(147)
    )
    assert kw["loss_kwargs"]["focal_gamma"] == 2.0


def test_non_rgb_obs_size_is_refused():
    with pytest.raises(AssertionError, match="not /3"):
        prediction_loss_kwargs(ArchPrnnCfg(loss=PredLoss.CE), _ObsSize(148))


# --- adapter surprisal parity, both layouts ---------------------------------


class _FakeCE:
    """Stands in for `pN.loss_fn`: only what `_prediction_errors` touches."""

    def __init__(self):
        self.vocab = vocab_tensor()

    def targets_for(self, pixels: torch.Tensor, *, check: bool) -> torch.Tensor:
        dist = (pixels.unsqueeze(-2) - self.vocab).abs().sum(-1)
        mindist, classes = dist.min(-1)
        assert float(mindist.max()) < 1e-3
        return classes


def _adapter_errors(obs_pred, obs_next, *, feature_dim):
    """`PRNNAdapter._prediction_errors` on a minimal stand-in instance."""
    from curious_george.models.prnn_adapter import PRNNAdapter

    stub = type("Stub", (), {})()
    stub.ce = True
    stub.vocab = vocab_tensor()
    stub.pN = type("PN", (), {})()
    stub.pN.loss_fn = _FakeCE()
    return PRNNAdapter._prediction_errors(
        stub, obs_pred, obs_next, feature_dim=feature_dim
    )


def _reference_surprisal(logits_t, targets_t):
    """Independent spelling: F.cross_entropy per tile, summed over tiles."""
    return torch.nn.functional.cross_entropy(
        logits_t.reshape(-1, C), targets_t.reshape(-1), reduction="none"
    ).reshape(targets_t.shape).sum(-1)


def _random_case(T: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    targets = torch.randint(0, C, (T, N_TILES), generator=g)
    pixels = vocab_tensor()[targets].reshape(T, -1)  # (T, 147)
    logits = torch.randn(T, N_TILES, C, generator=g)
    return targets, pixels, logits


def test_serial_layout_matches_the_reference():
    targets, pixels, logits = _random_case(T=12, seed=0)
    got = _adapter_errors(logits.reshape(12, -1), pixels, feature_dim=1)
    assert torch.allclose(got, _reference_surprisal(logits, targets), atol=1e-5)


def test_batched_layout_matches_the_reference():
    """(1, L, X, B): the layout the curiosity forward reduces at feature_dim=2."""
    B, L = 3, 4
    targets, pixels, logits = _random_case(T=B * L, seed=1)
    ref = _reference_surprisal(logits, targets).reshape(B, L)
    pix_b = pixels.reshape(1, B, L, -1).movedim(1, 3)  # (1, L, 147, B)
    log_b = logits.reshape(1, B, L, -1).movedim(1, 3)  # (1, L, 343, B)
    got = _adapter_errors(log_b, pix_b, feature_dim=2)  # (1, L, B)
    assert torch.allclose(got[0].transpose(0, 1), ref, atol=1e-5)


def test_out_of_vocabulary_target_asserts():
    _, pixels, logits = _random_case(T=2, seed=2)
    pixels[0, 0] = 0.5
    with pytest.raises(AssertionError):
        _adapter_errors(logits.reshape(2, -1), pixels, feature_dim=1)


# --- render round-trip (needs the installed fork) ---------------------------


def test_render_round_trips_one_hot_logits():
    """Perfect logits at vocab targets must render to the exact vocab pixels."""
    from prnn.utils.lossFuns import predCE

    loss = predCE(vocab_tensor())
    targets, pixels, _ = _random_case(T=5, seed=3)
    onehot = torch.full((5, N_TILES, C), -20.0)
    onehot.scatter_(2, targets.unsqueeze(-1), 20.0)
    rendered = loss.render(onehot.reshape(5, -1))
    assert torch.allclose(rendered, pixels, atol=1e-6)
    assert tuple(TILE_VOCABULARY) == TILE_CLASS_NAMES

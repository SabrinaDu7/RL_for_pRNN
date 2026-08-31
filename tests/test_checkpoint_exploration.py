"""The offline exploration series reads the archive the loop actually writes.

A real (tiny) training run archives two (pRNN, policy) pairs; the scorer must
pair them by step, rebuild the stack through the run's own checkpoint-loading
path, and produce seeded, repeatable rows. The archived-policy series was
write-only from 2026-08-28 until this consumer.
"""

import dataclasses
import shutil
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
from curious_george.evaluation.checkpoint_series import (
    exploration_points,
    score_exploration_series,
)
from curious_george.training.loop import save_checkpoint
from curious_george.training.setup import setup_training
from curious_george.utils.checkpoints import StatusCkptKeys
from tests.small_config import small_config


def _cfg():
    return small_config(
        backend=EnvBackend.DEVICE,
        num_envs=4,
        episodes_per_env=1,
        episode_steps=16,
        env=EnvCfg(
            content=EnvContent(
                kinds=tuple(LandmarkKind(s, impassable=True) for s in ("x", "plus", "block3"))
            ),
            source=Uniform(n=3, seed=7),
            set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
        ),
        evals=frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )


@pytest.fixture(scope="module")
def archived_run(tmp_path_factory):
    """A real two-checkpoint run: train a little, archive, train, archive."""
    run_dir = tmp_path_factory.mktemp("run")
    cfg = _cfg()
    comps = setup_training(cfg)
    run_ctx = SimpleNamespace(model_dir=str(run_dir) + "/")
    for step in (64, 128):
        exps, _ = comps.algo.collect_experiences()
        comps.algo.update_parameters(exps=exps)
        save_checkpoint(cfg, comps, run_ctx, num_frames=step, update=step // 64, archive=True)
    comps.envs.close()
    return cfg, run_dir


def test_points_pair_the_two_series_by_step(archived_run):
    _, run_dir = archived_run
    pairs = exploration_points(run_dir)
    assert [s for s, _, _ in pairs] == [64, 128]
    for step, prnn_path, policy_path in pairs:
        assert f"{step:010d}" in prnn_path.name and f"{step:010d}" in policy_path.name


def test_series_scores_every_pair_and_is_seeded_repeatable(archived_run):
    """One row per pair with the online metric keys - and byte-identical on a
    second scoring, because every point reseeds from cfg.run.seed. Rows that
    differ between scorings would mean each carries its own rollout noise,
    which is exactly what the fixed-probe design here exists to prevent."""
    cfg, run_dir = archived_run
    rows = score_exploration_series(cfg=cfg, run_dir=run_dir, collects_per_point=1)
    assert [r["step"] for r in rows] == [64, 128]
    for r in rows:
        assert r["episodes"] == cfg.collect.num_envs
        assert 0 < r["exploration/coverage"] <= 1
        assert 0 < r["exploration/nauc"] <= 1
        assert "exploration/room_entropy_norm" in r
        assert "exploration/t90_reached" in r
    again = score_exploration_series(cfg=cfg, run_dir=run_dir, collects_per_point=1)
    assert rows == again


def test_a_random_runs_archive_is_refused_with_directions(archived_run):
    """A RANDOM-agent run archives no policy weights; scoring it as a policy
    series would silently measure an untrained-init actor instead."""
    cfg, run_dir = archived_run
    step = 32
    src = exploration_points(run_dir)[0]
    shutil.copy(src[1], run_dir / "checkpoints" / f"predictiveNet_state_step{step:010d}.pt")
    torch.save(
        {StatusCkptKeys.NUM_FRAMES.value: step, StatusCkptKeys.UPDATE.value: 1},
        run_dir / "checkpoints" / f"policy_state_step{step:010d}.pt",
    )
    try:
        with pytest.raises(ValueError, match="no policy weights"):
            score_exploration_series(cfg=cfg, run_dir=run_dir, collects_per_point=1)
    finally:
        (run_dir / "checkpoints" / f"predictiveNet_state_step{step:010d}.pt").unlink()
        (run_dir / "checkpoints" / f"policy_state_step{step:010d}.pt").unlink()

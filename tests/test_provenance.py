"""Gates for `curious_george.log_and_store.provenance`.

The claim under test is not "the module works" but "an artifact cannot be
produced without a record of what produced it". Three commits in five days
changed what this code computes and every one was recorded as prose plus a
SLURM job number (docs/invalid-runs.md); these tests are what stops that
recurring.
"""

import json
import re

import pytest

from curious_george.log_and_store import provenance

SHA = re.compile(r"^[0-9a-f]{40}$")


def test_tracked_packages_all_resolve_to_a_commit():
    """Every repository whose code can change a result resolves to a sha.

    A None here means an artifact would be written claiming to know its
    provenance while not knowing it, which is worse than not writing one.
    """
    unresolved = {
        name: source
        for name in provenance.TRACKED_PACKAGES
        if not SHA.match((source := provenance.resolve_package(name)).commit or "")
    }
    assert not unresolved, f"no commit resolved for: {unresolved}"


def test_recorded_prnn_commit_is_the_installed_one():
    """The world model is the dependency most able to change a number silently,
    and it is pinned by BRANCH in pyproject.toml, so the recorded commit must
    come from the installed distribution rather than from the pin."""
    import importlib.metadata

    installed = json.loads(
        importlib.metadata.distribution("prnn").read_text("direct_url.json") or "{}"
    )
    assert provenance.resolve_package("prnn").commit == installed["vcs_info"]["commit_id"]


def test_a_floating_pin_is_recorded_as_such():
    """`prnn` is pinned by branch, so `requested` is a branch name and the
    resolved commit can move without anything in this tree changing. Recording
    it is how that risk stays visible in every artifact."""
    source = provenance.resolve_package("prnn")
    assert source.origin == "vcs"
    assert source.requested == "sdu/rl-integration", (
        f"prnn pin changed to {source.requested!r}; if it is now a sha, this "
        "test should assert that instead - the drift it guards is gone"
    )


def test_write_then_read_roundtrips(tmp_path):
    provenance.write(tmp_path, kind="dataset", params={"seed": 7})
    record = provenance.read(tmp_path)
    assert record["kind"] == "dataset"
    assert record["params"] == {"seed": 7}
    assert set(record["commits"]) == set(provenance.TRACKED_PACKAGES)
    assert record["created"].endswith("+00:00")


def test_write_leaves_no_partial_file(tmp_path):
    """Written through a temporary and os.replace, so a concurrent reader sees
    the whole record or no file. `envs/obs_bank.py:95` does the opposite and a
    reader there can load a truncated cache."""
    provenance.write(tmp_path, kind="checkpoint")
    assert [p.name for p in tmp_path.iterdir()] == [provenance.FILENAME]


def test_read_raises_when_absent(tmp_path):
    """An artifact with no provenance is one whose numbers cannot be placed on
    any timeline; returning a default would hide exactly that."""
    with pytest.raises(FileNotFoundError, match="predates provenance"):
        provenance.read(tmp_path)


def test_input_artifact_chains_through_a_producer(tmp_path):
    producer = tmp_path / "run"
    producer.mkdir()
    provenance.write(producer, kind="training", params={"run_name": "r"})
    (producer / "predictiveNet_state.pt").touch()

    described = provenance.input_artifact(producer / "predictiveNet_state.pt")
    assert described["exists"] is True
    assert described["provenance"]["kind"] == "training"
    assert set(described["provenance"]["commits"]) == set(provenance.TRACKED_PACKAGES)


def test_input_artifact_marks_a_broken_chain(tmp_path):
    """An input that predates provenance reports `None` rather than being
    omitted: the record should say the chain breaks HERE."""
    orphan = tmp_path / "old_ckpt"
    orphan.mkdir()
    described = provenance.input_artifact(orphan)
    assert described["provenance"] is None
    assert described["exists"] is True


def test_setup_run_writes_provenance(tmp_path, monkeypatch):
    """The gate the whole phase exists for: the training entry point cannot
    produce a run directory without one."""
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from curious_george.training.setup import setup_run

    monkeypatch.setenv("RL_STORAGE", str(tmp_path))
    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main", overrides=["logging.wandb_log=false"])

    run_ctx = setup_run(cfg)
    record = provenance.read(run_ctx.model_dir)
    assert record["kind"] == "training"
    assert record["params"]["run_name"] == run_ctx.run_name
    # the resolved config, not a config still carrying ${...}
    assert record["params"]["config"]["exp"]["seed"] == cfg.exp.seed
    assert "${" not in json.dumps(record["params"]["config"])

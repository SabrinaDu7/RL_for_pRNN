"""What produced an artifact, written beside it as `provenance.json`.

Every artifact this project makes - a training run, a task run, a checkpoint, a
collected dataset - gets one of these. It has two halves:

* a **fixed** half that is identical everywhere and makes the artifact
  self-identifying: when, what kind, and the resolved commit of every repository
  whose code ran;
* a **question-specific** half (`params`) carrying whatever it takes to re-run
  this exact thing - the resolved config, the seed, and every input artifact
  named by path.

Provenance composes. An artifact records its inputs' paths in `params`, and each
of those paths has its own `provenance.json`, so a chain of artifacts is a chain
of these files.

This exists because three (soon four) commits in five days changed what the code
computes, and every one of them was recorded as prose plus a SLURM job number -
so placing an existing checkpoint on a timeline required having written its job
number down at launch. See `docs/invalid-runs.md`. A file that names its own
commits turns each of those lines into a property of the artifact instead.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

FILENAME = "provenance.json"

#: Repositories whose code can affect what any artifact contains.
TRACKED_PACKAGES: tuple[str, ...] = ("rl-for-prnn", "prnn", "minigrid")

ArtifactKind = Literal["training", "task", "checkpoint", "dataset", "analysis"]


@dataclass(frozen=True)
class GitSource:
    """Where one package's code came from.

    commit:    resolved sha, or None when the package is not from git.
    requested: the ref that was ASKED for. A branch name here means the pin
               floats - `uv sync` can resolve it to a different commit without
               anything in the tree changing - so it is recorded rather than
               discarded.
    dirty:     uncommitted changes, knowable only for a local checkout.
    origin:    "vcs" (installed from git), "worktree" (editable checkout),
               "unknown" (neither, e.g. a wheel from PyPI).
    """

    commit: str | None
    requested: str | None
    dirty: bool | None
    origin: Literal["vcs", "worktree", "unknown"]


def _git(*args: str, cwd: Path) -> str | None:
    """`git *args` in `cwd`, or None if git fails or there is no repository."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _direct_url(package: str) -> dict[str, Any]:
    """The installer's record of where a distribution came from (PEP 610)."""
    try:
        blob = importlib.metadata.distribution(package).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return {}
    try:
        return json.loads(blob or "{}")
    except json.JSONDecodeError:
        return {}


def resolve_package(package: str) -> GitSource:
    """Resolve one installed package to a commit.

    Two cases, both real here: `prnn` and `minigrid` are installed FROM git and
    carry `vcs_info`; `rl-for-prnn` is an editable install of a local checkout
    and carries only a path, so the commit comes from git in that directory.
    """
    metadata = _direct_url(package)
    vcs = metadata.get("vcs_info", {})
    if vcs.get("commit_id"):
        return GitSource(
            commit=vcs["commit_id"],
            requested=vcs.get("requested_revision"),
            dirty=None,  # the installed copy is a checkout of that sha by construction
            origin="vcs",
        )

    url: str = metadata.get("url", "")
    if url.startswith("file://"):
        source_dir = Path(url[len("file://") :])
        commit = _git("rev-parse", "HEAD", cwd=source_dir)
        if commit is not None:
            status = _git("status", "--porcelain", cwd=source_dir)
            return GitSource(
                commit=commit,
                requested=_git("rev-parse", "--abbrev-ref", "HEAD", cwd=source_dir),
                dirty=bool(status),
                origin="worktree",
            )

    return GitSource(commit=None, requested=None, dirty=None, origin="unknown")


def _host() -> dict[str, Any]:
    """Where it ran. SLURM ids are how a cluster run is found in the logs."""
    accelerator = None
    try:
        import torch

        if torch.cuda.is_available():
            accelerator = torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001 - provenance must never break the run it describes
        pass
    return {
        "node": platform.node(),
        "accelerator": accelerator,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    }


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return versions


def build(
    *,
    kind: ArtifactKind,
    params: dict[str, Any] | None = None,
    packages: tuple[str, ...] = TRACKED_PACKAGES,
) -> dict[str, Any]:
    """The provenance record for one artifact, as a JSON-safe dict.

    `params` is the question-specific half: everything needed to re-run this,
    including the paths of any input artifacts.
    """
    return {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "kind": kind,
        "commits": {name: asdict(resolve_package(name)) for name in packages},
        "host": _host(),
        "versions": _versions(),
        "argv": sys.argv,
        "params": params or {},
    }


def write(
    directory: str | Path,
    *,
    kind: ArtifactKind,
    params: dict[str, Any] | None = None,
    packages: tuple[str, ...] = TRACKED_PACKAGES,
) -> Path:
    """Write `provenance.json` into `directory` and return its path.

    Writes via a temporary file and `os.replace`, so a reader either sees the
    complete record or no file at all. (`envs/obs_bank.py:95` gets this wrong
    for its cache and a concurrent reader can load a truncated file; do not
    repeat it here.)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(build(kind=kind, params=params, packages=packages), indent=2, default=str)
        + "\n"
    )
    os.replace(temporary, path)
    return path


def resolved_config(cfg: Any) -> dict[str, Any]:
    """A hydra/OmegaConf config as plain data, with interpolations resolved.

    A config that still contains `${...}` is not a record of what ran. Falls
    back to a string rather than losing the whole record to one odd value -
    provenance must never break the run it describes.
    """
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return {"unresolved": str(cfg)}


def input_artifact(path: str | Path) -> dict[str, Any]:
    """Describe one input artifact for the `params` of the thing it feeds.

    This is how provenance composes: the consumer records the producer's path
    AND the producer's own commits, so a chain can be walked without every
    intermediate directory still being on disk. `provenance` is null when the
    input predates this mechanism, which is itself worth recording - it says
    the chain is broken HERE rather than leaving the reader to guess.
    """
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    try:
        record = read(directory)
    except FileNotFoundError:
        record = None
    return {
        "path": str(path),
        "exists": path.exists(),
        "provenance": None if record is None else {
            "created": record.get("created"),
            "kind": record.get("kind"),
            "commits": record.get("commits"),
        },
    }


def read(directory: str | Path) -> dict[str, Any]:
    """Read `provenance.json` from `directory`.

    Raises rather than returning a default: an artifact without provenance is
    one whose numbers cannot be placed on any timeline, and silently treating
    that as "no metadata" is how the current run series lost its own history.
    """
    path = Path(directory) / FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {FILENAME} in {directory}. It predates provenance, or the code "
            f"that wrote it does not call curious_george.log_and_store.provenance.write."
        )
    return json.loads(path.read_text())

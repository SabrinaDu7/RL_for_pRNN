"""The cutover key map: every key a real run logged has an answer.

A query written against one config era finds nothing in the other, and finding
nothing looks exactly like a field that was never set. These check the map is
complete against the actual key sets, not against memory.
"""

import pytest

from curious_george.check.config_keys import (
    FOLDED,
    GONE,
    RENAMED,
    describe,
    to_new,
    to_old,
    translate_config,
)


def test_every_key_the_old_config_had_has_an_answer():
    """Completeness against the parked YAML tree, which is the real key set.

    Reads throwaway/hydra_era/Configs rather than a list typed here, so a key
    that existed and was forgotten fails this instead of silently having no
    translation.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "throwaway" / "hydra_era" / "Configs"
    assert root.is_dir(), "the parked config tree is what this checks against"

    known = set(RENAMED) | set(FOLDED) | set(GONE)
    missing = []
    for yaml in root.rglob("*.yaml"):
        text = yaml.read_text()
        m = re.search(r"^# @package (\S+)", text, re.M)
        ns = m.group(1) if m else None
        # Only the four real namespaces. main.yaml's prose mentions "@package
        # directives," and a looser regex reads that as a namespace - which is
        # how this test first failed, on its own instrument rather than on the map.
        if ns not in ("exp", "rl", "predNet", "logging"):
            continue
        for line in text.splitlines():
            key = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):", line)
            if key and f"{ns}.{key.group(1)}" not in known:
                missing.append(f"{ns}.{key.group(1)}")
    assert not missing, f"no cutover answer for: {sorted(set(missing))}"


def test_renames_round_trip():
    for old, new in RENAMED.items():
        assert to_new(old) == new
        assert to_old(new) == old


def test_folded_keys_are_forward_only():
    """Knowing `collect.backend` does not say whether a reader meant table_env,
    device_env or async_envs. Inventing an inverse would be a guess."""
    for old, (new, _) in FOLDED.items():
        assert to_new(old) == new
        assert to_old(new) is None, f"{new} must not claim a single ancestor"


def test_gone_keys_say_why():
    """A bare None cannot distinguish "derived" from "this never existed"."""
    for key in GONE:
        assert to_new(key) is None
        assert "nothing" in describe(key)


@pytest.mark.parametrize("key,expect", [
    ("predNet.seqdur", "collect.episode_steps"),
    ("exp.seed", "run.seed"),
    ("logging.analysis_every_steps", "eval.analysis_every_steps"),
])
def test_spot_checks(key, expect):
    assert to_new(key) == expect


def test_translate_drops_what_does_not_correspond():
    """Passing an untranslatable key through would put an old name in a new
    record, which is worse than omitting it."""
    out = translate_config({"exp.seed": 2, "rl.frames": 2048}, to="new")
    assert out == {"run.seed": 2}

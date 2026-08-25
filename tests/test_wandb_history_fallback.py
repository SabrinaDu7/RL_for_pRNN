"""`scan_history` refuses on this project's runs; the reader must not.

`Step column '_step' not found in schema` is a documented failure mode here
(docs/claude_logs/speed-30min-2026-08-23.md, methodology): runs whose media and
metrics are logged through their own `wandb.log()` calls rather than the update
log cannot be read with `scan_history`. Every OMT run is one, so the occupancy
reader Q1 depends on hit it immediately.

No network: a stub Run reproduces both branches.
"""

import pytest

from curious_george.io.wandb import _history_rows

METRIC = "Eval/OPA_Occupancy"
ROWS = [{"_step": 3, METRIC: {"path": "media/plotly/a.json"}}]


class _Refuses:
    """scan_history raises the way wandb does; history() serves the same rows."""

    def scan_history(self, keys=None):
        raise RuntimeError("error scanning step range: Step column '_step' not found in schema")

    def history(self, keys=None, pandas=True):
        import pandas as pd

        return pd.DataFrame(ROWS)


class _Works:
    def scan_history(self, keys=None):
        return iter(ROWS)

    def history(self, keys=None, pandas=True):
        raise AssertionError("history() must not be called when scan_history works")


def test_falls_back_when_scan_history_refuses():
    rows = _history_rows(_Refuses(), metric=METRIC, step_key="_step")
    assert [r["_step"] for r in rows] == [3]
    assert isinstance(rows[0][METRIC], dict)


def test_uses_scan_history_when_it_works():
    """The fallback is a fallback. Reading everything through `history()` would
    silently take the 10,000-sample cap on any dense series."""
    assert _history_rows(_Works(), metric=METRIC, step_key="_step") == ROWS


def test_empty_history_is_not_an_error():
    class _Empty(_Refuses):
        def history(self, keys=None, pandas=True):
            import pandas as pd

            return pd.DataFrame([])

    assert _history_rows(_Empty(), metric=METRIC, step_key="_step") == []


def test_non_media_rows_are_dropped():
    """A row whose metric is a scalar rather than a media reference is not an
    occupancy figure and must not be returned as one."""

    class _Mixed(_Refuses):
        def history(self, keys=None, pandas=True):
            import pandas as pd

            return pd.DataFrame([{"_step": 1, METRIC: 0.5}, *ROWS])

    rows = _history_rows(_Mixed(), metric=METRIC, step_key="_step")
    assert [r["_step"] for r in rows] == [3]

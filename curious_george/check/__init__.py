# Recompute a reported number and diff it against what was reported.
#
# `CLAUDE.md` states the checkability rule as `uv run exp check <QID>`. That CLI
# lives in the questions repo; what lives here is the part that belongs to the
# library - fetching a run's metrics and comparing two runs on a matched axis
# with a measured tolerance.

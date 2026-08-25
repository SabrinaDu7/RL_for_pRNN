# `throwaway/ported/` — the reference implementation, kept until its port is gated

Nothing here is deleted yet **on purpose**. You cannot prove a port is
bitwise-equal to an original you deleted, and §2 of
`docs/claude_logs/refactor_plan_2026-08-25.md` makes bitwise equality the
organising requirement of this refactor.

Every file here is either

* **discarded science** — the object / object-vector / trace line, whose results
  are not trusted and which is being restarted from scratch, or
* **a figure generator** for those results, or
* **superseded tooling** whose replacement lives in `curious_george/`.

`CLAUDE.md`'s rule applies with full force: **no result may depend on anything in
`throwaway/`.** These files are here to be diffed against, not run.

This directory is deleted in one commit at the end of the refactor, and that
commit names the gate that proved each file safe to drop.

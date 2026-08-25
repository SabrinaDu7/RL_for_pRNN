"""Where artifacts go, where they come from, and what produced them.

  storage.py     paths under RL_STORAGE, checkpoint read/write, model factories
  provenance.py  the `provenance.json` written beside every artifact
  wandb.py       reading finished runs back out of wandb

`provenance.py` is here because provenance is written beside an artifact and is
part of storing it. It is also the piece the questions repo calls directly:
`resolve_package` is what tells a rendered results document which library
produced it, and `input_artifact` is what lets a chain of artifacts be walked.
"""

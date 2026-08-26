"""What the CONFIG builds, pinned — the gate for the Hydra -> dataclass swap.

TEMPORARY. This exists for one migration and is deleted when it lands.

`capture_golden.py` and `capture_golden_eval.py` build their objects from
literal kwargs - no Hydra, no cfg (see their docstrings). That makes them a
valid oracle across a config rewrite, because they cannot be perturbed by one.
It also means they cover NONE of `training/setup.py`, where 77 of the ~143
config reads in this package live, and which is exactly where a config->
constructor translation bug would land: a key read from the wrong namespace, a
`.get` default that silently disagrees with the YAML, a preset that stops
composing.

So this pins the OTHER half: for each live composition, what `setup_training`
actually constructs.

    kwargs   every argument PredictivePPOAlgo received, captured by wrapping
             __init__ rather than re-deriving the expressions - re-deriving
             would reproduce a translation bug instead of catching it
    weights  sha256 of each module's state_dict, in sorted key order, so a
             changed width or init path shows up even when no scalar moved
    schedule the derived budget, which is what the rewrite re-roots

CAPTURE THIS ON THE HYDRA TREE FIRST, COMMIT IT, THEN MIGRATE. A fixture
captured after the change proves nothing.

    uv run python tests/golden/capture_golden_setup.py            # GATE: compare
    uv run python tests/golden/capture_golden_setup.py --recapture

RE-PINNED ONCE, on 2026-08-26, for exactly two leaves:

    multienv.kwargs.wm_pool_group: 0 -> 8
    ultra.kwargs.wm_pool_group:    0 -> 128

and nothing else - all 33 constructor kwargs, both weight hashes and the whole
derived schedule were identical in all three compositions. Those two are a
SPELLING change, not a semantic one: `prnn_adapter.py` resolves the group with
`g = int(group) if group and group > 0 else obs_b.size(0)`, so the old 0
sentinel meant "the whole batch" and the new value IS the whole batch. The
migration stopped emitting a sentinel whose meaning depended on the rollout
size.

`wandb_log=False` is forced on every composition: it reaches
`PredictiveNet(wandb_log=...)` and would otherwise open a network connection
during a test. It is identical on both sides of the migration, so it cancels.
"""

import hashlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "golden" / "golden_setup_v1.pt"

#: The compositions that actually get launched, from `slurm/` and `justfile`.
#: `episodes_total` is pinned small everywhere: it changes the derived budget
#: (which IS captured) but not a single constructed object, so it keeps the
#: fixture cheap without weakening it.
#: fixture key -> preset name. The keys are the ones the pre-migration fixture
#: was captured under, so the comparison still lines up; `default` was the bare
#: composition, which is now the `reference` preset.
COMPOSITIONS: dict[str, str] = {
    "default": "reference",
    "multienv": "multienv",
    "ultra": "ultra",
}


def _hash_state_dict(module) -> str:
    """sha256 over (name, bytes) in sorted key order.

    Sorted because `state_dict()` ordering follows module registration, which a
    refactor can permute without changing a single weight - that would be a
    false positive here.
    """
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd):
        value = sd[key]
        h.update(key.encode())
        tensor = value.detach().cpu().contiguous() if torch.is_tensor(value) else torch.as_tensor(value)
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _scalar(value):
    """Normalise so the fixture is stable and cross-comparable.

    An OmegaConf `ListConfig` and a plain list must compare equal, and a
    `str, Enum` member and its value must too - those are exactly the
    representation changes the migration makes deliberately, and flagging them
    would drown the translation bugs this is meant to catch.

    NON-SCALARS ARE NAMED, NEVER `repr`d. `preprocess_obss` is a closure whose
    repr carries its memory address, so a repr fallback made the fixture differ
    from itself on every run - caught on the first comparison. A qualified name
    still catches "a different function got passed", which is the only thing
    worth catching here.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value.value if isinstance(value, __import__("enum").Enum) else value
    if isinstance(value, (list, tuple)):
        return [_scalar(v) for v in value]
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', type(value).__name__)}>"
    return f"<{type(value).__name__}>"


def build_fixture() -> dict:
    """Compose each configuration, run `setup_training`, record what it built."""
    from dataclasses import replace

    from curious_george.configs import PRESETS
    from curious_george.rl.algo import PredictivePPOAlgo
    from curious_george.training.schedule import TrainingSchedule
    from curious_george.training.setup import setup_training

    captured: dict = {}
    original_init = PredictivePPOAlgo.__init__

    def recording_init(self, *args, **kwargs):
        # POSITIONAL args are envs/acmodel/prnn/device - objects, not config
        # values - so only the keywords are the config's translation.
        captured["kwargs"] = {k: _scalar(v) for k, v in kwargs.items()}
        return original_init(self, *args, **kwargs)

    fixture: dict = {"meta": {"torch": torch.__version__}}

    for name, preset in COMPOSITIONS.items():
        base = PRESETS[preset][1]
        # wandb off: it reaches PredictiveNet(wandb_log=...) and would otherwise
        # open a network connection. Identical on both sides, so it cancels.
        cfg = replace(base, run=replace(base.run, wandb=False))

        PredictivePPOAlgo.__init__ = recording_init
        try:
            comps = setup_training(cfg)
        finally:
            PredictivePPOAlgo.__init__ = original_init

        cfg = comps.cfg  # the EFFECTIVE config
        schedule = TrainingSchedule.from_config(cfg)
        fixture[name] = {
            "kwargs": captured["kwargs"],
            "weights": {
                "prnn": _hash_state_dict(comps.predictiveNet.pRNN),
                "acmodel": _hash_state_dict(comps.acmodel),
            },
            "schedule": {
                "total_env_steps": schedule.total_env_steps,
                "total_rollouts": schedule.total_rollouts,
                "prnn_grad_steps": schedule.total_prnn_steps,
                "policy_grad_steps": schedule.total_policy_steps,
                "env_steps_per_prnn_step": schedule.env_steps_per_prnn_step,
                "env_steps_per_policy_step": schedule.env_steps_per_policy_step,
            },
            "pastSR": comps.pastSR,
        }
        print(f"  captured {name}: {len(fixture[name]['kwargs'])} kwargs, "
              f"{fixture[name]['schedule']['prnn_grad_steps']:,} pRNN grad steps")

    return fixture


def compare(reference: dict, fresh: dict) -> list[str]:
    """Leaf-by-leaf, naming the composition and the key that moved."""
    bad: list[str] = []
    for name in sorted(set(reference) | set(fresh)):
        if name == "meta":
            continue
        if name not in reference or name not in fresh:
            bad.append(f"{name}: present in only one fixture")
            continue
        for section in ("kwargs", "weights", "schedule"):
            ref, new = reference[name][section], fresh[name][section]
            for key in sorted(set(ref) | set(new)):
                a, b = ref.get(key, "<absent>"), new.get(key, "<absent>")
                if a != b:
                    bad.append(f"{name}.{section}.{key}: {a!r} -> {b!r}")
        if reference[name]["pastSR"] != fresh[name]["pastSR"]:
            bad.append(f"{name}.pastSR: {reference[name]['pastSR']} -> {fresh[name]['pastSR']}")
    return bad


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recapture", action="store_true",
                    help="OVERWRITE the baseline. Only before the migration.")
    args = ap.parse_args()

    fixture = build_fixture()

    if OUT.exists() and not args.recapture:
        bad = compare(torch.load(OUT, weights_only=False), fixture)
        if bad:
            print(f"GOLDEN SETUP MISMATCH ({len(bad)} leaves):")
            for b in bad[:30]:
                print("  ", b)
            sys.exit(1)
        print(f"GOLDEN SETUP OK - every composition builds identically to {OUT}")
        return

    if OUT.exists():
        print(f"WARNING: --recapture given; OVERWRITING {OUT}")
    torch.save(fixture, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

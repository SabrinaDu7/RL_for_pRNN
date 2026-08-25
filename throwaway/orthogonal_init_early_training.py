"""THROWAWAY: does ORTHOGONAL INITIALISATION of the pRNN weights help EARLY training?

No repo result may depend on this file (CLAUDE.md). Nothing under
curious_george/, Configs/ or tests/ is touched: everything is built by the real
`curious_george.training.setup.setup_training`, and the weights are
re-initialised IN PLACE before the first update.

WHAT IS ACTUALLY THERE (traced, not assumed - see the report for file:line):
`predictiveNet.pRNN` is a `MaskedRNN(cell=LayerNormRNNCell, k=5)`. Its whole
parameter set, with the aliases prnn's own optimizer uses:

    W_in  = rnn.cell.weight_ih   (hidden, obs+act)  input projection
    W     = rnn.cell.weight_hh   (hidden, hidden)   RECURRENT matrix
    W_out = outlayer.0.weight    (obs, hidden)      readout, no bias
    bias  = rnn.cell.bias        (hidden,)          aliased to layernorm.mu

There is NO convolution and no encoder in this configuration.

TWO FACTS THAT GOVERN HOW W MAY BE RE-INITIALISED
1. `Architectures.py:146` adds `(1 - 1/neuralTimescale) * I` to W at
   construction. The neural timescale is IMPLEMENTED AS W's DIAGONAL and
   appears nowhere else, so an init that overwrites W without restoring that
   term changes the model, not its initialisation. Every W arm here re-adds it.
2. `LayerNormRNNCell` layer-norms the preactivation `W_in x + W h` before the
   ReLU, so the FORWARD pass is invariant to the overall scale of that sum.
   Gain therefore acts mainly on the BACKWARD pass - which is exactly the
   channel orthogonal recurrent init is supposed to fix over
   `predNet.seqdur=256` steps of BPTT.

Usage:
    uv run python throwaway/orthogonal_init_early_training.py --verify
    uv run python throwaway/orthogonal_init_early_training.py \
        --arm orth_W --seed 1 --updates 260 --spatial --out out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
)

# The run config the task fixes (fast path: world-model, policy and rollout
# CUDA graphs all on).
RUN_OVERRIDES: list[str] = [
    "env=lroom",
    "run=multienv",
    "exp.device_env=True",
    "predNet.batched_wm=True",
    "predNet.wm_pool_group=1",
    "predNet.compile_cell=layer",
    "predNet.cuda_graph=True",
    "rl.cuda_graph=True",
    "exp.rollout_cuda_graph=True",
    "exp.num_envs=128",
    "rl.frames=32768",
    "rl.ppo_batch_size=256",
    "rl.entropy_coef=0",
    "logging.wandb_log=false",
]

# The loop is driven here, so every periodic section of training/loop.py is
# off and cannot contaminate the timing. The spatial eval is run ONCE, at the
# end, explicitly.
QUIET_OVERRIDES: list[str] = [
    "logging.save_every_steps=0",
    "logging.analysis_every_steps=0",
    "logging.plot_every_steps=0",
    "logging.log_every_steps=0",
]

# Cheap enough to build on CPU in seconds; same world model, no device env.
VERIFY_OVERRIDES: list[str] = [
    "env=lroom",
    "exp.num_envs=1",
    "exp.device_env=False",
    "logging.wandb_log=false",
]

PROBE_SEED = 1234  # fixed spatial-eval probe, so arms differ by TRAINING only


# --------------------------------------------------------------------------- #
# the intervention
# --------------------------------------------------------------------------- #

# prnn's own aliases (Architectures.py:133-136). Reaching the parameters by
# these names rather than by module path is deliberate: they are the names
# `PredictiveNet.resetOptimizer` groups the RMSprop by, so an arm named
# "W_in" is exactly the group whose learning rate is scaled by rootk_i.
PRNN_WEIGHTS = ("W_in", "W", "W_out")


@dataclass(frozen=True)
class Arm:
    """One initialisation scheme: which pRNN weights to orthogonalise, at what gain."""

    name: str
    gains: dict[str, float] = field(default_factory=dict)  # subset of PRNN_WEIGHTS
    why: str = ""


ARMS: dict[str, Arm] = {
    a.name: a
    for a in [
        Arm(
            "baseline",
            {},
            "untouched. W_in/W come from thetaRNN.xavier_init - which is NOT "
            "Glorot but U(+/-1/sqrt(fan_in)), torch's default RNN init - and "
            "W_out from nn.Linear's default. Plus (1-1/neuralTimescale)*I on W.",
        ),
        Arm(
            "orth_W",
            {"W": 1.0},
            "THE experiment: norm-preserving orthogonal RECURRENT matrix "
            "(Saxe et al. 2014) at gain 1.0, with the neural-timescale "
            "diagonal re-added. BPTT runs over predNet.seqdur=256 steps.",
        ),
        Arm(
            "orth_W_in",
            {"W_in": 1.0},
            "The layer that consumes the flattened 7x7x3 observation plus the "
            "action code. 500 > input_size, so orthogonal_ gives ORTHONORMAL "
            "COLUMNS: the input norm is carried into the hidden layer exactly.",
        ),
        Arm(
            "orth_W_out",
            {"W_out": 1.0},
            "The readout into the sigmoid. Lowest prior: it is a single "
            "feedforward layer, outside the recurrence entirely.",
        ),
        Arm(
            "orth_all",
            {"W_in": 1.0, "W": 1.0, "W_out": 1.0},
            "All three at once - the arm that should be the union of any "
            "single-weight effect, and the one to read if none of the singles "
            "clears seed spread.",
        ),
    ]
}


def reinit_orthogonal(pN, arm: Arm) -> list[str]:
    """Orthogonalise the named pRNN weights IN PLACE; returns what was touched.

    In place matters twice over: `PredictiveNet.optimizer` already holds these
    Parameter objects, and `predNet.cuda_graph` fingerprints their
    `data_ptr()`s (`_GraphWMTrainer._fingerprint`), so a re-init that
    reallocated would either be ignored by the optimizer or strand the graph.
    `nn.init.orthogonal_` copies into the existing storage.

    W keeps its `(1 - 1/neuralTimescale) * I` term, which is the ONLY place
    the neural timescale is implemented (Architectures.py:146).

    The RNG state is saved and restored, so both arms consume the identical
    random stream from the first rollout onward: the arms differ in the weight
    VALUES and in nothing else.
    """
    prnn = pN.pRNN
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    touched: list[str] = []
    with torch.no_grad():
        for name, gain in arm.gains.items():
            w = getattr(prnn, name)
            nn.init.orthogonal_(w, gain=gain)
            note = ""
            if name == "W":
                decay = 1.0 - 1.0 / prnn.neuralTimescale
                w.add_(torch.eye(w.shape[0], device=w.device, dtype=w.dtype).mul_(decay))
                note = f" + {decay:g}*I (neuralTimescale={prnn.neuralTimescale})"
            touched.append(f"{name}{tuple(w.shape)} gain={gain:g}{note}")
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return touched


def spectrum(pN) -> dict:
    """Singular values of every pRNN weight - the evidence the init applied.

    An orthogonal matrix has ALL singular values equal to its gain (std 0).
    A random matrix's spread over a Marchenko-Pastur bulk is what
    distinguishes it, so `singular_std` is the discriminating number.
    """
    out = {}
    for name in PRNN_WEIGHTS:
        w = getattr(pN.pRNN, name).detach().float().cpu()
        s = torch.linalg.svdvals(w)
        ev = torch.linalg.eigvals(w) if w.shape[0] == w.shape[1] else None
        out[name] = {
            "shape": list(w.shape),
            "std": float(w.std()),
            "singular_min": float(s.min()),
            "singular_max": float(s.max()),
            "singular_mean": float(s.mean()),
            "singular_std": float(s.std()),
            "spectral_radius": None if ev is None else float(ev.abs().max()),
            "data_ptr": getattr(pN.pRNN, name).data_ptr(),
        }
    return out


def grad_norms(pN) -> dict[str, float]:
    """L2 norm of each pRNN weight's gradient, as left by the LAST world-model
    gradient step of the update just run.

    Readable on the graphed path too: `_GraphWMTrainer._capture` calls
    `zero_grad(set_to_none=False)` inside the captured region, so `.grad` is a
    persistent static buffer rather than being dropped after `step()`.
    """
    out = {}
    for name in PRNN_WEIGHTS:
        g = getattr(pN.pRNN, name).grad
        out[name] = float("nan") if g is None else float(g.detach().norm())
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def compose(overrides: list[str]):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
        return hydra_compose(config_name="main", overrides=overrides)


def run_arm(*, arm: Arm, seed: int, updates: int, spatial: bool, extra: list[str]) -> dict:
    from curious_george.training.schedule import TrainingSchedule
    from curious_george.training.setup import setup_training

    overrides = (
        RUN_OVERRIDES
        + QUIET_OVERRIDES
        + [f"exp.seed={seed}", f"rl.episodes_total={updates * 128}"]
        + extra
    )
    cfg = compose(overrides)

    t_setup = time.perf_counter()
    comps = setup_training(cfg)
    setup_s = time.perf_counter() - t_setup

    schedule = TrainingSchedule.from_config(cfg)
    print(schedule.summary(), flush=True)

    before = spectrum(comps.predictiveNet)
    touched = reinit_orthogonal(comps.predictiveNet, arm)
    after = spectrum(comps.predictiveNet)
    assert all(before[k]["data_ptr"] == after[k]["data_ptr"] for k in before), (
        "re-init reallocated a parameter; the optimizer and the CUDA graphs "
        "both hold the old storage"
    )
    print(f"[{arm.name}] re-initialised: {touched or 'nothing (baseline)'}", flush=True)
    for k in PRNN_WEIGHTS:
        print(
            f"  {k:>5} {tuple(before[k]['shape'])} singular values "
            f"before mean={before[k]['singular_mean']:.4f} sd={before[k]['singular_std']:.4f}"
            f"  ->  after mean={after[k]['singular_mean']:.4f} sd={after[k]['singular_std']:.4f}",
            flush=True,
        )

    algo = comps.algo
    saver_start = len(comps.predictiveNet.TrainingSaver)
    keys = ("entropy", "value", "policy_loss", "value_loss", "grad_norm")
    per_update: dict[str, list[float]] = {k: [] for k in (*keys, "seconds")}
    prnn_grad: dict[str, list[float]] = {k: [] for k in PRNN_WEIGHTS}

    t0 = time.perf_counter()
    for u in range(updates):
        t_u = time.perf_counter()
        exps, logs1 = algo.collect_experiences()
        logs = {**logs1, **algo.update_parameters(exps=exps)}
        per_update["seconds"].append(time.perf_counter() - t_u)
        for k in keys:
            per_update[k].append(float(logs[k]))
        for k, v in grad_norms(comps.predictiveNet).items():
            prnn_grad[k].append(v)
        if u % 25 == 0 or u == updates - 1:
            print(
                f"[{arm.name} s{seed}] update {u + 1}/{updates} "
                f"{per_update['seconds'][-1]:.2f}s "
                f"wm_loss={float(comps.predictiveNet.TrainingSaver['loss'].iloc[-1]):.6f} "
                f"|dW|={prnn_grad['W'][-1]:.4g} entropy={per_update['entropy'][-1]:.4f}",
                flush=True,
            )
    loop_s = time.perf_counter() - t0

    # EXACT per-gradient-step world-model loss: recordTrainingTrial appends one
    # row per trainStep, so this is the whole curve at gradient-step resolution
    # (wandb history is capped at 10,000 samples and would decimate it).
    ts = comps.predictiveNet.TrainingSaver
    wm_loss = [float(x) for x in ts["loss"].iloc[saver_start:].tolist()]

    metrics_spatial = None
    if spatial:
        from curious_george.evaluation.spatial import evaluate_spatial_representation

        t_s = time.perf_counter()
        m = evaluate_spatial_representation(
            comps.predictiveNet,
            comps.env,
            comps.ac_agent,
            n_trajs=cfg.exp.eval_trajs,
            traj_timesteps=cfg.predNet.seqdur,
            sleepstd=0.03,
            probe_seed=PROBE_SEED,
        )
        # SI comes back as a per-unit DataFrame; reduced exactly as
        # evaluation/spatial.py:281 reduces it for the multi-room logger.
        import numpy as np

        metrics_spatial = {
            "sRSA": float(m["sRSA"]),
            "SWdist": float(m["SWdist"]),
            "mean_SI": float(np.nanmean(m["SI"])),
            "seconds": round(time.perf_counter() - t_s, 2),
        }

    return {
        "arm": arm.name,
        "gains": arm.gains,
        "why": arm.why,
        "seed": seed,
        "updates": updates,
        "git": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "overrides": overrides,
        "schedule": {
            "wm_steps_per_update": schedule.world_model_steps_per_update,
            "policy_steps_per_update": schedule.policy_steps_per_update,
            "frames_per_update": schedule.frames_per_update,
            "seqdur": int(cfg.predNet.seqdur),
        },
        "setup_s": round(setup_s, 1),
        "loop_s": round(loop_s, 1),
        "loop_s_after_first": round(loop_s - per_update["seconds"][0], 1),
        "reinit": touched,
        "spectrum_before": before,
        "spectrum_after": after,
        "wm_loss_per_gradient_step": wm_loss,
        "prnn_grad_norm_per_update": prnn_grad,
        "per_update": per_update,
        "spatial": metrics_spatial,
    }


def verify() -> None:
    """Build the REAL world model (cheap config, CPU) and show what is in it,
    then what each arm does to its spectrum. No training, no GPU."""
    from curious_george.training.setup import setup_envs, setup_world_model

    cfg = compose(VERIFY_OVERRIDES)
    env = setup_envs(cfg)[0]

    for name, arm in ARMS.items():
        torch.manual_seed(0)
        pN = setup_world_model(cfg, env, wandb_log=False)
        prnn = pN.pRNN
        if name == "baseline":
            print(f"\npRNN class: {type(prnn).__name__}  cell: {type(prnn.rnn.cell).__name__}")
            print(f"obs_size={pN.obs_size} act_size={pN.act_size} hidden={pN.hidden_size} "
                  f"input_size={prnn.input_size} output_size={prnn.output_size} "
                  f"neuralTimescale={prnn.neuralTimescale} actfun={prnn.rnn.cell.actfun}")
            print("named_parameters():",
                  [(n, tuple(p.shape)) for n, p in prnn.named_parameters()])
            print("alias identity:", {
                k: getattr(prnn, k).data_ptr() for k in (*PRNN_WEIGHTS, "bias")
            })
            print("optimizer groups:", [
                (g.get("name"), round(g["lr"], 8), round(g["weight_decay"], 10))
                for g in pN.optimizer.param_groups
            ])
        print(f"\n=== {name}: {arm.why}")
        b = spectrum(pN)
        touched = reinit_orthogonal(pN, arm)
        a = spectrum(pN)
        print("  touched:", touched or "nothing")
        for k in PRNN_WEIGHTS:
            print(
                f"  {k:>5} {tuple(b[k]['shape'])}: std {b[k]['std']:.4f}->{a[k]['std']:.4f} "
                f"| singular mean {b[k]['singular_mean']:.4f}->{a[k]['singular_mean']:.4f} "
                f"sd {b[k]['singular_std']:.4f}->{a[k]['singular_std']:.4f} "
                f"range [{b[k]['singular_min']:.4f},{b[k]['singular_max']:.4f}]"
                f"->[{a[k]['singular_min']:.4f},{a[k]['singular_max']:.4f}]"
                + (
                    f" | spectral radius {b[k]['spectral_radius']:.4f}"
                    f"->{a[k]['spectral_radius']:.4f}"
                    if b[k]["spectral_radius"] is not None
                    else ""
                )
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="print the pRNN's parameters + spectra")
    p.add_argument("--arm", choices=sorted(ARMS), default="baseline")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--updates", type=int, default=260)
    p.add_argument("--spatial", action="store_true", help="one sRSA/SWdist eval at the end")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()

    if args.verify:
        verify()
        return

    import matplotlib

    matplotlib.use("Agg")
    r = run_arm(
        arm=ARMS[args.arm], seed=args.seed, updates=args.updates,
        spatial=args.spatial, extra=args.override,
    )
    print(
        f"\n{args.arm} seed={args.seed}: setup {r['setup_s']}s, loop {r['loop_s']}s "
        f"for {args.updates} updates ({len(r['wm_loss_per_gradient_step'])} world-model "
        f"gradient steps), final wm_loss={r['wm_loss_per_gradient_step'][-1]:.6f}, "
        f"spatial={r['spatial']}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

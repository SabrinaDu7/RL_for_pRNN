"""
`tyro.cli` builds a `Config` from the command line.
Frozen throughout, so the object handed to `provenance.write` is the object training ran on.

Philosophy: We separate content from process/transformation. For example, architecture
(content) and training (process)are separate; training acts on architecture.
Another example: environment (content) versus collecting episodes from the
environment (process) are separate.

KEYWORDS: gradient steps, environment steps, and episodes as measurement
units (instead of previous "rollouts" and "frames").

TRAINING SPEED: Fields marked (SPEED) change wall-clock without changing the
science. Fields marked (SPEED+SEMANTICS) change both - those need a learning
curve to justify them, not a benchmark.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Union

from prnn.utils import ActionEncodingsEnum, pRNNtypes

from curious_george.envs.layouts import (
    MULTI_ROOM_ID,
    Committed,
    Curated,
    EnvContent,
    Frozen,
    EnvDefault,
    Selected,
    EnvShape,
    LandmarkKind,
    RoomRules,
    RoomSetRules,
    RoomSource,
    Symmetry,
    Uniform,
    Vary,
)
from curious_george.utils.enums import AgentInputType, AgentType

#: The project's random-action distribution over (left, right, forward, pickup).
#: Forward-weighted: a uniform walker mostly spins on the spot and covers little
#: (measured: nAUC 0.108 vs 0.224 - `python -m curious_george.envs.action_graph`).
#: THE one home - `ArchPolicyCfg.random_action_probs` defaults to it, and every
#: other consumer (the collector, `get_agent`, the diagnostics probes) imports
#: it. It had four independent spellings until 2026-08-30.
RAND_ACT_PROBA: tuple[float, float, float, float] = (0.15, 0.15, 0.6, 0.1)

# ---------------------------------------------------------------------------
# Enums

class MinigridEnv(str, enum.Enum):
    """Registered gymnasium ids. The registry in minigrid is the authority;
    this mirrors the subset this project runs."""

    LROOM = "MiniGrid-LRoom-v0"
    LROOM_MULTI = "MiniGrid-LRoom-Multi-v0"
    SQUAREROOM = "MiniGrid-SquareRoom-v0"
    SQUAREROOM_MULTI = "MiniGrid-SquareRoom-Multi-v0"
    FOURROOMS_OBJECTS = "MiniGrid-FourRooms-Objects-v0"


class EnvBackend(str, enum.Enum):
    """How environment steps execute. ONE axis, so no combination is invalid.

    Replaces `table_env`, `device_env` and `async_envs` - three booleans whose
    eight states held four legal ones, with "device implies table" spelled as an
    `or` and "device needs more than one instance" as a raise.
    """

    SERIAL = "serial"  # in-process instances, observations rendered per step
    SERIAL_TABLE = "serial_table"  # in-process, static transition/observation tables
    ASYNC = "async"  # worker processes, rendered observations
    ASYNC_TABLE = "async_table"  # worker processes, table observations
    DEVICE = "device"  # one batched accelerator-resident table state machine

    @property
    def tabled(self) -> bool:
        return self in (EnvBackend.SERIAL_TABLE, EnvBackend.ASYNC_TABLE, EnvBackend.DEVICE)

    @property
    def batched(self) -> bool:
        return self is EnvBackend.DEVICE

    @property
    def measures_return(self) -> bool:
        """DEVICE cannot: it has no per-instance episode bookkeeping, so
        extrinsic return is unmeasurable there rather than zero."""
        return self is not EnvBackend.DEVICE


class PredLoss(str, enum.Enum):
    """The world model's prediction objective - ONE switch, and everything
    follows from it: the upstream loss (`predMSE`/`predCE`), the readout head
    (147-sigmoid pixels vs n_tiles x n_classes logits over
    `envs/palette.py::TILE_VOCABULARY`), and the curiosity reward's error
    measure (pixel MSE vs per-step summed surprisal in nats). No half-states:
    a run is one or the other, visible in its config and provenance."""

    MSE = "mse"
    CE = "ce"


class CompileMode(str, enum.Enum):
    """`torch.compile` on the recurrent cell. Was typed `bool` in YAML and read
    as `bool | str` in code, because a launcher passes the string `layer`."""

    OFF = "off"
    CELL = "cell"  # fuse the LayerNorm chain inside one step
    LAYER = "layer"  # compile the whole recurrence over an episode

    @property
    def adapter_arg(self) -> "bool | str":
        """What `PRNNAdapter` expects, which is a `bool | str` and not an enum.

        It tests `if compile_cell:` (models/prnn_adapter.py), so the STRING
        "off" is TRUTHY there and would silently enable compilation for a config
        that says off - caught by a stray inductor warning in a smoke run. OFF
        must cross that boundary as False.
        """
        return False if self is CompileMode.OFF else self.value


class RewardAlignment(str, enum.Enum):
    """Which observation the curiosity reward credits an action with."""

    LEGACY = "legacy"  # the pre-action observation (historical)
    NEXT_OBS = "next_obs"  # the observation the action produced


class EvalKind(str, enum.Enum):
    """Names the eval driver iterates over, replacing four independent booleans
    read at four call sites plus one implicit branch."""

    SPATIAL_ONPOLICY = "spatial_onpolicy"
    SPATIAL_OFFPOLICY = "spatial_offpolicy"
    SPATIAL_MULTIROOM = "spatial_multiroom"
    BEHAVIOUR = "behaviour"
    TRAJECTORY_PLOT = "trajectory_plot"


class SpatialEvalPath(str, enum.Enum):
    """Which spatial-evaluation implementation runs."""

    POOLED = "pooled"  # n_trajs trajectories of episode_steps, pooled
    LEGACY_DECODER = "legacy_decoder"  # prnn's own rollout plus a decoder fit


# ---------------------------------------------------------------------------
# Layouts: a union, so the pool's parameters cannot exist without the pool.


@dataclass(frozen=True)
class SingleLayout:
    """One room. The control for every multi-room number: same environment
    class, same landmarks, one room instead of several - so a change in sRSA is
    attributable to room COUNT rather than to scale or schedule.
    """

    index: int = 0 # selects the room in `envs/layouts.py`

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")


@dataclass(frozen=True)
class FrozenLayouts:
    """The frozen set for the base room (envs/layouts.py)."""


@dataclass(frozen=True)
class LayoutPool:
    """A seeded uniform sample of the admissible set. `seed` names the pool, so
    a run is reproducible from it alone."""

    size: int = 500
    seed: int = 20260813


#: Which rooms a multi-room run trains on. A UNION, not a mode string plus two
#: orphan fields: `size` and `seed` exist only under `LayoutPool`, so asking for
#: a pool size on a frozen set is not expressible. The old config carried both
#: on every env and raised AttributeError when `layouts` was not "pool".
EnvLayoutSpec = Union[SingleLayout, FrozenLayouts, LayoutPool]


# ---------------------------------------------------------------------------
# pRNN


@dataclass(frozen=True)
class EnvCfg:
    """The world: its SHAPE, its CONTENT, and which rooms a run trains over.

    One class, not a subclass per room. Shape is a FIELD, so a new room is a
    value rather than a type - which is what makes shape x content mixable. The
    hierarchy existed to hang multi-room-only fields off a subclass, and those
    fields now live on the source that owns them.
    """

    shape: EnvShape = field(default_factory=EnvShape)
    content: EnvContent = field(default_factory=EnvContent)
    room_rules: RoomRules = field(default_factory=RoomRules)
    """WITHIN one room: when is a placement legal?"""
    set_rules: RoomSetRules = field(default_factory=RoomSetRules)
    """BETWEEN rooms: what differs, and what is held constant?"""
    source: RoomSource = field(default_factory=EnvDefault)
    """Where the room set comes from. EnvDefault means the environment class
    supplies its own landmarks - what nearly every checkpoint here trained on."""
    indices: tuple[int, ...] | None = None
    """Subset of the resolved set. `(0,)` is the single-room control, and it
    works against ANY source rather than being its own."""

    see_through_walls: bool | None = None
    """None keeps the environment's own default (True). False is occlusion,
    which changes what the agent can perceive and needs its own baseline."""

    novel_object: tuple[int, int] | None = None
    """One extra object at this cell - the Object Memory Task's manipulation.

    NOT a landmark: the room's landmarks come from shape/content/source, and
    this is placed on top of them at a cell the experiment picks. The
    environment renders it as `FloorBright` and derives its colour itself
    (green), so colour is not a parameter here - see minigrid `envs/Lroom.py`.

    Restored 2026-08-27. It had been dropped from the config while
    `training/setup.py` kept reading it through `getattr(cfg.env,
    "new_obj_pos", None)` - a default that turned a missing field into "no
    object" instead of an error, so no run could place one and nothing said so.
    """

    def __post_init__(self) -> None:
        if self.novel_object is not None and self.novel_object not in self.shape.walkable:
            # A novel object outside the room is invisible, and an OMT run with
            # an invisible object looks exactly like a null result.
            raise ValueError(
                f"novel_object {self.novel_object} is not a walkable cell of "
                f"{self.shape.room}"
            )

    @property
    def has_room_set(self) -> bool:
        """False only for EnvDefault: there is no set, the env class owns the
        content."""
        return not isinstance(self.source, EnvDefault)

    @property
    def reachable_cells(self) -> frozenset[tuple[int, int]]:
        """Every cell the agent can occupy in AT LEAST ONE room this run holds.

        A property of the world, so it can be quoted without a training run. It
        is the support `loc_entropy` is taken over - see
        `envs.layouts.pooled_walkable` for why the union and not one room's set.
        """
        from curious_george.envs.layouts import pooled_walkable, resolve_rooms

        rooms = resolve_rooms(
            shape=self.shape, content=self.content, source=self.source,
            room_rules=self.room_rules, set_rules=self.set_rules, indices=self.indices,
        )
        return pooled_walkable(self.shape.walkable, rooms)

    @property
    def loc_entropy_ceiling(self) -> float:
        """`loc_entropy` a perfectly uniform explorer would score here, in bits.

        On the record so a cross-arm comparison is possible at all: impassable
        objects lower it, so an objects arm and a walkable control are NOT
        comparable on the raw number. Divide by this to compare them; within one
        design it is constant and can be ignored.
        """
        from curious_george.envs.layouts import entropy_ceiling

        return entropy_ceiling(self.reachable_cells)

    @property
    def env_name(self) -> str:
        """The registered id to build.

        Follows from the SOURCE. The plain id ships its own landmarks and takes
        no `landmarks=` argument; the `-Multi-v0` variant accepts one, so a
        specified room set needs it. That used to be a separate env config the
        caller had to keep in agreement with `layouts`.
        """
        return MULTI_ROOM_ID[self.shape.room] if self.has_room_set else self.shape.room




# ---------------------------------------------------------------------------
# How experience is collected.


@dataclass(frozen=True)
class CollectCfg:
    """How experience is collected. Every field here affects training speed.

    ROLLOUTS: One rollout is every parallel environment instance
    running `episodes_per_env` episodes of `episode_steps` steps each.
    The loop collects one rollout, hands it to both learners (prnn and policy)
    for them to take gradient STEPS (notice the plural).
    """

    num_envs: int = 8
    """(SPEED) Environment instances stepped in parallel. NOT the number of
    distinct environment configurations - that is `MultiRoomEnvCfg.layouts`, and
    the two are orthogonal."""

    episodes_per_env: int = 1
    """(SPEED) Episodes each instance runs per rollout."""

    episode_steps: int = 256
    """Steps per episode. Load-bearing three ways: the episode cut during
    collection, the evaluation trajectory length, and a factor of the
    environment-step budget."""

    backend: EnvBackend = EnvBackend.SERIAL
    """(SPEED+SEMANTICS) DEVICE cannot measure extrinsic return. The old default
    diverged between config (False) and code (`.get(..., True)`), a disagreement
    unreachable from a composed config and wrong if reached; SERIAL is the
    measured choice."""

    rollout_cuda_graph: bool = False
    """(SPEED+SEMANTICS) Capture one rollout timestep as a CUDA graph and replay
    it. Requires DEVICE. Not bitwise against eager once the policy samples: a
    captured region draws from CUDA's graph-safe generator, so the two realise
    different but equally valid streams."""

    @property
    def episodes_per_rollout(self) -> int:
        return self.num_envs * self.episodes_per_env

    @property
    def env_steps_per_rollout(self) -> int:
        """Was `rl.frames`. Derived, so it cannot disagree with its parts."""
        return self.episodes_per_rollout * self.episode_steps

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.episodes_per_env < 1:
            raise ValueError(f"episodes_per_env must be >= 1, got {self.episodes_per_env}")
        if self.episode_steps < 1:
            raise ValueError(f"episode_steps must be >= 1, got {self.episode_steps}")
        if self.backend is EnvBackend.DEVICE and self.num_envs < 2:
            raise ValueError("EnvBackend.DEVICE batches instances; it needs num_envs > 1")
        if self.rollout_cuda_graph and self.backend is not EnvBackend.DEVICE:
            raise ValueError("rollout_cuda_graph captures the device rollout; it requires DEVICE")


# ---------------------------------------------------------------------------
# Architectures: what the weights ARE.


@dataclass(frozen=True)
class ArchPrnnCfg:
    """The world model as a network object - every constructor argument that is
    not an optimizer setting. Changing anything here changes what the weights
    are, so nothing in this class is a speed knob."""

    hidden_size: int = 500
    """REQUESTED width. PredictiveNet may round it; the effective value is
    folded back into the config before provenance is written."""

    loss: PredLoss = PredLoss.MSE
    """See `PredLoss`. MSE constructs the network EXACTLY as before this field
    existed (same kwargs, same RNG stream - the goldens gate it bitwise)."""

    focal_gamma: float | None = None
    """Focal reweighting ((1-pt)^gamma * ce) for the CE TRAINING loss only -
    the curiosity reward stays plain surprisal (information is information;
    only the gradient allocation changes). CE-only; refused under MSE.

    Armed 2026-08-31 on the plan's named trigger, MEASURED under eval_mode
    (evaluation/surprisal_timing.py on the ce8full s2 checkpoint): background
    saturated at ~0.16 nats/tile while landmark tiles sit at 1.49 - near
    chance (ln 7 = 1.95) - EVEN at shown steps with the landmark in the
    input (docs/figures/surprisal_vs_time.png). Not an inference failure; a
    rare-class reconstruction failure, which is focal's exact use case. (The
    original one-off measured 1.78 through train-mode dropout; the audit's
    eval-mode rerun moved the number, not the conclusion.)"""

    action_encoding: ActionEncodingsEnum = ActionEncodingsEnum.SpeedHD
    n_timescale: int = 2
    dropout: float = 0.15
    noise_mean: float = 0.0
    noise_std: float = 0.05
    sparsity: float = 0.5

    action_offset: int = 0
    """WHICH ACTION SHARES A ROW WITH obs[t] - the whole circuit, in one integer.

        0  row t = (obs[t], a[t])    the action chosen AFTER seeing obs[t];
                                     the policy then acts on h[t-1]
        1  row t = (obs[t], a[t-1])  the action that PRODUCED obs[t];
                                     the policy acts on h[t], the state that
                                     already represents the current position

    Nothing else moves: same architecture, same `action_encoding`, same
    `predOffset`, same `actOffset=0`. Fingerprinting every tensor that reaches
    `pN.predict` under both settings differs in exactly one - the action input.

    NOT `actOffset`. That upstream parameter front-pads ZEROS and tail-drops, so
    it loses `HD[0]` from row 0 and discards each segment's last action. The
    shift is built where the rows are built, in `PRNNAdapter.action_rows`.
    """

    @property
    def prnn_type(self) -> pRNNtypes:
        """Fixed. The prevAct variant is retired; it is still named here because
        prnn's loader reads it and it belongs in provenance."""
        return pRNNtypes.masked

    def __post_init__(self) -> None:
        if self.action_offset not in (0, 1):
            raise ValueError(
                f"action_offset is which action shares a row with obs[t]; "
                f"only 0 and 1 mean anything, got {self.action_offset}"
            )
        if self.focal_gamma is not None:
            if self.loss is not PredLoss.CE:
                raise ValueError(
                    "focal_gamma reweights the CE loss; it means nothing under "
                    f"{self.loss} (storage.prediction_loss_kwargs would refuse "
                    "it later, but a config should fail where it is written)"
                )
            if not self.focal_gamma > 0:
                raise ValueError(
                    f"focal_gamma must be > 0 (gamma=0 IS plain CE - spell "
                    f"that as focal_gamma=None), got {self.focal_gamma}"
                )


@dataclass(frozen=True)
class ArchPolicyCfg:
    """Actor-critic on the pRNN's spatial representation.

    The plain actor-critic on raw observations is retired, so the boolean that
    chose between them has nothing left to choose and is gone.
    """

    input_type: AgentInputType = AgentInputType.H
    with_obs: bool = False
    with_head_direction: bool = True
    rgb: bool = True

    agent: AgentType = AgentType.AC
    """Who produces actions. Replaces a boolean PAIR that could disagree: a run
    with both `curious_agent` and `random_action_agent` set was NAMED curious
    and BEHAVED random. One field decides both."""

    random_action_probs: tuple[float, float, float, float] = RAND_ACT_PROBA
    """The distribution `agent=RANDOM` samples over (left, right, forward,
    pickup). The default is the project's forward-weighted walker; a UNIFORM
    baseline is `--arch-policy.random-action-probs 0.25 0.25 0.25 0.25`. The
    two random baselines differ ~2x on coverage, so which one a run used must
    be in its config, not in prose. Inert unless `agent` is RANDOM."""

    freeze_params: bool = False
    """Never update parameters (random-init control). Orthogonal to `agent`: an
    actor-critic that takes no gradient steps is a different control from a
    random walker."""


# ---------------------------------------------------------------------------
# Training. Two learners, two budgets, two rates.


@dataclass(frozen=True)
class TrainPrnnCfg:
    """World-model optimization. (SPEED) This section dominates wall-clock.

    HOW A ROLLOUT BECOMES GRADIENT STEPS. One gradient step consumes
    `episodes_per_grad_step` whole episodes out of the rollout. A rollout holds
    `num_envs * episodes_per_env` episodes, so it yields
    `episodes_per_rollout / episodes_per_grad_step` world-model gradient steps.
    Every episode feeds exactly one step: all fresh, no reuse. That is the
    difference from the policy, which replays the same transitions.
    """

    total_grad_steps: int = 80_000
    """GROUND TRUTH. Environment steps are derived from it:

        env_steps = total_grad_steps * episodes_per_grad_step * episode_steps

    Budgeting by experience instead made two arms incomparable without saying
    so: the same episode count trains the world model 8x less when 8 episodes
    are pooled per step. Budget the thing that varies; derive the thing that
    does not."""

    episodes_per_grad_step: int = 1
    """(SPEED+SEMANTICS) Episodes consumed per world-model optimizer step. Folds
    the old pool-group and segment-stride, which were the same number in two
    mutually exclusive fields plus a mode flag, with 0 meaning "all" - a
    sentinel whose value depended on the rollout size. Must divide
    `CollectCfg.episodes_per_rollout`."""

    batched: bool = False
    """(SPEED+SEMANTICS) What happens to the group: True AVERAGES the gradient
    over its episodes, False takes one and DROPS the rest. With one room,
    pooling loses on loss per gradient step; across several rooms it averages
    the gradient over rooms, which is the point of the multi-room design."""

    train: bool = True
    """Train the pRNN alongside RL."""

    lr: float = 3e-3
    weight_decay: float = 3e-3
    bptt_trunc: int = 10**8

    batched_curiosity: bool = False
    """(SPEED+SEMANTICS) One batched inference pass for equal-length episodes;
    changes dropout and noise RNG ordering."""

    cuda_graph: bool = False
    """(SPEED+SEMANTICS) Capture the per-episode training step. Fresh runs only:
    the optimizer is rebuilt capturable, which needs empty state."""

    curiosity_cuda_graph: bool = False
    """(SPEED) Forward-only, so unlike `cuda_graph` it needs no fresh optimizer."""

    compile: CompileMode = CompileMode.OFF
    """(SPEED+SEMANTICS) Accelerator only, and it may reorder float operations,
    so it needs a learning curve rather than a benchmark."""

    def env_steps_per_grad_step(self, episode_steps: int) -> int:
        """All fresh: a gradient step's episodes are used once and discarded."""
        return self.episodes_per_grad_step * episode_steps

    def total_env_steps(self, episode_steps: int) -> int:
        return self.total_grad_steps * self.env_steps_per_grad_step(episode_steps)

    def __post_init__(self) -> None:
        if self.total_grad_steps < 1:
            raise ValueError(f"total_grad_steps must be >= 1, got {self.total_grad_steps}")
        if self.episodes_per_grad_step < 1:
            raise ValueError(
                f"episodes_per_grad_step must be >= 1, got {self.episodes_per_grad_step}"
            )


@dataclass(frozen=True)
class TrainPolicyCfg:
    """PPO, and the reward it maximises.

    HOW A ROLLOUT BECOMES GRADIENT STEPS. The rollout's transitions are shuffled
    into minibatches of `ppo_batch_size` and replayed `ppo_epochs` times, so one
    rollout yields `ppo_epochs * env_steps_per_rollout / ppo_batch_size` steps
    and each transition is used `ppo_epochs` times. That reuse is why this
    learner has TWO rate properties and the pRNN has one: the transitions a step
    processes and the fresh experience behind it differ by exactly `ppo_epochs`.

    `ppo_batch_size` is DERIVED from the two budgets, not set. Scaling it with
    the rollout was how the policy got diluted 16x while the world model trained
    8x more per environment step - stating the step count makes that
    unrepresentable.
    """

    total_grad_steps: int = 320_000
    """GROUND TRUTH, with the pRNN's. Together they fix the minibatch:

        ppo_batch_size = ppo_epochs * env_steps / total_grad_steps
    """

    ppo_epochs: int = 4
    clip_eps: float = 0.2
    discount: float = 0.98
    gae_lambda: float = 0.95
    lr: float = 3e-4
    optim_betas: tuple[float, float] = (0.9, 0.999)
    optim_eps: float = 1e-8
    value_loss_coef: float = 1.0
    max_grad_norm: float = 0.5

    entropy_coef: float = 0.0
    entropy_coef_final: float | None = None
    """None is constant. Otherwise ramps LINEARLY in ENVIRONMENT STEPS, the axis
    policy collapse tracks. Collapse is late, so a rising coefficient puts the
    resistance where the drift is."""

    curious: bool = True
    k_curious: float = 1.0
    intrinsic: bool = False
    k_intrinsic: float = 1.0
    normalize_reward: bool = False
    """Divide the combined reward by a running std before GAE. The curiosity
    reward is the world model's own loss, which the world model is minimising,
    so the raw reward scale decays ~7x over a run and the critic's target
    drifts with it. Scale only, never centered. Gates and the exact-invariance
    property: `rl/update/advantage.py::RewardNormalizer`."""

    k_count: float = 0.0
    """Count-based novelty bonus: the m-th visit a stream makes to state
    (room, x, y, head direction) in a rollout earns k_count/sqrt(N + m), with N
    the LIFETIME visit count at rollout start (rl/update/rewards.py::CountBonus
    - the within-rollout term is what gives a fresh table a gradient at all).
    The curiosity CONTROL: the same novelty drive with the world model removed,
    so curious-vs-count isolates what prediction error adds over visitation
    statistics. 0 disables (the default; the agent stays AC). Requires the
    DEVICE backend; counts ride the policy checkpoint so a resume continues
    them. Scale it under normalize_advantage per the noise-floor protocol."""
    reward_alignment: RewardAlignment = RewardAlignment.NEXT_OBS
    """Composition made NEXT_OBS the effective default while the code fell back
    to LEGACY - a fallback reachable only from a config that omitted the key,
    which no live config did. NEXT_OBS is the corrected indexing and is now the
    single default."""

    normalize_advantage: bool = False
    """Whiten the advantage per PPO minibatch to mean 0, std 1.

    The policy gradient scales with |advantage| while `entropy_coef` is a fixed
    ADDITIVE term, so what governs exploration is the ratio
    `entropy_coef / |advantage|` - and that denominator moves: measured, |adv|
    falls from 0.525 to 0.120 within 30 rollouts, making a "constant" coefficient
    ~4x stronger by the end. Whitening pins the denominator at 1, so the
    coefficient means one fixed thing and transfers across circuits and seeds
    instead of being re-found for each.

    NOT equivalent to a learning-rate change: Adam is invariant to a global
    gradient rescale, so what this moves is the policy term's weight RELATIVE to
    the entropy and value terms - which is also why the effect should be small
    at entropy_coef=0 and large at 0.01.

    ⚠️ It rescales |adv| from ~0.12 to ~1, so every tuned `entropy_coef` here
    means something ~8x weaker with it on. The measured 0.003 knee does NOT
    carry over."""

    cuda_graph: bool = False
    """(SPEED) The minibatch step is almost pure dispatch on a small network."""

    def ppo_batch_size(self, total_env_steps: int) -> int:
        """DERIVED. See the class docstring."""
        return self.ppo_epochs * total_env_steps // self.total_grad_steps

    def fresh_env_steps_per_grad_step(self, total_env_steps: int) -> int:
        """New experience behind one step: `ppo_batch_size / ppo_epochs`. The
        same axis the pRNN's rate uses, so the two learners are comparable."""
        return self.ppo_batch_size(total_env_steps) // self.ppo_epochs

    def total_fresh_env_steps(self, total_env_steps: int) -> int:
        """Must equal the pRNN's `total_env_steps` - one run, measured twice."""
        return self.total_grad_steps * self.fresh_env_steps_per_grad_step(total_env_steps)

    def processed_transitions_per_grad_step(self, total_env_steps: int) -> int:
        """What one minibatch actually contains, counting reuse."""
        return self.ppo_batch_size(total_env_steps)

    def total_processed_transitions(self, total_env_steps: int) -> int:
        """Compute volume, not experience: every transition once per epoch."""
        return self.total_grad_steps * self.processed_transitions_per_grad_step(total_env_steps)

    def __post_init__(self) -> None:
        if self.total_grad_steps < 1:
            raise ValueError(f"total_grad_steps must be >= 1, got {self.total_grad_steps}")
        if self.ppo_epochs < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}")
        if self.entropy_coef_final is not None and self.cuda_graph:
            # A CAPTURED policy step cannot see a changing coefficient.
            # `algo.py` builds GraphPolicyTrainer ONCE with
            # loss_kwargs=dict(entropy_coef=self.entropy_coef) as a Python
            # float; `policy_graph._region` bakes that float into the capture;
            # and `rl/update/policy.py` routes graphed updates to
            # `_update_policy_epochs_graphed`, which never re-reads
            # `algo.entropy_coef`. So the ramp is silently pinned at its start
            # value - and `entropy_coef` is not logged, so nothing shows it.
            # One run (`mila-off1-e0.001to0.01-s2`) was wasted this way before
            # the combination was made unrepresentable.
            raise ValueError(
                "entropy_coef_final ramps the coefficient per update, but "
                "cuda_graph bakes it into the captured step, so the ramp would "
                "silently never happen. Choose one: drop the ramp, or set "
                "--train-policy.no-cuda-graph and pay the throughput."
            )


# ---------------------------------------------------------------------------
# Evaluation and run identity.


@dataclass(frozen=True)
class EvalCfg:
    """Which evaluations run, how they are parameterised, and how often.

    `evals` is a SET the driver iterates over, replacing four independent
    booleans read at four call sites plus one implicit branch that silently
    swapped the single-room evaluations for the multi-room one.
    """

    evals: frozenset[EvalKind] = frozenset(
        {EvalKind.SPATIAL_ONPOLICY, EvalKind.BEHAVIOUR, EvalKind.TRAJECTORY_PLOT}
    )

    rooms_max: int = 4
    """How many rooms the multi-room spatial eval scores, as a fixed PREFIX so
    the series stays comparable across checkpoints. The pooled estimate
    saturates at 2-3 because prnn caps the pairwise sample. Lived on the env
    before, which is where it was read with a code default of 8 that
    contradicted the config's 4.

    ⚠️ COST. One analysis event - this eval at rooms_max=4, plus the behaviour
    eval on the same cadence - is ~88 s measured 2026-08-27 on an RTX 4060 with
    `train_prnn.compile=LAYER`, and ~37 s with the compile off. It is therefore
    `eval.analysis_every_steps`, NOT the training loop, that sizes a short run:
    ten curve points is about half of a 30-minute budget. An earlier note here
    said 4.5-8.9 s per room; that was measured before the behaviour eval shared
    the cadence and before the compile, and runs were sized from it."""

    n_trajs: int = 8
    """Pooled trajectories of `CollectCfg.episode_steps` each, so evaluation
    statistics match training trajectory statistics."""

    spatial_path: SpatialEvalPath = SpatialEvalPath.POOLED
    legacy_decoder_timesteps: int = 15_000
    sleep_std: float = 0.03
    behaviour_timesteps: int = 25_000

    # Cadences are in ENVIRONMENT STEPS. A rollout is not a fixed amount of
    # experience - it scales with num_envs - so a rollout-counted interval
    # silently rescales with the collection shape. 0 disables the event.
    log_every_steps: int = 2_048
    plot_every_steps: int = 409_600
    analysis_every_steps: int = 409_600

    probe_seed: int | None = None
    """Seed for the spatial-eval probe rollouts (torch + numpy + env reset).

    None reproduces the historical unseeded eval bitwise. Set, it makes the
    probe FIXED: checkpoints within a run become comparable to each other
    instead of each carrying its own rollout noise (measured bands 0.068-0.114
    of sRSA per run unseeded - `docs/claude_logs/rl_tricks_2026-08-29.md` top).
    A CONSTANT shared across runs and training seeds on purpose: seed-to-seed
    CV then measures training variance, not probe variance. The multi-room
    eval seeds per room as `probe_seed + room_index`, so room k's probe does
    not depend on how much RNG the rooms before it consumed."""


@dataclass(frozen=True)
class RunCfg:
    """Identity, determinism, where artifacts come from and go."""

    exp_name: str = "pRNN"
    seed: int = 2

    wandb: bool = True
    wandb_entity: str = "blake-richards"
    wandb_project: str = "curious-george"
    video_every_episodes: int = 0

    save_every_steps: int = 245_760
    """Rolling checkpoint, overwritten. On its own a run leaves exactly ONE
    checkpoint, and "when did this representation appear" cannot be asked."""

    archive_every_steps: int = 0
    """Also keep a step-tagged copy. This is the developmental series that
    offline checkpoint scoring reads."""

    output_dir: Path | None = None
    """None resolves through `get_storage_dir()` - RL_STORAGE - which is what
    makes a cluster run write to its node-local scratch. Set it to override."""

    prnn_ckpt: Path | None = None
    """None initialises fresh; a path starts from that checkpoint. Replaces a
    boolean plus an environment variable, so "resume, but from nothing" stops
    being expressible - and the source checkpoint, previously absent from every
    run's record, now lands in provenance."""

    policy_ckpt: Path | None = None

    early_stop: bool = False
    """Stop when the environment is solved. Needs extrinsic return, which the
    DEVICE backend cannot measure."""


# ---------------------------------------------------------------------------
# The run.

@dataclass(frozen=True)
class Config:
    """One run.

    CROSS-SECTION PRECONDITIONS live here and nowhere else, because everything
    expressible inside one section is already a type error. What survives:

      1. intrinsic rewards need num_envs == 1
      2. early_stop needs a backend that measures return
      3. episodes_per_grad_step divides episodes_per_rollout
      4. SPATIAL_MULTIROOM iff the environment is multi-room
      5. the derived ppo_batch_size is a positive integer dividing the rollout
      6. a multi-room environment needs the DEVICE backend

    Typed away, so absent: device implies table; device needs more than one
    instance; rollout graphs need device; the base room matching the environment
    id; pool size and seed only under a pool; pool-group only when batched and
    stride only when not; the rollout dividing by instance count; the episode
    length dividing the rollout.
    """

    env: EnvCfg = field(default_factory=EnvCfg)
    collect: CollectCfg = field(default_factory=CollectCfg)
    arch_prnn: ArchPrnnCfg = field(default_factory=ArchPrnnCfg)
    arch_policy: ArchPolicyCfg = field(default_factory=ArchPolicyCfg)
    train_prnn: TrainPrnnCfg = field(default_factory=TrainPrnnCfg)
    train_policy: TrainPolicyCfg = field(default_factory=TrainPolicyCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    run: RunCfg = field(default_factory=RunCfg)

    # -- derived -----------------------------------------------------------

    @property
    def total_env_steps(self) -> int:
        return self.train_prnn.total_env_steps(self.collect.episode_steps)

    @property
    def total_episodes(self) -> int:
        return self.train_prnn.total_grad_steps * self.train_prnn.episodes_per_grad_step

    @property
    def total_rollouts(self) -> int:
        return self.total_env_steps // self.collect.env_steps_per_rollout

    @property
    def ppo_batch_size(self) -> int:
        return self.train_policy.ppo_batch_size(self.total_env_steps)

    @property
    def schedule(self):
        """Every derived count, as the shared dataclass. A property, not a
        field"""
        from curious_george.training.schedule import TrainingSchedule

        return TrainingSchedule.from_config(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, for provenance.json and the wandb config record.

        Replaces `provenance.resolved_config`, whose OmegaConf call sat under a
        bare `except` that wrote `{"unresolved": "<repr>"}` - a plausible file
        with no information in it.
        """
        return _jsonable(self)

    # -- validation --------------------------------------------------------

    def __post_init__(self) -> None:
        has_rooms = self.env.has_room_set

        if self.train_policy.intrinsic and self.collect.num_envs != 1:
            raise ValueError(
                f"intrinsic rewards are only implemented for one instance; got "
                f"num_envs == {self.collect.num_envs}"
            )
        if self.run.early_stop and not self.collect.backend.measures_return:
            raise ValueError(
                f"early_stop needs extrinsic return, which {self.collect.backend.value} "
                "cannot measure"
            )
        if self.collect.episodes_per_rollout % self.train_prnn.episodes_per_grad_step:
            raise ValueError(
                f"episodes_per_grad_step ({self.train_prnn.episodes_per_grad_step}) must "
                f"divide episodes_per_rollout ({self.collect.episodes_per_rollout})"
            )
        if (EvalKind.SPATIAL_MULTIROOM in self.eval.evals) != has_rooms:
            raise ValueError(
                "SPATIAL_MULTIROOM and a specified room set imply each other; got "
                f"eval={EvalKind.SPATIAL_MULTIROOM in self.eval.evals}, "
                f"room_set={has_rooms}"
            )
        if has_rooms and self.collect.backend is not EnvBackend.DEVICE:
            raise ValueError(
                "a room set is reassigned per episode across the batch; "
                f"it needs the DEVICE backend, got {self.collect.backend.value}"
            )

        batch = self.ppo_batch_size
        rollout = self.collect.env_steps_per_rollout
        if batch < 1 or rollout % batch:
            raise ValueError(
                f"derived ppo_batch_size ({batch}) must be a positive divisor of "
                f"env_steps_per_rollout ({rollout}); adjust total_grad_steps on either "
                "learner, or the rollout shape"
            )


def _jsonable(value: Any) -> Any:
    """Recursive, enum-aware, deterministic. `dataclasses.asdict` alone emits
    enums and Paths that `json.dumps` refuses, and frozensets whose order
    varies between runs."""
    import dataclasses

    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
        # Subclass identity is the whole point of the env union, and a plain
        # field dump would lose it - LRoomCfg and SquareRoomCfg have the same
        # fields and different meanings.
        out["_type"] = type(value).__name__
        return out
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Preset configurations: what `run=multienv` and `performance=ultra` were.


def _reference() -> Config:
    """The serial static L-room baseline: what a bare `main_train.py` ran."""
    return Config()


def _multienv() -> Config:
    """Multi-room training, pooled world-model steps.

    ⚠️ RETIRED BUDGET, kept because `tests/test_configs.py::EXPECTED` pins it as
    what this preset's YAML ancestor produced, and that pin is what says the
    Hydra migration changed no budget. It describes NO run that has ever
    happened: 491,520,000 environment steps at `num_envs=8`, and it predates the
    speed work, so a run launched from it is 3.4x slower than it needs to be
    (`docs/claude_logs/compaction-2026-08-28.md`).

    For actual multi-room training use `multienv-fast`.
    """
    base = Config()
    return replace(
        base,
        env=EnvCfg(source=Frozen()),
        collect=replace(base.collect, backend=EnvBackend.DEVICE),
        train_prnn=replace(
            base.train_prnn, batched=True, episodes_per_grad_step=8,
            batched_curiosity=True, total_grad_steps=240_000,
        ),
        train_policy=replace(base.train_policy, total_grad_steps=3_840_000, entropy_coef=0.01),
        eval=replace(
            base.eval,
            evals=frozenset({EvalKind.SPATIAL_MULTIROOM, EvalKind.BEHAVIOUR}),
            analysis_every_steps=4_194_304,
            plot_every_steps=20_971_520,
        ),
        run=replace(base.run, save_every_steps=1_048_576, archive_every_steps=10_485_760),
    )


def _multienv_fast() -> Config:
    """Multi-room training on the `parity` shape: 5 rooms, WALKABLE landmarks.

    A NEW preset rather than a redefinition of `multienv`, for the same reason
    `parity` did not redefine `reference`: `multienv`'s budget is pinned as what
    its YAML ancestor produced, and silently changing it would make every run
    labelled "multienv" mean two different things.

    THE ROOM SET IS PINNED BY ANCHOR, and that is the whole design. `Selected`
    carries the affordance, so one flag turns this into the impassable arm over
    the SAME five rooms:

        main_train.py multienv-fast                                   # walkable
        main_train.py multienv-fast env.source:selected \
            --env.source.n 5 --env.source.impassable                  # impassable

    `env.source` is a UNION, so the member is a SUBCOMMAND and its fields only
    exist after it - `--env.source.impassable False` is an unrecognized option
    plus a stray positional, and it cost one job.

    Selecting by INDEX cannot express "the same rooms, walkable": impassable
    landmarks admit 9,074 placements against walkable's 19,820, so the two pools
    are different sequences and 0 of the 5 indices name the same room in both.
    Measured, and the reason `Selected` exists.

    BEHAVIOUR is in `evals` because it is what logs `OPA_Occupancy_Map`, the
    visitation heatmap that says whether the policy collapsed.

    ⚠️ `rooms_max=5` and the analysis cadence, NOT the training loop, size this
    run: `EvalCfg.rooms_max` puts one event (multi-room at 4 plus behaviour) at
    ~88 s, and this scores five rooms.
    """
    base = _parity()
    return replace(
        base,
        env=EnvCfg(source=Selected(n=5, impassable=False)),
        eval=replace(
            base.eval,
            evals=frozenset({EvalKind.SPATIAL_MULTIROOM, EvalKind.BEHAVIOUR}),
            rooms_max=5,
            analysis_every_steps=15_000_000,
        ),
    )


def _ultra() -> Config:
    """Measured high-throughput preset for the static L-room.

    The device path is trajectory-equivalent to the table path, but the pooled
    world-model step, the rollout size and the optimizer settings intentionally
    change training semantics.
    """
    base = Config()
    return replace(
        base,
        collect=replace(base.collect, num_envs=128, backend=EnvBackend.DEVICE),
        train_prnn=replace(
            base.train_prnn, batched=True, episodes_per_grad_step=128,
            batched_curiosity=True, total_grad_steps=625,
        ),
        train_policy=replace(
            base.train_policy, total_grad_steps=5_000, lr=1e-3,
            optim_betas=(0.8, 0.97), entropy_coef=0.01,
        ),
    )


def _parity() -> Config:
    """The accelerated single L-room the circuit A/B is defined against.

    89,980,928 environment steps at 43,936 world-model and 175,744 policy
    gradient steps - the shape every `*-parity` run in wandb holds, starting
    from `mila-parity-e0.001_curious_26-08-27-14-32-32`. Everything here is a
    knob the arms hold FIXED.

    The two knobs UNDER TEST are deliberately left at their defaults so they
    have to be typed to be changed, and the changed variable is therefore always
    visible on the command line:

        main_train.py parity                                # offset 0, the unchanged circuit
        main_train.py parity --arch-prnn.action-offset 1     # the new circuit
        main_train.py parity --train-policy.entropy-coef 0.005

    2026-08-31: advantages are WHITENED by default and `entropy_coef` is 0.035
    - the coefficient means "fraction of a unit advantage scale", not a value
    tuned against a decaying |adv|. The raw-era knee was 0.003 (measured,
    `docs/entropy-sweep-and-noise-floor-2026-08-29.md`); it does not carry
    over: whitening rescales |adv| ~0.12 -> 1, and the 2026-08-30 rendering
    line (`docs/invalid-runs.md`) moved the reward scale ~5x - which is also
    why changing this preset IN PLACE costs nothing: every run the old values
    produced is pale-era and already incomparable. 0.035 is a STARTING point;
    the CE plan carries a scan. To reproduce the raw-era arms exactly:
    `--train-policy.no-normalize-advantage --train-policy.entropy-coef 0.003`.

    `probe_seed` is set here (constant, NOT derived from `run.seed` - see
    `EvalCfg.probe_seed`), so every parity-shaped run's sRSA series is a fixed
    probe rather than fresh rollout noise per event.

    NOT a replacement for `reference`, which names the SERIAL baseline and is
    what older runs and `tests/test_configs.py` mean by the word.
    """
    base = Config()
    return replace(
        base,
        collect=replace(
            base.collect, backend=EnvBackend.DEVICE, num_envs=256,
            episodes_per_env=1, episode_steps=256, rollout_cuda_graph=True,
        ),
        train_prnn=replace(
            base.train_prnn, batched=True, batched_curiosity=True,
            episodes_per_grad_step=8, compile=CompileMode.LAYER,
            cuda_graph=True, total_grad_steps=43_936,
        ),
        train_policy=replace(
            base.train_policy, total_grad_steps=175_744, cuda_graph=True,
            entropy_coef=0.035, normalize_advantage=True,
        ),
        eval=replace(
            base.eval, analysis_every_steps=3_333_328, plot_every_steps=7_499_989,
            probe_seed=10_007,
        ),
        run=replace(
            base.run, save_every_steps=8_388_608, archive_every_steps=8_388_608
        ),
    )


PRESETS: dict[str, tuple[str, Config]] = {
    "reference": ("serial static L-room baseline", _reference()),
    "parity": ("accelerated static L-room, 90.0M env steps (the A/B shape)", _parity()),
    "multienv": ("multi-room, pooled world model (RETIRED budget)", _multienv()),
    "multienv-fast": ("5 selected rooms on the parity shape", _multienv_fast()),
    "ultra": ("measured high-throughput static L-room", _ultra()),
}


def cli(args: list[str] | None = None) -> Config:
    """Build a `Config` from the command line.

        main_train.py reference
        main_train.py multienv --run.seed 3
        main_train.py ultra --train-prnn.total-grad-steps 1000

    Presets are named `Config` instances, so a mistake in one fails at import
    rather than at compose time, and every default still has exactly one home.
    """
    import tyro

    return tyro.cli(
        tyro.extras.subcommand_type_from_defaults(
            {name: cfg for name, (_, cfg) in PRESETS.items()},
            {name: doc for name, (doc, _) in PRESETS.items()},
        ),
        args=args,
    )

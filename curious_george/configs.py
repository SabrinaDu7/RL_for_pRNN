"""The whole run, as one typed value.

`tyro.cli` builds a `Config` from the command line; nothing else reads a loose
constant or a YAML file. Frozen throughout, so the object handed to
`provenance.write` is the object training ran on.

WHERE DOES A FIELD GO? One question decides it:

    Does this describe the WORLD, or how we GATHER EXPERIENCE from it?

An L-room is an L-room whether it is stepped serially, in async worker
processes, or as a GPU-resident table - so the backend is collection, and lives
in `CollectCfg`. Occlusion changes what there is to see, so it is the world, and
lives in `EnvCfg`.

THE VOCABULARY IS THREE WORDS: gradient steps, environment steps, episodes.
"Update" and "frames" are gone. An "update" was one rollout - 256 episodes, 32
pRNN gradient steps and 128 policy gradient steps at once - and naming that
single thing "update" made it read as one gradient step. `rl.frames` was the
environment steps in one rollout, written by hand in five places.

WHAT IS DERIVED. `TrainPrnnCfg.total_grad_steps` and
`TrainPolicyCfg.total_grad_steps` are the ground truth. Environment steps,
episode counts and `ppo_batch_size` all follow from them.

TRAINING SPEED. Fields marked (SPEED) change wall-clock without changing the
science. Fields marked (SPEED+SEMANTICS) change both - those need a learning
curve to justify them, not a benchmark. The speed-bearing sections are
`CollectCfg` (all of it), `TrainPrnnCfg` (the world model dominates a step),
and `TrainPolicyCfg.cuda_graph`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Union

from prnn.utils import ActionEncodingsEnum, pRNNtypes

from curious_george.utils.enums import AgentInputType, AgentType

# ---------------------------------------------------------------------------
# Enumerations. Reused where one already exists: AgentInputType and AgentType
# from utils/enums.py, ActionEncodingsEnum and pRNNtypes from prnn.


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


class CompileMode(str, enum.Enum):
    """`torch.compile` on the recurrent cell. Was typed `bool` in YAML and read
    as `bool | str` in code, because a launcher passes the string `layer`."""

    OFF = "off"
    CELL = "cell"  # fuse the LayerNorm chain inside one step
    LAYER = "layer"  # compile the whole recurrence over an episode


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
    class, same landmarks, one room instead of several."""


@dataclass(frozen=True)
class FrozenLayouts:
    """The frozen set for the base room (envs/layouts.py)."""


@dataclass(frozen=True)
class LayoutPool:
    """A seeded uniform sample of the admissible set. `seed` names the pool, so
    a run is reproducible from it alone."""

    size: int = 500
    seed: int = 20260813


LayoutSpec = Union[SingleLayout, FrozenLayouts, LayoutPool]


# ---------------------------------------------------------------------------
# The world.


@dataclass(frozen=True)
class EnvCfg:
    """What the world IS.

    Does this describe the world, or how we gather experience from it? Only the
    first kind belongs here. Nothing in this class affects speed; the rollout
    shape and the stepping backend are `CollectCfg`.
    """

    see_through_walls: bool | None = None
    """None keeps the environment's own default (True). False is occlusion,
    which changes what the agent can perceive and needs its own baseline."""

    @property
    def env_name(self) -> MinigridEnv:
        raise NotImplementedError("each environment subclass names itself")


@dataclass(frozen=True)
class LRoomCfg(EnvCfg):
    """Static L-room. Every checkpoint before the multi-room work is this."""

    new_obj_pos: tuple[int, int] | None = None
    """Cell of a FloorBright object, or None for none."""

    @property
    def env_name(self) -> MinigridEnv:
        return MinigridEnv.LROOM


@dataclass(frozen=True)
class SquareRoomCfg(EnvCfg):
    """Four-fold symmetric walls, so geometry carries no position information."""

    new_obj_pos: tuple[int, int] | None = None

    @property
    def env_name(self) -> MinigridEnv:
        return MinigridEnv.SQUAREROOM


@dataclass(frozen=True)
class FourRoomsCfg(EnvCfg):
    """The ONLY environment whose constructor reads these.

    `LEnv` and `SquareRoomEnv` inherit MiniGridEnv's `**kwargs` and discard them
    silently, so setting `door_poss` or `subroom_size` on an L-room was dead
    config that read as live. They live here alone for that reason.
    """

    subroom_size: int = 8
    door_poss: tuple[int, int, int, int] = (1, 5, 3, 4)
    start_room: int | None = 1
    """None randomises the start position. Folds the old `start_rand` and
    `start_room`, which stated one fact twice."""

    @property
    def env_name(self) -> MinigridEnv:
        return MinigridEnv.FOURROOMS_OBJECTS


@dataclass(frozen=True)
class MultiRoomEnvCfg(EnvCfg):
    """Rooms are reassigned at every synchronised episode boundary, so the same
    integrated trajectory lands at a different absolute position depending on
    which room an instance is in. That is the manipulation: it makes dead
    reckoning insufficient and a visible landmark necessary.

    `layouts` is the number of distinct environment CONFIGURATIONS, and it is
    orthogonal to `CollectCfg.num_envs`: N parallel instances each redraw a room
    from this pool at every episode boundary.
    """

    layouts: LayoutSpec = field(default_factory=FrozenLayouts)
    eval_rooms_max: int = 4
    """Spatial evaluation scores this many rooms, as a fixed prefix so the
    series stays comparable across checkpoints. Measured 4.5-8.9 s per room, and
    the pooled estimate saturates at 2-3 rooms because prnn caps the pairwise
    sample. The old code default of 8 contradicted the config's 4 and would have
    doubled the cost; 4 is what every multi-room run actually set."""

    @property
    def base_room(self) -> MinigridEnv:
        """Owns the wall geometry and selects the symmetry handling. Was a free
        string that had to agree with the environment id; the subclass now fixes
        both together."""
        raise NotImplementedError


@dataclass(frozen=True)
class LRoomMultiCfg(MultiRoomEnvCfg):
    @property
    def env_name(self) -> MinigridEnv:
        return MinigridEnv.LROOM_MULTI

    @property
    def base_room(self) -> MinigridEnv:
        return MinigridEnv.LROOM


@dataclass(frozen=True)
class SquareRoomMultiCfg(MultiRoomEnvCfg):
    """Closes both routes to localising without an object: geometry (square
    walls) and dead reckoning (the room changes every episode)."""

    @property
    def env_name(self) -> MinigridEnv:
        return MinigridEnv.SQUAREROOM_MULTI

    @property
    def base_room(self) -> MinigridEnv:
        return MinigridEnv.SQUAREROOM


AnyEnvCfg = Union[LRoomCfg, SquareRoomCfg, FourRoomsCfg, LRoomMultiCfg, SquareRoomMultiCfg]


# ---------------------------------------------------------------------------
# How experience is gathered.


@dataclass(frozen=True)
class CollectCfg:
    """How experience is GATHERED. Every field here is speed-bearing.

    WHAT A ROLLOUT IS. One rollout is every parallel environment instance
    running `episodes_per_env` episodes of `episode_steps` steps each, all
    stepped together. The loop collects one rollout, hands it to both learners,
    discards it, and collects the next. This is what the code used to call an
    "update", and `env_steps_per_rollout` is what `rl.frames` used to set by
    hand in five places.

    WHY A ROLLOUT EXISTS AT ALL - it is required by both algorithms, not a
    choice. PPO is on-policy: it needs a collected batch to compute GAE
    advantages and then runs several epochs of minibatches over it, so there is
    no such thing as one policy gradient step per environment step. And the pRNN
    backpropagates through a whole episode, so a complete episode must exist
    before any world-model gradient step.

    WHAT ITS SIZE BUYS: nothing. Holding everything else fixed and varying
    `num_envs` 32x moves both gradient-step totals by 0.02% - integer rounding
    on the final partial rollout. Rollout size sets how often cadenced events
    fire and how much memory a step needs, and that is all. The training budget
    is stated directly on each learner instead.
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

    action_encoding: ActionEncodingsEnum = ActionEncodingsEnum.SpeedHD
    n_timescale: int = 2
    dropout: float = 0.15
    noise_mean: float = 0.0
    noise_std: float = 0.05
    sparsity: float = 0.5

    @property
    def prnn_type(self) -> pRNNtypes:
        """Fixed. The prevAct variant is retired; it is still named here because
        prnn's loader reads it and it belongs in provenance."""
        return pRNNtypes.masked


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
    reward_alignment: RewardAlignment = RewardAlignment.NEXT_OBS
    """Composition made NEXT_OBS the effective default while the code fell back
    to LEGACY - a fallback reachable only from a config that omitted the key,
    which no live config did. NEXT_OBS is the corrected indexing and is now the
    single default."""

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

      1. a RANDOM agent needs num_envs == 1
      2. intrinsic rewards need num_envs == 1
      3. early_stop needs a backend that measures return
      4. episodes_per_grad_step divides episodes_per_rollout
      5. SPATIAL_MULTIROOM iff the environment is multi-room
      6. the derived ppo_batch_size is a positive integer dividing the rollout
      7. a multi-room environment needs the DEVICE backend

    Typed away, so absent: device implies table; device needs more than one
    instance; rollout graphs need device; the base room matching the environment
    id; pool size and seed only under a pool; pool-group only when batched and
    stride only when not; the rollout dividing by instance count; the episode
    length dividing the rollout.
    """

    env: AnyEnvCfg = field(default_factory=LRoomCfg)
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, for provenance.json and the wandb config record.

        Replaces `provenance.resolved_config`, whose OmegaConf call sat under a
        bare `except` that wrote `{"unresolved": "<repr>"}` - a plausible file
        with no information in it.
        """
        return _jsonable(self)

    # -- validation --------------------------------------------------------

    def __post_init__(self) -> None:
        multiroom = isinstance(self.env, MultiRoomEnvCfg)

        if self.arch_policy.agent is AgentType.RANDOM and self.collect.num_envs != 1:
            raise ValueError(
                f"a RANDOM agent pre-generates one action stream; it needs "
                f"num_envs == 1, got {self.collect.num_envs}"
            )
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
        if (EvalKind.SPATIAL_MULTIROOM in self.eval.evals) != multiroom:
            raise ValueError(
                "SPATIAL_MULTIROOM and a multi-room environment imply each other; got "
                f"eval={EvalKind.SPATIAL_MULTIROOM in self.eval.evals}, env={multiroom}"
            )
        if multiroom and self.collect.backend is not EnvBackend.DEVICE:
            raise ValueError(
                "a multi-room environment reassigns rooms per episode across the batch; "
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
# Presets: what `run=multienv` and `performance=ultra` were.


def _reference() -> Config:
    """The serial static L-room baseline: what a bare `main_train.py` ran."""
    return Config()


def _multienv() -> Config:
    """Multi-room training, pooled world-model steps.

    Pooling DESPITE the single-room sweep that found it loses on loss per
    gradient step: with one room that result holds, but with several rooms per
    rollout a pooled step averages the gradient ACROSS rooms, which is the
    design.
    """
    base = Config()
    return replace(
        base,
        env=LRoomMultiCfg(),
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


def _ultra() -> Config:
    """Measured high-throughput preset for the static L-room.

    The device path is trajectory-equivalent to the table path, but the pooled
    world-model step, the rollout size and the optimizer settings intentionally
    change training semantics. A fast local starting point, not a claim of
    scientific equivalence.
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


PRESETS: dict[str, tuple[str, Config]] = {
    "reference": ("serial static L-room baseline", _reference()),
    "multienv": ("multi-room, pooled world model", _multienv()),
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

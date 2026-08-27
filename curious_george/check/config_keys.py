"""Translate a run's config keys across the 2026-08-26 dataclass cutover.

wandb stores each run's config under the shape the code had at the time. Runs
before the cutover carry `exp.* / rl.* / predNet.* / logging.*`; runs after carry
`env / collect / arch_prnn / arch_policy / train_prnn / train_policy / eval /
run`. A query written against one era finds nothing in the other, and finding
nothing looks exactly like a field that was never set.

This maps between them, both directions, and is honest about the three kinds of
correspondence rather than pretending everything is one-to-one:

    RENAMED   a real 1:1 rename. Invertible.
    FOLDED    several old keys became one new field. NOT invertible - knowing
              `collect.backend` does not tell you which of table_env/device_env/
              async_envs a reader meant.
    GONE      derived or dropped. `rl.frames` is now computed from the
              collection shape; `rl.episodes_total` from the pRNN budget.

    from curious_george.check.config_keys import to_new, to_old, describe
    to_new("predNet.seqdur")        -> "collect.episode_steps"
    to_old("run.seed")              -> "exp.seed"
    to_new("rl.frames")             -> None      (see describe() for why)
"""

from __future__ import annotations

#: 1:1 renames. Invertible in both directions.
RENAMED: dict[str, str] = {
    # exp.* -> run / collect / env / arch_policy
    "exp.seed": "run.seed",
    "exp.exp_name": "run.exp_name",
    "exp.num_envs": "collect.num_envs",
    "exp.rollout_cuda_graph": "collect.rollout_cuda_graph",
    "exp.see_through_walls": "env.see_through_walls",
    "exp.input_type": "arch_policy.input_type",
    "exp.with_obs": "arch_policy.with_obs",
    "exp.with_HD": "arch_policy.with_head_direction",
    "exp.rgb": "arch_policy.rgb",
    "exp.random_init_control": "arch_policy.freeze_params",
    "exp.intrinsic": "train_policy.intrinsic",
    "exp.curious_agent": "train_policy.curious",
    "exp.eval_trajs": "eval.n_trajs",
    "exp.eval_timesteps": "eval.legacy_decoder_timesteps",
    "exp.eval_rooms_max": "eval.rooms_max",
    # rl.* -> train_policy
    "rl.discount": "train_policy.discount",
    "rl.lr": "train_policy.lr",
    "rl.optim_betas": "train_policy.optim_betas",
    "rl.gae_lambda": "train_policy.gae_lambda",
    "rl.entropy_coef": "train_policy.entropy_coef",
    "rl.entropy_coef_final": "train_policy.entropy_coef_final",
    "rl.value_loss_coef": "train_policy.value_loss_coef",
    "rl.max_grad_norm": "train_policy.max_grad_norm",
    "rl.optim_eps": "train_policy.optim_eps",
    "rl.ppo_epochs": "train_policy.ppo_epochs",
    "rl.ppo_clip_eps": "train_policy.clip_eps",
    "rl.cuda_graph": "train_policy.cuda_graph",
    "rl.k_int": "train_policy.k_intrinsic",
    "rl.k_curious": "train_policy.k_curious",
    "rl.reward_alignment": "train_policy.reward_alignment",
    # predNet.* -> arch_prnn / train_prnn / collect
    "predNet.hiddensize": "arch_prnn.hidden_size",
    "predNet.pRNNtype": "arch_prnn.prnn_type",
    "predNet.action_encoding": "arch_prnn.action_encoding",
    "predNet.ntimescale": "arch_prnn.n_timescale",
    "predNet.dropout": "arch_prnn.dropout",
    "predNet.noisemean": "arch_prnn.noise_mean",
    "predNet.noisestd": "arch_prnn.noise_std",
    "predNet.sparsity": "arch_prnn.sparsity",
    "predNet.lr": "train_prnn.lr",
    "predNet.bptttrunc": "train_prnn.bptt_trunc",
    "predNet.weight_decay": "train_prnn.weight_decay",
    "predNet.train": "train_prnn.train",
    "predNet.seqdur": "collect.episode_steps",
    "predNet.batched_wm": "train_prnn.batched",
    "predNet.cuda_graph": "train_prnn.cuda_graph",
    "predNet.curiosity_cuda_graph": "train_prnn.curiosity_cuda_graph",
    "predNet.batched_curiosity": "train_prnn.batched_curiosity",
    "predNet.compile_cell": "train_prnn.compile",
    # logging.* -> run / eval
    "logging.wandb_log": "run.wandb",
    "logging.wandb_entity": "run.wandb_entity",
    "logging.wandb_project": "run.wandb_project",
    "logging.video_log_freq": "run.video_every_episodes",
    "logging.save_every_steps": "run.save_every_steps",
    "logging.archive_every_steps": "run.archive_every_steps",
    "logging.early_stop": "run.early_stop",
    "logging.load_acmodel": "run.policy_ckpt",
    "logging.load_worldmodel": "run.prnn_ckpt",
    "logging.log_every_steps": "eval.log_every_steps",
    "logging.plot_every_steps": "eval.plot_every_steps",
    "logging.analysis_every_steps": "eval.analysis_every_steps",
}

#: Several old keys collapsed into one new field. Forward only: the new value
#: does not say which old key a reader had in mind.
FOLDED: dict[str, tuple[str, str]] = {
    "exp.table_env": ("collect.backend", "one axis replaced three booleans"),
    "exp.device_env": ("collect.backend", "one axis replaced three booleans"),
    "exp.async_envs": ("collect.backend", "one axis replaced three booleans"),
    "exp.random_action_agent": ("arch_policy.agent", "a boolean PAIR that could disagree"),
    "exp.onpolicy_prnn_eval": ("eval.evals", "four booleans became a set"),
    "exp.offpolicy_prnn_eval": ("eval.evals", "four booleans became a set"),
    "exp.analyze_agent_behav": ("eval.evals", "four booleans became a set"),
    "exp.eval_decoder": ("eval.spatial_path", "a boolean became a named path"),
    "exp.layouts": ("env.source", "a mode string became a typed source"),
    "exp.layout_pool_size": ("env.source.size", "only exists under a pool now"),
    "exp.layout_seed": ("env.source.seed", "only exists under a pool now"),
    "exp.room_id": ("env.shape.room", "the shape fixes the base room"),
    "exp.env_name": ("env.shape.room", "the env id follows from shape + source"),
    "predNet.wm_pool_group": ("train_prnn.episodes_per_grad_step", "two fields became one"),
    "predNet.wm_segment_stride": ("train_prnn.episodes_per_grad_step", "two fields became one"),
}

#: No new key at all: derived from the budget, or dropped with the feature.
GONE: dict[str, str] = {
    "rl.frames": "derived - collect.num_envs * episodes_per_env * episode_steps",
    "rl.episodes_total": "derived - train_prnn.total_grad_steps * episodes_per_grad_step",
    "rl.ppo_batch_size": "derived - ppo_epochs * env_steps / train_policy.total_grad_steps",
    "rl.algo": "dead key, read nowhere",
    "rl.loss": "dropped with a2c; one loss left is not an enumeration",
    "rl.trajs_per_batch": "tasks/ only; that code moved to the questions repo",
    "exp.pRNN": "dropped with the plain actor-critic it chose between",
    "exp.opt_return": "dead key, read nowhere",
    "exp.start_rand": "folded into a nullable start_room, then dropped with FourRooms",
    "exp.start_room": "dropped with the config-level FourRooms variant",
    "exp.env_subroom_size": "dropped with the config-level FourRooms variant",
    "exp.door_poss": "dropped with the config-level FourRooms variant",
    "exp.new_obj_pos": "dropped; a novel object is content, not an env field",
    "predNet.path": "dead key, and it named a different architecture than pRNNtype",
    "predNet.foder": "dead key (and a typo for folder)",
    "logging.save_params": "dead key, read nowhere",
    "logging.focus": "dead key, read nowhere",
}

_NEW_TO_OLD = {new: old for old, new in RENAMED.items()}


def to_new(old_key: str) -> str | None:
    """The post-cutover key, or None if there is not one. See `describe`."""
    if old_key in RENAMED:
        return RENAMED[old_key]
    if old_key in FOLDED:
        return FOLDED[old_key][0]
    return None


def to_old(new_key: str) -> str | None:
    """The pre-cutover key, or None.

    Only RENAMED is invertible: a folded field had several ancestors and
    picking one would be a guess.
    """
    return _NEW_TO_OLD.get(new_key)


def describe(key: str) -> str:
    """Why a key has no counterpart - the thing a bare None cannot say."""
    if key in RENAMED:
        return f"{key} -> {RENAMED[key]} (renamed)"
    if key in FOLDED:
        new, why = FOLDED[key]
        return f"{key} -> {new} (folded: {why}; not invertible)"
    if key in GONE:
        return f"{key} -> nothing ({GONE[key]})"
    if key in _NEW_TO_OLD:
        return f"{key} <- {_NEW_TO_OLD[key]} (renamed)"
    return f"{key} is not a key this cutover touched"


def translate_config(config: dict, *, to: str = "new") -> dict:
    """A flat {dotted_key: value} config mapped across the cutover.

    Keys with no counterpart are DROPPED rather than passed through, so the
    result is only what actually corresponds. Ask `describe` about the rest.
    """
    if to not in ("new", "old"):
        raise ValueError(f"to must be 'new' or 'old', got {to!r}")
    fn = to_new if to == "new" else to_old
    return {mapped: v for k, v in config.items() if (mapped := fn(k)) is not None}

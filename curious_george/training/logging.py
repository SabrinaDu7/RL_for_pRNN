"""wandb logging for training runs.

Replaces the fragile header/data zip of the historical script with explicit
dict construction. Key names are IDENTICAL to the historical ones (dashboard
continuity), with one additive fix: the per-action curiosity and advantage
keys computed by the collector are now actually forwarded (they were computed
but silently dropped before).
"""

from dataclasses import dataclass

import wandb
from omegaconf import OmegaConf

from curious_george.utils.common import synthesize


@dataclass
class UpdateStats:
    """Everything log_update needs beyond the algo's logs dict."""

    update: int
    num_frames: int
    fps: float
    duration: int
    random_agent: bool


def init_wandb(cfg, run_ctx):
    return wandb.init(
        entity=cfg.logging.wandb_entity,
        project=cfg.logging.wandb_project,
        group=cfg.exp.exp_name,
        name=run_ctx.run_name,
        id=run_ctx.run_name,
        dir=run_ctx.model_dir,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )


def _prefixed(prefix: str, stats: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def build_update_log(logs: dict, stats: UpdateStats, mi_policy: float | None) -> dict:
    """Flat scalar dict with the historical key names."""
    out = {
        "update": stats.update,
        **_prefixed("steps_per_trial", synthesize(logs["num_frames_per_episode"])),
        "num_episodes": logs["num_episodes"],
        "policy_entropy": logs["entropy"],
        "loc_entropy": logs["loc_entropy"],
        "loc_entropy_5": logs["loc_entropy_5"],
        "frames": stats.num_frames,
        "FPS": stats.fps,
        "duration": stats.duration,
    }
    if not stats.random_agent:
        # Absent under exp.device_env, which cannot measure extrinsic reward.
        # No series at all beats a series of zeros nothing can distinguish from
        # a real one (collector.py::_close_device_segment).
        if "return_per_episode" in logs:
            out.update(
                _prefixed("return", synthesize(logs["return_per_episode"], signs=True))
            )
        out.update(
            _prefixed("int_reward", synthesize(logs["intrinsic_rewards"], abs=True))
        )
        out.update(
            _prefixed("cur_reward", synthesize(logs["curious_rewards"], abs=True))
        )
        out.update(_prefixed("values", synthesize(logs["values"])))
        out.update(_prefixed("advantages", synthesize(logs["advantages"])))
        out["policy_loss"] = logs["policy_loss"]
        out["value_loss"] = logs["value_loss"]
        out["grad_norm"] = logs["grad_norm"]
        if mi_policy is not None:
            out["MI_policy"] = mi_policy
        # previously computed but never forwarded to wandb:
        out.update(
            {
                k: v
                for k, v in logs.items()
                if k.startswith("curious_reward_") or k.startswith("avg_adv_")
            }
        )
    return out


def log_update(logs: dict, stats: UpdateStats, mi_policy: float | None) -> None:
    wandb.log(build_update_log(logs, stats, mi_policy))
    wandb.log({"subroom_ids": wandb.Histogram(logs["subroom_ids"])})
    wandb.log({"dist_travelled": logs["dist_travelled"]})


def log_spatial(metrics: dict, nameext: str) -> None:
    # sRSA/SI/SWdist are pRNN metrics and are logged BY prnn, not here
    # (Sabrina, 2026-07-09). Until the pooled-eval metrics move into prnn
    # (next refactor phase), the pooled path's values reach wandb only via
    # this repo's own historical key, SWdist_direct.
    wandb.log({f"SWdist_direct{nameext}": metrics["SWdist"]})


def log_multi_room(result: dict, layout_episodes) -> None:
    """Per-room and pooled spatial metrics, plus how much each room was trained on.

    prnn logs the raw sRSA/SI/SWdist under its own per-room name extensions; what
    is added here is the COMPARISON between them, which is the actual measurement
    (see evaluation/spatial.py::evaluate_multi_room_representation), and the
    exposure counts without which a low per-room number cannot be told apart from
    a room the agent barely visited.
    """
    out = {
        "multiroom/mean_room_sRSA": result["mean_room_sRSA"],
        "multiroom/pooled_sRSA": result["pooled"]["sRSA"],
        "multiroom/remapping_index": result["remapping_index"],
        "multiroom/pooled_SWdist": result["pooled"]["SWdist"],
        "multiroom/pooled_SI": result["pooled"]["SI"],
        "multiroom/episodes_min": int(min(layout_episodes)),
        "multiroom/episodes_max": int(max(layout_episodes)),
    }
    for i, room in enumerate(result["per_room"]):
        out[f"multiroom/room{i}_sRSA"] = room["sRSA"]
        out[f"multiroom/room{i}_SWdist"] = room["SWdist"]
        out[f"multiroom/room{i}_SI"] = room["SI"]
    # Only meaningful when rooms are few enough to name individually.
    if len(layout_episodes) <= 16:
        for i, n in enumerate(layout_episodes):
            out[f"multiroom/room{i}_episodes"] = int(n)
    wandb.log(out)


def log_behavior(opa, with_figures: bool = True) -> None:
    wandb.log({"MI_policy_eval": opa.mi})
    if not with_figures:
        return
    wandb.log({"OPA_Advantages": wandb.Plotly(opa.plot_advantages())})
    wandb.log({"OPA_Policy_Heatmaps": wandb.Plotly(opa.plot_policy_heatmaps())})
    wandb.log({"OPA_Occupancy_Map": wandb.Plotly(opa.plot_occupancy())})

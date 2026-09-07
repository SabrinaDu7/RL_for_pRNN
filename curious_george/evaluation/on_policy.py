import numpy as np
from jaxtyping import Integer
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import entropy

from curious_george.rl.algo import PredictivePPOAlgo

SCALES = {
    "viridis": plotly.colors.sequential.Viridis,
    "plasma": plotly.colors.sequential.Plasma,
    "default": plotly.colors.sequential.Plasma,
}


def mutual_info_policy(joint_dist):
    """
    Compute I(S;A) from the un-normalized joint_probs array
    S is state, and A is action

    Input is (hd,x,y,a)
    """
    _, _, _, A = joint_dist.shape
    new_joint = joint_dist.copy().reshape(-1, A)
    mask = new_joint.sum(axis=1) > 0
    new_joint = new_joint[mask]

    p_sa = new_joint / new_joint.sum()  # normalise
    p_s = p_sa.sum(axis=1)
    p_a = p_sa.sum(axis=0)

    H_s = entropy(p_s, base=2)
    H_a = entropy(p_a, base=2)
    H_sa = entropy(p_sa.flatten(), base=2)
    mi = H_s + H_a - H_sa

    return mi


def plot_heatmaps(feature, title="", zmin=None, zmax=None, HDs=True, scale="default"):
    """
    Plot the heatmaps of a feature.
    """
    if not isinstance(zmin, (int, float)):
        zmin = np.nanmin(feature)
    if not isinstance(zmax, (int, float)):
        zmax = np.nanmax(feature)

    if HDs:
        fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{"colspan": 2, "rowspan": 2}, None],
                [None, None],
                [{}, {}],
                [{}, {}],
            ],
        )

        fig.add_trace(
            go.Heatmap(z=np.nanmean(feature, axis=0).T, colorscale=SCALES[scale]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(z=feature[0].T, colorscale=SCALES[scale]), row=3, col=1
        )
        fig.add_trace(
            go.Heatmap(z=feature[1].T, colorscale=SCALES[scale]), row=3, col=2
        )
        fig.add_trace(
            go.Heatmap(z=feature[2].T, colorscale=SCALES[scale]), row=4, col=1
        )
        fig.update_traces(showscale=False)
        fig.add_trace(
            go.Heatmap(z=feature[3].T, colorscale=SCALES[scale]), row=4, col=2
        )
        fig.update_layout(
            height=800,
            width=600,
            title_text=title,
            title_x=0.5,
        )
    else:
        fig = go.Figure(
            data=go.Heatmap(z=np.nanmean(feature, axis=0).T, colorscale=SCALES[scale])
        )
        fig.update_layout(
            height=400,
            width=500,
            title_text=title,
            title_x=0.5,
        )
    fig.update_traces(zmin=zmin, zmax=zmax)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(font_family="Courier New")
    if HDs:
        fig.add_annotation(
            x=-0.1,
            y=0.37,
            xref="paper",
            yref="paper",
            text="→",
            font={"size": 24, "family": "Courier"},
            showarrow=False,
        )
        fig.add_annotation(
            x=0.5,
            y=0.37,
            xref="paper",
            yref="paper",
            text="↓",
            font={"size": 24, "family": "Courier"},
            showarrow=False,
        )
        fig.add_annotation(
            x=-0.1,
            y=0.08,
            xref="paper",
            yref="paper",
            text="←",
            font={"size": 24, "family": "Courier"},
            showarrow=False,
        )
        fig.add_annotation(
            x=0.5,
            y=0.08,
            xref="paper",
            yref="paper",
            text="↑",
            font={"size": 24, "family": "Courier"},
            showarrow=False,
        )

    # fig.show()
    return fig


class OnPolicyAnalysis:
    """
    Class for analyzing the on-policy representations of the environment learned or used by RL agent.
    """

    def __init__(self, PPOalgo=None, timesteps=10000, reuse_last_rollout=False, **kwargs):
        """With reuse_last_rollout=True, analyze the rollout already sitting in
        PPOalgo's buffers (the last collect_experiences) instead of building a
        fresh algo and collecting `timesteps` new steps - this is free, whereas
        the fresh collection used to dominate analysis wall-clock time.
        """
        if reuse_last_rollout:
            assert PPOalgo is not None, "reuse_last_rollout requires PPOalgo"
            assert hasattr(PPOalgo, "last_joint_dist"), (
                "PPOalgo has no collected rollout yet - call collect_experiences first"
            )
            self.algo = PPOalgo
            self.timesteps = PPOalgo.num_frames
            self.joint_probs = PPOalgo.last_joint_dist
            self.mi = mutual_info_policy(self.joint_probs)
            return

        self.timesteps = timesteps
        if PPOalgo is not None:
            # Build a new algo with same params, just shorter timesteps
            self.algo = PredictivePPOAlgo(
                env=PPOalgo.env,
                acmodel=PPOalgo.acmodel,
                predictiveNet=PPOalgo.pN,
                device=PPOalgo.device,
                num_frames=timesteps,
                discount=PPOalgo.discount,
                lr=PPOalgo.lr,
                gae_lambda=PPOalgo.gae_lambda,
                entropy_coef=PPOalgo.entropy_coef,
                value_loss_coef=PPOalgo.value_loss_coef,
                max_grad_norm=PPOalgo.max_grad_norm,
                preprocess_obss=PPOalgo.preprocess_obss,
                train_pN=PPOalgo.train_pN,
                prnn_seqdur=PPOalgo.prnn_seqdur,
                action_offset=PPOalgo.action_offset,
                curious_agent=PPOalgo.curious_agent,
                k_curious=PPOalgo.k_curious,
                # Without these the clone was a DIFFERENT agent: legacy
                # alignment (asserts under action_offset=1) sampling the
                # policy even for a RANDOM run.
                reward_alignment=PPOalgo.reward_alignment,
                random_actions=PPOalgo.random_actions,
                random_action_probs=PPOalgo.random_action_probs,
                # Same bug class, next generation (audit 2026-08-31): without
                # these a normalized/count run's fresh-clone analysis computed
                # advantages on raw rewards - a different scale from the run
                # it claims to describe.
                normalize_advantage=PPOalgo.normalize_advantage,
                normalize_reward=PPOalgo.reward_normalizer is not None,
                k_count=PPOalgo.k_count,
            )
        else:
            required_keys = [
                "env",
                "acmodel",
                "predictiveNet",
                "device",
                "discount",
                "gae_lambda",
                "prnn_seqdur",
                "preprocess_obss",
                "action_offset",
                "curious_agent",
                "k_curious",
            ]
            for key in required_keys:
                if key not in kwargs:
                    raise ValueError(f"Missing required argument: {key}")
            self.algo = PredictivePPOAlgo(num_frames=timesteps, **kwargs)

        _, logs = self.algo.collect_experiences()
        self.joint_probs = logs["joint_dist"]
        self.mi = mutual_info_policy(self.joint_probs)

    def plot_advantages(self, zmin=None, zmax=None, HDs=True, scale="default"):
        """
        Plot the heatmaps of advantages.
        """
        instances_map = np.zeros((4, self.algo.env.width - 2, self.algo.env.height - 2))
        adv_map = np.zeros((4, self.algo.env.width - 2, self.algo.env.height - 2))
        for t in range(self.timesteps):
            adv_map[
                self.algo.directions[t],
                self.algo.locs[t][0] - 1,
                self.algo.locs[t][1] - 1,
            ] += self.algo.advantages[t].cpu().numpy()

            instances_map[
                self.algo.directions[t],
                self.algo.locs[t][0] - 1,
                self.algo.locs[t][1] - 1,
            ] += 1

        adv_map /= np.maximum(instances_map, 1e-6)

        return plot_heatmaps(adv_map, "Advantages", zmin, zmax, HDs, scale)

    def plot_values(self, zmin=None, zmax=None, HDs=True, scale="default"):
        """
        Plot the heatmaps of values.
        """
        instances_map = np.zeros((4, self.algo.env.width - 2, self.algo.env.height - 2))
        values_map = np.zeros((4, self.algo.env.width - 2, self.algo.env.height - 2))
        for t in range(self.timesteps):
            values_map[
                self.algo.directions[t],
                self.algo.locs[t][0] - 1,
                self.algo.locs[t][1] - 1,
            ] += self.algo.values[t].cpu().numpy()

            instances_map[
                self.algo.directions[t],
                self.algo.locs[t][0] - 1,
                self.algo.locs[t][1] - 1,
            ] += 1

        values_map /= np.maximum(instances_map, 1e-6)

        return plot_heatmaps(values_map, "Values", zmin, zmax, HDs, scale)

    def plot_policy_heatmaps(self, scale="default"):
        """
        Visualise π(a|s). A 4×4 grid:
            Rows = head-direction  (↑, →, ↓, ←)
            Cols = actions         (↺, ↻, ↑, ·)
        """
        A = self.algo.acmodel.act_dim
        assert A == 4, "Only supports 4 actions for now"

        # Arrow labels
        hd_labels = ["↑", "→", "↓", "←"]  # rows (rotated layout)
        act_labels = ["↺", "↻", "↑", "·"]  # columns

        joint = self.joint_probs.copy()
        denom = joint.sum(axis=3, keepdims=True)
        denom[denom == 0] = 1.0
        policy = joint / denom  # [hd, x, y, a]

        fig = make_subplots(
            rows=4,
            cols=4,
            specs=[[{}] * 4] * 4,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
            row_titles=[f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)],
            column_titles=[f"A {i}: {lbl}" for i, lbl in enumerate(act_labels)],
        )

        for hd in range(4):
            for a in range(4):
                z = policy[hd, :, :, a].T
                fig.add_trace(
                    go.Heatmap(z=z, showscale=False, colorscale=SCALES[scale]),
                    row=hd + 1,
                    col=a + 1,
                )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, autorange="reversed")
        fig.update_layout(
            height=900,
            width=900,
            title="Policy π(a|s)  (rows = HD, cols = action)",
            title_x=0.5,
            font_family="Courier New",
        )

        # Manually set font for subplot labels (row/col titles)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font = dict(
                size=24, family="Courier New", color="black"
            )

        # fig.show()
        return fig

    def plot_occupancy(self, scale="plasma"):
        return get_occupancy_fig(self.algo, self.timesteps, scale)


# Distinct function
def occupancy_counts(
    algo: PredictivePPOAlgo, timesteps: int
) -> Integer[np.ndarray, "hd width height"]:
    """State-occupancy counts per head-direction, `[hd, x, y]`, 0-indexed.

    THE MEASUREMENT. `get_occupancy_fig` draws this and nothing else, so the
    numbers a figure shows and the numbers an analysis reads are the same
    array by construction.

    It exists because they were not: occupancy was built inside the plotting
    function and thrown away into a plotly figure, so the only way to analyse
    it later was to scrape the figure's `z` back out of wandb - a number
    recovered from a picture. Log this instead and the picture is downstream of
    the data.

    Indexing is `[hd, x, y]` with x horizontal, matching `get_walkable_mask`
    and MiniGrid's `grid.get(x, y)`. Note that the FIGURE transposes to `[y, x]`
    for display, which is the convention any consumer of the logged plotly JSON
    inherits.
    """
    occ = np.zeros((4, algo.env.width - 2, algo.env.height - 2), dtype=np.int64)
    for t in range(timesteps):
        x, y = algo.locs[t][0] - 1, algo.locs[t][1] - 1
        occ[algo.directions[t], x, y] += 1
    return occ


def get_occupancy_fig(algo: PredictivePPOAlgo, timesteps: int, scale="plasma") -> go.Figure:
    """
    Show state-occupancy counts (no action dimension).
    1x4 layout - one heat-map per head-direction.

    Draws `occupancy_counts` and adds nothing to it.
    """
    hd_labels = ["\u2192", "\u2193", "\u2190", "\u2191"]

    occ = occupancy_counts(algo, timesteps)

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{}] * 4],
        horizontal_spacing=0.03,
        column_titles=[f"HD {i}: {lbl}" for i, lbl in enumerate(hd_labels)],
    )

    for hd in range(4):
        fig.add_trace(
            go.Heatmap(z=occ[hd].T, showscale=False, colorscale=SCALES[scale]),
            row=1,
            col=hd + 1,
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(
        height=280,
        width=900,
        title="State-occupancy per head-direction",
        title_x=0.5,
        font_family="Courier New",
    )

    # Make HD labels bigger + bold
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font = dict(
            size=24, family="Courier New", color="black"
        )

    # fig.show()
    return fig

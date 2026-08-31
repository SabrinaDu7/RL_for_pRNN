"""Exploration metrics of the POLICY, from positions alone.

Analysis only: every function is a pure map from tensors already collected -
``positions (E, T, 2)``, per-episode room ids, per-room geometry - to one
result dataclass. Nothing here steps an environment or a network; real
rollouts, reference walkers (`envs.action_graph`) and fabricated fixtures all
score through the same code.

Conventions, shared with the collector and `on_policy.occupancy_counts`:
``positions[e, t]`` is the PRE-action MiniGrid ``(x, y)`` (1-indexed, walls
never visited) of episode ``e`` at step ``t``; grids are indexed ``[x, y]``
with shape ``(W, H)``. The unit of replication is the EPISODE, and every
denominator is the episode's own room's walkable-cell count - asked of the
`Layout`, never hardcoded (see docs/claude_logs/plan-policy-exploration-evals
-2026-08-30.md for why the pooled ceiling cannot compare arms).
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int


@dataclass(frozen=True)
class ThresholdStat:
    """Time-to-cover at one level, with censoring FIRST: a mean over only the
    episodes that got there describes the luckiest few and hides the rest."""

    level: float
    reached_fraction: float
    mean_step: float  # mean env steps to reach, over reaching episodes; nan if none

    def __str__(self) -> str:
        if self.reached_fraction == 0.0:
            return "censored"
        return f"{self.mean_step:5.0f} ({self.reached_fraction:4.0%})"


@dataclass(frozen=True)
class CoverageCurves:
    """Cumulative distinct cells visited, per episode - the source of truth
    every scalar below derives from, so a figure and a table cannot disagree."""

    curves: Int[torch.Tensor, "E T"]  # distinct cells after k+1 steps
    denominators: Int[torch.Tensor, "E"]  # the episode's room's walkable count

    @property
    def normalized(self) -> Float[torch.Tensor, "E T"]:
        return self.curves.float() / self.denominators.float().unsqueeze(1)

    @property
    def nauc(self) -> Float[torch.Tensor, "E"]:
        """Mean of the normalized curve: 1.0 = everything visited at step one."""
        return self.normalized.mean(dim=1)

    @property
    def final_coverage(self) -> Float[torch.Tensor, "E"]:
        return self.normalized[:, -1]

    def threshold(self, level: float) -> ThresholdStat:
        norm = self.normalized
        reached = norm[:, -1] >= level  # curves are monotone
        if not reached.any():
            return ThresholdStat(level, 0.0, math.nan)
        steps = (norm[reached] >= level).int().argmax(dim=1) + 1  # env steps taken
        return ThresholdStat(
            level, reached.float().mean().item(), steps.float().mean().item()
        )


def first_visit_steps(
    positions: Int[torch.Tensor, "E T 2"], *, width: int, height: int
) -> Int[torch.Tensor, "E cells"]:
    """Step index of each cell's first visit; T where never visited.

    Cells are flattened ``x * height + y``, matching the ``[x, y]`` grid
    convention.
    """
    E, T, _ = positions.shape
    cell = (positions[..., 0] * height + positions[..., 1]).long()
    first = torch.full((E, width * height), T, dtype=torch.long, device=cell.device)
    steps = torch.arange(T, device=cell.device).expand(E, T)
    return first.scatter_reduce_(1, cell, steps, reduce="amin")


def coverage_curves(
    positions: Int[torch.Tensor, "E T 2"],
    *,
    denominators: Int[torch.Tensor, "E"],
    width: int,
    height: int,
) -> CoverageCurves:
    """Distinct-cells-visited curve per episode, normalized by the episode's
    own room. All scatter/cumsum - no Python loop over E*T."""
    E, T, _ = positions.shape
    first = first_visit_steps(positions, width=width, height=height)
    ones = torch.ones_like(first)
    newly = torch.zeros((E, T + 1), dtype=torch.long, device=first.device)
    newly.scatter_add_(1, first, ones)  # bin T collects the never-visited
    return CoverageCurves(
        curves=newly[:, :T].cumsum(dim=1), denominators=denominators
    )


@dataclass(frozen=True)
class RoomVisitation:
    """Visit histograms with a room channel - the mixture `loc_entropy` pools
    away, and the only form whose entropy is comparable across arms."""

    counts: Int[torch.Tensor, "R W H"]
    supports: Bool[torch.Tensor, "R W H"]  # the room's own walkable cells

    @property
    def entropy_bits(self) -> Float[torch.Tensor, "R"]:
        """Shannon entropy of each room's histogram; nan for unvisited rooms
        (untested is not zero)."""
        counts = self.counts.flatten(1).float()
        totals = counts.sum(dim=1, keepdim=True)
        p = counts / totals
        ent = -(p * torch.where(p > 0, p.log2(), p.new_zeros(()))).sum(dim=1)
        return torch.where(totals.squeeze(1) > 0, ent, torch.full_like(ent, math.nan))

    @property
    def ceilings_bits(self) -> Float[torch.Tensor, "R"]:
        return self.supports.flatten(1).sum(dim=1).float().log2()

    @property
    def normalized(self) -> Float[torch.Tensor, "R"]:
        """entropy / log2(room's walkable count): 1.0 = uniform over the room."""
        return self.entropy_bits / self.ceilings_bits


def visitation_by_room(
    positions: Int[torch.Tensor, "E T 2"],
    *,
    layout_ids: Int[torch.Tensor, "E"],
    supports: Bool[torch.Tensor, "R W H"],
) -> RoomVisitation:
    """Pool episode visits into per-room histograms.

    A visit outside its room's support is a wiring error upstream (wrong room
    channel, wrong table), never data - so it raises rather than being
    silently masked away, which is how the old one-room `loc_mask` lost 839
    visits before anyone noticed.
    """
    R, W, H = supports.shape
    cell = (positions[..., 0] * H + positions[..., 1]).long()
    rooms = layout_ids.long().unsqueeze(1).expand_as(cell)
    counts = torch.zeros((R, W * H), dtype=torch.long, device=cell.device)
    counts.index_put_(
        (rooms.reshape(-1), cell.reshape(-1)),
        torch.ones_like(cell.reshape(-1)),
        accumulate=True,
    )
    counts = counts.reshape(R, W, H)
    stray = counts[~supports]
    if int(stray.sum()) != 0:
        raise ValueError(
            f"{int(stray.sum())} visits landed outside their room's walkable "
            "set; the positions/layout_ids/supports triple is inconsistent"
        )
    return RoomVisitation(counts=counts, supports=supports)


@dataclass(frozen=True)
class DistanceCoverage:
    """Coverage as a function of BFS distance-from-spawn, in actions.

    The control for spawn placement: aggregate coverage lets a policy cover
    90% of nearby cells and 5% of far ones and still look fine.
    """

    reachable: Int[torch.Tensor, "E D"]  # cells at each distance, per episode
    visited: Int[torch.Tensor, "E D"]

    @property
    def distances(self) -> Int[torch.Tensor, "D"]:
        return torch.arange(self.reachable.shape[1], device=self.reachable.device)

    @property
    def fraction(self) -> Float[torch.Tensor, "E D"]:
        """visited / reachable; nan where the episode has no cells that far."""
        reach = self.reachable.float()
        return torch.where(
            reach > 0, self.visited.float() / reach, torch.full_like(reach, math.nan)
        )

    def pooled_fraction(self) -> Float[torch.Tensor, "D"]:
        """All episodes' cells pooled per distance - the headline curve."""
        reach = self.reachable.sum(dim=0).float()
        return torch.where(
            reach > 0,
            self.visited.sum(dim=0).float() / reach,
            torch.full_like(reach, math.nan),
        )


def rollout_summary(
    *,
    positions: Int[torch.Tensor, "E T 2"],
    layout_ids: Int[torch.Tensor, "E"],
    supports: Bool[torch.Tensor, "R W H"],
    denominators: Int[torch.Tensor, "R"],
    thresholds: tuple[float, ...] = (0.5, 0.9),
) -> dict[str, float]:
    """One rollout's exploration scalars, keyed for wandb.

    The log-cadence surface: coverage@T and nAUC means, time-to-cover with the
    reach fraction always present and the conditional mean OMITTED when no
    episode reached the level (a missing series cannot be misread; a fabricated
    number can), and per-room normalized visitation entropy (nan-mean over
    rooms, since an unvisited room is untested rather than zero).
    """
    R, W, H = supports.shape
    curves = coverage_curves(
        positions,
        denominators=denominators[layout_ids],
        width=W,
        height=H,
    )
    rooms = visitation_by_room(positions, layout_ids=layout_ids, supports=supports)
    out = {
        "exploration/coverage": curves.final_coverage.mean().item(),
        "exploration/nauc": curves.nauc.mean().item(),
        "exploration/room_entropy_norm": rooms.normalized.nanmean().item(),
    }
    for level in thresholds:
        stat = curves.threshold(level)
        out[f"exploration/t{int(level * 100)}_reached"] = stat.reached_fraction
        if stat.reached_fraction > 0:
            out[f"exploration/t{int(level * 100)}_steps"] = stat.mean_step
    return out


def distance_binned_coverage(
    positions: Int[torch.Tensor, "E T 2"],
    *,
    distance_maps: Int[torch.Tensor, "E W H"],  # -1 = unreachable in that room
) -> DistanceCoverage:
    """Bin each episode's visited/reachable cells by its own spawn-distance map
    (`envs.action_graph.ActionGraph.distance_map`)."""
    E, T, _ = positions.shape
    W, H = distance_maps.shape[1:]
    visited = first_visit_steps(positions, width=W, height=H) < T  # (E, cells)
    dists = distance_maps.reshape(E, W * H).long()
    reachable_mask = dists >= 0
    D = int(dists.max().item()) + 1
    bins = dists.clamp(min=0)
    reachable = torch.zeros((E, D), dtype=torch.long, device=dists.device)
    reachable.scatter_add_(1, bins, reachable_mask.long())
    hit = torch.zeros((E, D), dtype=torch.long, device=dists.device)
    hit.scatter_add_(1, bins, (reachable_mask & visited).long())
    return DistanceCoverage(reachable=reachable, visited=hit)

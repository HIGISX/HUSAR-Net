"""Run learned candidate screening and objective-based allocation."""

from dataclasses import dataclass
from importlib.resources import files
import math
from pathlib import Path

import numpy as np

from ._objective import Instance, greedy_restricted
from ._policy import HUSARNet, load_model, make_cache, rollout, save_model
from ._search import swap_search_adaptive


@dataclass(frozen=True)
class Allocation:
    """A complete allocation and its objective components."""

    selected: tuple[int, ...]
    objective: float
    normalized_return: float
    hierarchical_coverage: float
    spatial_protection: float
    candidate_count: int


def load_policy(path: str | Path) -> HUSARNet:
    """Load a saved riskmean policy for CPU inference."""
    return load_model(path)[0]


def load_example_policy() -> HUSARNet:
    """Load the included policy trained on synthetic instances."""
    resource = files("husarnet").joinpath("_assets", "synthetic_policy.pt")
    with resource.open("rb") as stream:
        return load_model(stream)[0]


def save_policy(policy: HUSARNet, path: str | Path) -> None:
    """Write a trained policy checkpoint."""
    save_model(policy, path)


def _candidate_pool(instance: Instance, policy: HUSARNet, seed: int) -> np.ndarray:
    cache = make_cache(instance)
    visits = np.zeros(instance.N, dtype=np.int64)
    for sample in range(4):
        selected = rollout(policy, instance, cache, stochastic=sample > 0, seed=seed + sample)[0]
        visits[selected] += 1
    visited = np.flatnonzero(visits)
    ranked = visited[np.argsort(-visits[visited], kind="stable")]
    high_risk = instance.qstatic_order[:2 * instance.K]
    return np.union1d(ranked[:instance.K], high_risk)


def solve(
    instance: Instance,
    policy: HUSARNet | None = None,
    *,
    swap_seconds: float = 5.0,
    seed: int = 5100,
) -> Allocation:
    """Select K candidates using the supplied or bundled policy.

    swap_seconds is a soft limit for local search. Use zero to return the
    greedy construction without local exchanges.
    """
    if not math.isfinite(swap_seconds) or swap_seconds < 0:
        raise ValueError("swap_seconds must be finite and nonnegative")
    if policy is None:
        policy = load_example_policy()
    pool = _candidate_pool(instance, policy, seed)
    block = 1 if instance.K <= 1000 else 25
    selected = greedy_restricted(instance, pool, block=block)
    if swap_seconds > 0 and instance.K < instance.N:
        if instance.K <= 100:
            remove_pool, add_pool = 24, 96
        elif instance.K <= 500:
            remove_pool, add_pool = 20, 80
        else:
            remove_pool, add_pool = 32, 120
        selected = swap_search_adaptive(
            instance, selected, max_seconds=swap_seconds, min_rounds=3,
            patience=8, remove_pool=remove_pool, add_pool=add_pool, seed=seed,
        )
    objective, normalized_return, coverage, protection = instance.evaluate(selected)
    return Allocation(
        selected=tuple(int(candidate) for candidate in selected),
        objective=objective,
        normalized_return=normalized_return,
        hierarchical_coverage=coverage,
        spatial_protection=protection,
        candidate_count=len(pool),
    )

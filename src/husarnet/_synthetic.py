"""Generate synthetic instances for examples and training."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from ._objective import Instance

def zipf_sizes(rng: np.random.Generator, N: int, G: int, a: float = 1.1) -> np.ndarray:
    ranks = np.arange(1, G + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, a)
    w /= w.sum()
    raw = w * N
    size = np.maximum(np.floor(raw).astype(np.int64), 2)
    diff = N - int(size.sum())
    if diff > 0:
        order = np.argsort(-(raw - size))
        size[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(size)[::-1]
        k = 0
        while diff < 0:
            j = order[k % G]
            if size[j] > 2:
                size[j] -= 1
                diff += 1
            k += 1
    assert size.sum() == N and size.min() >= 2
    return size


def delaunay_edges(coords: np.ndarray) -> np.ndarray:
    tri = Delaunay(coords)
    edge_set = set()
    for triangle in tri.simplices:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            a, b = sorted((int(triangle[i]), int(triangle[j])))
            edge_set.add((a, b))
    return np.array(sorted(edge_set), dtype=np.int64)


def make_synthetic(seed: int = 20261117, N: int = 1000, G: int = 60, P: int = 10,
                   B: int = 150, K: int = 50, zipf_a: float = 1.1) -> Instance:
    """Generate a hierarchy and jointly simulated risks from a seed."""
    if not (3 <= G <= N // 4 and 1 <= P <= G and B >= 4 and 1 <= K <= N):
        raise ValueError("require 3 <= G <= N/4, 1 <= P <= G, B >= 4, and 1 <= K <= N")
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, (G, 2))
    edges = delaunay_edges(coords)
    ordering = np.argsort(coords[:, 0])
    city_prov = np.empty(G, dtype=np.int64)
    for p, idx in enumerate(np.array_split(ordering, P)):
        city_prov[idx] = p

    size = zipf_sizes(rng, N, G, zipf_a)
    city = np.repeat(np.arange(G), size)
    rng.shuffle(city)
    prov = city_prov[city]

    base = rng.normal(-0.35, 0.55, N)
    region_effect = rng.normal(0, 0.25, G)
    base += region_effect[city]
    space_d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    spatial_cov = 0.45 * np.exp(-space_d / 0.26) + 0.55 * np.eye(G)
    spatial = rng.multivariate_normal(np.zeros(G), spatial_cov, size=B).T   # (G,B)
    global_shock = rng.normal(0, 1, B)
    for b in range(B):
        if b % 6 == 0:
            epicenter = int(rng.integers(G))
            spatial[:, b] += 0.65 * np.exp(-space_d[epicenter] / 0.24)
    idio = rng.normal(0, 1, (N, B))
    q = np.exp(base[:, None] + 0.16 * global_shock[None, :]
               + 0.60 * spatial[city, :] + 0.24 * idio)
    regime_count = 6
    exposure = rng.dirichlet(np.full(regime_count, 0.55), size=N)
    regimes = np.arange(B) % regime_count
    rng.shuffle(regimes)
    q *= 0.25 + 4.5 * exposure[:, regimes]
    q = np.ascontiguousarray(q)

    rho = np.exp(region_effect + np.array([base[city == g].mean() for g in range(G)]) * 0.3)
    hist_city = np.array([np.exp(base[city == g]).mean() for g in range(G)])
    vol_city = np.array([np.std(q[city == g], axis=1).mean() for g in range(G)])
    a_city = np.clip((hist_city / np.mean(hist_city)) * (1 + 0.5 * vol_city / np.mean(vol_city)), 0.25, 4.0)
    a_city /= a_city.mean()
    a_prov = np.array([np.mean(a_city[city_prov == p]) for p in range(P)])
    a_prov /= a_prov.mean()

    return Instance(q=q, city=city, prov=prov, city_prov=city_prov, edges=edges,
                    rho=rho, a_city=a_city, a_prov=a_prov, K=K)

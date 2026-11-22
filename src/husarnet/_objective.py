"""HUSAR objective, incremental evaluation, and greedy construction.

The objective combines normalized lower-tail return, risk-weighted hierarchical
coverage, and protection against the upper tail of adjacent residual risk.
"""
from __future__ import annotations

from operator import index

import numpy as np
from scipy import sparse

ALPHA, BETA, TAU_R, TAU_G = 0.10, 0.50, 0.25, 0.25
OMEGA_CITY = OMEGA_PROV = 0.50


def tails(v, tau: float, upper: bool = False) -> np.ndarray:
    """Compute empirical CVaR with fractional weight at the tail boundary."""
    a = np.sort(np.asarray(v, dtype=np.float64), axis=-1)
    if upper:
        a = np.flip(a, axis=-1)
    k = tau * a.shape[-1]
    n = int(np.floor(k + 1e-12))
    frac = k - n
    out = np.sum(a[..., :n], axis=-1) if n else np.zeros(a.shape[:-1])
    if frac > 1e-12:
        out = out + frac * a[..., n]
    return out / k


def _indices(values, name, shape=None, upper=None):
    a = np.asarray(values)
    expected = (None,) if shape is None else shape
    if a.ndim != len(expected) or (shape is not None and a.shape != shape) or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {expected}")
    if upper is not None and np.any((a < 0) | (a >= upper)):
        raise ValueError(f"{name} indices must be between 0 and {upper - 1}")
    return np.array(a, dtype=np.int64, copy=True)


def _weights(values, name, size):
    a = np.ones(size) if values is None else np.array(values, dtype=np.float64, copy=True)
    if a.shape != (size,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(f"{name} must contain {size} finite, positive weights")
    return a


class Instance:
    """Allocation inputs with zero-based candidate and region indices."""

    def __init__(self, q, city, prov, city_prov, edges, K, *,
                 rho=None, a_city=None, a_prov=None, alpha=ALPHA, beta=BETA):
        self.alpha = float(alpha)
        self.beta = float(beta)
        if not np.isfinite([self.alpha, self.beta]).all() or min(self.alpha, self.beta) < 0:
            raise ValueError("alpha and beta must be finite and nonnegative")
        self.q = np.array(q, dtype=np.float64, order="C", copy=True)
        if self.q.ndim != 2 or min(self.q.shape) == 0:
            raise ValueError("q must be a nonempty candidate-by-scenario matrix")
        if not np.isfinite(self.q).all() or np.any(self.q < 0):
            raise ValueError("q must contain finite, nonnegative risks")
        self.N, self.B = self.q.shape
        self.K = index(K)
        if not 1 <= self.K <= self.N:
            raise ValueError("K must be between 1 and the number of candidates")
        self.qbar = float(self.q.mean())
        if not np.isfinite(self.qbar) or self.qbar <= 0:
            raise ValueError("mean risk must be finite and positive")

        self.city_prov = _indices(city_prov, "city_prov")
        self.G = len(self.city_prov)
        if not 1 <= self.G <= self.N or np.any((self.city_prov < 0) | (self.city_prov >= self.G)):
            raise ValueError("city_prov must assign each prefecture a valid province index")
        self.P = int(self.city_prov.max()) + 1
        self.city = _indices(city, "city", (self.N,), self.G)
        self.prov = _indices(prov, "prov", (self.N,), self.P)
        if not np.array_equal(self.city_prov[self.city], self.prov):
            raise ValueError("prov must agree with city_prov for every candidate")

        self.N_city = np.bincount(self.city, minlength=self.G)
        self.N_prov = np.bincount(self.prov, minlength=self.P)
        if np.any(self.N_city == 0) or np.any(self.N_prov == 0):
            raise ValueError("prefecture and province indices must be contiguous and populated")

        raw_edges = np.asarray(edges)
        if raw_edges.ndim != 2 or raw_edges.shape[1] != 2 or len(raw_edges) == 0:
            raise ValueError("edges must contain at least one pair of adjacent prefectures")
        self.edges = _indices(raw_edges, "edges", raw_edges.shape, self.G)
        undirected = np.sort(self.edges, axis=1)
        if np.any(undirected[:, 0] == undirected[:, 1]) or len(np.unique(undirected, axis=0)) != len(undirected):
            raise ValueError("list each undirected edge once and omit self-edges")
        self.rho = _weights(rho, "rho", self.G)
        self.a_city = _weights(a_city, "a_city", self.G)
        self.a_prov = _weights(a_prov, "a_prov", self.P)
        self.city_ids = [np.flatnonzero(self.city == g) for g in range(self.G)]
        self.prov_ids = [np.flatnonzero(self.prov == p) for p in range(self.P)]
        self.D = np.stack([self.q[ids].sum(axis=0) for ids in self.city_ids])   # (G,B)
        self.U0 = (self.D > 0).astype(np.float64)
        self.inv_D = np.divide(1.0, self.D, out=np.zeros_like(self.D), where=self.D > 0)

        e0, e1 = self.edges[:, 0], self.edges[:, 1]
        self.e0, self.e1 = e0, e1
        w = np.sqrt(self.rho[e0] * self.rho[e1])
        self.W = float(w.sum())
        if not np.isfinite(self.W) or not self.W > 0:
            raise ValueError("adjacency weight sum must be positive")
        self.edge_w = w / self.W
        # Store both directions of each undirected edge.
        rows = np.r_[e0, e1]
        cols = np.r_[e1, e0]
        vals = np.r_[self.edge_w, self.edge_w]
        order = np.argsort(rows, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        bounds = np.searchsorted(rows, np.arange(self.G + 1))
        self._nbr_idx = [cols[bounds[g]:bounds[g + 1]] for g in range(self.G)]
        self._nbr_w = [vals[bounds[g]:bounds[g + 1]] for g in range(self.G)]
        self.degree = np.array([len(v) for v in self._nbr_idx])
        self.neigh = sparse.csr_matrix((vals, (rows, cols)), shape=(self.G, self.G))
        self.adj_rowsum = np.asarray(self.neigh.sum(axis=1)).ravel()
        self.adj_rowsum_safe = np.maximum(self.adj_rowsum, 1e-12)

        # Tabulate logarithmic coverage at each possible allocation count.
        self.m_city = np.maximum(1, np.ceil(self.K * self.N_city / self.N)).astype(np.int64)
        self.m_prov = np.maximum(1, np.ceil(self.K * self.N_prov / self.N)).astype(np.int64)
        self.city_coef = OMEGA_CITY * self.a_city / self.a_city.sum()
        self.prov_coef = OMEGA_PROV * self.a_prov / self.a_prov.sum()
        n_c = np.arange(self.K + 2, dtype=np.float64)
        self.phi_city = np.stack([
            np.minimum(1.0, np.log1p(n_c) / np.log1p(m)) for m in self.m_city])
        self.phi_prov = np.stack([
            np.minimum(1.0, np.log1p(n_c) / np.log1p(m)) for m in self.m_prov])
        self.city_risk_mean = np.array([self.q[ids].mean() for ids in self.city_ids])
        self.qmean = self.q.mean(axis=1)
        self.qstatic_order = np.argsort(-self.qmean)
        self._arangeG = np.arange(self.G)
        self._arangeP = np.arange(self.P)

    def _rank_sort(self, S):
        S = np.asarray(S, dtype=np.int64)
        return S[np.argsort(self.city[S], kind="stable")]

    def _parts(self, S):
        S = self._rank_sort(S)
        qs = np.ascontiguousarray(self.q[S, :].T)          # (B,K)
        F = qs.sum(axis=1)
        r = float(tails(F, TAU_R)) / (self.K * self.qbar)
        cs = self.city[S]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        Zseg = np.add.reduceat(qs, starts, axis=1)          # (B, nseg)
        seg_city = cs[starts]
        U = self.U0.copy()
        U[seg_city] -= Zseg.T * self.inv_D[seg_city]
        H = np.einsum("e,eb,eb->b", self.edge_w, U[self.e0], U[self.e1])
        cg = 1.0 - float(tails(H, TAU_G, upper=True))
        cnt_c = np.bincount(cs, minlength=self.G)
        cnt_p = np.bincount(self.prov[S], minlength=self.P)
        ch = float(self.city_coef @ self.phi_city[self._arangeG, cnt_c]
                   + self.prov_coef @ self.phi_prov[self._arangeP, cnt_p])
        J = r + self.alpha * ch + self.beta * cg
        return J, r, ch, cg

    def evaluate(self, selection):
        """Return objective, normalized return, coverage, and protection for K distinct candidates."""
        selection = _indices(selection, "selection", (self.K,), self.N)
        if len(np.unique(selection)) != self.K:
            raise ValueError("selection must contain distinct indices")
        return self._parts(selection)

    def state(self):
        return State(self)

class State:
    """Incremental risk, coverage, and adjacency state for a partial allocation."""

    __slots__ = ("inst", "selected", "nc", "np_", "F", "U", "Umean", "Unbr", "H",
                 "ch", "cg", "J", "t", "arangeG", "arangeP")

    def __init__(self, inst: Instance):
        self.inst = inst
        self.selected = np.zeros(inst.N, dtype=bool)
        self.nc = np.zeros(inst.G, dtype=np.int64)
        self.np_ = np.zeros(inst.P, dtype=np.int64)
        self.F = np.zeros(inst.B)
        self.U = inst.U0.copy()
        self.Umean = self.U.mean(axis=1)
        self.Unbr = inst.neigh @ self.Umean
        self.H = np.einsum("e,eb,eb->b", inst.edge_w, self.U[inst.e0], self.U[inst.e1])
        self.ch = 0.0
        self.cg = 1.0 - float(tails(self.H, TAU_G, upper=True))
        self.J = inst.beta * self.cg
        self.t = 0
        self.arangeG = inst._arangeG
        self.arangeP = inst._arangeP

    def take(self, i: int) -> float:
        if self.selected[i]:
            raise AssertionError("duplicate action")
        I = self.inst
        g, p = I.city[i], I.prov[i]
        self.F += I.q[i]
        di = I.q[i] * I.inv_D[g]
        nbr = I._nbr_idx[g]
        if len(nbr):
            self.H -= di * (I._nbr_w[g] @ self.U[nbr])
        self.U[g] -= di
        dm = float(di.mean())
        self.Umean[g] -= dm
        if len(nbr):
            self.Unbr[nbr] -= dm * I._nbr_w[g]
        if self.nc[g] + 1 <= I.K:
            self.ch += I.city_coef[g] * (I.phi_city[g, self.nc[g] + 1] - I.phi_city[g, self.nc[g]])
        if self.np_[p] + 1 <= I.K:
            self.ch += I.prov_coef[p] * (I.phi_prov[p, self.np_[p] + 1] - I.phi_prov[p, self.np_[p]])
        self.nc[g] += 1
        self.np_[p] += 1
        self.selected[i] = True
        self.t += 1
        self.cg = 1.0 - float(tails(self.H, TAU_G, upper=True))
        self.J = (float(tails(self.F, TAU_R)) / (I.K * I.qbar)
                  + I.alpha * self.ch + I.beta * self.cg)
        return self.J

    def clone(self):
        o = State.__new__(State)
        o.inst = self.inst
        o.selected = self.selected.copy()
        o.nc = self.nc.copy()
        o.np_ = self.np_.copy()
        o.F = self.F.copy()
        o.U = self.U.copy()
        o.Umean = self.Umean.copy()
        o.Unbr = self.Unbr.copy()
        o.H = self.H.copy()
        o.ch = self.ch
        o.cg = self.cg
        o.J = self.J
        o.t = self.t
        o.arangeG = self.arangeG
        o.arangeP = self.arangeP
        return o

    def gains(self, cand=None):
        """Evaluate exact marginal gains for the supplied candidates."""
        I = self.inst
        if cand is None:
            cand = np.flatnonzero(~self.selected)
        cand = np.asarray(cand, dtype=np.int64)
        if len(cand) == 0:
            return np.zeros(0)
        fb = self.F[None, :] + I.q[cand]                       # (M,B)
        r2 = tails(fb, TAU_R) / (I.K * I.qbar)
        nbr_sum = I.neigh @ self.U                              # (G,B)
        dh = -(I.q[cand] * I.inv_D[I.city[cand]]) * nbr_sum[I.city[cand]]
        cg2 = 1.0 - tails(self.H[None, :] + dh, TAU_G, upper=True)
        gidx = I.city[cand]
        pidx = I.prov[cand]
        nc1 = self.nc[gidx] + 1
        np1 = self.np_[pidx] + 1
        oldc = I.phi_city[gidx, self.nc[gidx]]
        newc = I.phi_city[gidx, nc1]
        oldp = I.phi_prov[pidx, self.np_[pidx]]
        newp = I.phi_prov[pidx, np1]
        ch2 = self.ch + I.city_coef[gidx] * (newc - oldc) + I.prov_coef[pidx] * (newp - oldp)
        J2 = r2 + I.alpha * ch2 + I.beta * cg2
        return J2 - self.J


def greedy(inst: Instance, K: int | None = None, screen: int | None = None,
           static_order=None):
    """Construct an allocation greedily using gains in the full objective."""
    K = inst.K if K is None else int(K)
    st = inst.state()
    if static_order is None:
        static_order = inst.qstatic_order
    if screen is None or screen >= inst.N:
        for _ in range(K):
            cand = np.flatnonzero(~st.selected)
            g = st.gains(cand)
            st.take(int(cand[int(np.argmax(g))]))
    else:
        pool = static_order[:max(screen, 2 * K)]
        for _ in range(K):
            cand = pool[~st.selected[pool]]
            if len(cand) < 8:
                cand = np.flatnonzero(~st.selected)
            g = st.gains(cand)
            st.take(int(cand[int(np.argmax(g))]))
    return np.flatnonzero(st.selected)


def greedy_restricted(inst: Instance, pool, K: int | None = None, block: int = 1):
    """Construct an allocation within the pool, optionally selecting in blocks."""
    K = inst.K if K is None else int(K)
    st = inst.state()
    pool = np.asarray(pool, dtype=np.int64)
    while st.t < K:
        cand = pool[~st.selected[pool]]
        if len(cand) == 0:
            break
        g = st.gains(cand)
        take = cand[np.argsort(-g)[:max(1, block)]]
        for i in take:
            if st.t >= K:
                break
            st.take(int(i))
    return np.flatnonzero(st.selected)

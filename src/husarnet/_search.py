"""Objective-based local search."""

import time

import numpy as np


def swap_search_adaptive(inst, selected, max_seconds: float = 10.0, min_rounds: int = 4,
                         patience: int = 8, remove_pool: int = 16, add_pool: int = 40,
                         seed: int = 0):
    """Improve a complete allocation through one-for-one exchanges."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    S = np.sort(np.asarray(selected, dtype=np.int64)).copy()
    cur = inst._parts(S)[0]
    no_imp = 0
    rounds = 0
    hard_stop = False
    while rounds < 200 and not hard_stop:
        if rounds >= 1 and time.perf_counter() - t0 >= max_seconds:
            break
        if rounds >= min_rounds and no_imp >= patience:
            break
        rounds += 1
        mask = np.zeros(inst.N, dtype=bool)
        mask[S] = True
        st = inst.state()
        for i in S:
            st.take(int(i))
        pool = np.flatnonzero(~mask)
        if len(pool) > add_pool:
            screen = pool
            if len(pool) > 2000:
                screen = pool[np.argsort(-inst.qmean[pool])[:2000]]
            scores = st.gains(screen)
            hi = screen[np.argsort(-scores)[: max(1, add_pool * 3 // 4)]]
            others = np.setdiff1d(screen, hi)
            extra = rng.choice(others, size=min(len(others), add_pool - len(hi)), replace=False)
            pool = np.concatenate([hi, extra])
        if len(S) > remove_pool:
            least = S[np.argsort(inst.qmean[S])[: max(1, remove_pool * 3 // 4)]]
            rest = np.setdiff1d(S, least)
            extra = rng.choice(rest, size=min(len(rest), remove_pool - len(least)), replace=False)
            rem = np.concatenate([least, extra])
        else:
            rem = S
        vals = np.empty(len(rem) * len(pool))
        pairs = []
        k = 0
        for r_i in rem:
            cand_small = S[S != r_i]
            for a_i in pool:
                if a_i == r_i:
                    continue
                vals[k] = inst._parts(np.r_[cand_small, a_i])[0]
                pairs.append((int(r_i), int(a_i)))
                k += 1
            if (rounds >= 2 and k >= 256 and time.perf_counter() - t0 >= max_seconds):
                hard_stop = True
                break
        vals = vals[:k]
        if k and vals.max() > cur + 1e-12:
            r_i, a_i = pairs[int(np.argmax(vals))]
            S = np.sort(np.r_[S[S != r_i], a_i])
            cur = float(vals.max())
            no_imp = 0
        else:
            no_imp += 1
    return S

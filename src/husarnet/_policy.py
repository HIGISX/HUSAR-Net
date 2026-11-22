"""Hierarchical mean-risk policy, imitation learning, and PPO."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ._objective import TAU_R, tails


class NetConfig:
    def __init__(self, hidden: int = 64):
        self.hidden = hidden
        self.feat = 8
        self.encoder = "riskmean"

    def to_dict(self):
        return dict(hidden=self.hidden, feat=self.feat, encoder=self.encoder)


class HUSARNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        h = self.cfg.hidden
        self.firm_enc = nn.Sequential(nn.Linear(self.cfg.feat, h), nn.Tanh(),
                                      nn.Linear(h, h), nn.Tanh())
        self.city_enc = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(),
                                      nn.Linear(h, h), nn.Tanh())
        self.city_actor = nn.Sequential(nn.Linear(h + 5 + 4, h), nn.Tanh(), nn.Linear(h, 1))
        self.firm_actor = nn.Sequential(nn.Linear(2 * h + 8, h), nn.Tanh(), nn.Linear(h, 1))
        self.critic = nn.Sequential(nn.Linear(h + 4, h), nn.Tanh(), nn.Linear(h, 1))

    def encode(self, inst, cache):
        f = self.firm_enc(cache["features"])
        city_sum = torch.zeros((inst.G, self.cfg.hidden))
        city_sum.index_add_(0, cache["city"], f)
        city_avg = city_sum / cache["city_norm"][:, None]
        prov_sum = torch.zeros((inst.P, self.cfg.hidden))
        prov_sum.index_add_(0, cache["city_prov"], city_avg)
        p_count = torch.bincount(cache["city_prov"], minlength=inst.P).float()[:, None]
        prov_avg = prov_sum / p_count
        nbr = cache["adj_norm"] @ city_avg
        c = self.city_enc(torch.cat([city_avg, nbr, prov_avg[cache["city_prov"]]], dim=-1))
        return f, c

    @staticmethod
    def dynamic(inst, st):
        nbr_um = st.Unbr / inst.adj_rowsum_safe
        remain = (inst.N_city - st.nc) / np.maximum(inst.N_city, 1)
        citydyn = np.stack([st.nc / np.maximum(inst.m_city, 1), st.Umean, nbr_um, remain,
                            inst.city_risk_mean / inst.qbar], axis=1).astype(np.float32)
        glob = np.array([st.t / inst.K,
                         float(tails(st.F, TAU_R)) / (inst.K * inst.qbar),
                         st.ch, st.cg], dtype=np.float32)
        return torch.from_numpy(citydyn), torch.from_numpy(glob)

    def city_logits(self, inst, st, c, citydyn, glob):
        logits = self.city_actor(torch.cat([c, citydyn, glob.expand(inst.G, -1)], dim=1)).squeeze(-1)
        avail = torch.from_numpy(st.nc < inst.N_city)
        return logits.masked_fill(~avail, -1e9)

    def firm_logits(self, inst, st, f, c, g, glob, cache, pool_cap=None):
        ids = inst.city_ids[g]
        if pool_cap is not None and len(ids) > pool_cap:
            keep = np.argsort(-inst.qmean[ids])[:pool_cap]
            screened = ids[np.sort(keep)]
            if np.any(~st.selected[screened]):
                ids = screened
        dyn_g = np.array([st.nc[g] / max(1, inst.m_city[g]), st.Umean[g]], dtype=np.float32)
        dyn = torch.from_numpy(np.repeat(dyn_g[None, :], len(ids), axis=0))
        x = torch.cat([f[ids], c[g].expand(len(ids), -1), dyn,
                       cache["firm_qd"][ids], glob.expand(len(ids), -1)], dim=1)
        logits = self.firm_actor(x).squeeze(-1)
        mask = torch.from_numpy(st.selected[ids])
        return ids, logits.masked_fill(mask, -1e9)

    def critic_value(self, c, glob):
        return self.critic(torch.cat([c.mean(dim=0), glob], dim=0)).squeeze(-1)


def make_cache(inst) -> dict:
    mean = inst.qmean / inst.qbar
    dmean = (inst.q * inst.inv_D[inst.city]).mean(axis=1) * inst.G
    feats = np.zeros((inst.N, 8), dtype=np.float32)
    feats[:, 0] = np.clip(mean, 0.0, 8.0)
    adj = inst.neigh.toarray().astype(np.float32)
    adj_norm = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-12)
    return {
        "features": torch.from_numpy(feats),
        "city": torch.from_numpy(inst.city.astype(np.int64)),
        "city_prov": torch.from_numpy(inst.city_prov.astype(np.int64)),
        "city_norm": torch.from_numpy(inst.N_city.astype(np.float32)),
        "adj_norm": torch.from_numpy(adj_norm),
        "firm_qd": torch.from_numpy(np.stack([np.clip(mean, 0, 8), np.clip(dmean, 0, 8)],
                                             axis=1).astype(np.float32)),
    }


@torch.no_grad()
def rollout(model: HUSARNet, inst, cache=None, stochastic: bool = False, seed=None,
            temperature: float = 1.0, pool_cap: int | None = 1500):
    cache = cache or make_cache(inst)
    f, c = model.encode(inst, cache)
    st = inst.state()
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    acts, logps, vals, rews, ents = [], [], [], [], []
    for _ in range(inst.K):
        citydyn, glob = model.dynamic(inst, st)
        cl = model.city_logits(inst, st, c, citydyn, glob)
        cp = torch.distributions.Categorical(logits=cl / temperature)
        g = int(torch.multinomial(cp.probs, 1, generator=rng).item()) if stochastic else int(torch.argmax(cl))
        ids, fl = model.firm_logits(inst, st, f, c, g, glob, cache, pool_cap)
        fp = torch.distributions.Categorical(logits=fl / temperature)
        j = int(torch.multinomial(fp.probs, 1, generator=rng).item()) if stochastic else int(torch.argmax(fl))
        act = int(ids[j])
        lp = float(cp.log_prob(torch.tensor(g)) + fp.log_prob(torch.tensor(j)))
        ent = float(cp.entropy() + fp.entropy())
        v = float(model.critic_value(c, glob))
        before = st.J
        st.take(act)
        acts.append(act)
        logps.append(lp)
        vals.append(v)
        rews.append(st.J - before)
        ents.append(ent)
    S = np.flatnonzero(st.selected)
    return S, acts, logps, vals, rews, ents


def greedy_rollout(model, inst, cache=None, pool_cap: int | None = 1500):
    S, acts, _, _, rews, _ = rollout(model, inst, cache, stochastic=False, pool_cap=pool_cap)
    return S, acts, float(np.sum(rews))


def boundary_states(inst, acts, chunk: int):
    """Save the state at the start of each training chunk."""
    st = inst.state()
    out = [st.clone()]
    for t, a in enumerate(acts):
        st.take(int(a))
        if (t + 1) % chunk == 0 and t + 1 < len(acts):
            out.append(st.clone())
    return out


def expert_order(inst, expert):
    """Order teacher candidates by their marginal gain at each step."""
    st = inst.state()
    remain = list(map(int, np.sort(np.asarray(expert, dtype=np.int64))))
    order = []
    for _ in range(inst.K):
        cand = np.asarray(remain, dtype=np.int64)
        g = st.gains(cand)
        k = int(np.argmax(g))
        i = int(cand[k])
        order.append(i)
        st.take(i)
        remain.remove(i)
    return np.asarray(order, dtype=np.int64)


def _chunk_loss(model, inst, cache, st, acts, need_value=False, returns=None,
                old_logp=None, adv=None, clip=None):
    f, c = model.encode(inst, cache)
    loss = torch.zeros(())
    lps, vals, ents = [], [], []
    for t, act in enumerate(acts):
        citydyn, glob = model.dynamic(inst, st)
        cl = model.city_logits(inst, st, c, citydyn, glob)
        g = int(inst.city[act])
        ids, fl = model.firm_logits(inst, st, f, c, g, glob, cache)
        loc = int(np.flatnonzero(ids == act)[0])
        cp = torch.distributions.Categorical(logits=cl)
        fp = torch.distributions.Categorical(logits=fl)
        lp = cp.log_prob(torch.tensor(g)) + fp.log_prob(torch.tensor(loc))
        lps.append(lp)
        ents.append(cp.entropy() + fp.entropy())
        if need_value:
            vals.append(model.critic_value(c, glob))
        if not need_value:
            loss = loss - lp
        st.take(int(act))
    if need_value:
        lp_t = torch.stack(lps)
        v_t = torch.stack(vals)
        ent_t = torch.stack(ents)
        old = torch.tensor(old_logp, dtype=torch.float32)
        advantage = torch.tensor(adv, dtype=torch.float32)
        ret = torch.tensor(returns, dtype=torch.float32)
        ratio = torch.exp(lp_t - old)
        pol = -torch.minimum(ratio * advantage,
                             torch.clamp(ratio, 1 - clip, 1 + clip) * advantage).mean()
        loss = pol + 0.4 * (v_t - ret).square().mean() - 0.005 * ent_t.mean()
    return loss


def imitation_train(model, instances, orders, caches=None, epochs: int = 12,
                    lr: float = 1e-3, seed: int = 2026, chunk: int = 256,
                    verbose: bool = False):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    caches = caches or [make_cache(I) for I in instances]
    bounds = [boundary_states(instances[j], orders[j], chunk) for j in range(len(instances))]
    history = []
    for ep in range(epochs):
        losses = []
        for j in rng.permutation(len(instances)):
            inst, cache, order = instances[j], caches[j], orders[j]
            model.train()
            n_chunk = len(bounds[j])
            ep_loss = 0.0
            for b in range(n_chunk):
                st = bounds[j][b].clone()
                seg = order[b * chunk:(b + 1) * chunk]
                loss = _chunk_loss(model, inst, cache, st, seg)
                loss = loss / max(1, len(order))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                ep_loss += float(loss.detach()) * len(seg) / max(1, len(order))
            losses.append(ep_loss)
        history.append(dict(stage="imitation", iteration=ep + 1, loss=float(np.mean(losses))))
        if verbose:
            print(f"[BC] epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    return history


def ppo_train(model, train_instances, val_instances, rounds: int = 16,
              rollouts_per_round: int = 8, epochs_per_round: int = 2,
              lr: float = 2e-4,
              clip: float = 0.2, seed: int = 2026, verbose: bool = False,
              train_caches=None, val_caches=None, chunk: int = 256,
              time_budget: float | None = None, pool_cap: int | None = 1500):
    import time
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_caches = train_caches or [make_cache(I) for I in train_instances]
    val_caches = val_caches or [make_cache(I) for I in val_instances]

    def val_score():
        s = []
        for j, inst in enumerate(val_instances):
            S, _, _ = greedy_rollout(model, inst, val_caches[j], pool_cap)
            s.append(inst._parts(S)[0])
        return float(np.mean(s))

    best_score = val_score()
    best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    t_start = time.perf_counter()
    for r in range(rounds):
        if time_budget and time.perf_counter() - t_start > time_budget:
            break
        trajectories = []
        for _ in range(rollouts_per_round):
            j = int(rng.integers(len(train_instances)))
            inst, cache = train_instances[j], train_caches[j]
            S, acts, logps, vals, rews, _ = rollout(model, inst, cache, stochastic=True,
                                                    seed=int(rng.integers(1 << 30)),
                                                    pool_cap=pool_cap)
            rews = np.asarray(rews, dtype=np.float64)
            vals = np.asarray(vals, dtype=np.float64)
            ret = np.cumsum(rews[::-1])[::-1].copy()
            adv = ret - vals
            trajectories.append(dict(j=j, acts=acts, logps=np.asarray(logps),
                                     adv=adv, returns=ret,
                                     bounds=boundary_states(inst, acts, chunk)))
        all_adv = np.concatenate([t["adv"] for t in trajectories])
        mu, sd = float(all_adv.mean()), float(all_adv.std() + 1e-8)
        losses = []
        for ep in range(epochs_per_round):
            for tr in trajectories:
                inst, cache = train_instances[tr["j"]], train_caches[tr["j"]]
                model.train()
                for b in range(len(tr["bounds"])):
                    st = tr["bounds"][b].clone()
                    sl = slice(b * chunk, (b + 1) * chunk)
                    loss = _chunk_loss(model, inst, cache, st, tr["acts"][sl],
                                       need_value=True, returns=tr["returns"][sl],
                                       old_logp=tr["logps"][sl],
                                       adv=(tr["adv"][sl] - mu) / sd, clip=clip)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                    losses.append(float(loss.detach()))
        score = val_score()
        history.append(dict(stage="ppo", iteration=r + 1, loss=float(np.mean(losses)),
                            validation_J=score))
        if verbose:
            print(f"[PPO] round {r+1}/{rounds} loss={np.mean(losses):.4f} val_J={score:.5f} "
                  f"best={best_score:.5f}", flush=True)
        if score > best_score:
            best_score = score
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return history, best_score


def save_model(model, path, extra=None):
    torch.save(dict(model_state_dict=model.state_dict(), cfg=model.cfg.to_dict(),
                    extra=extra or {}), path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    c = ck["cfg"]
    if c.get("encoder") != "riskmean" or c.get("feat") != 8:
        raise ValueError("checkpoint must be a HUSAR-Net riskmean policy")
    cfg = NetConfig(hidden=c["hidden"])
    m = HUSARNet(cfg)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck.get("extra", {})

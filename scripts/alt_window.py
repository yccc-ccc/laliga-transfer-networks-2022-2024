# -*- coding: utf-8 -*-
"""Alternative-window robustness: July 1 - August 31 instead of June 30 - September 2.

Rebuilds the directed club networks from the harmonized event samples under the
narrower window, re-estimates the binary Borgatti-Everett core-periphery fit
(same search protocol as the main analysis: degree-initialized + 30 random
restarts per k, kmax=40, seed=42), and runs a degree-preserving fixed-k null
test (double-edge swaps, 10*m swaps per simulation, plus-one p-value).

import-safe for multiprocessing on Windows.
"""
import os, sys, time
import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
OUT = os.path.join(BASE, "..", "results")
os.makedirs(OUT, exist_ok=True)

WIN_START, WIN_END = "07-01", "08-31"   # alternative window (month-day)
YEARS = (2022, 2023, 2024)

# ---------- core-periphery estimation (same algorithm as scripts/coreperiphery.py) ----------
def pearson_cp(A, c):
    iu = np.triu_indices(len(c), 1)
    x = A[iu]
    y = (1 - (1 - c[iu[0]]) * (1 - c[iu[1]])).astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def hill_climb(A, k, c0):
    c = c0.copy(); d = A.sum(axis=1)
    while True:
        s = A @ c
        ci = np.where(c == 1)[0]; pi = np.where(c == 0)[0]
        if len(pi) == 0: break
        base = 2*(d[pi][None,:]-d[ci][:,None]) - 2*(s[pi][None,:]-A[np.ix_(pi,ci)].T) + 2*s[ci][:,None]
        i, j = np.unravel_index(np.argmax(base), base.shape)
        if base[i, j] <= 1e-12: break
        c[ci[i]] = 0; c[pi[j]] = 1
    return c

def estimate(A, restarts=30, seed=42, kmax=40):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    kmax = min(kmax, n - 1)
    deg = A.sum(axis=1)
    best = (-1.0, 0, None)
    for k in range(1, kmax + 1):
        c = np.zeros(n, dtype=int)
        c[np.argsort(-deg)[:k]] = 1
        c = hill_climb(A, k, c)
        rho = pearson_cp(A, c)
        if rho > best[0]:
            best = (rho, k, c.copy())
        for _ in range(restarts):
            c = np.zeros(n, dtype=int)
            c[rng.choice(n, size=k, replace=False)] = 1
            c = hill_climb(A, k, c)
            rho = pearson_cp(A, c)
            if rho > best[0]:
                best = (rho, k, c.copy())
    return best

# ---------- null model (same machinery as scripts/null_model.py) ----------
def best_at_k(A, k, rng, restarts=3):
    n = A.shape[0]; deg = A.sum(axis=1)
    c = np.zeros(n, dtype=int); c[np.argsort(-deg)[:k]] = 1
    c = hill_climb(A, k, c); best = pearson_cp(A, c)
    for _ in range(restarts):
        c = np.zeros(n, dtype=int); c[rng.choice(n, k, replace=False)] = 1
        c = hill_climb(A, k, c)
        r = pearson_cp(A, c)
        if r > best: best = r
    return best

def swap_edges(eu, ev, nswap, rng):
    m = len(eu)
    eset = set(zip(eu.tolist(), ev.tolist()))
    for _ in range(nswap):
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = eu[i], ev[i]; c, d = eu[j], ev[j]
        if rng.random() < 0.5:
            na, nb, nc, nd = a, d, c, b
        else:
            na, nb, nc, nd = a, c, d, b
        if na == nb or nc == nd: continue
        e1 = (min(na, nb), max(na, nb)); e2 = (min(nc, nd), max(nc, nd))
        if e1 == e2 or e1 in eset or e2 in eset: continue
        eset.discard((a, b)); eset.discard((c, d))
        eset.add(e1); eset.add(e2)
        eu[i], ev[i] = e1; eu[j], ev[j] = e2

# ---------- network construction under the alternative window ----------
def build_alt(year):
    ev = pd.read_csv(os.path.join(DATA, f"events_{year}.csv"), dtype=str)
    d = pd.to_datetime(ev["transfer_date"])
    lo = pd.Timestamp(f"{year}-{WIN_START}"); hi = pd.Timestamp(f"{year}-{WIN_END}")
    ev = ev[(d >= lo) & (d <= hi)]
    G = nx.DiGraph()
    for _, r in ev.iterrows():
        f, t = r["from_team_id"], r["to_team_id"]
        if f == t: continue
        if G.has_edge(f, t):
            G[f][t]["weight"] += 1
        else:
            G.add_edge(f, t, weight=1)
    nodes = sorted(G.nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    seen = set(); eu, evv = [], []
    for u, v in G.edges:
        a, b = idx[u], idx[v]
        key = (min(a, b), max(a, b))
        A[key] = 1.0; A[key[1], key[0]] = 1.0
        if key not in seen:
            seen.add(key); eu.append(key[0]); evv.append(key[1])
    return ev, G, nodes, A, np.array(eu), np.array(evv)

def run_chunk(args):
    year, n_iter, k, seed = args
    _, _, nodes, A_obs, eu0, ev0 = build_alt(year)
    n = len(nodes); m = len(eu0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_iter)
    for i in range(n_iter):
        eu, ev = eu0.copy(), ev0.copy()
        swap_edges(eu, ev, 10*m, rng)
        A = np.zeros((n, n)); A[eu, ev] = 1.0; A[ev, eu] = 1.0
        out[i] = best_at_k(A, k, rng, restarts=3)
    return out

def main(n_null=1000, nproc=8):
    from multiprocessing import Pool
    rows = []
    for year in YEARS:
        ev, G, nodes, A, eu, evv = build_alt(year)
        print(f"{year}: 替代窗口事件 {len(ev)} | 节点 {G.number_of_nodes()} | 有向边 {G.number_of_edges()} "
              f"| 无向简单边 {len(eu)}", flush=True)
        t0 = time.time()
        rho, k, c = estimate(A, restarts=30, seed=42, kmax=40)
        print(f"  观测拟合: k={k} ρ={rho:.4f} ({time.time()-t0:.0f}s)，启动 {n_null} 次固定k零模型...", flush=True)
        chunk = 250
        n_chunks = (n_null + chunk - 1) // chunk
        tasks = [(year, min(chunk, n_null - i*chunk), k, 777000 + year*1000 + i) for i in range(n_chunks)]
        with Pool(nproc) as pool:
            nulls = np.concatenate(list(pool.imap(run_chunk, tasks)))
        b = int((nulls >= rho).sum()); p = (b + 1) / (len(nulls) + 1)
        print(f"  零模型: 均值 {nulls.mean():.4f} | 最大 {nulls.max():.4f} | b={b} p={p:.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(window=year, events=len(ev), nodes=G.number_of_nodes(),
                         dir_edges=G.number_of_edges(), k=k, rho=round(rho, 6),
                         null_mean=round(float(nulls.mean()), 6),
                         null_max=round(float(nulls.max()), 6),
                         n_null=len(nulls), b=b, p=round(p, 6)))
        np.save(os.path.join(OUT, f"null_altwin_{year}.npy"), nulls)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "alt_window_robustness.csv"), index=False)
    print(df.to_string(index=False))
    print(f"saved -> {os.path.join(OUT, 'alt_window_robustness.csv')}")

if __name__ == "__main__":
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n_null)

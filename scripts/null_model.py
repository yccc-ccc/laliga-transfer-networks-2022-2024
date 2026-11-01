# -*- coding: utf-8 -*-
"""保度零模型：二元无向简单图上双边交换（10×边数），固定 k 重估。
核对论文 4.4：零模型均值/最大值与经验 p 值。
import-safe：主流程在 __main__ 保护下，支持多进程与分块运行。"""
import numpy as np
import networkx as nx
import time, os, sys

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

def pearson_cp(A, c):
    iu = np.triu_indices(len(c), 1)
    x = A[iu]; y = (1-(1-c[iu[0]])*(1-c[iu[1]])).astype(float)
    if x.std() == 0 or y.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

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

def load_graph(year):
    G = nx.read_graphml(f"data/network_{year}.graphml")
    nodes = sorted(G.nodes); n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    eu, ev = [], []
    seen = set()
    for u, v in G.edges:
        a, b = idx[u], idx[v]
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key); eu.append(key[0]); ev.append(key[1])
    return nodes, np.array(eu), np.array(ev)

def swap_edges(eu, ev, nswap, rng):
    """边列表上的保度双边交换（无自环/重边），就地修改 eu/ev。"""
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

def edges_to_A(eu, ev, n):
    A = np.zeros((n, n))
    A[eu, ev] = 1.0; A[ev, eu] = 1.0
    return A

def run_chunk(args):
    """跑一批零模型，返回 rho 数组。args=(year, n_iter, k, seed)"""
    year, n_iter, k, seed = args
    nodes, eu0, ev0 = load_graph(year)
    n = len(nodes); m = len(eu0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_iter)
    for i in range(n_iter):
        eu, ev = eu0.copy(), ev0.copy()
        swap_edges(eu, ev, 10*m, rng)
        A = edges_to_A(eu, ev, n)
        out[i] = best_at_k(A, k, rng, restarts=3)
    return out

EXP = {
    2022: dict(k=18, obs=0.298, nulmean=0.251460, nulmax=0.263911),
    2023: dict(k=18, obs=0.295, nulmean=0.246780, nulmax=0.259089),
    2024: dict(k=19, obs=0.296, nulmean=0.246720, nulmax=0.259759),
}

def main(year, total=5000, nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(8, os.cpu_count() or 4)
    k = EXP[year]["k"]
    chunk = 250
    n_chunks = (total + chunk - 1) // chunk
    tasks = [(year, min(chunk, total - c*chunk), k, 2024000 + year*1000 + c)
             for c in range(n_chunks)]
    t0 = time.time(); results = []
    with Pool(nproc) as pool:
        for ci, r in enumerate(pool.imap(run_chunk, tasks)):
            results.append(r)
            if (ci+1) % 4 == 0 or ci == n_chunks-1:
                done = sum(len(x) for x in results)
                el = time.time()-t0
                print(f"  [{year}] {done}/{total} 完成，耗时 {el:.0f}s，"
                      f"预计剩余 {el/done*(total-done):.0f}s", flush=True)
    nulls = np.concatenate(results)
    np.save(f"data/null_fixedk_{year}.npy", nulls)
    obs = EXP[year]["obs"]
    b = int((nulls >= obs).sum()); p = (b+1)/(total+1)
    print(f"{year}: {total}次固定k零模型 | 均值 {nulls.mean():.6f} (论文 {EXP[year]['nulmean']}) | "
          f"最大 {nulls.max():.6f} (论文 {EXP[year]['nulmax']}) | 观测 {obs} | b={b} p={p:.6f} (论文 0.000200) | "
          f"总耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    main(year, total)

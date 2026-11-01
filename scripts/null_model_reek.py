# -*- coding: utf-8 -*-
"""重估 k 零模型：每个保度零网络上重新搜索最优 k（1..40），取最大 ρ。
500 次模拟，核对论文 4.4 的经验 p 值（论文 p=0.001996，即 b=0/500）。"""
import numpy as np
import time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from null_model import load_graph, swap_edges, edges_to_A, EXP
from coreperiphery import estimate

OBS_RHO = {2022: 0.298, 2023: 0.295, 2024: 0.296}

def run_chunk(args):
    year, n_iter, seed = args
    nodes, eu0, ev0 = load_graph(year)
    n = len(nodes); m = len(eu0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_iter)
    for i in range(n_iter):
        eu, ev = eu0.copy(), ev0.copy()
        swap_edges(eu, ev, 10*m, rng)
        A = edges_to_A(eu, ev, n)
        rho, k, _ = estimate(A, restarts=3, seed=int(rng.integers(2**31)), kmax=40)
        out[i] = rho
    return out

def main(year, total=500, nproc=None):
    from multiprocessing import Pool
    nproc = nproc or min(12, os.cpu_count() or 4)
    chunk = 25
    n_chunks = (total + chunk - 1) // chunk
    tasks = [(year, min(chunk, total - c*chunk), 3033000 + year*100 + c)
             for c in range(n_chunks)]
    t0 = time.time(); results = []
    with Pool(nproc) as pool:
        for ci, r in enumerate(pool.imap(run_chunk, tasks)):
            results.append(r)
            done = sum(len(x) for x in results)
            el = time.time() - t0
            print(f"  [{year}] {done}/{total} 完成，耗时 {el:.0f}s，"
                  f"预计剩余 {el/done*(total-done):.0f}s", flush=True)
    nulls = np.concatenate(results)
    np.save(f"data/null_reek_{year}.npy", nulls)
    obs = OBS_RHO[year]
    b = int((nulls >= obs).sum()); p = (b+1)/(total+1)
    print(f"{year}: {total}次重估k零模型 | 均值 {nulls.mean():.6f} | 最大 {nulls.max():.6f} | "
          f"观测 {obs} | b={b} p={p:.6f} (论文 0.001996) | 总耗时 {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(year, total)

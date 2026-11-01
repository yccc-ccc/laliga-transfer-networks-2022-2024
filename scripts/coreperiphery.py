# -*- coding: utf-8 -*-
"""Borgatti–Everett 离散核心—边缘模型：搜索核心规模 k，节点交换爬山，最大化 Pearson ρ。
向量化交换评估，支持二元/次数/费用三种无向矩阵。核对论文 4.4 / 表4。"""
import numpy as np
import networkx as nx

def cp_objective_S(A, c):
    """S = sum_{i!=j} A_ij * delta_ij, delta=1 iff i or j in core。"""
    d = A.sum(axis=1)
    return 2.0 * c @ d - c @ A @ c

def pearson_cp(A, c):
    n = len(c)
    iu = np.triu_indices(n, k=1)
    x = A[iu]
    y = (1 - (1 - c[iu[0]]) * (1 - c[iu[1]])).astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def hill_climb(A, k, c0):
    """从初始归属 c0（含 k 个核心）出发，向量化节点交换至局部最优。"""
    n = len(c0)
    c = c0.copy()
    d = A.sum(axis=1)
    while True:
        s = A @ c                      # s_i = sum_j A_ij c_j
        core_idx = np.where(c == 1)[0]
        per_idx = np.where(c == 0)[0]
        # dS(u,v) = 2(d_v - d_u) - 2(s_v - A_vu) + 2 s_u，行=core u，列=periphery v
        base = 2 * (d[per_idx][None, :] - d[core_idx][:, None]) \
               - 2 * (s[per_idx][None, :] - A[np.ix_(per_idx, core_idx)].T) \
               + 2 * s[core_idx][:, None]
        i, j = np.unravel_index(np.argmax(base), base.shape)
        if base[i, j] <= 1e-12:
            break
        c[core_idx[i]] = 0
        c[per_idx[j]] = 1
    return c

def estimate(A, restarts=100, seed=42, kmax=None):
    """对每个 k 做多次重启，返回全局最优 (rho, k, c)。kmax 限制搜索范围（观测 k≤19，取 40 已宽裕）。"""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    if kmax is None:
        kmax = n - 1
    kmax = min(kmax, n - 1)
    deg = A.sum(axis=1)
    best = (-1.0, 0, None)
    for k in range(1, kmax + 1):
        # 确定性起点：度/强度前 k
        c = np.zeros(n, dtype=int)
        c[np.argsort(-deg)[:k]] = 1
        c = hill_climb(A, k, c)
        rho = pearson_cp(A, c)
        if rho > best[0]:
            best = (rho, k, c.copy())
        # 随机重启
        for _ in range(restarts):
            c = np.zeros(n, dtype=int)
            c[rng.choice(n, size=k, replace=False)] = 1
            c = hill_climb(A, k, c)
            rho = pearson_cp(A, c)
            if rho > best[0]:
                best = (rho, k, c.copy())
    return best

EXP = {  # (k_bin, rho_bin), (k_cnt, rho_cnt), (k_fee, rho_fee)
    2022: ((18, 0.298), (17, 0.264), (3, 0.173)),
    2023: ((18, 0.295), (18, 0.274), (7, 0.154)),
    2024: ((19, 0.296), (19, 0.264), (6, 0.175)),
}

def main():
    results = {}
    for year in (2022, 2023, 2024):
        G = nx.read_graphml(f"data/network_{year}.graphml")
        nodes = sorted(G.nodes)
        n = len(nodes)
        # 二元无向
        Ab = np.zeros((n, n))
        Ac = np.zeros((n, n))
        for u, v, d_ in G.edges(data=True):
            i, j = nodes.index(u), nodes.index(v)
            Ab[i, j] = Ab[j, i] = 1
            Ac[i, j] += d_["weight"]; Ac[j, i] += d_["weight"]
        # 费用子网络（仅正费用边及非孤立节点）
        fee_edges = [(u, v, d_["fee"]) for u, v, d_ in G.edges(data=True) if d_["fee"] > 0]
        Gf = nx.Graph(); Gf.add_weighted_edges_from(fee_edges)
        fnodes = sorted(Gf.nodes)
        nf = len(fnodes)
        Af = np.zeros((nf, nf))
        for u, v, w in fee_edges:
            i, j = fnodes.index(u), fnodes.index(v)
            Af[i, j] += w; Af[j, i] += w

        out = {}
        for tag, M, nodelist in [("二元", Ab, nodes), ("次数", Ac, nodes), ("费用", Af, fnodes)]:
            rho, k, c = estimate(M, restarts=30, kmax=40)
            out[tag] = (rho, k, c, nodelist)
            print(f"  {year} {tag} 完成: k={k} ρ={rho:.4f}", flush=True)
        e = EXP[year]
        rm_bin = "核心" if out["二元"][2][nodes.index("418")] else "外围"
        rm_cnt = "核心" if out["次数"][2][nodes.index("418")] else "外围"
        rm_fee = "核心" if out["费用"][2][fnodes.index("418")] else "外围"
        os_bin = "核心" if out["二元"][2][nodes.index("331")] else "外围"
        print(f"{year}:")
        print(f"  二元 k={out['二元'][1]} ρ={out['二元'][0]:.3f} (论文 k={e[0][0]} ρ={e[0][1]}) | 皇马 {rm_bin} | 奥萨苏纳 {os_bin}")
        print(f"  次数 k={out['次数'][1]} ρ={out['次数'][0]:.3f} (论文 k={e[1][0]} ρ={e[1][1]}) | 皇马 {rm_cnt}")
        print(f"  费用 k={out['费用'][1]} ρ={out['费用'][0]:.3f} (论文 k={e[2][0]} ρ={e[2][1]}) | 皇马 {rm_fee}")
        results[year] = out
    np.save("data/cp_results.npy", results, allow_pickle=True)

if __name__ == "__main__":
    main()

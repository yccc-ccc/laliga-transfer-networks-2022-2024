# -*- coding: utf-8 -*-
"""时间一致性与敏感性分析（论文 4.5 / 4.6）。
1) 选定可复现种子的 Infomap 划分并保存成员表；
2) 共同节点上的 NMI/ARI；核心集合 Jaccard（二元/次数/费用）；
3) 敏感性：移除租借归还、仅永久转会、西甲内部子网络 Infomap；
4) 种子稳定性：30 种子两两平均 NMI。"""
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import random, itertools, sys, os
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coreperiphery import estimate

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}
EXP = {
    "common": {(2022,2023): 131, (2023,2024): 127, (2022,2024): 114},
    "nmi": {(2022,2023): 0.448, (2023,2024): 0.505, (2022,2024): 0.407},
    "ari": {(2022,2023): 0.131, (2023,2024): 0.160, (2022,2024): 0.040},
    "jac_bin": {(2022,2023): 0.636, (2023,2024): 0.682, (2022,2024): 0.609},
    "jac_cnt": {(2022,2023): 0.667, (2023,2024): 0.682, (2022,2024): 0.636},
    "jac_fee": {(2022,2023): 0.250, (2023,2024): 0.444, (2022,2024): 0.500},
    "no_loanret": {2022: 0.274, 2023: 0.275, 2024: 0.275},
    "transfer_only": {2022: 0.247, 2023: 0.255, 2024: 0.255},
    "internal_coms": {2022: 1, 2023: 1, 2024: 3},
    "seed_nmi": {2022: 0.949, 2023: 0.999, 2024: 0.990},
}

def to_igraph(G, directed=True):
    g = ig.Graph(directed=directed)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    g.es["weight"] = [G[u][v]["weight"] for u, v in G.edges]
    return g

def infomap_partition(G, seed, trials=100):
    ig.set_random_number_generator(random.Random(seed))
    g = to_igraph(G, directed=True)
    return g.community_infomap(edge_weights="weight", trials=trials).membership

def build_net(year, keep_types=None, drop_types=None):
    ev = pd.read_csv(f"data/events_{year}.csv", dtype=str)
    if keep_types is not None:
        ev = ev[ev["transfer_type"].isin(keep_types)]
    if drop_types is not None:
        ev = ev[~ev["transfer_type"].isin(drop_types)]
    ev["fee_num"] = pd.to_numeric(ev["transfer_fee"], errors="coerce")
    G = nx.DiGraph()
    for _, r in ev.iterrows():
        f, t = r["from_team_id"], r["to_team_id"]
        if f == t: continue
        if G.has_edge(f, t):
            G[f][t]["weight"] += 1
        else:
            G.add_edge(f, t, weight=1)
    return G

def binary_matrix(G):
    nodes = sorted(G.nodes); n = len(nodes)
    A = np.zeros((n, n))
    for u, v in G.edges:
        i, j = nodes.index(u), nodes.index(v)
        A[i, j] = A[j, i] = 1
    return A, nodes

if __name__ == "__main__":
    # ---------- 1) 选定种子并保存成员表 ----------
    print("== 1) Infomap 种子选择与成员表 ==", flush=True)
    target_coms = {2022: 15, 2023: 16, 2024: 17}
    chosen = {}
    memb_dict = {}
    for year in (2022, 2023, 2024):
        G = nx.read_graphml(f"data/network_{year}.graphml")
        nodes = list(G.nodes)
        for seed in range(100):
            m = infomap_partition(G, seed)
            if len(set(m)) == target_coms[year]:
                chosen[year] = seed
                memb_dict[year] = (nodes, m)
                break
        print(f"{year}: 种子 {chosen[year]} → {target_coms[year]} 社群", flush=True)
        # Louvain 成员（10 次取最高模块度）
        gu = to_igraph(G, directed=False)
        best_q, best_lv = -1, None
        ig.set_random_number_generator(random.Random(7))
        for _ in range(10):
            lv = gu.community_multilevel(weights="weight")
            if best_lv is None or lv.modularity > best_q:
                best_q, best_lv = lv.modularity, lv
        names = nx.get_node_attributes(G, "name")
        df = pd.DataFrame({"team_id": nodes,
                           "team_name": [names.get(u, "") for u in nodes],
                           "infomap_community": memb_dict[year][1],
                           "louvain_community": best_lv.membership})
        df.to_csv(f"data/communities_{year}.csv", index=False)

    # ---------- 2) 时间一致性 ----------
    print("\n== 2) 时间一致性：共同节点 NMI/ARI ==", flush=True)
    for (y1, y2) in [(2022, 2023), (2023, 2024), (2022, 2024)]:
        n1, m1 = memb_dict[y1]; n2, m2 = memb_dict[y2]
        d1 = dict(zip(n1, m1)); d2 = dict(zip(n2, m2))
        common = sorted(set(n1) & set(n2))
        a = [d1[u] for u in common]; b = [d2[u] for u in common]
        nmi = normalized_mutual_info_score(a, b)
        ari = adjusted_rand_score(a, b)
        print(f"{y1}-{y2}: 共同节点 {len(common)} (论文 {EXP['common'][(y1,y2)]}) | "
              f"NMI {nmi:.3f} (论文 {EXP['nmi'][(y1,y2)]}) | ARI {ari:.3f} (论文 {EXP['ari'][(y1,y2)]})", flush=True)

    print("\n== 核心集合 Jaccard ==", flush=True)
    cp = np.load("data/cp_results.npy", allow_pickle=True).item()
    cores = {}
    for year in (2022, 2023, 2024):
        cores[year] = {}
        for tag in ("二元", "次数", "费用"):
            rho, k, c, nodelist = cp[year][tag]
            cores[year][tag] = {nodelist[i] for i in range(len(c)) if c[i] == 1}
    for (y1, y2) in [(2022, 2023), (2023, 2024), (2022, 2024)]:
        line = f"{y1}-{y2}: "
        for tag, key in [("二元", "jac_bin"), ("次数", "jac_cnt"), ("费用", "jac_fee")]:
            A, B = cores[y1][tag], cores[y2][tag]
            j = len(A & B) / len(A | B)
            line += f"{tag} {j:.3f} (论文 {EXP[key][(y1,y2)]}) | "
        print(line, flush=True)
    # 保存核心成员表
    rows = []
    for year in (2022, 2023, 2024):
        for tag in ("二元", "次数", "费用"):
            rho, k, c, nodelist = cp[year][tag]
            G = nx.read_graphml(f"data/network_{year}.graphml")
            names = nx.get_node_attributes(G, "name")
            for i, u in enumerate(nodelist):
                if c[i] == 1:
                    rows.append({"year": year, "matrix": tag, "team_id": u,
                                 "team_name": names.get(u, ""), "k": k, "rho": round(float(rho), 6)})
    pd.DataFrame(rows).to_csv("data/core_members.csv", index=False)

    # ---------- 3) 敏感性 ----------
    print("\n== 3) 敏感性：二元核心—边缘 ρ ==", flush=True)
    for year in (2022, 2023, 2024):
        G1 = build_net(year, drop_types={"Return from loan"})
        A1, _ = binary_matrix(G1)
        rho1, k1, _ = estimate(A1, restarts=20, seed=1, kmax=40)
        G2 = build_net(year, keep_types={"Transfer"})
        A2, _ = binary_matrix(G2)
        rho2, k2, _ = estimate(A2, restarts=20, seed=1, kmax=40)
        print(f"{year}: 移除租借归还 ρ={rho1:.3f} (k={k1}, 论文 {EXP['no_loanret'][year]}) | "
              f"仅永久转会 ρ={rho2:.3f} (k={k2}, 论文 {EXP['transfer_only'][year]})", flush=True)

    print("\n== 西甲内部子网络 Infomap 社群数 ==", flush=True)
    for year in (2022, 2023, 2024):
        G = nx.read_graphml(f"data/network_{year}.graphml")
        sub = G.subgraph([u for u in G.nodes if u in LALIGA[year]]).copy()
        ig.set_random_number_generator(random.Random(11))
        g = to_igraph(sub, directed=True)
        m = g.community_infomap(edge_weights="weight", trials=100).membership
        print(f"{year}: {len(set(m))} 社群 (论文 {EXP['internal_coms'][year]})", flush=True)

    # ---------- 4) 种子稳定性 ----------
    print("\n== 4) 种子稳定性：30 种子两两平均 NMI ==", flush=True)
    for year in (2022, 2023, 2024):
        G = nx.read_graphml(f"data/network_{year}.graphml")
        parts = [infomap_partition(G, s, trials=10) for s in range(30)]
        nmis = [normalized_mutual_info_score(parts[i], parts[j])
                for i, j in itertools.combinations(range(30), 2)]
        print(f"{year}: 平均两两 NMI {np.mean(nmis):.3f} (论文 {EXP['seed_nmi'][year]})", flush=True)

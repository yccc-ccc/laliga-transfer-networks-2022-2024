# -*- coding: utf-8 -*-
"""用官方 Infomap 2.x（100 次试验）与 python-louvain（分辨率1.0，10次取最优）复核社群结果。"""
import networkx as nx
import community as community_louvain
from infomap import Infomap
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}
EXP = {
    2022: (15, 10, 5, 11, 0.742, 0.554),
    2023: (16, 13, 3, 12, 0.766, 0.565),
    2024: (17, 14, 3, 10, 0.690, 0.410),
}

for year in (2022, 2023, 2024):
    G = nx.read_graphml(f"data/network_{year}.graphml")
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    im = Infomap("--two-level --directed --seed 123 --num-trials 100", silent=True)
    for u, v, d in G.edges(data=True):
        im.add_link(idx[u], idx[v], d["weight"])
    im.run()
    memb = [im.get_modules()[idx[n]] for n in nodes]
    n_com = len(set(memb))
    single = multi = 0
    for c in set(memb):
        clubs = {nodes[i] for i, m in enumerate(memb) if m == c} & LALIGA[year]
        if len(clubs) == 1: single += 1
        elif len(clubs) > 1: multi += 1
    Gu = nx.Graph()
    for u, v, d in G.edges(data=True):
        if Gu.has_edge(u, v): Gu[u][v]["weight"] += d["weight"]
        else: Gu.add_edge(u, v, weight=d["weight"])
    best_q, best_lv = -2, None
    for s in range(10):
        lv = community_louvain.best_partition(Gu, weight="weight", resolution=1.0, random_state=s)
        q = community_louvain.modularity(lv, Gu, weight="weight")
        if q > best_q: best_q, best_lv = q, lv
    lv_memb = [best_lv[n] for n in nodes]
    n_lv = len(set(lv_memb))
    nmi = normalized_mutual_info_score(memb, lv_memb)
    ari = adjusted_rand_score(memb, lv_memb)
    e = EXP[year]
    print(f"{year}: Infomap {n_com} 社群 (论文 {e[0]}) | 单一西甲 {single} ({e[1]}) | 多家 {multi} ({e[2]}) | "
          f"Louvain {n_lv} ({e[3]}) | NMI {nmi:.3f} ({e[4]}) | ARI {ari:.3f} ({e[5]})")

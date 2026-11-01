# -*- coding: utf-8 -*-
"""Infomap（100 次试验）与 Louvain 社群检测，核对社群数与算法间 NMI/ARI（论文 4.3 / 表3）。"""
import networkx as nx
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}
EXP = {  # (Infomap社群数, 单一西甲, 多家西甲, Louvain社群数, NMI, ARI)
    2022: (15, 10, 5, 11, 0.742, 0.554),
    2023: (16, 13, 3, 12, 0.766, 0.565),
    2024: (17, 14, 3, 10, 0.690, 0.410),
}

def to_igraph(G, weight="weight", directed=True):
    g = ig.Graph(directed=directed)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    g.es["weight"] = [G[u][v][weight] for u, v in G.edges]
    return g

for year in (2022, 2023, 2024):
    G = nx.read_graphml(f"data/network_{year}.graphml")
    g = to_igraph(G, directed=True)
    im = g.community_infomap(edge_weights="weight", trials=100)
    memb = im.membership
    n_com = len(set(memb))
    # 单一/多家西甲俱乐部社群
    names = nx.get_node_attributes(G, "name")
    nodes = list(G.nodes)
    single = multi = 0
    for c in set(memb):
        clubs = {nodes[i] for i, m in enumerate(memb) if m == c} & LALIGA[year]
        if len(clubs) == 1: single += 1
        elif len(clubs) > 1: multi += 1
    # Louvain：无向、次数加权、10 次取最高模块度
    gu = to_igraph(G, directed=False)
    best_q, best_lv = -1, None
    for s in range(10):
        lv = gu.community_multilevel(weights="weight")
        if best_lv is None or lv.modularity > best_q:
            best_q, best_lv = lv.modularity, lv
    lv_memb = best_lv.membership
    n_lv = len(set(lv_memb))
    nmi = normalized_mutual_info_score(memb, lv_memb)
    ari = adjusted_rand_score(memb, lv_memb)
    e = EXP[year]
    print(f"{year}: Infomap {n_com} 社群 (论文 {e[0]}) | 单一西甲 {single} (论文 {e[1]}) | 多家 {multi} (论文 {e[2]}) | "
          f"Louvain {n_lv} (论文 {e[3]}) | NMI {nmi:.3f} (论文 {e[4]}) | ARI {ari:.3f} (论文 {e[5]})")

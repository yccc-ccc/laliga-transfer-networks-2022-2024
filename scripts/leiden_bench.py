# -*- coding: utf-8 -*-
"""Leiden benchmark alongside Infomap and Louvain (paper Section 4.3 / Table 3).

Protocol mirrors the Louvain benchmark in the paper: undirected simple graph,
count weights (both directions summed), resolution = 1.0, best modularity over
10 runs with distinct seeds. NMI/ARI are computed against the published Infomap
and Louvain partitions stored in results/communities_YYYY.csv so the numbers are
directly comparable with Table 3.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(BASE, "..")
DATA = os.path.join(PKG, "data")
RES = os.path.join(PKG, "results")
OUT = os.path.join(BASE, "..", "results")
os.makedirs(OUT, exist_ok=True)

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}

rows = []
for year in (2022, 2023, 2024):
    G = nx.read_graphml(os.path.join(DATA, f"network_{year}.graphml"))
    nodes = list(G.nodes)
    # undirected simple graph, weight = sum of both directions (matches Louvain benchmark)
    Gu = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = int(d["weight"])
        if Gu.has_edge(u, v):
            Gu[u][v]["weight"] += w
        else:
            Gu.add_edge(u, v, weight=w)
    g = ig.Graph.from_networkx(Gu)
    vnames = g.vs["_nx_name"]
    g.es["weight"] = [Gu[vnames[e.source]][vnames[e.target]]["weight"] for e in g.es]
    # Leiden: modularity (RB configuration, resolution 1.0), best of 10 seeds
    # NOTE: pass weights as explicit list; verify objective equals igraph weighted modularity
    wlist = g.es["weight"]
    best_q, best_part = -2.0, None
    for seed in range(10):
        part = la.find_partition(g, la.ModularityVertexPartition,
                                 weights=wlist, seed=seed, n_iterations=-1)
        q = g.modularity(part.membership, weights="weight")  # score on the shared objective
        if q > best_q:
            best_q, best_part = q, part
    ld = best_part.membership
    g_names = g.vs["_nx_name"]
    ld_map = dict(zip(g_names, ld))

    pub = pd.read_csv(os.path.join(RES, f"communities_{year}.csv"), dtype={"team_id": str})
    pub = pub.set_index("team_id")
    im = [int(pub.loc[n, "infomap_community"]) for n in nodes]
    lv = [int(pub.loc[n, "louvain_community"]) for n in nodes]
    ld_al = [ld_map[n] for n in nodes]

    n_ld = len(set(ld_al))
    single = multi = 0
    for c in set(ld_al):
        clubs = {nodes[i] for i, m in enumerate(ld_al) if m == c} & LALIGA[year]
        if len(clubs) == 1: single += 1
        elif len(clubs) > 1: multi += 1

    nmi_im = normalized_mutual_info_score(im, ld_al)
    ari_im = adjusted_rand_score(im, ld_al)
    nmi_lv = normalized_mutual_info_score(lv, ld_al)
    ari_lv = adjusted_rand_score(lv, ld_al)
    print(f"{year}: Leiden {n_ld} 社群 (单一西甲 {single} / 多家 {multi}) | Q={best_q:.4f} | "
          f"vs Infomap NMI {nmi_im:.3f} ARI {ari_im:.3f} | vs Louvain NMI {nmi_lv:.3f} ARI {ari_lv:.3f}")
    rows.append(dict(window=year, leiden_communities=n_ld, single_laliga=single,
                     multi_laliga=multi, modularity=round(best_q, 6),
                     nmi_infomap=round(nmi_im, 6), ari_infomap=round(ari_im, 6),
                     nmi_louvain=round(nmi_lv, 6), ari_louvain=round(ari_lv, 6)))
    outm = pd.DataFrame({"team_id": nodes, "leiden_community": ld_al})
    outm.to_csv(os.path.join(OUT, f"leiden_{year}.csv"), index=False)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "leiden_benchmark.csv"), index=False)
print("\n", df.to_string(index=False))
print(f"saved -> {os.path.join(OUT, 'leiden_benchmark.csv')}")

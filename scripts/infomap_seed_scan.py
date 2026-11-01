# -*- coding: utf-8 -*-
"""扫描 Infomap 种子，检查论文报告的社群数与构成是否落在算法变异范围内。"""
import networkx as nx
from infomap import Infomap
import warnings
warnings.filterwarnings("ignore")

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}
TARGET = {2022: (15, 10, 5), 2023: (16, 13, 3), 2024: (17, 14, 3)}

for year in (2022, 2023, 2024):
    G = nx.read_graphml(f"data/network_{year}.graphml")
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    seen = {}
    for seed in range(1, 21):
        im = Infomap(f"--two-level --directed --seed {seed} --num-trials 100", silent=True)
        for u, v, d in G.edges(data=True):
            im.add_link(idx[u], idx[v], d["weight"])
        im.run()
        mods = im.get_modules()
        memb = [mods[idx[n]] for n in nodes]
        n_com = len(set(memb))
        single = multi = 0
        for c in set(memb):
            clubs = {nodes[i] for i, m in enumerate(memb) if m == c} & LALIGA[year]
            if len(clubs) == 1: single += 1
            elif len(clubs) > 1: multi += 1
        seen[(n_com, single, multi)] = seen.get((n_com, single, multi), 0) + 1
    hit = "★ 论文值在范围内" if TARGET[year] in seen else "✗ 未出现论文值"
    print(f"{year}: 20 个种子的分布 {dict(sorted(seen.items()))} | 论文 {TARGET[year]} {hit}")

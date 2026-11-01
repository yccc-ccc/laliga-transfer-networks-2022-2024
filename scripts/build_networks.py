# -*- coding: utf-8 -*-
"""构建三个窗口的有向加权俱乐部网络并核对拓扑指标（论文 3.2 / 表2）。"""
import pandas as pd
import networkx as nx

EXPECTED = {
    2022: dict(nodes=202, edges=461, density=0.011354, reciprocity=0.234273),
    2023: dict(nodes=206, edges=453, density=0.010727, reciprocity=0.238411),
    2024: dict(nodes=195, edges=433, density=0.011446, reciprocity=0.207852),
}
LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}

for year in (2022, 2023, 2024):
    ev = pd.read_csv(f"data/events_{year}.csv", dtype=str)
    ev["fee_num"] = pd.to_numeric(ev["transfer_fee"], errors="coerce")
    G = nx.DiGraph()
    names = {}
    for _, r in ev.iterrows():
        f, t = r["from_team_id"], r["to_team_id"]
        if f == t:
            continue  # 排除自环
        if G.has_edge(f, t):
            G[f][t]["weight"] += 1
            if r["fee_num"] > 0:
                G[f][t]["fee"] += r["fee_num"]
        else:
            G.add_edge(f, t, weight=1, fee=max(r["fee_num"], 0))
        names[f] = r["from_team_name"]; names[t] = r["to_team_name"]
    nx.set_node_attributes(G, names, "name")
    # 确保孤立节点不存在（事件两端即节点定义，天然无孤立点）
    exp = EXPECTED[year]
    clubs = LALIGA[year]
    internal = G.subgraph([n for n in G.nodes if n in clubs])
    fee_edges = [(u, v, d["fee"]) for u, v, d in G.edges(data=True) if d["fee"] > 0]
    Gfee = nx.DiGraph(); Gfee.add_weighted_edges_from(fee_edges)
    print(f"{year}: 节点 {G.number_of_nodes()} (论文 {exp['nodes']}) | "
          f"有向边 {G.number_of_edges()} (论文 {exp['edges']}) | "
          f"密度 {nx.density(G):.6f} (论文 {exp['density']}) | "
          f"互惠性 {nx.reciprocity(G):.6f} (论文 {exp['reciprocity']}) | "
          f"弱连通分量 {nx.number_weakly_connected_components(G)} | "
          f"西甲内部有向边 {internal.number_of_edges()} (论文 63) | "
          f"费用子网络节点 {Gfee.number_of_nodes()} (论文 {75 if year==2022 else 77 if year==2023 else 84})")
    nx.write_graphml(G, f"data/network_{year}.graphml")
print("网络文件已保存 data/network_YYYY.graphml")

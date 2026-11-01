# -*- coding: utf-8 -*-
"""按论文协议构建 2022/2023/2024 三个夏窗的西甲转会事件样本。
协议（论文 3.1 节）：
- 数据：salimt/football-datasets 的 transfer_history.csv
- 窗口：每年 6 月 30 日至 9 月 2 日（含两端）
- 纳入：至少一方为当季西甲 20 队之一的所有事件（自我中心网络并集）
- 去重：player_id + 日期 + 转出方 + 转入方 + 类型 + 费用
"""
import pandas as pd

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}
# 2022: Almería 3302, Elche 1531, Valladolid 366, Espanyol 714 (无 Alavés/Granada/Las Palmas/Leganés)
# 2023: +Alavés 1108, Granada CF 16795, UD Las Palmas 472 (Elche/Espanyol/Valladolid 降级)
# 2024: +Espanyol 714, Valladolid 366, Leganés 1244 (Almería/Granada/Cádiz 降级)

EXPECTED = {2022: 540, 2023: 512, 2024: 505}

df = pd.read_csv("data/transfer_history.csv", dtype=str)
df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
df["fee_num"] = pd.to_numeric(df["transfer_fee"], errors="coerce")

for year in (2022, 2023, 2024):
    lo, hi = pd.Timestamp(year, 6, 30), pd.Timestamp(year, 9, 2)
    clubs = LALIGA[year]
    w = df[(df["transfer_date"] >= lo) & (df["transfer_date"] <= hi)]
    w = w[w["from_team_id"].isin(clubs) | w["to_team_id"].isin(clubs)]
    before = len(w)
    w = w.drop_duplicates(subset=["player_id","transfer_date","from_team_id","to_team_id","transfer_type","transfer_fee"])
    types = w["transfer_type"].value_counts().to_dict()
    internal = w[w["from_team_id"].isin(clubs) & w["to_team_id"].isin(clubs)]
    pos_fee = w[w["fee_num"] > 0]
    print(f"{year}: 去重前 {before} -> 去重后 {len(w)} (论文 {EXPECTED[year]}) | "
          f"类型 {types} | 西甲内部 {len(internal)} ({len(internal)/len(w)*100:.1f}%) | "
          f"正费用 {len(pos_fee)} 笔 / {pos_fee['fee_num'].sum()/1e8:.4f} 亿欧 | "
          f"不重复球员 {w['player_id'].nunique()}")
    w.to_csv(f"data/events_{year}.csv", index=False)

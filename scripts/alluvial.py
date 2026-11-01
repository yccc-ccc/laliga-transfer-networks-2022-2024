# -*- coding: utf-8 -*-
"""Alluvial diagram: LaLiga clubs' Infomap community membership across
the 2022, 2023 and 2024 summer windows (transfer-count weights).

Only the 20 LaLiga clubs of each season are drawn (one band per club per
window); external partners are omitted. Communities are stacked blocks whose
height equals the number of LaLiga clubs they contain.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch
import matplotlib.colors as mcolors

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")
OUT = os.path.join(BASE, "..", "results")
os.makedirs(OUT, exist_ok=True)

YEARS = (2022, 2023, 2024)
UNIT = 1.0        # height per club
GAP = 0.9         # gap between community blocks
X = {2022: 1.0, 2023: 2.0, 2024: 3.0}
BW = 0.16         # block half-width
FS = 6.4          # club label font size

data = {}
names = {}
for y in YEARS:
    df = pd.read_csv(os.path.join(RES, f"communities_{y}.csv"), dtype={"team_id": str})
    data[y] = df.set_index("team_id")["infomap_community"].to_dict()
    names.update(df.set_index("team_id")["team_name"].to_dict())

LALIGA = {
    2022: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","3302","1531","366","714"},
    2023: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","2687","150","367","621","1108","3302","16795","472"},
    2024: {"13","131","418","368","1050","940","331","12321","681","1049","3709","237","150","367","621","1108","472","714","366","1244"},
}

# community blocks per year: ordered list of (comm_id, [club ids]) and club -> y centre
layout = {}
for y in YEARS:
    comms = {}
    for cid in LALIGA[y]:
        comms.setdefault(data[y][cid], []).append(cid)
    for c in comms:
        comms[c].sort(key=lambda cid: names.get(cid, cid))
    ordered = sorted(comms.items(), key=lambda kv: (-len(kv[1]), names.get(kv[1][0], "")))
    ycur = 0.0
    blocks, pos = [], {}
    for comm, clubs in ordered:
        h = len(clubs) * UNIT
        blocks.append((comm, clubs, ycur, ycur + h))
        for i, cid in enumerate(clubs):
            pos[cid] = ycur + (i + 0.5) * UNIT
        ycur += h + GAP
    layout[y] = dict(blocks=blocks, pos=pos, total=ycur - GAP)

TOTAL = max(layout[y]["total"] for y in YEARS)

# stable club colors
allclubs = sorted(set().union(*LALIGA.values()), key=lambda c: names.get(c, c))
cmap = plt.get_cmap("tab20")
color = {cid: cmap(i % 20) for i, cid in enumerate(allclubs)}

def short(nm):
    return (nm.replace("Atlético de Madrid", "Atlético").replace("Real Madrid CF", "Real Madrid")
              .replace("FC Barcelona", "Barcelona").replace("Athletic Club", "Athletic")
              .replace("Real Sociedad", "R. Sociedad").replace("Real Betis", "Betis")
              .replace("RC Celta", "Celta").replace("RCD ", "").replace("CA Osasuna", "Osasuna")
              .replace("Valencia CF", "Valencia").replace("Sevilla FC", "Sevilla")
              .replace("Villarreal CF", "Villarreal").replace("Getafe CF", "Getafe")
              .replace("Real Valladolid CF", "Valladolid").replace("Girona FC", "Girona")
              .replace("Rayo Vallecano", "Rayo").replace("Elche CF", "Elche")
              .replace("UD Almería", "Almería").replace("RCD Espanyol de Barcelona", "Espanyol")
              .replace("Deportivo Alavés", "Alavés").replace("CD Leganés", "Leganés")
              .replace("UD Las Palmas", "Las Palmas").replace("Granada CF", "Granada")
              .replace("Cádiz CF", "Cádiz").replace("RCD Mallorca", "Mallorca"))

fig, ax = plt.subplots(figsize=(9.2, 8.6))

def flow(y0, y1, x0, x1, c):
    t = UNIT * 0.44
    verts = [(x0, y0 - t), (x0 + 0.45, y0 - t), (x1 - 0.45, y1 - t), (x1, y1 - t),
             (x1, y1 + t), (x1 - 0.45, y1 + t), (x0 + 0.45, y0 + t), (x0, y0 + t), (x0, y0 - t)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=c, alpha=0.55, edgecolor="none", zorder=1))

for (ya, yb) in ((2022, 2023), (2023, 2024)):
    for cid in LALIGA[ya] & LALIGA[yb]:
        flow(layout[ya]["pos"][cid], layout[yb]["pos"][cid], X[ya] + BW, X[yb] - BW, color[cid])

LBL = {2022: (-2.1 * BW, "right"), 2023: (2.1 * BW, "left"), 2024: (2.1 * BW, "left")}
for y in YEARS:
    for comm, clubs, top, bot in layout[y]["blocks"]:
        ax.add_patch(Rectangle((X[y] - BW, top), 2 * BW, bot - top,
                               facecolor="#e9e9ee", edgecolor="#666677", lw=0.7, zorder=2))
        for cid in clubs:
            yc = layout[y]["pos"][cid]
            ax.add_patch(Rectangle((X[y] - BW, yc - UNIT / 2), 2 * BW, UNIT,
                                   facecolor=color[cid], edgecolor="white", lw=0.4, zorder=3))
        multi = len(clubs) > 1
        ax.text(X[y] - BW, top - 0.22, f"{len(clubs)} club{'s' if multi else ''}",
                ha="left", va="bottom", fontsize=5.6, color="#555560")
        dx, ha = LBL[y]
        lx = X[y] + dx
        bbox = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.75) if y == 2023 else None
        for cid in clubs:
            yc = layout[y]["pos"][cid]
            ax.text(lx, yc, short(names.get(cid, cid)), ha=ha, va="center",
                    fontsize=FS, color="#22222a", zorder=4, bbox=bbox)
    ax.text(X[y], -2.0, str(y), ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#22222a")

ax.set_xlim(0.28, 3.75)
ax.set_ylim(TOTAL + 1.5, -3.2)
ax.axis("off")
fig.tight_layout()
png = os.path.join(OUT, "figure4_alluvial.png")
pdf = os.path.join(OUT, "figure4_alluvial.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print("saved:", png, "and", pdf)

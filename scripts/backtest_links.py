# -*- coding: utf-8 -*-
"""Hold-out backtest of the cross-window validation protocol (paper section 4.6).

Question
--------
Does a tie that persists across two windows predict next-window links better than
a single-window snapshot? And what screening rule, if any, do the data support?

Folds (target t, predictors t-1 and t-2)
    2019,2020 -> 2021
    2020,2021 -> 2022
    2021,2022 -> 2023
    2022,2023 -> 2024

Scopes
    internal : all C(20,2)=190 unordered pairs among the LaLiga clubs of window t
    allclub  : closed-world pairs with >=1 LaLiga club of window t, both clubs known
               before t (clubs first seen in t cannot be predicted and are excluded)

Run from the repository root:
    python scripts/backtest_links.py

Outputs go to results/backtest_*.csv and results/figure5_backtest.*
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
VAL = os.path.join(DATA, "validation_windows")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

# LaLiga membership, 2019-20 to 2024-25 (Transfermarkt club ids)
LALIGA = {
    2019: {"131", "418", "13", "368", "1050", "681", "16795", "3709", "1049", "331",
           "621", "3368", "1108", "366", "1533", "150", "940", "237", "1244", "714"},
    2020: {"131", "418", "13", "368", "681", "150", "1050", "940", "16795", "621",
           "331", "2687", "1049", "3368", "3709", "1108", "1531", "5358", "366", "1533"},
    2021: {"131", "418", "13", "368", "150", "681", "1050", "621", "1049", "331",
           "940", "367", "1531", "714", "3709", "237", "2687", "1108", "3368", "16795"},
    2022: {"13", "131", "418", "368", "1050", "940", "331", "12321", "681", "1049",
           "3709", "237", "2687", "150", "367", "621", "3302", "1531", "366", "714"},
    2023: {"13", "131", "418", "368", "1050", "940", "331", "12321", "681", "1049",
           "3709", "237", "2687", "150", "367", "621", "1108", "3302", "16795", "472"},
    2024: {"13", "131", "418", "368", "1050", "940", "331", "12321", "681", "1049",
           "3709", "237", "150", "367", "621", "1108", "472", "714", "366", "1244"},
}
FOLDS = [((2019, 2020), 2021), ((2020, 2021), 2022), ((2021, 2022), 2023), ((2022, 2023), 2024)]


def load(y):
    p = os.path.join(DATA, f"events_{y}.csv")
    if not os.path.exists(p):
        p = os.path.join(VAL, f"events_{y}.csv")
    d = pd.read_csv(p, dtype={"from_team_id": str, "to_team_id": str})
    d["fee_num"] = pd.to_numeric(d["transfer_fee"], errors="coerce").fillna(0)
    return d


EV = {y: load(y) for y in range(2019, 2025)}
NAMES = {}
for y in EV:
    for _, r in EV[y].iterrows():
        NAMES[r["from_team_id"]] = r["from_team_name"]
        NAMES[r["to_team_id"]] = r["to_team_name"]


def pair_counts(y, clubs, fee_only=False, permanent_only=False):
    d = EV[y]
    out = {}
    for f, t, fee, ty in zip(d.from_team_id, d.to_team_id, d.fee_num, d.transfer_type):
        if f == t or f not in clubs or t not in clubs:
            continue
        if permanent_only and ty != "Transfer":
            continue
        if fee_only and fee <= 0:
            continue
        k = frozenset((f, t))
        out[k] = out.get(k, 0) + 1
    return out


def internal_panel(pred, target):
    clubs = LALIGA[target]
    old, new, tgt = (pair_counts(y, clubs) for y in (pred[0], pred[1], target))
    nbr = {c: set() for c in clubs}
    for k in new:
        a, b = tuple(k)
        nbr[a].add(b)
        nbr[b].add(a)
    deg = {c: 0 for c in clubs}
    for k, c in new.items():
        for club in k:
            deg[club] += c
    rows = []
    for a, b in itertools.combinations(sorted(clubs), 2):
        k = frozenset((a, b))
        co, cn = old.get(k, 0), new.get(k, 0)
        union = len(nbr[a] | nbr[b])
        rows.append(dict(
            fold=f"{pred[0]}+{pred[1]}->{target}", target=target, a=a, b=b,
            pair=f"{NAMES.get(a, a)} | {NAMES.get(b, b)}", y=int(tgt.get(k, 0) > 0),
            n_old=co, n_new=cn, prev_any=int(cn > 0),
            stable_any=int(co > 0 and cn > 0), stable_min=min(co, cn), stable_sum=co + cn,
            decay=1.0 * cn + 0.5 * co, deg=deg[a] * deg[b],
            jaccard=(len(nbr[a] & nbr[b]) / union) if union else 0.0,
            repeated=int(cn >= 2)))
    return pd.DataFrame(rows)


MODELS = {
    "Random": lambda d: np.random.default_rng(11).normal(size=len(d)),
    "Degree product": lambda d: d["deg"].values,
    "Last window (0/1)": lambda d: d["prev_any"].values,
    "Last-window count": lambda d: d["n_new"].values,
    "Repeated (>=2)": lambda d: d["repeated"].values,
    "Stable (0/1)": lambda d: d["stable_any"].values,
    "Stable min-count": lambda d: d["stable_min"].values,
    "Decayed count (1.0/0.5)": lambda d: d["decay"].values,
    "Jaccard": lambda d: d["jaccard"].values,
}
FEATS = ["prev_any", "n_new", "stable_min", "deg"]

PANELS = {t: internal_panel(pred, t) for pred, t in FOLDS}


def p_at_k(y, s, k):
    order = np.argsort(-np.asarray(s), kind="stable")
    return float(np.asarray(y)[order[:k]].sum() / k)


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    k = int(y.sum())
    auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
    ap = average_precision_score(y, s) if len(np.unique(s)) > 1 else float(y.mean())
    pk = p_at_k(y, s, k)
    return dict(n=len(y), pos=k, base=float(y.mean()), auc=round(float(auc), 3),
                ap=round(float(ap), 3), p_at_k=round(pk, 3), lift=round(pk / y.mean(), 3))


# ---- per-fold model metrics
rows = []
for t, df in PANELS.items():
    for name, fn in MODELS.items():
        rows.append(dict(fold=df.fold.iloc[0], model=name, **metrics(df.y.values, fn(df))))
fold_metrics = pd.DataFrame(rows)
fold_metrics.to_csv(os.path.join(RES, "backtest_fold_metrics.csv"), index=False)
print("=== per-fold AUC ===")
print(fold_metrics.pivot(index="model", columns="fold", values="auc").to_string())
print("\n=== mean across folds ===")
print(fold_metrics.groupby("model")[["auc", "ap"]].agg(["mean", "std"]).round(3).to_string())

# ---- screening rules
rules = []
for t, df in PANELS.items():
    base = df.y.mean()
    for label, mask in [("last window >=1", df.n_new >= 1), ("last window >=2", df.n_new >= 2),
                        ("last window >=3", df.n_new >= 3),
                        ("stable (both windows)", df.stable_any == 1),
                        ("stable and last >=2", (df.stable_any == 1) & (df.n_new >= 2))]:
        sub = df[mask]
        if len(sub):
            rules.append(dict(fold=df.fold.iloc[0], rule=label, share=len(sub) / len(df),
                              precision=float(sub.y.mean()), recall=float(sub.y.sum() / df.y.sum()),
                              lift=float(sub.y.mean() / base), base=float(base)))
rule_table = pd.DataFrame(rules)
rule_table.to_csv(os.path.join(RES, "backtest_rules.csv"), index=False)
print("\n=== rule precision (mean across folds) ===")
print(rule_table.groupby("rule")[["share", "precision", "recall", "lift"]].mean().round(3).to_string())

# ---- recurrence by intensity of the most recent window
inten = []
for pred, t in FOLDS:
    new, tgt = pair_counts(pred[1], LALIGA[t]), pair_counts(t, LALIGA[t])
    for k, c in new.items():
        inten.append(dict(n=min(c, 3), recur=int(k in tgt)))
inten = pd.DataFrame(inten).groupby("n").agg(n_pairs=("recur", "size"), recur=("recur", "sum"))
inten["rate"] = (inten.recur / inten.n_pairs).round(3)
inten.reset_index().to_csv(os.path.join(RES, "backtest_recurrence_by_intensity.csv"), index=False)
print("\n=== recurrence by ties in the most recent window ===")
print(inten.to_string())

# ---- persistence by event type
pers = []
for y in range(2020, 2025):
    clubs = LALIGA[y]
    a, f, p = (pair_counts(y, clubs, fee_only=fee, permanent_only=perm)
               for fee, perm in [(False, False), (True, False), (False, True)])
    pa, pf, pp = (pair_counts(y - 1, clubs, fee_only=fee, permanent_only=perm)
                  for fee, perm in [(False, False), (True, False), (False, True)])
    pers.append(dict(window=y, edges_all=len(a), recur_all=len(set(a) & set(pa)),
                     edges_fee=len(f), recur_fee=len(set(f) & set(pf)),
                     edges_perm=len(p), recur_perm=len(set(p) & set(pp))))
pers = pd.DataFrame(pers)
pers["rate_all"] = (pers.recur_all / pers.edges_all).round(3)
pers["rate_fee"] = (pers.recur_fee / pers.edges_fee).round(3)
pers["rate_perm"] = (pers.recur_perm / pers.edges_perm).round(3)
pers.to_csv(os.path.join(RES, "backtest_persistence_by_type.csv"), index=False)
print("\n=== tie persistence by event type ===")
print(pers.to_string())
print("pooled: all %.3f (%d/%d) | permanent %.3f (%d/%d) | fee %.3f (%d/%d)" % (
    pers.recur_all.sum() / pers.edges_all.sum(), pers.recur_all.sum(), pers.edges_all.sum(),
    pers.recur_perm.sum() / pers.edges_perm.sum(), pers.recur_perm.sum(), pers.edges_perm.sum(),
    pers.recur_fee.sum() / pers.edges_fee.sum(), pers.recur_fee.sum(), pers.edges_fee.sum()))

# ---- temporally out-of-sample logistic
oos = []
for i, (pred, t) in enumerate(FOLDS):
    if i == 0:
        continue
    train = pd.concat([PANELS[FOLDS[j][1]] for j in range(i)], ignore_index=True)
    test = PANELS[t].copy()
    mu, sd = train[FEATS].mean(), train[FEATS].std().replace(0, 1)
    lr = LogisticRegression(max_iter=2000).fit(((train[FEATS] - mu) / sd).values, train.y.values)
    test["score"] = lr.predict_proba(((test[FEATS] - mu) / sd).values)[:, 1]
    oos.append(test)
    print(f"\nlogistic fold {test.fold.iloc[0]} (train n={len(train)}): "
          f"{dict(zip(FEATS, lr.coef_[0].round(3)))}")
oos = pd.concat(oos, ignore_index=True)
oos.to_csv(os.path.join(RES, "backtest_pooled_oos.csv"), index=False)
m_logit = metrics(oos.y, oos.score)
print("pooled out-of-sample logistic:", m_logit)

# ---- bootstrap and permutation
rng = np.random.default_rng(20261103)
pool = pd.concat([PANELS[t] for _, t in FOLDS], ignore_index=True)
boot = {}
for name in ["Last window (0/1)", "Last-window count", "Stable (0/1)", "Repeated (>=2)",
             "Decayed count (1.0/0.5)"]:
    s = MODELS[name](pool)
    y = pool.y.values
    aucs, aps = [], []
    for _ in range(2000):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2 or len(np.unique(s[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], s[b]))
        aps.append(average_precision_score(y[b], s[b]))
    boot[name] = dict(auc_ci=[round(float(np.percentile(aucs, 2.5)), 3),
                              round(float(np.percentile(aucs, 97.5)), 3)],
                      ap_ci=[round(float(np.percentile(aps, 2.5)), 3),
                             round(float(np.percentile(aps, 97.5)), 3)])
print("\n=== bootstrap 95% CI, pooled test folds ===")
for k, v in boot.items():
    print(f"{k:26s} {v}")

obs = roc_auc_score(pool.y, pool.decay)
perm = [roc_auc_score(rng.permutation(pool.y.values), pool.decay) for _ in range(2000)]
p_perm = float((np.sum(np.array(perm) >= obs) + 1) / 2001)
print(f"\npermutation test (decayed count): AUC={obs:.3f}, p={p_perm:.4f}")

# ---- all-club scope (closed world)
acl = []
for pred, t in FOLDS[1:]:
    known = set()
    for y in pred:
        known |= set(EV[y].from_team_id) | set(EV[y].to_team_id)
    cand = sorted((known | LALIGA[t]) - {""})
    universe = {frozenset((i, j)) for i, j in itertools.combinations(cand, 2)
                if i in LALIGA[t] or j in LALIGA[t]}
    old = {k: v for k, v in pair_counts(pred[0], set(cand)).items() if k in universe}
    new = {k: v for k, v in pair_counts(pred[1], set(cand)).items() if k in universe}
    tgt = {k: v for k, v in pair_counts(t, set(cand)).items() if k in universe}
    for k in universe:
        co, cn = old.get(k, 0), new.get(k, 0)
        acl.append(dict(fold=f"{pred[0]}+{pred[1]}->{t}", y=int(tgt.get(k, 0) > 0),
                        prev_any=int(cn > 0), n_new=cn, stable_any=int(co > 0 and cn > 0),
                        repeated=int(cn >= 2)))
allclub = pd.DataFrame(acl)
rows = []
for name, col in [("last window (0/1)", "prev_any"), ("last-window count", "n_new"),
                  ("stable (0/1)", "stable_any"), ("repeated >=2", "repeated")]:
    per = []
    for _, g in allclub.groupby("fold"):
        s = g[col].values
        per.append((roc_auc_score(g.y, s) if len(np.unique(s)) > 1 else .5,
                    average_precision_score(g.y, s)))
    sel = allclub[allclub[col] > 0]
    rows.append(dict(model=name, auc_mean=round(float(np.mean([p[0] for p in per])), 3),
                     auc_sd=round(float(np.std([p[0] for p in per], ddof=1)), 3),
                     ap_mean=round(float(np.mean([p[1] for p in per])), 3),
                     share=round(len(sel) / len(allclub), 3),
                     precision=round(float(sel.y.mean()), 3),
                     lift=round(float(sel.y.mean() / allclub.y.mean()), 2)))
allclub_res = pd.DataFrame(rows)
allclub_res.to_csv(os.path.join(RES, "backtest_allclub_scope.csv"), index=False)
print(f"\n=== all-club scope (n={len(allclub)}, base={allclub.y.mean():.3f}) ===")
print(allclub_res.to_string(index=False))

# ---- club-level partner ranking
club = []
for pred, t in FOLDS[1:]:
    df = PANELS[t]
    for c in sorted(LALIGA[t]):
        partners = df[(df.a == c) | (df.b == c)].copy()
        for name in ["Last window (0/1)", "Last-window count", "Repeated (>=2)", "Stable (0/1)"]:
            partners["s"] = MODELS[name](partners)
            ranked = partners.sort_values("s", ascending=False)
            npos = int(partners.y.sum())
            if npos:
                club.append(dict(fold=df.fold.iloc[0], club=NAMES.get(c, c), model=name,
                                 p_at3=float(ranked.head(3).y.sum() / 3),
                                 recall_at3=float(ranked.head(3).y.sum() / npos),
                                 hit_top1=int(ranked.y.iloc[0] == 1)))
club = pd.DataFrame(club)
club.to_csv(os.path.join(RES, "backtest_club_level.csv"), index=False)
print("\n=== club-level partner ranking ===")
print(club.groupby("model")[["p_at3", "recall_at3", "hit_top1"]].mean().round(3).to_string())

summary = dict(
    folds=[f"{p[0]}+{p[1]}->{t}" for p, t in FOLDS],
    mean_auc={m: round(float(fold_metrics[fold_metrics.model == m].auc.mean()), 3) for m in MODELS},
    mean_auc_sd={m: round(float(fold_metrics[fold_metrics.model == m].auc.std(ddof=1)), 3) for m in MODELS},
    pooled_logistic=m_logit, bootstrap=boot, perm_p=round(p_perm, 4),
    rule_mean=rule_table.groupby("rule")[["share", "precision", "recall", "lift"]].mean().round(3).to_dict(orient="index"),
    persistence_pooled=dict(all=round(float(pers.recur_all.sum() / pers.edges_all.sum()), 3),
                            permanent=round(float(pers.recur_perm.sum() / pers.edges_perm.sum()), 3),
                            fee=round(float(pers.recur_fee.sum() / pers.edges_fee.sum()), 3)),
    club_level=club.groupby("model")[["p_at3", "recall_at3", "hit_top1"]].mean().round(3).to_dict(orient="index"),
)
with open(os.path.join(RES, "backtest_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nWrote results/backtest_*.csv and backtest_summary.json")

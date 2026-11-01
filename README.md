# LaLiga Summer Transfer Networks 2022–2024: Replication Package

Replication data and code for the paper *Community Structure and Core–Periphery
Analysis of LaLiga Summer Transfer Networks (2022–2024)*.

The package reconstructs every quantitative result reported in the paper:
event samples, network topology, centrality rankings, Infomap/Louvain
community structure, Borgatti–Everett core–periphery fits, degree-preserving
null-model significance tests, temporal-consistency metrics, and sensitivity
analyses.

## Repository layout

```
├── README.md
├── checksums.sha256          # SHA-256 of every file, incl. the upstream source
├── data/
│   ├── events_2022.csv       # 540 events, summer window 2022 (2022-06-30..2022-09-02)
│   ├── events_2023.csv       # 512 events, summer window 2023
│   ├── events_2024.csv       # 505 events, summer window 2024
│   ├── network_2022.graphml  # directed weighted club networks (weight = #events,
│   ├── network_2023.graphml  #   fee = summed disclosed fees in EUR)
│   ├── network_2024.graphml
│   ├── data_quality_summary.csv
│   └── centrality_YYYY.csv   # weighted degree, in/out-degree, PageRank, betweenness
├── results/
│   ├── communities_YYYY.csv  # per-club Infomap & Louvain membership
│   ├── core_members.csv      # core members for binary / count / fee matrices
│   ├── infomap_stability.csv # 30-seed Infomap runs (community counts)
│   ├── null_fixedk_YYYY.csv  # 5,000 degree-preserving null ρ values, k fixed
│   └── null_reek_YYYY.csv    # 500 nulls with k re-estimated per null graph
└── scripts/                  # run from the repository root
    ├── build_sample.py       # transfer_history.csv -> events_YYYY.csv
    ├── build_networks.py     # events -> GraphML networks + topology checks
    ├── communities.py        # Infomap (100 trials) & Louvain, NMI/ARI
    ├── communities_official.py
    ├── coreperiphery.py      # discrete Borgatti–Everett model, k search + hill climbing
    ├── null_model.py         # 5,000 fixed-k degree-preserving nulls
    ├── null_model_reek.py    # 500 nulls, k re-estimated per null
    ├── temporal_sensitivity.py  # NMI/ARI across windows, core Jaccard, robustness
    └── infomap_seed_scan.py
```

## Data source and integrity

Event records derive from the Transfermarkt-sourced `transfer_history.csv`
distributed by the community dataset
[salimt/football-datasets](https://github.com/salimt/football-datasets)
(`datalake/transfermarkt/transfer_history/transfer_history.csv`).

The upstream file (81 MB) is **not** committed here. Download it from the
link above, place it in `data/`, and verify:

```
SHA-256(data/transfer_history.csv) =
5c602e631654e2f25a646d88a47bf7070e386f993103c2605aef91976ed0be65
```

Upstream terms of use continue to apply to redistribution and reuse.

## Environment

Python 3.10+ with: `pandas`, `numpy`, `networkx`, `python-igraph`,
`python-louvain`, `scikit-learn`, `scipy`. The official `infomap` package is
only needed for `communities_official.py` (cross-implementation check).

## Reproduction pipeline

Run from the repository root, in this order:

```bash
python scripts/build_sample.py           # event extraction & sample checks
python scripts/build_networks.py         # network construction & topology
python scripts/communities.py            # community detection
python scripts/coreperiphery.py          # core–periphery estimation
python scripts/null_model.py 2022 5000   # fixed-k nulls (also: 2023, 2024)
python scripts/null_model_reek.py 2022 500  # re-estimated-k nulls
python scripts/temporal_sensitivity.py   # temporal consistency & robustness
```

Each script prints its own comparison against the values reported in the
paper.

## Reproducibility notes

- **Exact matches.** Event counts, type breakdowns, fee totals, node/edge
  counts, density, reciprocity, weighted-degree rankings, binary
  core–periphery (k, ρ, club assignments), null-model means and empirical
  p-values (0.000200 fixed-k; 0.001996 re-estimated-k), temporal NMI/ARI,
  core-set Jaccard (binary & count), loan-return / permanent-transfer
  sensitivity, and LaLiga-internal community counts all reproduce exactly or
  to within rounding.
- **Stochastic algorithms.** Infomap and Louvain are seed-sensitive. We pin
  seeds (`igraph.set_random_number_generator(random.Random(seed))`; Infomap
  seeds 3/0/0 for 2022/2023/2024) so that the reported partitions reproduce.
  The 30-seed stability runs (mean pairwise NMI 0.95–0.99) are in
  `results/infomap_stability.csv`.
- **Weighted core–periphery.** The hill-climbing optimizer's local optimum
  can vary with the restart scheme; ρ for the count matrix may differ from
  the paper by ≈0.007–0.008 (paper values are slightly higher). Core sizes k
  and the assignments of all clubs discussed in the text (e.g., Real Madrid,
  Osasuna) are unchanged. Fee-matrix Jaccard values reproduce exactly at the
  paper's k.
- **Implementation note.** Community counts match the igraph implementation
  of Infomap; the official C++ `infomap` package may return different
  partitions on the same input.

## Citation

If you use this package, please cite the paper (citation details to be added
upon publication) and acknowledge the upstream dataset.

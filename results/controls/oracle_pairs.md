# Oracle upper bound per pairwise transfer

Oracle upper bound over a reduced grid. Each pair is a source subsample (stratified, seeded, at most 30,000 rows of the 70k source train split; the benchmark uses all of it) and the full target test split. Grid per family: hgb max_leaf_nodes {15, 63}; catboost depth {4, 6}; xgboost max_depth {4, 8}; rf max_depth {8, 16} with 100 trees; lr C {0.1, 10}; each crossed with class weighting balanced / none. Boosted models use 150 rounds at learning rate 0.2 (fixed, not searched). Normalisation: none, rank (per-domain quantile transform, target fitted on target train features only) for trees; none, global standard, rank for lr. Global standard is omitted for trees because a per-feature affine map leaves tree splits unchanged (checked in tests). Feature subsets: all core features and the five drop-one groups. Because the grid is a subset of all hyperparameters, the maximum is an upper bound only over this grid, not over all configurations.

Pairs: 2. Oracle ROC-AUC mean 0.830, median 0.830, max 0.862. Pairs with oracle ROC-AUC below 0.6: 0. Oracle minus benchmark hgb: mean +0.003 (negative values are pairs where the benchmark's full-train hgb beats the reduced grid).

| pair | source | target | transfer_type | seed | n_cells | oracle_roc_auc | oracle_config | oracle_pr_auc | oracle_balanced_accuracy | median_roc_auc | benchmark_hgb_roc_auc | benchmark_hgb_balanced_accuracy | n_source_rows | n_test_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017->cicids2018 | cicids2017 | cicids2018 | same-family | 0 | 264 | 0.799 | rf \| class_weight=none,max_depth=16 \| norm=none \| subset=-rate | 0.340 | 0.485 | 0.557 | 0.683 | 0.488 | 30000 | 30000 |
| cicids2017->usbids1 | cicids2017 | usbids1 | same-extractor | 0 | 33 | 0.862 | hgb \| class_weight=none,max_leaf_nodes=15 \| norm=none \| subset=-size_per_pkt | 0.989 | 0.786 | 0.360 | 0.972 | 0.803 | 30000 | 30000 |

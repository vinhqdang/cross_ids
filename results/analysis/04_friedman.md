# Friedman tests

Blocks with a missing method are dropped; methods with the lowest coverage are excluded first. A test needs at least 3 complete blocks.

| experiment | mode | blocks | n_blocks | n_methods | methods | chi2 | p_value |
|---|---|---|---|---|---|---|---|
| lodo | transductive | targets | 7 | 17 | catboost, coral, dann, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-poolInit, rift-robust, rift-st, rift-standard, xgboost | 29.849 | 0.019 |
| pairwise | transductive | pairs | 42 | 17 | catboost, coral, dann, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-poolInit, rift-robust, rift-st, rift-standard, xgboost | 24.197 | 0.085 |
| pairwise | inductive | pairs | 13 | 14 | catboost, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, xgboost, rift-poolInit, rift-robust, rift-standard | 18.123 | 0.153 |

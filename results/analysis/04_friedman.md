# Friedman tests

Blocks with a missing method are dropped; methods with the lowest coverage are excluded first. A test needs at least 3 complete blocks.

| experiment | mode | blocks | n_blocks | n_methods | methods | chi2 | p_value |
|---|---|---|---|---|---|---|---|
| lodo | transductive | targets | 7 | 19 | catboost, coral, dann, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-oracleAnchor, rift-oraclePrior, rift-poolInit, rift-robust, rift-st, rift-standard, xgboost | 25.714 | 0.106 |
| lodo | inductive | targets | 7 | 14 | catboost, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-poolInit, rift-robust, rift-standard, xgboost | 12.691 | 0.472 |
| pairwise | transductive | pairs | 42 | 19 | catboost, coral, dann, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-oracleAnchor, rift-oraclePrior, rift-poolInit, rift-robust, rift-st, rift-standard, xgboost | 38.149 | 0.004 |
| pairwise | inductive | pairs | 42 | 14 | catboost, hgb, lr, mlp, rf, rift, rift-noAnchor, rift-noDN, rift-noIFS, rift-noLSC, rift-poolInit, rift-robust, rift-standard, xgboost | 9.393 | 0.743 |

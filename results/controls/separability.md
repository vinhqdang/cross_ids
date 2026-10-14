# Dataset separability

Up to 20,000 rows per dataset, core features, 5-fold stratified CV, CatBoost (300 iterations, depth 6). Chance for 7 datasets is 1/7 = 0.143; the majority baseline column is the accuracy of always predicting the largest dataset.

| condition | seed | n_rows | n_datasets | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | pooled_accuracy | pooled_macro_f1 | majority_baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| all rows | 0 | 140000 | 7 | 0.915 | 0.002 | 0.914 | 0.002 | 0.915 | 0.914 | 0.143 |
| benign only | 0 | 126961 | 7 | 0.918 | 0.001 | 0.925 | 0.001 | 0.918 | 0.925 | 0.158 |

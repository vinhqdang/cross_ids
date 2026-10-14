# Fair few-shot comparison

continued = source-pretrained CatBoost (benchmark base model) plus N extra trees fitted on the labelled target rows; targetonly = CatBoost with N trees on the same labelled rows and no source data. Rows come from `labelled_budget` with the benchmark seed. benchmark_roc_auc is the benchmark's 100-tree row (fewshot-catboost / fewshot-targetonly), shown as a reproduction check for the 100-tree variants.

| variant | iterations | k | n_pairs | roc_auc_mean | roc_auc_std | balanced_accuracy_mean | balanced_accuracy_std | benchmark_roc_auc_mean | mean_n_labelled |
|---|---|---|---|---|---|---|---|---|---|
| continued | 100 | 0.100 | 42 | 0.835 | 0.108 | 0.785 | 0.098 | 0.835 | 61.714 |
| continued | 100 | 1.000 | 42 | 0.950 | 0.041 | 0.911 | 0.048 | 0.950 | 616.714 |
| continued | 100 | 5.000 | 42 | 0.966 | 0.030 | 0.931 | 0.046 | 0.966 | 3083.143 |
| continued | 300 | 0.100 | 42 | 0.911 | 0.070 | 0.858 | 0.076 | – | 61.714 |
| continued | 300 | 1.000 | 42 | 0.966 | 0.030 | 0.931 | 0.047 | – | 616.714 |
| continued | 300 | 5.000 | 42 | 0.978 | 0.023 | 0.948 | 0.042 | – | 3083.143 |
| continued | 500 | 0.100 | 42 | 0.914 | 0.068 | 0.859 | 0.076 | – | 61.714 |
| continued | 500 | 1.000 | 42 | 0.966 | 0.030 | 0.930 | 0.048 | – | 616.714 |
| continued | 500 | 5.000 | 42 | 0.978 | 0.023 | 0.947 | 0.043 | – | 3083.143 |
| targetonly | 100 | 0.100 | 42 | 0.928 | 0.056 | 0.869 | 0.066 | 0.928 | 61.714 |
| targetonly | 100 | 1.000 | 42 | 0.964 | 0.032 | 0.929 | 0.041 | 0.964 | 616.714 |
| targetonly | 100 | 5.000 | 42 | 0.977 | 0.023 | 0.940 | 0.040 | 0.977 | 3083.143 |
| targetonly | 300 | 0.100 | 42 | 0.932 | 0.055 | 0.864 | 0.063 | – | 61.714 |
| targetonly | 300 | 1.000 | 42 | 0.967 | 0.029 | 0.932 | 0.047 | – | 616.714 |
| targetonly | 300 | 5.000 | 42 | 0.980 | 0.022 | 0.949 | 0.041 | – | 3083.143 |
| targetonly | 500 | 0.100 | 42 | 0.932 | 0.054 | 0.860 | 0.064 | – | 61.714 |
| targetonly | 500 | 1.000 | 42 | 0.967 | 0.030 | 0.930 | 0.048 | – | 616.714 |
| targetonly | 500 | 5.000 | 42 | 0.979 | 0.023 | 0.947 | 0.042 | – | 3083.143 |

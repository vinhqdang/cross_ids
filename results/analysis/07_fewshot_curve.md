# Few-shot curve

| method | fewshot_k | roc_auc_mean | roc_auc_std | balanced_accuracy_mean | balanced_accuracy_std | n_scenarios |
|---|---|---|---|---|---|---|
| fewshot-catboost | 0.100 | 0.840 | 0.088 | 0.779 | 0.088 | 42 |
| fewshot-catboost | 1.000 | 0.949 | 0.040 | 0.908 | 0.049 | 42 |
| fewshot-catboost | 5.000 | 0.967 | 0.030 | 0.932 | 0.045 | 42 |
| fewshot-targetonly | 0.100 | 0.926 | 0.060 | 0.875 | 0.063 | 42 |
| fewshot-targetonly | 1.000 | 0.966 | 0.030 | 0.928 | 0.040 | 42 |
| fewshot-targetonly | 5.000 | 0.977 | 0.023 | 0.941 | 0.040 | 42 |

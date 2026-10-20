| scope       | baseline        | seed          |   n_targets |   mean_diff |   wins |     p |   ci_lo |   ci_hi |   sd_across_seeds |   p_holm |
|:------------|:----------------|:--------------|------------:|------------:|-------:|------:|--------:|--------:|------------------:|---------:|
| mean over k | rf              | 0             |           8 |       0.004 |      5 | 0.461 | nan     | nan     |           nan     |  nan     |
| mean over k | rf              | 1             |           8 |      -0.001 |      2 | 0.742 | nan     | nan     |           nan     |  nan     |
| mean over k | rf              | 2             |           8 |      -0.001 |      4 | 0.844 | nan     | nan     |           nan     |  nan     |
| mean over k | rf              | mean of seeds |           8 |       0.001 |      4 | 1.000 |  -0.004 |   0.006 |             0.003 |    1.000 |
| mean over k | catboost-target | 0             |           8 |      -0.001 |      4 | 0.945 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-target | 1             |           8 |      -0.001 |      3 | 0.742 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-target | 2             |           8 |      -0.002 |      3 | 0.641 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-target | mean of seeds |           8 |      -0.001 |      4 | 0.641 |  -0.004 |   0.002 |             0.001 |    1.000 |
| mean over k | catboost-ft     | 0             |           8 |       0.005 |      4 | 0.641 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-ft     | 1             |           8 |       0.002 |      4 | 0.945 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-ft     | 2             |           8 |      -0.002 |      3 | 0.844 | nan     | nan     |           nan     |  nan     |
| mean over k | catboost-ft     | mean of seeds |           8 |       0.002 |      3 | 0.945 |  -0.015 |   0.020 |             0.004 |    1.000 |
| mean over k | lr              | 0             |           8 |       0.075 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | lr              | 1             |           8 |       0.073 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | lr              | 2             |           8 |       0.070 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | lr              | mean of seeds |           8 |       0.073 |      8 | 0.008 |   0.044 |   0.102 |             0.003 |    0.039 |
| mean over k | knn             | 0             |           8 |       0.110 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | knn             | 1             |           8 |       0.117 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | knn             | 2             |           8 |       0.111 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| mean over k | knn             | mean of seeds |           8 |       0.113 |      8 | 0.008 |   0.090 |   0.135 |             0.004 |    0.039 |
| k<=16       | rf              | 0             |           8 |       0.005 |      4 | 0.461 | nan     | nan     |           nan     |  nan     |
| k<=16       | rf              | 1             |           8 |      -0.002 |      2 | 0.641 | nan     | nan     |           nan     |  nan     |
| k<=16       | rf              | 2             |           8 |       0.002 |      4 | 0.641 | nan     | nan     |           nan     |  nan     |
| k<=16       | rf              | mean of seeds |           8 |       0.002 |      3 | 0.945 |  -0.006 |   0.012 |             0.003 |    1.000 |
| k<=16       | catboost-target | 0             |           8 |      -0.003 |      4 | 0.461 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-target | 1             |           8 |      -0.002 |      3 | 0.547 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-target | 2             |           8 |      -0.001 |      3 | 0.742 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-target | mean of seeds |           8 |      -0.002 |      3 | 0.547 |  -0.008 |   0.004 |             0.001 |    1.000 |
| k<=16       | catboost-ft     | 0             |           8 |      -0.002 |      4 | 0.844 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-ft     | 1             |           8 |      -0.001 |      4 | 0.945 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-ft     | 2             |           8 |      -0.007 |      3 | 0.844 | nan     | nan     |           nan     |  nan     |
| k<=16       | catboost-ft     | mean of seeds |           8 |      -0.004 |      3 | 0.742 |  -0.034 |   0.030 |             0.003 |    1.000 |
| k<=16       | lr              | 0             |           8 |       0.087 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k<=16       | lr              | 1             |           8 |       0.078 |      7 | 0.016 | nan     | nan     |           nan     |  nan     |
| k<=16       | lr              | 2             |           8 |       0.082 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k<=16       | lr              | mean of seeds |           8 |       0.082 |      8 | 0.008 |   0.046 |   0.118 |             0.005 |    0.039 |
| k<=16       | knn             | 0             |           8 |       0.165 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k<=16       | knn             | 1             |           8 |       0.182 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k<=16       | knn             | 2             |           8 |       0.175 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k<=16       | knn             | mean of seeds |           8 |       0.174 |      8 | 0.008 |   0.131 |   0.215 |             0.009 |    0.039 |
| k>=32       | rf              | 0             |           8 |       0.002 |      4 | 0.547 | nan     | nan     |           nan     |  nan     |
| k>=32       | rf              | 1             |           8 |      -0.000 |      2 | 0.742 | nan     | nan     |           nan     |  nan     |
| k>=32       | rf              | 2             |           8 |      -0.004 |      2 | 0.148 | nan     | nan     |           nan     |  nan     |
| k>=32       | rf              | mean of seeds |           8 |      -0.001 |      2 | 0.641 |  -0.004 |   0.004 |             0.003 |    1.000 |
| k>=32       | catboost-target | 0             |           8 |       0.002 |      4 | 0.742 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-target | 1             |           8 |       0.000 |      3 | 0.844 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-target | 2             |           8 |      -0.003 |      3 | 0.383 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-target | mean of seeds |           8 |      -0.000 |      3 | 0.945 |  -0.003 |   0.002 |             0.003 |    1.000 |
| k>=32       | catboost-ft     | 0             |           8 |       0.012 |      7 | 0.016 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-ft     | 1             |           8 |       0.005 |      4 | 0.312 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-ft     | 2             |           8 |       0.003 |      3 | 0.742 | nan     | nan     |           nan     |  nan     |
| k>=32       | catboost-ft     | mean of seeds |           8 |       0.007 |      5 | 0.109 |   0.002 |   0.012 |             0.005 |    0.328 |
| k>=32       | lr              | 0             |           8 |       0.064 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | lr              | 1             |           8 |       0.069 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | lr              | 2             |           8 |       0.058 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | lr              | mean of seeds |           8 |       0.064 |      8 | 0.008 |   0.035 |   0.095 |             0.005 |    0.039 |
| k>=32       | knn             | 0             |           8 |       0.056 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | knn             | 1             |           8 |       0.052 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | knn             | 2             |           8 |       0.046 |      8 | 0.008 | nan     | nan     |           nan     |  nan     |
| k>=32       | knn             | mean of seeds |           8 |       0.051 |      8 | 0.008 |   0.037 |   0.066 |             0.005 |    0.039 |
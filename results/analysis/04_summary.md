# Summary: mean ± std over scenarios

Values are mean ± std over scenarios; each scenario value is the mean over seeds. `prior_abs_error` is |est_prior - true_prior| and is blank for methods without a prior estimate.

| experiment | mode | transfer_type | method | n_scenarios | roc_auc | pr_auc | balanced_accuracy | f1 | mcc | fpr | prior_abs_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fewshot | fewshot | cross-extractor | fewshot-catboost@0.1 | 26 | 0.828 ± 0.089 | 0.748 ± 0.220 | 0.767 ± 0.087 | 0.706 ± 0.183 | 0.503 ± 0.162 | 0.192 ± 0.141 | – |
| fewshot | fewshot | cross-extractor | fewshot-catboost@1 | 26 | 0.942 ± 0.040 | 0.903 ± 0.096 | 0.900 ± 0.049 | 0.871 ± 0.094 | 0.776 ± 0.096 | 0.099 ± 0.064 | – |
| fewshot | fewshot | cross-extractor | fewshot-catboost@5 | 26 | 0.962 ± 0.030 | 0.944 ± 0.057 | 0.925 ± 0.045 | 0.904 ± 0.078 | 0.837 ± 0.094 | 0.071 ± 0.062 | – |
| fewshot | fewshot | cross-extractor | fewshot-targetonly@0.1 | 26 | 0.919 ± 0.059 | 0.890 ± 0.109 | 0.869 ± 0.062 | 0.840 ± 0.119 | 0.733 ± 0.132 | 0.116 ± 0.067 | – |
| fewshot | fewshot | cross-extractor | fewshot-targetonly@1 | 26 | 0.961 ± 0.031 | 0.946 ± 0.052 | 0.922 ± 0.040 | 0.913 ± 0.064 | 0.846 ± 0.083 | 0.081 ± 0.067 | – |
| fewshot | fewshot | cross-extractor | fewshot-targetonly@5 | 26 | 0.973 ± 0.024 | 0.965 ± 0.034 | 0.935 ± 0.040 | 0.924 ± 0.061 | 0.869 ± 0.083 | 0.063 ± 0.062 | – |
| fewshot | fewshot | same-extractor | fewshot-catboost@0.1 | 12 | 0.880 ± 0.085 | 0.785 ± 0.225 | 0.815 ± 0.092 | 0.743 ± 0.210 | 0.550 ± 0.165 | 0.165 ± 0.145 | – |
| fewshot | fewshot | same-extractor | fewshot-catboost@1 | 12 | 0.968 ± 0.035 | 0.938 ± 0.087 | 0.929 ± 0.045 | 0.891 ± 0.114 | 0.820 ± 0.104 | 0.070 ± 0.036 | – |
| fewshot | fewshot | same-extractor | fewshot-catboost@5 | 12 | 0.979 ± 0.028 | 0.961 ± 0.064 | 0.950 ± 0.042 | 0.922 ± 0.092 | 0.878 ± 0.098 | 0.038 ± 0.026 | – |
| fewshot | fewshot | same-extractor | fewshot-targetonly@0.1 | 12 | 0.947 ± 0.059 | 0.920 ± 0.118 | 0.894 ± 0.063 | 0.860 ± 0.145 | 0.782 ± 0.149 | 0.093 ± 0.035 | – |
| fewshot | fewshot | same-extractor | fewshot-targetonly@1 | 12 | 0.979 ± 0.025 | 0.968 ± 0.046 | 0.942 ± 0.036 | 0.931 ± 0.074 | 0.886 ± 0.080 | 0.049 ± 0.025 | – |
| fewshot | fewshot | same-extractor | fewshot-targetonly@5 | 12 | 0.985 ± 0.021 | 0.976 ± 0.038 | 0.956 ± 0.037 | 0.942 ± 0.070 | 0.909 ± 0.082 | 0.029 ± 0.017 | – |
| fewshot | fewshot | same-family | fewshot-catboost@0.1 | 4 | 0.793 ± 0.058 | 0.652 ± 0.171 | 0.751 ± 0.057 | 0.684 ± 0.106 | 0.484 ± 0.092 | 0.200 ± 0.146 | – |
| fewshot | fewshot | same-family | fewshot-catboost@1 | 4 | 0.937 ± 0.043 | 0.879 ± 0.099 | 0.896 ± 0.049 | 0.849 ± 0.087 | 0.761 ± 0.090 | 0.110 ± 0.088 | – |
| fewshot | fewshot | same-family | fewshot-catboost@5 | 4 | 0.959 ± 0.033 | 0.918 ± 0.078 | 0.922 ± 0.046 | 0.880 ± 0.097 | 0.816 ± 0.104 | 0.083 ± 0.076 | – |
| fewshot | fewshot | same-family | fewshot-targetonly@0.1 | 4 | 0.910 ± 0.069 | 0.842 ± 0.139 | 0.857 ± 0.073 | 0.786 ± 0.149 | 0.687 ± 0.146 | 0.120 ± 0.083 | – |
| fewshot | fewshot | same-family | fewshot-targetonly@1 | 4 | 0.960 ± 0.032 | 0.928 ± 0.062 | 0.920 ± 0.045 | 0.893 ± 0.081 | 0.837 ± 0.093 | 0.081 ± 0.088 | – |
| fewshot | fewshot | same-family | fewshot-targetonly@5 | 4 | 0.974 ± 0.024 | 0.955 ± 0.045 | 0.934 ± 0.042 | 0.909 ± 0.078 | 0.862 ± 0.092 | 0.068 ± 0.079 | – |
| indomain | inductive | in-domain | catboost | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.041 | 0.947 ± 0.058 | 0.914 ± 0.082 | 0.032 ± 0.041 | – |
| indomain | inductive | in-domain | hgb | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.041 | 0.947 ± 0.058 | 0.914 ± 0.082 | 0.033 ± 0.042 | – |
| indomain | inductive | in-domain | lr | 7 | 0.923 ± 0.054 | 0.850 ± 0.166 | 0.891 ± 0.064 | 0.845 ± 0.136 | 0.747 ± 0.146 | 0.114 ± 0.104 | – |
| indomain | inductive | in-domain | mlp | 7 | 0.981 ± 0.023 | 0.975 ± 0.031 | 0.951 ± 0.044 | 0.948 ± 0.052 | 0.912 ± 0.081 | 0.038 ± 0.049 | – |
| indomain | inductive | in-domain | rf | 7 | 0.981 ± 0.021 | 0.972 ± 0.035 | 0.944 ± 0.058 | 0.936 ± 0.068 | 0.891 ± 0.115 | 0.047 ± 0.058 | – |
| indomain | inductive | in-domain | rift | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-noAnchor | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-noDN | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-noIFS | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-noLSC | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-poolInit | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-robust | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | rift-standard | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.049 | 0.921 ± 0.076 | 0.029 ± 0.037 | – |
| indomain | inductive | in-domain | xgboost | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.040 | 0.947 ± 0.059 | 0.914 ± 0.083 | 0.032 ± 0.041 | – |
| indomain | transductive | in-domain | catboost | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.041 | 0.947 ± 0.058 | 0.914 ± 0.082 | 0.032 ± 0.041 | – |
| indomain | transductive | in-domain | coral | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.041 | 0.947 ± 0.058 | 0.914 ± 0.082 | 0.033 ± 0.043 | – |
| indomain | transductive | in-domain | dann | 7 | 0.980 ± 0.023 | 0.974 ± 0.032 | 0.948 ± 0.045 | 0.932 ± 0.066 | 0.888 ± 0.091 | 0.045 ± 0.056 | – |
| indomain | transductive | in-domain | hgb | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.041 | 0.947 ± 0.058 | 0.914 ± 0.082 | 0.033 ± 0.043 | – |
| indomain | transductive | in-domain | lr | 7 | 0.923 ± 0.054 | 0.850 ± 0.166 | 0.891 ± 0.064 | 0.845 ± 0.136 | 0.747 ± 0.146 | 0.114 ± 0.104 | – |
| indomain | transductive | in-domain | mlp | 7 | 0.981 ± 0.023 | 0.975 ± 0.031 | 0.951 ± 0.044 | 0.948 ± 0.052 | 0.912 ± 0.081 | 0.038 ± 0.049 | – |
| indomain | transductive | in-domain | rf | 7 | 0.981 ± 0.021 | 0.972 ± 0.035 | 0.944 ± 0.058 | 0.936 ± 0.068 | 0.891 ± 0.115 | 0.047 ± 0.058 | – |
| indomain | transductive | in-domain | rift | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-noAnchor | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-noDN | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-noIFS | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-noLSC | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-poolInit | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-robust | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-st | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | rift-standard | 7 | 0.985 ± 0.020 | 0.979 ± 0.028 | 0.955 ± 0.042 | 0.952 ± 0.050 | 0.921 ± 0.076 | 0.029 ± 0.038 | – |
| indomain | transductive | in-domain | xgboost | 7 | 0.985 ± 0.020 | 0.980 ± 0.027 | 0.958 ± 0.040 | 0.947 ± 0.059 | 0.914 ± 0.083 | 0.032 ± 0.041 | – |
| lodo | inductive | lodo | catboost | 7 | 0.652 ± 0.208 | 0.613 ± 0.280 | 0.611 ± 0.133 | 0.435 ± 0.232 | 0.173 ± 0.182 | 0.194 ± 0.248 | – |
| lodo | inductive | lodo | hgb | 7 | 0.630 ± 0.211 | 0.591 ± 0.275 | 0.596 ± 0.148 | 0.419 ± 0.240 | 0.157 ± 0.218 | 0.208 ± 0.249 | – |
| lodo | inductive | lodo | lr | 7 | 0.480 ± 0.148 | 0.522 ± 0.299 | 0.507 ± 0.100 | 0.361 ± 0.177 | 0.025 ± 0.174 | 0.333 ± 0.224 | – |
| lodo | inductive | lodo | mlp | 7 | 0.503 ± 0.155 | 0.525 ± 0.314 | 0.478 ± 0.097 | 0.293 ± 0.143 | -0.052 ± 0.181 | 0.288 ± 0.237 | – |
| lodo | inductive | lodo | rf | 7 | 0.574 ± 0.230 | 0.534 ± 0.306 | 0.556 ± 0.109 | 0.336 ± 0.242 | 0.087 ± 0.139 | 0.186 ± 0.249 | – |
| lodo | inductive | lodo | rift | 7 | 0.586 ± 0.179 | 0.565 ± 0.270 | 0.514 ± 0.054 | 0.116 ± 0.130 | 0.018 ± 0.146 | 0.063 ± 0.060 | – |
| lodo | inductive | lodo | rift-noAnchor | 7 | 0.604 ± 0.242 | 0.569 ± 0.326 | 0.516 ± 0.143 | 0.289 ± 0.175 | 0.010 ± 0.283 | 0.236 ± 0.226 | – |
| lodo | inductive | lodo | rift-noDN | 7 | 0.620 ± 0.219 | 0.579 ± 0.281 | 0.592 ± 0.182 | 0.414 ± 0.255 | 0.141 ± 0.283 | 0.201 ± 0.250 | – |
| lodo | inductive | lodo | rift-noIFS | 7 | 0.586 ± 0.179 | 0.565 ± 0.270 | 0.514 ± 0.054 | 0.116 ± 0.130 | 0.018 ± 0.146 | 0.063 ± 0.060 | – |
| lodo | inductive | lodo | rift-noLSC | 7 | 0.586 ± 0.179 | 0.565 ± 0.270 | 0.514 ± 0.054 | 0.116 ± 0.130 | 0.018 ± 0.146 | 0.063 ± 0.060 | – |
| lodo | inductive | lodo | rift-poolInit | 7 | 0.586 ± 0.179 | 0.565 ± 0.270 | 0.514 ± 0.054 | 0.116 ± 0.130 | 0.018 ± 0.146 | 0.063 ± 0.060 | – |
| lodo | inductive | lodo | rift-robust | 7 | 0.502 ± 0.247 | 0.564 ± 0.288 | 0.517 ± 0.071 | 0.165 ± 0.177 | 0.035 ± 0.151 | 0.098 ± 0.095 | – |
| lodo | inductive | lodo | rift-standard | 7 | 0.605 ± 0.197 | 0.604 ± 0.249 | 0.550 ± 0.130 | 0.244 ± 0.240 | 0.084 ± 0.263 | 0.099 ± 0.087 | – |
| lodo | inductive | lodo | xgboost | 7 | 0.612 ± 0.207 | 0.570 ± 0.290 | 0.561 ± 0.150 | 0.364 ± 0.204 | 0.090 ± 0.256 | 0.189 ± 0.253 | – |
| lodo | transductive | lodo | catboost | 7 | 0.652 ± 0.208 | 0.613 ± 0.280 | 0.611 ± 0.133 | 0.435 ± 0.232 | 0.173 ± 0.182 | 0.194 ± 0.248 | – |
| lodo | transductive | lodo | coral | 7 | 0.631 ± 0.177 | 0.617 ± 0.198 | 0.540 ± 0.111 | 0.271 ± 0.153 | 0.061 ± 0.191 | 0.193 ± 0.157 | – |
| lodo | transductive | lodo | dann | 7 | 0.613 ± 0.184 | 0.620 ± 0.281 | 0.539 ± 0.132 | 0.435 ± 0.170 | 0.063 ± 0.210 | 0.391 ± 0.119 | – |
| lodo | transductive | lodo | hgb | 7 | 0.630 ± 0.211 | 0.591 ± 0.275 | 0.596 ± 0.148 | 0.419 ± 0.240 | 0.157 ± 0.218 | 0.208 ± 0.249 | – |
| lodo | transductive | lodo | lr | 7 | 0.480 ± 0.148 | 0.522 ± 0.299 | 0.507 ± 0.100 | 0.361 ± 0.177 | 0.025 ± 0.174 | 0.333 ± 0.224 | – |
| lodo | transductive | lodo | mlp | 7 | 0.503 ± 0.155 | 0.525 ± 0.314 | 0.478 ± 0.097 | 0.293 ± 0.143 | -0.052 ± 0.181 | 0.288 ± 0.237 | – |
| lodo | transductive | lodo | rf | 7 | 0.574 ± 0.230 | 0.534 ± 0.306 | 0.556 ± 0.109 | 0.336 ± 0.242 | 0.087 ± 0.139 | 0.186 ± 0.249 | – |
| lodo | transductive | lodo | rift | 7 | 0.519 ± 0.129 | 0.522 ± 0.262 | 0.500 ± 0.000 | 0.002 ± 0.005 | 0.001 ± 0.001 | 0.001 ± 0.002 | 0.507 ± 0.289 |
| lodo | transductive | lodo | rift-noAnchor | 7 | 0.543 ± 0.115 | 0.539 ± 0.309 | 0.501 ± 0.083 | 0.282 ± 0.128 | 0.009 ± 0.171 | 0.255 ± 0.191 | 0.359 ± 0.251 |
| lodo | transductive | lodo | rift-noDN | 7 | 0.620 ± 0.219 | 0.579 ± 0.281 | 0.598 ± 0.167 | 0.349 ± 0.375 | 0.212 ± 0.333 | 0.149 ± 0.284 | 0.280 ± 0.250 |
| lodo | transductive | lodo | rift-noIFS | 7 | 0.519 ± 0.129 | 0.522 ± 0.262 | 0.500 ± 0.000 | 0.002 ± 0.005 | 0.001 ± 0.001 | 0.001 ± 0.002 | 0.507 ± 0.289 |
| lodo | transductive | lodo | rift-noLSC | 7 | 0.519 ± 0.129 | 0.522 ± 0.262 | 0.500 ± 0.048 | 0.094 ± 0.106 | -0.004 ± 0.147 | 0.058 ± 0.064 | – |
| lodo | transductive | lodo | rift-oracleAnchor | 7 | 0.596 ± 0.159 | 0.541 ± 0.300 | 0.516 ± 0.121 | 0.191 ± 0.356 | 0.037 ± 0.283 | 0.141 ± 0.208 | 0.000 ± 0.000 |
| lodo | transductive | lodo | rift-oraclePrior | 7 | 0.562 ± 0.099 | 0.539 ± 0.298 | 0.502 ± 0.025 | 0.226 ± 0.284 | 0.000 ± 0.079 | 0.166 ± 0.224 | 0.000 ± 0.000 |
| lodo | transductive | lodo | rift-poolInit | 7 | 0.498 ± 0.107 | 0.507 ± 0.269 | 0.498 ± 0.004 | 0.002 ± 0.005 | -0.013 ± 0.028 | 0.004 ± 0.008 | 0.504 ± 0.287 |
| lodo | transductive | lodo | rift-robust | 7 | 0.487 ± 0.130 | 0.506 ± 0.294 | 0.514 ± 0.041 | 0.057 ± 0.140 | 0.013 ± 0.054 | 0.008 ± 0.013 | 0.471 ± 0.245 |
| lodo | transductive | lodo | rift-st | 7 | 0.500 ± 0.142 | 0.514 ± 0.276 | 0.500 ± 0.000 | 0.000 ± 0.001 | -0.002 ± 0.005 | 0.000 ± 0.000 | 0.508 ± 0.291 |
| lodo | transductive | lodo | rift-standard | 7 | 0.510 ± 0.102 | 0.519 ± 0.279 | 0.497 ± 0.013 | 0.029 ± 0.058 | -0.017 ± 0.045 | 0.022 ± 0.037 | 0.481 ± 0.268 |
| lodo | transductive | lodo | xgboost | 7 | 0.612 ± 0.207 | 0.570 ± 0.290 | 0.561 ± 0.150 | 0.364 ± 0.204 | 0.090 ± 0.256 | 0.189 ± 0.253 | – |
| pairwise | inductive | cross-extractor | catboost | 26 | 0.542 ± 0.192 | 0.555 ± 0.277 | 0.514 ± 0.127 | 0.347 ± 0.282 | 0.027 ± 0.226 | 0.377 ± 0.349 | – |
| pairwise | inductive | cross-extractor | hgb | 26 | 0.528 ± 0.171 | 0.562 ± 0.274 | 0.520 ± 0.115 | 0.352 ± 0.270 | 0.045 ± 0.209 | 0.347 ± 0.279 | – |
| pairwise | inductive | cross-extractor | lr | 26 | 0.503 ± 0.220 | 0.555 ± 0.260 | 0.499 ± 0.145 | 0.441 ± 0.290 | 0.004 ± 0.314 | 0.509 ± 0.293 | – |
| pairwise | inductive | cross-extractor | mlp | 26 | 0.539 ± 0.211 | 0.557 ± 0.286 | 0.511 ± 0.127 | 0.376 ± 0.277 | 0.036 ± 0.247 | 0.409 ± 0.337 | – |
| pairwise | inductive | cross-extractor | rf | 26 | 0.522 ± 0.204 | 0.550 ± 0.273 | 0.508 ± 0.145 | 0.301 ± 0.301 | 0.012 ± 0.275 | 0.361 ± 0.369 | – |
| pairwise | inductive | cross-extractor | rift | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-noAnchor | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-noDN | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-noIFS | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-noLSC | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-poolInit | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-robust | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | rift-standard | 26 | 0.519 ± 0.191 | 0.557 ± 0.272 | 0.518 ± 0.110 | 0.341 ± 0.281 | 0.037 ± 0.206 | 0.352 ± 0.294 | – |
| pairwise | inductive | cross-extractor | xgboost | 26 | 0.532 ± 0.190 | 0.560 ± 0.282 | 0.517 ± 0.114 | 0.338 ± 0.262 | 0.034 ± 0.213 | 0.347 ± 0.308 | – |
| pairwise | inductive | same-extractor | catboost | 12 | 0.635 ± 0.180 | 0.606 ± 0.300 | 0.572 ± 0.095 | 0.380 ± 0.234 | 0.134 ± 0.202 | 0.294 ± 0.307 | – |
| pairwise | inductive | same-extractor | hgb | 12 | 0.638 ± 0.146 | 0.602 ± 0.322 | 0.583 ± 0.103 | 0.366 ± 0.271 | 0.149 ± 0.187 | 0.262 ± 0.294 | – |
| pairwise | inductive | same-extractor | lr | 12 | 0.472 ± 0.221 | 0.559 ± 0.295 | 0.479 ± 0.127 | 0.403 ± 0.246 | -0.019 ± 0.230 | 0.524 ± 0.357 | – |
| pairwise | inductive | same-extractor | mlp | 12 | 0.621 ± 0.153 | 0.603 ± 0.317 | 0.543 ± 0.098 | 0.347 ± 0.235 | 0.080 ± 0.191 | 0.385 ± 0.414 | – |
| pairwise | inductive | same-extractor | rf | 12 | 0.597 ± 0.184 | 0.585 ± 0.303 | 0.511 ± 0.061 | 0.252 ± 0.249 | 0.037 ± 0.146 | 0.337 ± 0.391 | – |
| pairwise | inductive | same-extractor | rift | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-noAnchor | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-noDN | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-noIFS | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-noLSC | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-poolInit | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-robust | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | rift-standard | 12 | 0.638 ± 0.142 | 0.604 ± 0.321 | 0.580 ± 0.105 | 0.341 ± 0.285 | 0.143 ± 0.188 | 0.260 ± 0.296 | – |
| pairwise | inductive | same-extractor | xgboost | 12 | 0.642 ± 0.149 | 0.605 ± 0.314 | 0.589 ± 0.079 | 0.374 ± 0.240 | 0.170 ± 0.133 | 0.256 ± 0.306 | – |
| pairwise | inductive | same-family | catboost | 4 | 0.438 ± 0.226 | 0.373 ± 0.238 | 0.452 ± 0.046 | 0.320 ± 0.326 | -0.105 ± 0.081 | 0.443 ± 0.446 | – |
| pairwise | inductive | same-family | hgb | 4 | 0.473 ± 0.172 | 0.379 ± 0.256 | 0.490 ± 0.026 | 0.288 ± 0.281 | -0.031 ± 0.056 | 0.308 ± 0.334 | – |
| pairwise | inductive | same-family | lr | 4 | 0.341 ± 0.091 | 0.311 ± 0.240 | 0.417 ± 0.077 | 0.294 ± 0.269 | -0.166 ± 0.116 | 0.532 ± 0.394 | – |
| pairwise | inductive | same-family | mlp | 4 | 0.453 ± 0.214 | 0.366 ± 0.308 | 0.494 ± 0.091 | 0.295 ± 0.315 | -0.035 ± 0.181 | 0.307 ± 0.366 | – |
| pairwise | inductive | same-family | rf | 4 | 0.485 ± 0.156 | 0.361 ± 0.233 | 0.437 ± 0.090 | 0.291 ± 0.302 | -0.133 ± 0.155 | 0.463 ± 0.506 | – |
| pairwise | inductive | same-family | rift | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-noAnchor | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-noDN | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-noIFS | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-noLSC | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-poolInit | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-robust | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | rift-standard | 4 | 0.494 ± 0.154 | 0.382 ± 0.253 | 0.497 ± 0.010 | 0.285 ± 0.302 | -0.011 ± 0.027 | 0.320 ± 0.401 | – |
| pairwise | inductive | same-family | xgboost | 4 | 0.397 ± 0.208 | 0.351 ± 0.241 | 0.437 ± 0.058 | 0.273 ± 0.266 | -0.129 ± 0.093 | 0.403 ± 0.388 | – |
| pairwise | transductive | cross-extractor | catboost | 26 | 0.541 ± 0.193 | 0.554 ± 0.276 | 0.514 ± 0.127 | 0.343 ± 0.286 | 0.022 ± 0.231 | 0.376 ± 0.349 | – |
| pairwise | transductive | cross-extractor | coral | 26 | 0.542 ± 0.181 | 0.576 ± 0.267 | 0.512 ± 0.102 | 0.306 ± 0.284 | 0.022 ± 0.209 | 0.348 ± 0.310 | – |
| pairwise | transductive | cross-extractor | dann | 26 | 0.521 ± 0.179 | 0.544 ± 0.266 | 0.493 ± 0.105 | 0.385 ± 0.244 | -0.007 ± 0.219 | 0.450 ± 0.244 | – |
| pairwise | transductive | cross-extractor | hgb | 26 | 0.529 ± 0.172 | 0.561 ± 0.273 | 0.515 ± 0.111 | 0.346 ± 0.262 | 0.036 ± 0.201 | 0.350 ± 0.281 | – |
| pairwise | transductive | cross-extractor | lr | 26 | 0.503 ± 0.220 | 0.555 ± 0.260 | 0.499 ± 0.145 | 0.441 ± 0.290 | 0.004 ± 0.314 | 0.509 ± 0.293 | – |
| pairwise | transductive | cross-extractor | mlp | 26 | 0.539 ± 0.211 | 0.557 ± 0.286 | 0.512 ± 0.127 | 0.376 ± 0.277 | 0.036 ± 0.246 | 0.410 ± 0.337 | – |
| pairwise | transductive | cross-extractor | rf | 26 | 0.521 ± 0.203 | 0.549 ± 0.273 | 0.508 ± 0.147 | 0.303 ± 0.299 | 0.013 ± 0.276 | 0.361 ± 0.370 | – |
| pairwise | transductive | cross-extractor | rift | 26 | 0.491 ± 0.135 | 0.537 ± 0.241 | 0.514 ± 0.051 | 0.182 ± 0.188 | 0.023 ± 0.109 | 0.138 ± 0.178 | 0.418 ± 0.216 |
| pairwise | transductive | cross-extractor | rift-noAnchor | 26 | 0.560 ± 0.139 | 0.569 ± 0.261 | 0.532 ± 0.089 | 0.457 ± 0.273 | 0.049 ± 0.163 | 0.517 ± 0.326 | 0.340 ± 0.210 |
| pairwise | transductive | cross-extractor | rift-noDN | 26 | 0.544 ± 0.192 | 0.570 ± 0.278 | 0.529 ± 0.121 | 0.330 ± 0.335 | 0.066 ± 0.227 | 0.364 ± 0.358 | 0.377 ± 0.234 |
| pairwise | transductive | cross-extractor | rift-noIFS | 26 | 0.484 ± 0.139 | 0.526 ± 0.239 | 0.496 ± 0.049 | 0.137 ± 0.177 | -0.011 ± 0.098 | 0.150 ± 0.207 | 0.449 ± 0.196 |
| pairwise | transductive | cross-extractor | rift-noLSC | 26 | 0.491 ± 0.135 | 0.537 ± 0.241 | 0.516 ± 0.068 | 0.251 ± 0.207 | 0.026 ± 0.120 | 0.220 ± 0.213 | – |
| pairwise | transductive | cross-extractor | rift-oracleAnchor | 26 | 0.536 ± 0.167 | 0.553 ± 0.243 | 0.498 ± 0.126 | 0.251 ± 0.235 | 0.000 ± 0.252 | 0.224 ± 0.211 | 0.000 ± 0.000 |
| pairwise | transductive | cross-extractor | rift-oraclePrior | 26 | 0.476 ± 0.149 | 0.523 ± 0.247 | 0.497 ± 0.085 | 0.284 ± 0.251 | -0.006 ± 0.166 | 0.253 ± 0.259 | 0.000 ± 0.000 |
| pairwise | transductive | cross-extractor | rift-poolInit | 26 | 0.489 ± 0.122 | 0.542 ± 0.245 | 0.506 ± 0.055 | 0.174 ± 0.199 | 0.010 ± 0.112 | 0.178 ± 0.245 | 0.436 ± 0.219 |
| pairwise | transductive | cross-extractor | rift-robust | 26 | 0.466 ± 0.118 | 0.523 ± 0.239 | 0.501 ± 0.045 | 0.182 ± 0.226 | -0.008 ± 0.104 | 0.223 ± 0.291 | 0.452 ± 0.195 |
| pairwise | transductive | cross-extractor | rift-st | 26 | 0.486 ± 0.124 | 0.531 ± 0.243 | 0.508 ± 0.051 | 0.196 ± 0.187 | 0.015 ± 0.092 | 0.186 ± 0.220 | 0.423 ± 0.190 |
| pairwise | transductive | cross-extractor | rift-standard | 26 | 0.459 ± 0.129 | 0.511 ± 0.245 | 0.493 ± 0.058 | 0.194 ± 0.224 | -0.024 ± 0.107 | 0.254 ± 0.295 | 0.436 ± 0.215 |
| pairwise | transductive | cross-extractor | xgboost | 26 | 0.532 ± 0.190 | 0.560 ± 0.282 | 0.517 ± 0.114 | 0.338 ± 0.262 | 0.034 ± 0.213 | 0.347 ± 0.308 | – |
| pairwise | transductive | same-extractor | catboost | 12 | 0.637 ± 0.179 | 0.606 ± 0.300 | 0.572 ± 0.095 | 0.379 ± 0.234 | 0.133 ± 0.205 | 0.294 ± 0.307 | – |
| pairwise | transductive | same-extractor | coral | 12 | 0.494 ± 0.180 | 0.562 ± 0.327 | 0.475 ± 0.076 | 0.272 ± 0.244 | -0.037 ± 0.158 | 0.409 ± 0.366 | – |
| pairwise | transductive | same-extractor | dann | 12 | 0.678 ± 0.158 | 0.621 ± 0.310 | 0.585 ± 0.110 | 0.428 ± 0.163 | 0.135 ± 0.158 | 0.385 ± 0.357 | – |
| pairwise | transductive | same-extractor | hgb | 12 | 0.643 ± 0.161 | 0.601 ± 0.321 | 0.587 ± 0.104 | 0.373 ± 0.264 | 0.156 ± 0.186 | 0.264 ± 0.296 | – |
| pairwise | transductive | same-extractor | lr | 12 | 0.472 ± 0.221 | 0.559 ± 0.295 | 0.479 ± 0.127 | 0.403 ± 0.246 | -0.019 ± 0.230 | 0.524 ± 0.357 | – |
| pairwise | transductive | same-extractor | mlp | 12 | 0.621 ± 0.153 | 0.603 ± 0.317 | 0.543 ± 0.098 | 0.347 ± 0.235 | 0.080 ± 0.191 | 0.385 ± 0.415 | – |
| pairwise | transductive | same-extractor | rf | 12 | 0.597 ± 0.187 | 0.585 ± 0.304 | 0.512 ± 0.062 | 0.254 ± 0.247 | 0.039 ± 0.146 | 0.336 ± 0.391 | – |
| pairwise | transductive | same-extractor | rift | 12 | 0.465 ± 0.130 | 0.535 ± 0.309 | 0.496 ± 0.050 | 0.100 ± 0.158 | -0.028 ± 0.082 | 0.102 ± 0.160 | 0.471 ± 0.300 |
| pairwise | transductive | same-extractor | rift-noAnchor | 12 | 0.491 ± 0.159 | 0.543 ± 0.316 | 0.495 ± 0.113 | 0.343 ± 0.264 | -0.045 ± 0.195 | 0.522 ± 0.394 | 0.531 ± 0.241 |
| pairwise | transductive | same-extractor | rift-noDN | 12 | 0.625 ± 0.151 | 0.602 ± 0.318 | 0.558 ± 0.101 | 0.296 ± 0.305 | 0.103 ± 0.172 | 0.243 ± 0.296 | 0.457 ± 0.272 |
| pairwise | transductive | same-extractor | rift-noIFS | 12 | 0.493 ± 0.167 | 0.545 ± 0.322 | 0.486 ± 0.069 | 0.084 ± 0.125 | -0.049 ± 0.144 | 0.085 ± 0.173 | 0.468 ± 0.308 |
| pairwise | transductive | same-extractor | rift-noLSC | 12 | 0.465 ± 0.130 | 0.535 ± 0.309 | 0.489 ± 0.079 | 0.190 ± 0.200 | -0.041 ± 0.126 | 0.193 ± 0.243 | – |
| pairwise | transductive | same-extractor | rift-oracleAnchor | 12 | 0.570 ± 0.168 | 0.588 ± 0.318 | 0.520 ± 0.087 | 0.286 ± 0.286 | 0.022 ± 0.188 | 0.200 ± 0.235 | 0.000 ± 0.000 |
| pairwise | transductive | same-extractor | rift-oraclePrior | 12 | 0.528 ± 0.138 | 0.564 ± 0.313 | 0.522 ± 0.098 | 0.306 ± 0.330 | 0.014 ± 0.149 | 0.231 ± 0.287 | 0.000 ± 0.000 |
| pairwise | transductive | same-extractor | rift-poolInit | 12 | 0.480 ± 0.141 | 0.545 ± 0.311 | 0.483 ± 0.049 | 0.084 ± 0.095 | -0.057 ± 0.099 | 0.097 ± 0.135 | 0.464 ± 0.316 |
| pairwise | transductive | same-extractor | rift-robust | 12 | 0.513 ± 0.185 | 0.564 ± 0.329 | 0.516 ± 0.091 | 0.141 ± 0.240 | 0.017 ± 0.180 | 0.089 ± 0.137 | 0.449 ± 0.311 |
| pairwise | transductive | same-extractor | rift-st | 12 | 0.448 ± 0.119 | 0.532 ± 0.313 | 0.495 ± 0.064 | 0.153 ± 0.189 | -0.028 ± 0.095 | 0.147 ± 0.236 | 0.438 ± 0.282 |
| pairwise | transductive | same-extractor | rift-standard | 12 | 0.537 ± 0.211 | 0.567 ± 0.328 | 0.500 ± 0.066 | 0.141 ± 0.184 | -0.010 ± 0.118 | 0.102 ± 0.236 | 0.434 ± 0.310 |
| pairwise | transductive | same-extractor | xgboost | 12 | 0.642 ± 0.149 | 0.605 ± 0.314 | 0.589 ± 0.079 | 0.374 ± 0.240 | 0.170 ± 0.133 | 0.256 ± 0.306 | – |
| pairwise | transductive | same-family | catboost | 4 | 0.440 ± 0.225 | 0.373 ± 0.238 | 0.452 ± 0.046 | 0.320 ± 0.326 | -0.105 ± 0.081 | 0.443 ± 0.446 | – |
| pairwise | transductive | same-family | coral | 4 | 0.494 ± 0.066 | 0.387 ± 0.312 | 0.478 ± 0.032 | 0.361 ± 0.368 | -0.046 ± 0.070 | 0.486 ± 0.498 | – |
| pairwise | transductive | same-family | dann | 4 | 0.516 ± 0.204 | 0.394 ± 0.326 | 0.527 ± 0.069 | 0.320 ± 0.321 | 0.042 ± 0.140 | 0.254 ± 0.212 | – |
| pairwise | transductive | same-family | hgb | 4 | 0.480 ± 0.172 | 0.383 ± 0.260 | 0.489 ± 0.027 | 0.284 ± 0.280 | -0.034 ± 0.059 | 0.306 ± 0.331 | – |
| pairwise | transductive | same-family | lr | 4 | 0.341 ± 0.091 | 0.311 ± 0.240 | 0.417 ± 0.077 | 0.294 ± 0.269 | -0.166 ± 0.116 | 0.532 ± 0.394 | – |
| pairwise | transductive | same-family | mlp | 4 | 0.453 ± 0.214 | 0.367 ± 0.308 | 0.493 ± 0.093 | 0.293 ± 0.313 | -0.037 ± 0.183 | 0.304 ± 0.361 | – |
| pairwise | transductive | same-family | rf | 4 | 0.478 ± 0.145 | 0.359 ± 0.236 | 0.437 ± 0.091 | 0.291 ± 0.301 | -0.132 ± 0.156 | 0.463 ± 0.506 | – |
| pairwise | transductive | same-family | rift | 4 | 0.501 ± 0.061 | 0.388 ± 0.324 | 0.522 ± 0.043 | 0.275 ± 0.308 | 0.033 ± 0.090 | 0.229 ± 0.303 | 0.211 ± 0.142 |
| pairwise | transductive | same-family | rift-noAnchor | 4 | 0.541 ± 0.087 | 0.401 ± 0.324 | 0.531 ± 0.051 | 0.340 ± 0.354 | 0.062 ± 0.120 | 0.314 ± 0.343 | 0.193 ± 0.131 |
| pairwise | transductive | same-family | rift-noDN | 4 | 0.445 ± 0.198 | 0.366 ± 0.273 | 0.516 ± 0.041 | 0.306 ± 0.343 | 0.025 ± 0.074 | 0.307 ± 0.405 | 0.222 ± 0.106 |
| pairwise | transductive | same-family | rift-noIFS | 4 | 0.452 ± 0.072 | 0.373 ± 0.318 | 0.518 ± 0.048 | 0.270 ± 0.312 | 0.020 ± 0.092 | 0.212 ± 0.229 | 0.177 ± 0.091 |
| pairwise | transductive | same-family | rift-noLSC | 4 | 0.501 ± 0.061 | 0.388 ± 0.324 | 0.524 ± 0.049 | 0.307 ± 0.316 | 0.034 ± 0.097 | 0.257 ± 0.308 | – |
| pairwise | transductive | same-family | rift-oracleAnchor | 4 | 0.510 ± 0.139 | 0.401 ± 0.346 | 0.529 ± 0.052 | 0.312 ± 0.343 | 0.043 ± 0.103 | 0.235 ± 0.239 | 0.000 ± 0.000 |
| pairwise | transductive | same-family | rift-oraclePrior | 4 | 0.449 ± 0.043 | 0.370 ± 0.289 | 0.484 ± 0.030 | 0.313 ± 0.313 | -0.034 ± 0.052 | 0.345 ± 0.353 | 0.000 ± 0.000 |
| pairwise | transductive | same-family | rift-poolInit | 4 | 0.493 ± 0.063 | 0.387 ± 0.332 | 0.507 ± 0.049 | 0.301 ± 0.339 | 0.000 ± 0.097 | 0.301 ± 0.357 | 0.191 ± 0.076 |
| pairwise | transductive | same-family | rift-robust | 4 | 0.460 ± 0.049 | 0.371 ± 0.302 | 0.490 ± 0.012 | 0.233 ± 0.258 | -0.020 ± 0.033 | 0.212 ± 0.211 | 0.175 ± 0.129 |
| pairwise | transductive | same-family | rift-st | 4 | 0.519 ± 0.058 | 0.406 ± 0.322 | 0.530 ± 0.022 | 0.295 ± 0.260 | 0.060 ± 0.047 | 0.247 ± 0.309 | 0.235 ± 0.193 |
| pairwise | transductive | same-family | rift-standard | 4 | 0.497 ± 0.065 | 0.385 ± 0.309 | 0.524 ± 0.033 | 0.294 ± 0.332 | 0.043 ± 0.067 | 0.257 ± 0.328 | 0.213 ± 0.109 |
| pairwise | transductive | same-family | xgboost | 4 | 0.397 ± 0.208 | 0.351 ± 0.241 | 0.437 ± 0.058 | 0.273 ± 0.266 | -0.129 ± 0.093 | 0.403 ± 0.388 | – |

# Concept-shift test (kNN overlap region)

Standardised with source-train statistics; k = 10 nearest source-train neighbours (at most 50,000 reference rows); queries are at most 10,000 rows of the target test split. Overlap region: mean neighbour distance at or below the 90th percentile of leave-self-out source-to-source distances. Neighbour label is the majority vote (5-5 ties are benign). The in-domain reference is source test rows against the same source-train index. Kappa near or below 0 inside the overlap means the same flow statistics carry different labels across the two datasets.

| pair | source | target | transfer_type | seed | n_reference | n_target | n_source_test | threshold | cross_overlap_fraction | cross_agreement | cross_kappa | cross_n_overlap | indomain_overlap_fraction | indomain_agreement | indomain_kappa | indomain_n_overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cicids2018->iot23 | cicids2018 | iot23 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.174 | 0.460 | 0.291 | -0.076 | 4599 | 0.896 | 0.971 | 0.885 | 8957 |
| cicids2017->cicids2018 | cicids2017 | cicids2018 | same-family | 0 | 50000 | 10000 | 10000 | 0.323 | 0.401 | 0.792 | -0.066 | 4010 | 0.896 | 0.995 | 0.980 | 8957 |
| unswnb15->cicids2018 | unswnb15 | cicids2018 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.212 | 0.004 | 0.000 | -0.053 | 39 | 0.894 | 0.895 | 0.786 | 8942 |
| insdn->cicids2018 | insdn | cicids2018 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.017 | 0.855 | -0.050 | 166 | 0.895 | 0.978 | 0.948 | 8950 |
| iot23->cicids2017 | iot23 | cicids2017 | cross-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.016 | 0.425 | -0.042 | 160 | 0.896 | 0.896 | 0.792 | 8964 |
| cicids2017->iot23 | cicids2017 | iot23 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.323 | 0.512 | 0.353 | -0.040 | 5118 | 0.896 | 0.995 | 0.980 | 8957 |
| cicids2018->usbids1 | cicids2018 | usbids1 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.174 | 0.001 | 0.143 | 0.000 | 7 | 0.896 | 0.971 | 0.885 | 8957 |
| cicids2017->unswnb15 | cicids2017 | unswnb15 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.323 | 0.033 | 0.642 | 0.000 | 330 | 0.896 | 0.995 | 0.980 | 8957 |
| usbids1->iot23 | usbids1 | iot23 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.001 | 0.300 | 0.000 | 10 | 0.903 | 1.000 | 1.000 | 9028 |
| usbids1->cicids2018 | usbids1 | cicids2018 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.002 | 0.062 | 0.000 | 16 | 0.903 | 1.000 | 1.000 | 9028 |
| usbids1->cicids2017 | usbids1 | cicids2017 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.002 | 0.600 | 0.000 | 20 | 0.903 | 1.000 | 1.000 | 9028 |
| toniot->unswnb15 | toniot | unswnb15 | same-family | 0 | 50000 | 10000 | 10000 | 0.167 | 0.001 | 0.000 | 0.000 | 7 | 0.905 | 0.997 | 0.992 | 9049 |
| unswnb15->iot23 | unswnb15 | iot23 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.212 | 0.002 | 0.167 | 0.000 | 18 | 0.894 | 0.895 | 0.786 | 8942 |
| cicids2018->unswnb15 | cicids2018 | unswnb15 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.174 | 0.000 | 0.000 | 0.000 | 3 | 0.896 | 0.971 | 0.885 | 8957 |
| cicids2018->insdn | cicids2018 | insdn | same-extractor | 0 | 50000 | 10000 | 10000 | 0.174 | 0.059 | 0.362 | 0.000 | 585 | 0.896 | 0.971 | 0.885 | 8957 |
| cicids2017->insdn | cicids2017 | insdn | same-extractor | 0 | 50000 | 10000 | 10000 | 0.323 | 0.128 | 0.384 | 0.000 | 1282 | 0.896 | 0.995 | 0.980 | 8957 |
| unswnb15->cicids2017 | unswnb15 | cicids2017 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.212 | 0.000 | 0.000 | 0.000 | 4 | 0.894 | 0.895 | 0.786 | 8942 |
| iot23->usbids1 | iot23 | usbids1 | cross-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.000 | 0.000 | 0.000 | 3 | 0.896 | 0.896 | 0.792 | 8964 |
| toniot->cicids2017 | toniot | cicids2017 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.167 | 0.243 | 0.391 | 0.006 | 2433 | 0.905 | 0.997 | 0.992 | 9049 |
| cicids2017->toniot | cicids2017 | toniot | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.323 | 0.194 | 0.258 | 0.031 | 1939 | 0.896 | 0.995 | 0.980 | 8957 |
| iot23->cicids2018 | iot23 | cicids2018 | cross-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.005 | 0.449 | 0.032 | 49 | 0.896 | 0.896 | 0.792 | 8964 |
| insdn->toniot | insdn | toniot | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.102 | 0.350 | 0.033 | 1016 | 0.895 | 0.978 | 0.948 | 8950 |
| toniot->iot23 | toniot | iot23 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.167 | 0.513 | 0.275 | 0.035 | 5134 | 0.905 | 0.997 | 0.992 | 9049 |
| insdn->cicids2017 | insdn | cicids2017 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.043 | 0.834 | 0.040 | 433 | 0.895 | 0.978 | 0.948 | 8950 |
| cicids2018->toniot | cicids2018 | toniot | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.174 | 0.156 | 0.330 | 0.054 | 1559 | 0.896 | 0.971 | 0.885 | 8957 |
| unswnb15->toniot | unswnb15 | toniot | same-family | 0 | 50000 | 10000 | 10000 | 0.212 | 0.001 | 0.300 | 0.054 | 10 | 0.894 | 0.895 | 0.786 | 8942 |
| iot23->toniot | iot23 | toniot | same-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.038 | 0.472 | 0.061 | 377 | 0.896 | 0.896 | 0.792 | 8964 |
| toniot->cicids2018 | toniot | cicids2018 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.167 | 0.160 | 0.515 | 0.100 | 1595 | 0.905 | 0.997 | 0.992 | 9049 |
| cicids2017->usbids1 | cicids2017 | usbids1 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.323 | 0.007 | 0.826 | 0.119 | 69 | 0.896 | 0.995 | 0.980 | 8957 |
| toniot->insdn | toniot | insdn | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.167 | 0.112 | 0.736 | 0.142 | 1123 | 0.905 | 0.997 | 0.992 | 9049 |
| insdn->iot23 | insdn | iot23 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.036 | 0.501 | 0.159 | 357 | 0.895 | 0.978 | 0.948 | 8950 |
| cicids2018->cicids2017 | cicids2018 | cicids2017 | same-family | 0 | 50000 | 10000 | 10000 | 0.174 | 0.327 | 0.980 | 0.247 | 3267 | 0.896 | 0.971 | 0.885 | 8957 |
| iot23->insdn | iot23 | insdn | cross-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.002 | 0.667 | 0.393 | 18 | 0.896 | 0.896 | 0.792 | 8964 |
| toniot->usbids1 | toniot | usbids1 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.167 | 0.001 | 1.000 | 1.000 | 6 | 0.905 | 0.997 | 0.992 | 9049 |
| usbids1->insdn | usbids1 | insdn | same-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.000 | – | – | 0 | 0.903 | 1.000 | 1.000 | 9028 |
| usbids1->unswnb15 | usbids1 | unswnb15 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.000 | – | – | 0 | 0.903 | 1.000 | 1.000 | 9028 |
| usbids1->toniot | usbids1 | toniot | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.140 | 0.001 | 1.000 | – | 9 | 0.903 | 1.000 | 1.000 | 9028 |
| insdn->usbids1 | insdn | usbids1 | same-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.000 | – | – | 0 | 0.895 | 0.978 | 0.948 | 8950 |
| insdn->unswnb15 | insdn | unswnb15 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.134 | 0.000 | – | – | 0 | 0.895 | 0.978 | 0.948 | 8950 |
| unswnb15->usbids1 | unswnb15 | usbids1 | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.212 | 0.000 | – | – | 0 | 0.894 | 0.895 | 0.786 | 8942 |
| unswnb15->insdn | unswnb15 | insdn | cross-extractor | 0 | 50000 | 10000 | 10000 | 0.212 | 0.000 | – | – | 0 | 0.894 | 0.895 | 0.786 | 8942 |
| iot23->unswnb15 | iot23 | unswnb15 | cross-extractor | 0 | 27565 | 10000 | 10000 | 0.027 | 0.000 | – | – | 0 | 0.896 | 0.896 | 0.792 | 8964 |

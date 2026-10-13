# Quick CatBoost: cic17+unsw__to__ton+iot23__core

> **Note on sampling.** Rows are water-filled per attack type (each attack type is drawn up to a common budget, with a per-label floor), so the attack prior in these samples is not the natural prior of any dataset. The numbers are exploratory and are not benchmark results. The benchmark (DESIGN.md §2) samples distinct rows and keeps natural priors.

- Train datasets: cicids2017, unswnb15
- Test datasets: toniot, iot23
- Feature mode: `core` (13 features)
- Sampling: at most 100000 rows per dataset, water-filled per attack type (floor 1000 per label), seed 0
- Cleaning: inf replaced by NaN, NaN rows dropped on model features, duplicates dropped
- Model: CatBoostClassifier(iterations=500, depth=6, auto_class_weights='Balanced', random_seed=0); early stopping on a 10% slice of pooled 70% splits; best iteration 499; threshold 0.5
- Training rows: 115825 fit, 12870 early-stopping; fit time 6.8 s; total runtime 26.7 s

## Feature set

duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, total_pkts, total_bytes, bytes_per_s, pkts_per_s, fwd_bytes_per_pkt, bwd_bytes_per_pkt, fwd_bwd_pkt_ratio, fwd_bwd_byte_ratio

## Sample composition

| Dataset | Role | Raw rows scanned | Clean rows | Sampled | Benign | Attack | Attack types |
|---|---|---|---|---|---|---|---|
| cicids2017 | train | 2830743 | 1946389 | 100000 | 22397 | 77603 | 15 |
| unswnb15 | train | 175341 | 83851 | 83851 | 41384 | 42467 | 10 |
| toniot | test | 211043 | 91858 | 91858 | 22885 | 68973 | 10 |
| iot23 | test | 6046623 | 86813 | 70922 | 36697 | 34225 | 13 |

Per attack type in sample for cicids2017:

| Attack type | Binary label | n |
|---|---|---|
| BENIGN | 0 | 22397 |
| DDoS | 1 | 22397 |
| DoS Hulk | 1 | 22397 |
| DoS GoldenEye | 1 | 10281 |
| DoS Slowhttptest | 1 | 5172 |
| DoS slowloris | 1 | 4867 |
| FTP-Patator | 1 | 4259 |
| SSH-Patator | 1 | 3149 |
| PortScan | 1 | 1612 |
| Web Attack � Brute Force | 1 | 1426 |
| Bot | 1 | 1325 |
| Web Attack � XSS | 1 | 651 |
| Infiltration | 1 | 36 |
| Web Attack � Sql Injection | 1 | 20 |
| Heartbleed | 1 | 11 |

Per attack type in sample for unswnb15:

| Attack type | Binary label | n |
|---|---|---|
| Normal | 0 | 41384 |
| Exploits | 1 | 18706 |
| Fuzzers | 1 | 12827 |
| Reconnaissance | 1 | 5363 |
| DoS | 1 | 2744 |
| Shellcode | 1 | 954 |
| Generic | 1 | 689 |
| Backdoor | 1 | 545 |
| Analysis | 1 | 517 |
| Worms | 1 | 122 |

Per attack type in sample for toniot:

| Attack type | Binary label | n |
|---|---|---|
| normal | 0 | 22885 |
| injection | 1 | 19787 |
| password | 1 | 15233 |
| ddos | 1 | 11875 |
| xss | 1 | 8544 |
| dos | 1 | 6291 |
| scanning | 1 | 4425 |
| backdoor | 1 | 1812 |
| mitm | 1 | 1004 |
| ransomware | 1 | 2 |

Per attack type in sample for iot23:

| Attack type | Binary label | n |
|---|---|---|
| Benign | 0 | 36697 |
| PartOfAHorizontalPortScan | 1 | 26231 |
| C&C | 1 | 5529 |
| Okiru | 1 | 878 |
| C&C-HeartBeat | 1 | 799 |
| Attack | 1 | 538 |
| DDoS | 1 | 162 |
| C&C-FileDownload | 1 | 46 |
| C&C-Torii | 1 | 17 |
| FileDownload | 1 | 13 |
| C&C-HeartBeat-FileDownload | 1 | 8 |
| Okiru-Attack | 1 | 3 |
| C&C-Mirai | 1 | 1 |

## Metrics

| Kind | Eval set | n | Attack prior | ROC-AUC | PR-AUC | Accuracy | Bal. acc | F1 (attack) | Precision | Recall (DR) | FPR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in-domain | cicids2017 30% holdout | 30000 | 0.7760 | 0.9992 | 0.9998 | 0.9912 | 0.9854 | 0.9943 | 0.9928 | 0.9958 | 0.0250 | 0.9745 |
| in-domain | unswnb15 30% holdout | 25156 | 0.5064 | 0.9741 | 0.9722 | 0.9007 | 0.9010 | 0.8995 | 0.9230 | 0.8772 | 0.0751 | 0.8026 |
| cross-dataset | toniot | 91858 | 0.7509 | 0.5558 | 0.8063 | 0.3659 | 0.5232 | 0.3319 | 0.7946 | 0.2097 | 0.1633 | 0.0503 |
| cross-dataset | iot23 | 70922 | 0.4826 | 0.2810 | 0.4651 | 0.5358 | 0.5486 | 0.6554 | 0.5107 | 0.9146 | 0.8174 | 0.1418 |

## Confusion matrices [[TN, FP], [FN, TP]]

### cicids2017 30% holdout

TN=6551, FP=168, FN=97, TP=23184

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 6551 | 168 |
| True attack (1) | 97 | 23184 |

### unswnb15 30% holdout

TN=11484, FP=932, FN=1565, TP=11175

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 11484 | 932 |
| True attack (1) | 1565 | 11175 |

### toniot

TN=19147, FP=3738, FN=54508, TP=14465

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 19147 | 3738 |
| True attack (1) | 54508 | 14465 |

### iot23

TN=6701, FP=29996, FN=2923, TP=31302

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 6701 | 29996 |
| True attack (1) | 2923 | 31302 |

## Per-attack-type breakdown

### cicids2017 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| BENIGN | 6551 | 168 | 6719 | 2.5004 |
| Bot | 2 | 395 | 397 | 99.4962 |
| DDoS | 5 | 6714 | 6719 | 99.9256 |
| DoS GoldenEye | 17 | 3067 | 3084 | 99.4488 |
| DoS Hulk | 29 | 6690 | 6719 | 99.5684 |
| DoS Slowhttptest | 10 | 1542 | 1552 | 99.3557 |
| DoS slowloris | 6 | 1454 | 1460 | 99.5890 |
| FTP-Patator | 2 | 1276 | 1278 | 99.8435 |
| Heartbleed | 0 | 3 | 3 | 100.0000 |
| Infiltration | 3 | 8 | 11 | 72.7273 |
| PortScan | 4 | 480 | 484 | 99.1736 |
| SSH-Patator | 10 | 935 | 945 | 98.9418 |
| Web Attack � Brute Force | 4 | 424 | 428 | 99.0654 |
| Web Attack � Sql Injection | 1 | 5 | 6 | 83.3333 |
| Web Attack � XSS | 4 | 191 | 195 | 97.9487 |

### unswnb15 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Normal | 11484 | 932 | 12416 | 7.5064 |
| Analysis | 78 | 77 | 155 | 49.6774 |
| Backdoor | 13 | 150 | 163 | 92.0245 |
| DoS | 57 | 766 | 823 | 93.0741 |
| Exploits | 236 | 5376 | 5612 | 95.7947 |
| Fuzzers | 1100 | 2748 | 3848 | 71.4137 |
| Generic | 15 | 192 | 207 | 92.7536 |
| Reconnaissance | 6 | 1603 | 1609 | 99.6271 |
| Shellcode | 56 | 230 | 286 | 80.4196 |
| Worms | 4 | 33 | 37 | 89.1892 |

### toniot

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| normal | 19147 | 3738 | 22885 | 16.3338 |
| backdoor | 1596 | 216 | 1812 | 11.9205 |
| ddos | 7373 | 4502 | 11875 | 37.9116 |
| dos | 5777 | 514 | 6291 | 8.1704 |
| injection | 14084 | 5703 | 19787 | 28.8220 |
| mitm | 918 | 86 | 1004 | 8.5657 |
| password | 13935 | 1298 | 15233 | 8.5210 |
| ransomware | 2 | 0 | 2 | 0.0000 |
| scanning | 4368 | 57 | 4425 | 1.2881 |
| xss | 6455 | 2089 | 8544 | 24.4499 |

### iot23

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Benign | 6701 | 29996 | 36697 | 81.7397 |
| Attack | 519 | 19 | 538 | 3.5316 |
| C&C | 756 | 4773 | 5529 | 86.3266 |
| C&C-FileDownload | 46 | 0 | 46 | 0.0000 |
| C&C-HeartBeat | 338 | 461 | 799 | 57.6971 |
| C&C-HeartBeat-FileDownload | 8 | 0 | 8 | 0.0000 |
| C&C-Mirai | 1 | 0 | 1 | 0.0000 |
| C&C-Torii | 8 | 9 | 17 | 52.9412 |
| DDoS | 109 | 53 | 162 | 32.7160 |
| FileDownload | 13 | 0 | 13 | 0.0000 |
| Okiru | 803 | 75 | 878 | 8.5421 |
| Okiru-Attack | 1 | 2 | 3 | 66.6667 |
| PartOfAHorizontalPortScan | 321 | 25910 | 26231 | 98.7763 |

## Top-10 CatBoost feature importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | bwd_bytes_per_pkt | 14.3254 |
| 2 | duration | 10.0622 |
| 3 | fwd_bytes_per_pkt | 9.5637 |
| 4 | pkts_per_s | 9.4520 |
| 5 | total_pkts | 8.3385 |
| 6 | fwd_bwd_pkt_ratio | 8.2967 |
| 7 | fwd_bytes | 7.6628 |
| 8 | fwd_bwd_byte_ratio | 7.0457 |
| 9 | total_bytes | 6.5580 |
| 10 | bwd_bytes | 5.8023 |

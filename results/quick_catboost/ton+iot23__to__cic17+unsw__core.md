# Quick CatBoost: ton+iot23__to__cic17+unsw__core

- Train datasets: toniot, iot23
- Test datasets: cicids2017, unswnb15
- Feature mode: `core` (13 features)
- Sampling: at most 100000 rows per dataset, stratified by original label, seed 0
- Cleaning: inf replaced by NaN, NaN rows dropped on model features, duplicates dropped
- Model: CatBoostClassifier(iterations=500, depth=6, auto_class_weights='Balanced', random_seed=0); early stopping on a 10% slice of pooled 70% splits; best iteration 499; threshold 0.5
- Training rows: 102550 fit, 11395 early-stopping; fit time 6.0 s; total runtime 26.2 s

## Feature set

duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, total_pkts, total_bytes, bytes_per_s, pkts_per_s, fwd_bytes_per_pkt, bwd_bytes_per_pkt, fwd_bwd_pkt_ratio, fwd_bwd_byte_ratio

## Sample composition

| Dataset | Role | Raw rows scanned | Clean rows | Sampled | Benign | Attack | Attack types |
|---|---|---|---|---|---|---|---|
| toniot | train | 211043 | 91858 | 91858 | 22885 | 68973 | 10 |
| iot23 | train | 6046623 | 86813 | 70922 | 36697 | 34225 | 13 |
| cicids2017 | test | 2830743 | 1946389 | 100000 | 22397 | 77603 | 15 |
| unswnb15 | test | 175341 | 83851 | 83851 | 41384 | 42467 | 10 |

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

## Metrics

| Kind | Eval set | n | Attack prior | ROC-AUC | PR-AUC | Accuracy | Bal. acc | F1 (attack) | Precision | Recall (DR) | FPR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in-domain | toniot 30% holdout | 27558 | 0.7509 | 0.9991 | 0.9997 | 0.9890 | 0.9895 | 0.9926 | 0.9968 | 0.9885 | 0.0095 | 0.9710 |
| in-domain | iot23 30% holdout | 21277 | 0.4826 | 0.9465 | 0.9602 | 0.9242 | 0.9225 | 0.9176 | 0.9659 | 0.8739 | 0.0288 | 0.8514 |
| cross-dataset | cicids2017 | 100000 | 0.7760 | 0.6410 | 0.8359 | 0.6916 | 0.5764 | 0.7980 | 0.8114 | 0.7850 | 0.6323 | 0.1472 |
| cross-dataset | unswnb15 | 83851 | 0.5065 | 0.5560 | 0.5345 | 0.5493 | 0.5487 | 0.5716 | 0.5511 | 0.5937 | 0.4962 | 0.0979 |

## Confusion matrices [[TN, FP], [FN, TP]]

### toniot 30% holdout

TN=6801, FP=65, FN=238, TP=20454

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 6801 | 65 |
| True attack (1) | 238 | 20454 |

### iot23 30% holdout

TN=10692, FP=317, FN=1295, TP=8973

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 10692 | 317 |
| True attack (1) | 1295 | 8973 |

### cicids2017

TN=8235, FP=14162, FN=16682, TP=60921

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 8235 | 14162 |
| True attack (1) | 16682 | 60921 |

### unswnb15

TN=20848, FP=20536, FN=17255, TP=25212

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 20848 | 20536 |
| True attack (1) | 17255 | 25212 |

## Per-attack-type breakdown

### toniot 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| normal | 6801 | 65 | 6866 | 0.9467 |
| backdoor | 4 | 540 | 544 | 99.2647 |
| ddos | 118 | 3445 | 3563 | 96.6882 |
| dos | 11 | 1876 | 1887 | 99.4171 |
| injection | 26 | 5910 | 5936 | 99.5620 |
| mitm | 33 | 268 | 301 | 89.0365 |
| password | 5 | 4565 | 4570 | 99.8906 |
| ransomware | 0 | 1 | 1 | 100.0000 |
| scanning | 4 | 1323 | 1327 | 99.6986 |
| xss | 37 | 2526 | 2563 | 98.5564 |

### iot23 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Benign | 10692 | 317 | 11009 | 2.8795 |
| Attack | 2 | 167 | 169 | 98.8166 |
| C&C | 1151 | 535 | 1686 | 31.7319 |
| C&C-FileDownload | 3 | 19 | 22 | 86.3636 |
| C&C-HeartBeat | 117 | 107 | 224 | 47.7679 |
| C&C-HeartBeat-FileDownload | 0 | 2 | 2 | 100.0000 |
| C&C-Mirai | 0 | 1 | 1 | 100.0000 |
| C&C-Torii | 3 | 1 | 4 | 25.0000 |
| DDoS | 1 | 54 | 55 | 98.1818 |
| FileDownload | 0 | 2 | 2 | 100.0000 |
| Okiru | 8 | 279 | 287 | 97.2125 |
| Okiru-Attack | 1 | 0 | 1 | 0.0000 |
| PartOfAHorizontalPortScan | 9 | 7806 | 7815 | 99.8848 |

### cicids2017

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| BENIGN | 8235 | 14162 | 22397 | 63.2317 |
| Bot | 19 | 1306 | 1325 | 98.5660 |
| DDoS | 7134 | 15263 | 22397 | 68.1475 |
| DoS GoldenEye | 1121 | 9160 | 10281 | 89.0964 |
| DoS Hulk | 7630 | 14767 | 22397 | 65.9329 |
| DoS Slowhttptest | 382 | 4790 | 5172 | 92.6141 |
| DoS slowloris | 325 | 4542 | 4867 | 93.3224 |
| FTP-Patator | 22 | 4237 | 4259 | 99.4834 |
| Heartbleed | 11 | 0 | 11 | 0.0000 |
| Infiltration | 3 | 33 | 36 | 91.6667 |
| PortScan | 3 | 1609 | 1612 | 99.8139 |
| SSH-Patator | 2 | 3147 | 3149 | 99.9365 |
| Web Attack � Brute Force | 7 | 1419 | 1426 | 99.5091 |
| Web Attack � Sql Injection | 0 | 20 | 20 | 100.0000 |
| Web Attack � XSS | 23 | 628 | 651 | 96.4670 |

### unswnb15

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Normal | 20848 | 20536 | 41384 | 49.6230 |
| Analysis | 196 | 321 | 517 | 62.0890 |
| Backdoor | 372 | 173 | 545 | 31.7431 |
| DoS | 1167 | 1577 | 2744 | 57.4708 |
| Exploits | 7913 | 10793 | 18706 | 57.6981 |
| Fuzzers | 5260 | 7567 | 12827 | 58.9927 |
| Generic | 359 | 330 | 689 | 47.8955 |
| Reconnaissance | 1345 | 4018 | 5363 | 74.9208 |
| Shellcode | 582 | 372 | 954 | 38.9937 |
| Worms | 61 | 61 | 122 | 50.0000 |

## Top-10 CatBoost feature importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | total_pkts | 27.0456 |
| 2 | fwd_bwd_pkt_ratio | 13.3720 |
| 3 | fwd_pkts | 9.0536 |
| 4 | duration | 8.4745 |
| 5 | pkts_per_s | 7.6443 |
| 6 | fwd_bytes | 5.5994 |
| 7 | bwd_bytes_per_pkt | 4.7157 |
| 8 | bytes_per_s | 4.4863 |
| 9 | bwd_bytes | 4.3133 |
| 10 | fwd_bwd_byte_ratio | 4.2454 |

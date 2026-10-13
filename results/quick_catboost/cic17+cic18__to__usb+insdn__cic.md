# Quick CatBoost: cic17+cic18__to__usb+insdn__cic

- Train datasets: cicids2017, cicids2018
- Test datasets: usbids1, insdn
- Feature mode: `cic` (76 features)
- Sampling: at most 100000 rows per dataset, stratified by original label, seed 0
- Cleaning: inf replaced by NaN, NaN rows dropped on model features, duplicates dropped
- Model: CatBoostClassifier(iterations=500, depth=6, auto_class_weights='Balanced', random_seed=0); early stopping on a 10% slice of pooled 70% splits; best iteration 469; threshold 0.5
- Training rows: 125955 fit, 13996 early-stopping; fit time 15.2 s; total runtime 370.0 s

## Feature set

Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min

## Sample composition

| Dataset | Role | Raw rows scanned | Clean rows | Sampled | Benign | Attack | Attack types |
|---|---|---|---|---|---|---|---|
| cicids2017 | train | 2830743 | 2321493 | 100000 | 21456 | 78544 | 15 |
| cicids2018 | train | 8284254 | 5935762 | 99931 | 12474 | 87457 | 14 |
| usbids1 | test | 4813395 | 4813296 | 99630 | 6905 | 92725 | 17 |
| insdn | test | 343889 | 184896 | 99930 | 32223 | 67707 | 8 |

Per attack type in sample for cicids2017:

| Attack type | Binary label | n |
|---|---|---|
| BENIGN | 0 | 21456 |
| DDoS | 1 | 21456 |
| DoS Hulk | 1 | 21456 |
| DoS GoldenEye | 1 | 10286 |
| FTP-Patator | 1 | 5931 |
| DoS slowloris | 1 | 5385 |
| DoS Slowhttptest | 1 | 5228 |
| SSH-Patator | 1 | 3219 |
| PortScan | 1 | 1956 |
| Web Attack � Brute Force | 1 | 1470 |
| Bot | 1 | 1437 |
| Web Attack � XSS | 1 | 652 |
| Infiltration | 1 | 36 |
| Web Attack � Sql Injection | 1 | 21 |
| Heartbleed | 1 | 11 |

Per attack type in sample for cicids2018:

| Attack type | Binary label | n |
|---|---|---|
| Benign | 0 | 12474 |
| Bot | 1 | 12474 |
| DDOS attack-HOIC | 1 | 12474 |
| DoS attacks-GoldenEye | 1 | 12474 |
| DoS attacks-Hulk | 1 | 12474 |
| Infilteration | 1 | 12474 |
| SSH-Bruteforce | 1 | 12474 |
| DoS attacks-Slowloris | 1 | 9908 |
| DDOS attack-LOIC-UDP | 1 | 1730 |
| Brute Force -Web | 1 | 555 |
| Brute Force -XSS | 1 | 228 |
| SQL Injection | 1 | 84 |
| DoS attacks-SlowHTTPTest | 1 | 55 |
| FTP-BruteForce | 1 | 53 |

Per attack type in sample for usbids1:

| Attack type | Binary label | n |
|---|---|---|
| BENIGN | 0 | 6905 |
| Hulk-Evasive | 1 | 6905 |
| Hulk-NoDefense | 1 | 6905 |
| Hulk-Reqtimeout | 1 | 6905 |
| Hulk-Security2 | 1 | 6905 |
| TCPFlood-Evasive | 1 | 6905 |
| TCPFlood-NoDefense | 1 | 6905 |
| TCPFlood-Reqtimeout | 1 | 6905 |
| TCPFlood-Security2 | 1 | 6905 |
| Slowloris-Reqtimeout | 1 | 6781 |
| Slowhttptest-Evasive | 1 | 6656 |
| Slowhttptest-Security2 | 1 | 6653 |
| Slowhttptest-NoDefense | 1 | 6646 |
| Slowhttptest-Reqtimeout | 1 | 5405 |
| Slowloris-NoDefense | 1 | 1782 |
| Slowloris-Evasive | 1 | 1781 |
| Slowloris-Security2 | 1 | 1781 |

Per attack type in sample for insdn:

| Attack type | Binary label | n |
|---|---|---|
| Normal | 0 | 32223 |
| DoS | 1 | 32223 |
| Probe | 1 | 32223 |
| DDoS | 1 | 1520 |
| BFA | 1 | 1377 |
| Web-Attack | 1 | 183 |
| BOTNET | 1 | 164 |
| U2R | 1 | 17 |

## Metrics

| Kind | Eval set | n | Attack prior | ROC-AUC | PR-AUC | Accuracy | Bal. acc | F1 (attack) | Precision | Recall (DR) | FPR | MCC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in-domain | cicids2017 30% holdout | 30000 | 0.7854 | 0.9999 | 1.0000 | 0.9971 | 0.9970 | 0.9982 | 0.9991 | 0.9972 | 0.0033 | 0.9914 |
| in-domain | cicids2018 30% holdout | 29980 | 0.8752 | 0.9795 | 0.9971 | 0.9121 | 0.9247 | 0.9476 | 0.9909 | 0.9079 | 0.0585 | 0.7044 |
| cross-dataset | usbids1 | 99630 | 0.9307 | 0.4869 | 0.9427 | 0.7706 | 0.4305 | 0.8701 | 0.9200 | 0.8254 | 0.9644 | -0.0951 |
| cross-dataset | insdn | 99930 | 0.6775 | 0.3854 | 0.6121 | 0.3396 | 0.4840 | 0.1372 | 0.5979 | 0.0775 | 0.1095 | -0.0529 |

## Confusion matrices [[TN, FP], [FN, TP]]

### cicids2017 30% holdout

TN=6416, FP=21, FN=66, TP=23497

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 6416 | 21 |
| True attack (1) | 66 | 23497 |

### cicids2018 30% holdout

TN=3523, FP=219, FN=2416, TP=23822

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 3523 | 219 |
| True attack (1) | 2416 | 23822 |

### usbids1

TN=246, FP=6659, FN=16194, TP=76531

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 246 | 6659 |
| True attack (1) | 16194 | 76531 |

### insdn

TN=28695, FP=3528, FN=62461, TP=5246

|  | Pred benign (0) | Pred attack (1) |
|---|---|---|
| True benign (0) | 28695 | 3528 |
| True attack (1) | 62461 | 5246 |

## Per-attack-type breakdown

### cicids2017 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| BENIGN | 6416 | 21 | 6437 | 0.3262 |
| Bot | 25 | 406 | 431 | 94.1995 |
| DDoS | 9 | 6428 | 6437 | 99.8602 |
| DoS GoldenEye | 2 | 3084 | 3086 | 99.9352 |
| DoS Hulk | 6 | 6431 | 6437 | 99.9068 |
| DoS Slowhttptest | 0 | 1568 | 1568 | 100.0000 |
| DoS slowloris | 5 | 1610 | 1615 | 99.6904 |
| FTP-Patator | 2 | 1777 | 1779 | 99.8876 |
| Heartbleed | 1 | 2 | 3 | 66.6667 |
| Infiltration | 2 | 9 | 11 | 81.8182 |
| PortScan | 2 | 585 | 587 | 99.6593 |
| SSH-Patator | 2 | 964 | 966 | 99.7930 |
| Web Attack � Brute Force | 3 | 438 | 441 | 99.3197 |
| Web Attack � Sql Injection | 1 | 5 | 6 | 83.3333 |
| Web Attack � XSS | 6 | 190 | 196 | 96.9388 |

### cicids2018 30% holdout

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Benign | 3523 | 219 | 3742 | 5.8525 |
| Bot | 13 | 3729 | 3742 | 99.6526 |
| Brute Force -Web | 5 | 162 | 167 | 97.0060 |
| Brute Force -XSS | 0 | 69 | 69 | 100.0000 |
| DDOS attack-HOIC | 0 | 3742 | 3742 | 100.0000 |
| DDOS attack-LOIC-UDP | 0 | 519 | 519 | 100.0000 |
| DoS attacks-GoldenEye | 0 | 3742 | 3742 | 100.0000 |
| DoS attacks-Hulk | 0 | 3742 | 3742 | 100.0000 |
| DoS attacks-SlowHTTPTest | 0 | 17 | 17 | 100.0000 |
| DoS attacks-Slowloris | 0 | 2973 | 2973 | 100.0000 |
| FTP-BruteForce | 0 | 16 | 16 | 100.0000 |
| Infilteration | 2398 | 1344 | 3742 | 35.9166 |
| SQL Injection | 0 | 25 | 25 | 100.0000 |
| SSH-Bruteforce | 0 | 3742 | 3742 | 100.0000 |

### usbids1

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| BENIGN | 246 | 6659 | 6905 | 96.4374 |
| Hulk-Evasive | 11 | 6894 | 6905 | 99.8407 |
| Hulk-NoDefense | 0 | 6905 | 6905 | 100.0000 |
| Hulk-Reqtimeout | 0 | 6905 | 6905 | 100.0000 |
| Hulk-Security2 | 0 | 6905 | 6905 | 100.0000 |
| Slowhttptest-Evasive | 4649 | 2007 | 6656 | 30.1532 |
| Slowhttptest-NoDefense | 4650 | 1996 | 6646 | 30.0331 |
| Slowhttptest-Reqtimeout | 2117 | 3288 | 5405 | 60.8326 |
| Slowhttptest-Security2 | 4651 | 2002 | 6653 | 30.0917 |
| Slowloris-Evasive | 32 | 1749 | 1781 | 98.2033 |
| Slowloris-NoDefense | 32 | 1750 | 1782 | 98.2043 |
| Slowloris-Reqtimeout | 1 | 6780 | 6781 | 99.9853 |
| Slowloris-Security2 | 32 | 1749 | 1781 | 98.2033 |
| TCPFlood-Evasive | 4 | 6901 | 6905 | 99.9421 |
| TCPFlood-NoDefense | 8 | 6897 | 6905 | 99.8841 |
| TCPFlood-Reqtimeout | 1 | 6904 | 6905 | 99.9855 |
| TCPFlood-Security2 | 6 | 6899 | 6905 | 99.9131 |

### insdn

| True attack type | Pred benign | Pred attack | n | % flagged |
|---|---|---|---|---|
| Normal | 28695 | 3528 | 32223 | 10.9487 |
| BFA | 1369 | 8 | 1377 | 0.5810 |
| BOTNET | 164 | 0 | 164 | 0.0000 |
| DDoS | 1516 | 4 | 1520 | 0.2632 |
| DoS | 27821 | 4402 | 32223 | 13.6610 |
| Probe | 31396 | 827 | 32223 | 2.5665 |
| U2R | 14 | 3 | 17 | 17.6471 |
| Web-Attack | 181 | 2 | 183 | 1.0929 |

## Top-10 CatBoost feature importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Init_Win_bytes_backward | 19.8931 |
| 2 | Init_Win_bytes_forward | 12.6258 |
| 3 | Fwd Header Length | 6.8593 |
| 4 | min_seg_size_forward | 4.1803 |
| 5 | Min Packet Length | 3.6858 |
| 6 | Fwd Packet Length Max | 3.1144 |
| 7 | Bwd Packet Length Mean | 3.0747 |
| 8 | Bwd Packet Length Std | 2.9684 |
| 9 | Bwd Packet Length Min | 2.6243 |
| 10 | Fwd IAT Min | 2.4638 |

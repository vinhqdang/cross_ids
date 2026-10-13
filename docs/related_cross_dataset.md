# Related work: quantitative cross-dataset results for flow-based NIDS

Scope: papers that report train-on-A / test-on-B results for pairs among CIC-IDS2017 (CIC17), CSE-CIC-IDS2018 (CIC18), USB-IDS-1, InSDN, UNSW-NB15, ToN-IoT, IoT-23, CTU-13, and their NetFlow-v2 "NFv2-*" versions.

Verification: every entry's title, authors, venue, year and DOI were checked against Crossref (https://api.crossref.org/works/{DOI}) on 2026-10-13. For the preprints, the arXiv identifier was checked against the arXiv PDF header. Numbers marked "full text" were read from the PDF (arXiv, author-accepted manuscript or open-access MDPI). Numbers marked "abstract only" were not checked against tables. Page numbers refer to the PDF version that was read (arXiv numbering unless stated); they may differ from the journal pagination.

Metric conventions differ between papers (F1 of the attack class, MCC, balanced accuracy, macro-F1). Values are therefore not directly interchangeable.

## 1. Verified table of quantitative results

All rows are binary (benign vs attack) unless the "Notes" column says otherwise.

| # | Paper (key) | Train -> Test | Features | Model(s) | Metric | In-domain | Cross-dataset | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | layeghy2023explainable (CAEE 2023; arXiv 2205.04112) | NFv2-CIC18 -> NFv2-UNSW-NB15 | NetFlow v2 | Extra Tree / Random Forest / FF-NN / LSTM | F1 (attack class, %) | 84.62 / 95.44 / 46.27 / 90.17 | 0.57 / 0.84 / 0.05 / 9.63 | Tab. 2, Tab. 4, Fig. 1 (full text) |
| 2 | layeghy2023explainable | NFv2-UNSW-NB15 -> NFv2-CIC18 | NetFlow v2 | ET / RF / FF / LSTM | F1 (%) | 91.73 / 92.17 / 90.63 / 92.82 (UNSW in-domain) | 17.47 / 7.70 / 34.89 / 14.20 | Tab. 2, Tab. 4 (full text) |
| 3 | layeghy2023explainable | NFv2-CIC18 -> NFv2-ToN-IoT | NetFlow v2 | ET / RF / FF / LSTM | F1 (%) | 84.62 / 95.44 / 46.27 / 90.17 | 62.25 / 30.28 / 69.01 / 48.03 | Tab. 4 (full text) |
| 4 | layeghy2023explainable | NFv2-ToN-IoT -> NFv2-CIC18 | NetFlow v2 | ET / RF / FF / LSTM | F1 (%) | 77.63 / 77.33 / 93.98 / 76.38 (ToN in-domain) | 44.83 / 4.50 / 57.55 / 21.82 | Tab. 4 (full text) |
| 5 | layeghy2023explainable | NFv2-UNSW-NB15 -> NFv2-ToN-IoT | NetFlow v2 | ET / RF / FF / LSTM | F1 (%) | 91.73 / 92.17 / 90.63 / 92.82 | 73.39 / 0.00 / 3.82 / 81.40 | Tab. 4 (full text) |
| 6 | layeghy2023explainable | NFv2-ToN-IoT -> NFv2-UNSW-NB15 | NetFlow v2 | ET / RF / FF / LSTM | F1 (%) | 77.63 / 77.33 / 93.98 / 76.38 | 0.00 / 0.25 / 0.00 / 0.40 | Tab. 4 (full text) |
| 7 | layeghy2023explainable (balanced "-b" sets, unsupervised) | CIC18-b <-> UNSW-b, CIC18-b <-> ToN-b, UNSW-b <-> ToN-b | NetFlow v2 | IsolationForest / oSVM / SGD-oSVM | F1 (%) | Tab. 3 | e.g. CIC18-b -> ToN-b 56.24 / 58.12 (oSVM/SGD-oSVM); UNSW-b -> ToN-b 64.85 (oSVM); ToN-b -> UNSW-b 7.10 (oSVM); average decay 28.33% | Tab. 6, Tab. 7 (full text) |
| 8 | layeghy2023dinids (KBS 2023; arXiv 2210.08252) | NFv2-CIC18 -> NFv2-UNSW-NB15 | NetFlow v2 | RF / ET / LSTM / FF / OSVM / DANN / DI-NIDS | F1 (%) | 95.44 / 84.62 / 90.17 / 97.72 / 92.97 / 97.81 / 93.23 | 0.84 / 0.57 / 9.63 / 3.09 / 86.15 / 17.31 / 85.79 | Tab. 4, Tab. 5 (full text) |
| 9 | layeghy2023dinids | NFv2-UNSW-NB15 -> NFv2-CIC18 | NetFlow v2 | same | F1 (%) | 92.17 / 91.73 / 92.82 / 92.24 / 98.28 / 93.38 / 98.68 | 7.70 / 17.47 / 14.20 / 30.79 / 15.74 / 61.94 / 93.29 | Tab. 4, Tab. 6 (full text) |
| 10 | cantone2024crossdataset (IEEE Access 2024; arXiv 2402.10974) | CIC17 -> CIC18 | CICFlowMeter-style (84 features before exclusions; MinMax) | LDA / DT / RF / XGB | MCC (%) | 64.86 / 99.72 / 99.74 / 99.74 (CIC17 in-domain) | 32.62 / 30.82 / 25.96 / 35.68 | Tab. 3 (full text, arXiv version) |
| 11 | cantone2024crossdataset | CIC17 -> CIC18 | same | LDA / DT / RF / XGB | F1 (%) | 69.02 / 99.78 / 99.79 / 99.79 | 39.12 / 31.20 / 25.76 / 26.43 | Tab. 3 (arXiv version) |
| 12 | cantone2024crossdataset | CIC18 -> CIC17 | same | LDA / DT / RF / XGB | MCC (%) | 76.52 / 96.33 / 96.45 / 96.48 (CIC18 in-domain) | 39.29 / 35.50 / 44.44 / 41.75 | Tab. 3 (arXiv version) |
| 13 | cantone2024crossdataset | CIC18 -> CIC17 | same | LDA / DT / RF / XGB | F1 (%) | 79.82 / 96.89 / 96.99 / 97.02 | 48.41 / 33.86 / 51.21 / 44.58 | Tab. 3 (arXiv version) |
| 14 | dhooge2023damaged (J. Cybersecur. Priv. 2023) | CIC17 -> CIC18, all attack classes pooled ("Global") | CICFlowMeter | xgboost (baseline) vs knn (best generalised) | Balanced acc. / F1 (%) | xgboost 99.82 / 99.73 (best baseline row) | knn 81.36 / 71.98 (best of top-3 generalised rows) | Tab. 3, p. 140-141 (full text) |
| 15 | dhooge2023damaged | CIC17 -> CIC18, L7-DoS | CICFlowMeter | xgboost vs linsvc | Balanced acc. / F1 (%) | 99.85 / 99.79 | 97.80 / 82.59 | Tab. 3 (full text) |
| 16 | dhooge2023damaged | CIC17 -> CIC18, Botnet | CICFlowMeter | xgboost vs ada | Balanced acc. / F1 (%) | 98.42 / 98.19 | 98.90 / 98.40 (best); other top-3 rows 92.11 / 82.67 | Tab. 3 (full text) |
| 17 | dhooge2023damaged | CIC17 -> CIC18, Web attacks / Infiltration | CICFlowMeter | various | Balanced acc. / F1 (%) | Web 97.58 / 96.92; Infiltration 88.89 / 71.79 | Web 64.23 / 43.37 (best BA row); Infiltration 50.19 / 17.50 | Tab. 3 (full text) |
| 18 | dhooge2023damaged | CIC17 -> CIC18, DDoS (two matched subsets) | CICFlowMeter | extratree vs dtree / binlr | Balanced acc. / F1 (%) | 99.89 / 99.89 | 96.30 / 96.75 and 99.86 / 99.87 | Tab. 3 (full text) |
| 19 | catillo2022transferability (Softw. Qual. J. 2022) | CIC17 (DoS subset) -> USB-IDS-1 (hulk, TCP flood, slowloris, slowhttptest; no-defense / evasive / reqtimeout) | CICFlowMeter (84 features, same extractor) | DT / RF / DNN | Precision, recall, F1 | CIC17 test: DT P/R/F1 = 0.99/0.98/0.99; RF 1/0.99/0.99 | DT: hulk P/R/F1 1/0.97/0.99 (no defense), recall 0.06 (evasive); TCP flood not detected; slowloris recall 0.79 (no defense, evasive), 0.24 (reqtimeout); slowhttptest not detected. RF: hulk recall 0.99 (no defense, reqtimeout), 0 (evasive); TCP flood 0; slowloris recall 0.80 / 0.13; slowhttptest recall 0.7 / 0.3 | Sec. 5.1-5.3, Fig. 5-7 (author-accepted manuscript; values quoted from text, bar charts not read) |
| 20 | hossain2026assessing (arXiv preprint 2605.04407, May 2026) | UNSW-NB15 -> ToN-IoT | 7 common features (duration, proto, service, src_bytes, dst_bytes, src_pkts, dst_pkts) | RF / LR / NB | Accuracy / F1 | RF: UNSW 0.9508 / 0.9614; ToN 0.9979 / 0.9987 | RF 0.3807 / 0.4898; LR 0.7779 / 0.8751; NB 0.2585 / 0.1020 | Tab. 1, 2, 4 (full text) |
| 21 | hossain2026assessing | ToN-IoT -> UNSW-NB15 | same 7 features | RF / LR / NB | Accuracy / F1 | as above | RF 0.3954 / 0.5268; LR 0.3824 / 0.4902; NB 0.3658 / 0.0573 | Tab. 5 (full text) |
| 22 | kilic2026shap (IEEE Access 2026) | CIC17 -> CIC18 | 80 CICFlowMeter features; SHAP-selected subsets | RF / XGB | macro-F1 (multi-class, not binary) | about 0.90 on CIC17 | RF 0.127, XGB 0.119; SHAP reduction RF 0.166; plus port removal RF 0.207 | Abstract only (repository record) |
| 23 | apruzzese2022cross (TNSM 2022; arXiv 2203.04686) | Per-attack detectors trained on one dataset, tested on attack families (Botnet / DoS / Other) from the other networks; datasets CTU-13, NB15, IDS18, DDOS19 (heterogeneous) and UF-NB15, UF-IDS18, UF-ToNIoT, UF-BotIoT (uniform NetFlow) | Flow features without IPs/ports (Tab. IV) | RF ensemble | F1 per attack family (%) | Tab. V: CTU13 botnet 98.1; NB15 88.6/98.5/98.1; IDS18 90.1/99.9/96.1 | Tab. VI (C4, unknown attacks from other networks): CTU13 botnet 80.0, DoS 38.1, other 49.7; NB15 65.8/40.7/75.2; IDS18 54.9/49.4/76.1; UF-NB15 72.2/52.3/64.1; UF-IDS18 68.2/81.0/63.3; UF-ToNIoT 82.1/89.3/85.1 | Tab. V, VI (full text). Not a dataset-to-dataset matrix: scores are averaged over all other datasets' attacks; benign comes from the origin dataset |

Out-of-scope datasets (reported for completeness, not in our benchmark):

| Paper | Train -> Test | Result | Source |
|---|---|---|---|
| al-riyami2021cross (LNNS 2021) | NSL-KDD <-> gureKDD | Before fixing a feature-definition mismatch: F1 0.2969 (NSL-KDD -> gureKDD, RF) and 0.0362 (reverse). After the fix: best F1 0.9955 / 0.9511 | Tab. 1, 2 (author PDF, Univ. Liverpool) |
| layeghy2023explainable | includes NFv2-BoT-IoT as a fourth domain, e.g. BoT-IoT -> CIC18 F1 22.32 (ET, RF) | Fig. 1, Tab. 4 | |
| hakim2026crossdomain (arXiv 2607.00553) | Edge-IIoTset -> Gotham 2025 / WUSTL-IIoT-2021 | Not read in detail; datasets do not overlap with ours | |

## 2. Per-paper notes

### Sarhan, Layeghy, Portmann: NetFlow standard feature set
- `sarhan2022standard` (Mobile Netw. Appl. 27(1):357-370, DOI 10.1007/s11036-021-01843-0; arXiv 2101.11315). Proposes NetFlow-based feature sets (NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2 and NF-UQ-NIDS-v2) and evaluates them with 70/30 splits and 5-fold cross-validation inside each dataset. Full text was searched for train/test across datasets: none is reported. Its role is to make cross-dataset evaluation possible. No cross-dataset numbers to cite.
- `sarhan2022evaluating` (Big Data Res. 30:100359, DOI 10.1016/j.bdr.2022.100359; arXiv 2104.07183). Compares NetFlow and CICFlowMeter feature sets on CIC18, BoT-IoT and ToN-IoT (70/30 split, 5-fold) and uses SHAP. Evaluation is per dataset (in-domain). Full text searched: no train-on-A/test-on-B results. The generalisation claim is "same feature set performs across several datasets", not transfer. Do not cite as a cross-dataset result.
- `sarhan2021netflow` (LNICST, DOI 10.1007/978-3-030-72802-1_9) introduces NF-UNSW-NB15, NF-BoT-IoT, NF-ToN-IoT, NF-CSE-CIC-IDS2018 (NFv1); dataset paper, abstract not checked for cross-dataset results. Cited here as the source of the NF-* datasets only.
- The cross-dataset numbers for the NFv2 datasets come from Layeghy and Portmann (below).

### Layeghy and Portmann, cross-domain evaluation (CAEE 2023)
- Published title: "Explainable Cross-domain Evaluation of ML-based Network Intrusion Detection Systems" (Comput. Electr. Eng. 108:108692). The arXiv version 2205.04112 is titled "On Generalisability of Machine Learning-based Network Intrusion Detection Systems". Numbers above were read from the arXiv PDF; check them against the journal version before quoting.
- Datasets: NFv2-BoT-IoT, NFv2-CIC-2018, NFv2-ToN-IoT, NFv2-UNSW-NB15, each stratified-sampled to 1,000,000 flows. Binary Benign vs Attack. Default hyperparameters, no tuning (to avoid overfitting to the source). Supervised runs use the natural class ratio; the unsupervised runs use down-sampled balanced "-b" versions.
- Main findings: no model generalises over all pairs; average decay 56.28% (supervised) and 28.33% (unsupervised); generalisation is asymmetric (e.g. ToN -> CIC18 ET 44.83 vs CIC18 -> ToN ET 62.25, but UNSW -> ToN ET 73.39 vs ToN -> UNSW 0.00); the cause is attributed with SHAP to features that correlate with the label in only one dataset.
- Pairs overlapping with our benchmark: CIC18 <-> UNSW-NB15 <-> ToN-IoT (NFv2 features).

### Layeghy, Baktashmotlagh, Portmann: DI-NIDS (KBS 2023)
- DOI 10.1016/j.knosys.2023.110626, vol. 273, art. 110626. Numbers from arXiv 2210.08252 v1 (Oct 2022); the journal version may differ.
- NFv2-CIC18 <-> NFv2-UNSW-NB15, F1. Baselines are copied from the CAEE paper above (RF, ET, LSTM) plus a new FF-NN. DI-NIDS (domain-adversarial feature extractor + one-class SVM) keeps F1 at 85.79 (CIC18 -> UNSW) and 93.29 (UNSW -> CIC18), degradation 7.44 and 5.39 points. The paper states that DI-NIDS is trained on labelled source data and tests on unlabelled target.
- Also cites Pontes et al. [14] (CIC-2017 and CIC-2019 cross-domain; degradation 11.10 and 27.50 points); that work is not in our dataset set and was not read directly.

### D'hooge et al.
- `dhooge2020inter` (J. Inf. Secur. Appl. 54:102564, DOI 10.1016/j.jisa.2020.102564): twelve supervised algorithms, CIC17 and CIC18, DoS/SSL and botnet classes used for inter-dataset testing. Publisher full text was not accessible; only the abstract was read ("Exposure of these models to unseen, but related samples ... assumption is shown to be erroneous"). Tables not verified. See the follow-up below for numbers.
- `dhooge2023damaged` (J. Cybersecur. Priv. 3(2):118-144, DOI 10.3390/jcp3020008, open access): pretrained on CIC17, classified CIC18 without retraining, per attack class, with flat performance losses of 0 to 50% reported in the abstract. Table 3 lists the best three models per class for the baseline (B, in-dataset) and generalised (G, cross-dataset) setting. Caveat: G rows are selected as the best-generalising of many trained models (selection on the target data), so they are optimistic upper bounds; the paper itself says Table 3 "obfuscates" the unstable behaviour across models. Column order in the table is Balanced Acc., F1, Precision, Recall.

### Engelen, Rimmer, Joosen (IEEE SPW 2021)
- DOI 10.1109/SPW53761.2021.00009, pp. 7-12. Audits CIC17 (flow construction, labelling, over 20% of flows relabelled or reconstructed, per Cantone et al.'s description) and re-evaluates classifiers on the corrected data. Publisher full text could not be downloaded (connection reset); only the abstract was read. No cross-dataset train/test results are known to be reported. Cite for CIC17 label noise, not for transfer numbers.

### Apruzzese, Pajola, Conti (TNSM 2022)
- DOI 10.1109/TNSM.2022.3157344, 19(4):5152-5169; arXiv 2203.04686. Proposes XeNIDS, a cross-evaluation model with ten contexts. Experiments in two scenarios: "heterogeneous" (CTU13, NB15, IDS18, DDOS19) and "uniform" (UF-NB15, UF-IDS18, UF-ToNIoT, UF-BotIoT, i.e. NetFlow-converted versions of the Sarhan et al. datasets). Detectors are ensembles of per-attack RFs. Results are F1 per attack family, not per dataset pair. In Tab. VI the same detectors drop from 83-99 (own attacks, Tab. V) to 38-99 (attacks of other networks). Tab. VIII ("surrogation", benign from own network, malicious from others) reaches 99.6-99.9 F1, which the authors flag as needing investigation. Comparison made in the paper with an earlier study (their ref. [19], not identified here): DDOS19 -> IDS18 about 64 F1 and IDS18 -> DDOS19 about 78 F1 in that study, against their own similar values.
- The related `apruzzese2023sok` (EuroS&P 2023, DOI 10.1109/EuroSP57164.2023.00042) was found on Crossref but not read; it is a SoK on assessment practices and is not claimed to contain pair-wise numbers.

### Arp et al. (USENIX Security 2022)
- "Dos and Don'ts of Machine Learning in Computer Security", pp. 3971-3988 (pages from the USENIX listing; arXiv 2010.09470 states "To appear at USENIX Security Symposium 2022"). Crossref has no record for this proceedings item; verification is by arXiv header and USENIX listing. The NIDS case study (Sec. 4.4) uses Kitsune on the Mirai capture and compares to a simple boxplot baseline (AUC 0.968 vs 0.998; TPR at FPR 0.001: 0.882 vs 0.996). It contains no cross-dataset results. Use only as a methodological reference (pitfalls P1 sampling bias, P4 spurious correlations, P6 inappropriate baselines, P9 lab-only evaluation).

### Al-Riyami, Lisitsa, Coenen
- LNNS 2021, DOI 10.1007/978-981-16-2102-4_73, pp. 815-828 (full text from the Liverpool author PDF). Datasets are NSL-KDD and gureKDD, which are outside our benchmark. Result worth citing for methodology: the initial cross-dataset collapse (F1 below 0.4) was caused by a feature-definition mismatch; after correction, F1 rose above 0.9. This is a caution for any cross-dataset study that merges feature schemas. The earlier version, "A Re-evaluation of Intrusion Detection Accuracy" (ACM CCS 2018, DOI 10.1145/3243734.3278490), appears on Crossref but was not read.

### Catillo et al.
- `catillo2022transferability` (Softw. Qual. J. 30(4):955-981, DOI 10.1007/s11219-022-09587-0): trains DT, RF and DNN on CIC17 (DoS attacks plus benign; 691,406 flows, 70/15/15 split) and tests on USB-IDS-1, which was collected on a simpler network with httperf as workload generator, with four DoS attacks under three defence settings (no defense, evasive, reqtimeout). Numbers above are those written in the running text of the author-accepted manuscript; Fig. 5-7 values (bar charts) were not extracted. The DNN is described as agreeing with the DT and RF findings.
- `catillo2021usbids1` (DSN-W 2021, DOI 10.1109/DSN-W52860.2021.00012) is the dataset paper. `catillo2025usbidstc` (ICISSP 2025, DOI 10.5220/0013248600003899) is a follow-up dataset paper on DoS attacks in different network scenarios. Neither was read for cross-dataset numbers.
- Related: Dang, "Multi-layer Intrusion Detection on the USB-IDS-1 Dataset", LNNS 2023, DOI 10.1007/978-3-031-27409-1_102 (Crossref verified; content not read).

### Cantone, Marrocco, Bria
- IEEE Access 12:144489-144508 (2024), DOI 10.1109/ACCESS.2024.3472907. Crossref title: "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study". The arXiv version 2402.10974 (Feb 2024) is titled "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection". The numbers above are from the arXiv version; confirm against the IEEE Access version before quoting.
- Datasets: CIC17, CIC18, LycoS-IDS2017, LycoS-Unicas-IDS2018 (the last two are re-extracted versions of the CIC raw traffic with corrected features). Training: whole dataset A, test on whole dataset B; within-dataset 80/20; MinMax scaling; grid search on 20% of training data. Average MCC: within 94.63%, cross 29.35%; CIC18 -> CIC17 averages 40.24% (best), CIC17 -> CIC18 averages 31.27%; LycoS17 -> LycoS18 is worst at 10.83%. The paper states that the "corrected" LycoS datasets show no improved transfer. Table 3 does not state explicitly in the extracted text that the labels are binary; the structure (4 within + 4 cross experiments, plus separate single-attack experiments) and the benign shares quoted in Sec. 3.1 indicate benign vs all attacks. Verify before relying on it.

### Dang, "Learning to Transfer Knowledge Between Datasets to Enhance Intrusion Detection Systems"
- Lecture Notes in Electrical Engineering, Computational Intelligence (Springer Singapore, 2023), pp. 39-46, DOI 10.1007/978-981-19-7346-8_4. Crossref and the Springer page agree on title, author and DOI; the Springer page lists the author as Industrial University of Ho Chi Minh City (affiliation as printed in 2023).
- Accessible content: abstract and reference list only (paywalled, the preview showed no body text). The abstract concerns software-defined networks: models perform well in single-dataset cross-validation but "their performance reduces significantly in the presence of concept drift, i.e., if the testing dataset is collected from a different source"; concept-drift detection techniques are applied and the authors "claim that our presented models can improve the performance in the cross-dataset scenario". No numbers appear in the abstract. The reference list includes the InSDN paper (Elsayed et al., IEEE Access 2020), CIC-IDS2017 (Sharafaldin et al., ICISSP 2018), Isolation Forest, and the author's own "Detecting intrusion using multiple datasets in software-defined networks" (ICRCICN 2022). It is plausible but NOT confirmed that the experiments use InSDN and CIC-IDS2017; the exact train/test pairs, models and metrics need the full text.

### Other 2023-2026 items
- `hossain2026assessing`: arXiv preprint (not peer reviewed, May 2026); UNSW-NB15 <-> ToN-IoT with only seven shared features; numbers in table row 20-21.
- `kilic2026shap` (IEEE Access 14:90926-90936, DOI 10.1109/ACCESS.2026.3703481): CIC17 -> CIC18, multi-class macro-F1; abstract only (repository record).
- `salman2026domain` (SSRN preprint, DOI 10.2139/ssrn.7154703): CIC17 vs a self-generated testbed; asymmetric transfer; no numbers read. Preprint, not peer reviewed; not an overlap with our datasets beyond CIC17.
- Kim, Applied Sciences 16:6957 (2026), DOI 10.3390/app16146957: few-shot domain adaptation, "two evaluated transfer directions"; abstract does not name the datasets; not read further.
- `hakim2026crossdomain` (arXiv 2607.00553): IIoT datasets only.

## 3. Unverified / needs access

| Item | What is missing |
|---|---|
| Dang (LNEE 2023), DOI 10.1007/978-981-19-7346-8_4 | Citation is verified; body text paywalled. Need datasets, train/test pairs, models and numbers. |
| D'hooge et al. 2020 (JISA 54:102564) | Citation verified; only the abstract was read. Tables (per-algorithm F1 for CIC17 -> CIC18, DoS/SSL and botnet) not seen. |
| Engelen et al. 2021 (SPW) | Citation verified; abstract only. Confirm no cross-dataset experiments. |
| Layeghy and Portmann, CAEE version | Numbers taken from arXiv 2205.04112; confirm against the journal version (Elsevier page returned 403). |
| DI-NIDS, KBS version | Numbers taken from arXiv v1. |
| Cantone et al., IEEE Access version | Numbers taken from arXiv 2402.10974; Table 3 binary labelling inferred. |
| Catillo et al., Fig. 5-7 values | Only values written in the text were used. |
| Kilic and Sengul 2026 | Abstract only. |
| Pontes et al. (CIC-2017 / CIC-2019 cross-domain, cited by DI-NIDS) | Not located or verified. |
| "Al-Riyami et al." CCS 2018 poster | Crossref record found (DOI 10.1145/3243734.3278490), not read. |
| Chuang and Ye, Sustainability 15(12):9395 (2023), DOI 10.3390/su15129395 | SDN transfer learning; abstract does not name datasets or give cross-dataset train/test pairs; not confirmed as InSDN-based. |
| Any InSDN or IoT-23 cross-dataset result | No verified paper found that reports train-on-InSDN / test-on-another-dataset or IoT-23 pairs, other than the unconfirmed Dang abstract. |

## 4. BibTeX (verified entries)

```bibtex
@article{sarhan2022standard,
  author  = {Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
  title   = {Towards a Standard Feature Set for Network Intrusion Detection System Datasets},
  journal = {Mobile Networks and Applications},
  volume  = {27},
  number  = {1},
  pages   = {357--370},
  year    = {2022},
  doi     = {10.1007/s11036-021-01843-0}
}

@article{sarhan2022evaluating,
  author  = {Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
  title   = {Evaluating Standard Feature Sets Towards Increased Generalisability and Explainability of {ML}-Based Network Intrusion Detection},
  journal = {Big Data Research},
  volume  = {30},
  pages   = {100359},
  year    = {2022},
  doi     = {10.1016/j.bdr.2022.100359}
}

@inproceedings{sarhan2021netflow,
  author    = {Sarhan, Mohanad and Layeghy, Siamak and Moustafa, Nour and Portmann, Marius},
  title     = {{NetFlow} Datasets for Machine Learning-Based Network Intrusion Detection Systems},
  booktitle = {Lecture Notes of the Institute for Computer Sciences, Social Informatics and Telecommunications Engineering},
  pages     = {117--135},
  year      = {2021},
  doi       = {10.1007/978-3-030-72802-1_9}
}

@article{layeghy2023explainable,
  author  = {Layeghy, Siamak and Portmann, Marius},
  title   = {Explainable Cross-domain Evaluation of {ML}-based Network Intrusion Detection Systems},
  journal = {Computers and Electrical Engineering},
  volume  = {108},
  pages   = {108692},
  year    = {2023},
  doi     = {10.1016/j.compeleceng.2023.108692},
  note    = {arXiv:2205.04112, titled ``On Generalisability of Machine Learning-based Network Intrusion Detection Systems''}
}

@article{layeghy2023dinids,
  author  = {Layeghy, Siamak and Baktashmotlagh, Mahsa and Portmann, Marius},
  title   = {{DI-NIDS}: Domain Invariant Network Intrusion Detection System},
  journal = {Knowledge-Based Systems},
  volume  = {273},
  pages   = {110626},
  year    = {2023},
  doi     = {10.1016/j.knosys.2023.110626},
  note    = {arXiv:2210.08252}
}

@article{layeghy2024benchmarking,
  author  = {Layeghy, Siamak and Gallagher, Marcus and Portmann, Marius},
  title   = {Benchmarking the Benchmark -- Comparing Synthetic and Real-World Network {IDS} Datasets},
  journal = {Journal of Information Security and Applications},
  volume  = {80},
  pages   = {103689},
  year    = {2024},
  doi     = {10.1016/j.jisa.2023.103689}
}

@article{dhooge2020inter,
  author  = {D'hooge, Laurens and Wauters, Tim and Volckaert, Bruno and De Turck, Filip},
  title   = {Inter-dataset Generalization Strength of Supervised Machine Learning Methods for Intrusion Detection},
  journal = {Journal of Information Security and Applications},
  volume  = {54},
  pages   = {102564},
  year    = {2020},
  doi     = {10.1016/j.jisa.2020.102564}
}

@article{dhooge2023damaged,
  author  = {D'hooge, Laurens and Verkerken, Miel and Wauters, Tim and De Turck, Filip and Volckaert, Bruno},
  title   = {Characterizing the Impact of Data-Damaged Models on Generalization Strength in Intrusion Detection},
  journal = {Journal of Cybersecurity and Privacy},
  volume  = {3},
  number  = {2},
  pages   = {118--144},
  year    = {2023},
  doi     = {10.3390/jcp3020008}
}

@inproceedings{engelen2021troubleshooting,
  author    = {Engelen, Gints and Rimmer, Vera and Joosen, Wouter},
  title     = {Troubleshooting an Intrusion Detection Dataset: the {CICIDS2017} Case Study},
  booktitle = {2021 IEEE Security and Privacy Workshops (SPW)},
  pages     = {7--12},
  year      = {2021},
  doi       = {10.1109/SPW53761.2021.00009}
}

@article{apruzzese2022cross,
  author  = {Apruzzese, Giovanni and Pajola, Luca and Conti, Mauro},
  title   = {The Cross-Evaluation of Machine Learning-Based Network Intrusion Detection Systems},
  journal = {IEEE Transactions on Network and Service Management},
  volume  = {19},
  number  = {4},
  pages   = {5152--5169},
  year    = {2022},
  doi     = {10.1109/TNSM.2022.3157344},
  note    = {arXiv:2203.04686}
}

@inproceedings{apruzzese2023sok,
  author    = {Apruzzese, Giovanni and Laskov, Pavel and Schneider, Johannes},
  title     = {{SoK}: Pragmatic Assessment of Machine Learning for Network Intrusion Detection},
  booktitle = {2023 IEEE 8th European Symposium on Security and Privacy (EuroS\&P)},
  pages     = {592--614},
  year      = {2023},
  doi       = {10.1109/EuroSP57164.2023.00042}
}

@inproceedings{arp2022dos,
  author    = {Arp, Daniel and Quiring, Erwin and Pendlebury, Feargus and Warnecke, Alexander and Pierazzi, Fabio and Wressnegger, Christian and Cavallaro, Lorenzo and Rieck, Konrad},
  title     = {Dos and Don'ts of Machine Learning in Computer Security},
  booktitle = {31st USENIX Security Symposium (USENIX Security 22)},
  pages     = {3971--3988},
  year      = {2022},
  note      = {arXiv:2010.09470}
}

@inbook{alriyami2021cross,
  author    = {Al-Riyami, Said and Lisitsa, Alexei and Coenen, Frans},
  title     = {Cross-Datasets Evaluation of Machine Learning Models for Intrusion Detection Systems},
  booktitle = {Lecture Notes in Networks and Systems},
  pages     = {815--828},
  year      = {2021},
  doi       = {10.1007/978-981-16-2102-4_73}
}

@article{catillo2022transferability,
  author  = {Catillo, Marta and Del Vecchio, Andrea and Pecchia, Antonio and Villano, Umberto},
  title   = {Transferability of Machine Learning Models Learned from Public Intrusion Detection Datasets: the {CICIDS2017} Case Study},
  journal = {Software Quality Journal},
  volume  = {30},
  number  = {4},
  pages   = {955--981},
  year    = {2022},
  doi     = {10.1007/s11219-022-09587-0}
}

@inproceedings{catillo2021usbids1,
  author    = {Catillo, Marta and Del Vecchio, Andrea and Ocone, Luciano and Pecchia, Antonio and Villano, Umberto},
  title     = {{USB-IDS-1}: a Public Multilayer Dataset of Labeled Network Flows for {IDS} Evaluation},
  booktitle = {2021 51st Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)},
  pages     = {1--6},
  year      = {2021},
  doi       = {10.1109/DSN-W52860.2021.00012}
}

@inproceedings{catillo2025usbidstc,
  author    = {Catillo, Marta and Pecchia, Antonio and Villano, Umberto},
  title     = {{USB-IDS-TC}: A Flow-Based Intrusion Detection Dataset of {DoS} Attacks in Different Network Scenarios},
  booktitle = {Proceedings of the 11th International Conference on Information Systems Security and Privacy (ICISSP)},
  pages     = {302--309},
  year      = {2025},
  doi       = {10.5220/0013248600003899}
}

@article{cantone2024crossdataset,
  author  = {Cantone, Marco and Marrocco, Claudio and Bria, Alessandro},
  title   = {Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {144489--144508},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3472907},
  note    = {arXiv:2402.10974, titled ``On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection''}
}

@incollection{dang2023learning,
  author    = {Dang, Quang-Vinh},
  title     = {Learning to Transfer Knowledge Between Datasets to Enhance Intrusion Detection Systems},
  booktitle = {Computational Intelligence},
  series    = {Lecture Notes in Electrical Engineering},
  publisher = {Springer Nature Singapore},
  pages     = {39--46},
  year      = {2023},
  doi       = {10.1007/978-981-19-7346-8_4}
}

@article{kilic2026shap,
  author  = {K{\i}l{\i}{\c{c}}, Can and {\c{S}}eng{\"u}l, G{\"o}khan},
  title   = {{SHAP}-Guided Feature Selection for Cross-Dataset Generalization in Network Intrusion Detection Systems},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {90926--90936},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3703481}
}

@misc{hossain2026assessing,
  author       = {Hossain, Md Zakir and Khan, Md Ayshik Rahman and Islam, Md Rafiqul and Islam, Syed Mohammed Shamsul and Gedeon, Tom},
  title        = {Assessing Generalisation Capability of Machine Learning Models for Intrusion Detection},
  year         = {2026},
  eprint       = {2605.04407},
  archiveprefix = {arXiv},
  note         = {Preprint, not peer reviewed}
}
```

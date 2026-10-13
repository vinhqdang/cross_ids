# Prior work of the author relevant to this project

Works by Quang-Vinh Dang (ORCID 0000-0002-3877-8024) that this project builds
on or should be compared with. DOIs in the first two tables were checked
against Crossref; the remaining ones are taken from the ORCID record.

## Directly related: transfer between IDS datasets

| Year | Title | Venue | DOI | Relation to this project |
|---|---|---|---|---|
| 2023 | Learning to Transfer Knowledge Between Datasets to Enhance Intrusion Detection Systems | Lecture Notes in Electrical Engineering | 10.1007/978-981-19-7346-8_4 | Earlier approach: an Isolation-Forest-based distance from the new dataset to the training data, then CatBoost fine-tuned on the new dataset. That step needs labels from the target network. This project removes that need (label-free adaptation) and keeps the earlier approach as the few-shot reference setting (DESIGN.md §4.5). |
| 2022 | Detecting Intrusion Using Multiple Datasets in Software-Defined Networks | Communications in Computer and Information Science | 10.1007/978-981-19-8069-5_55 | Multi-dataset training in SDN. Motivates the LODO protocol and the InSDN target. |
| 2023 | Multi-layer Intrusion Detection on the USB-IDS-1 Dataset | Lecture Notes in Networks and Systems | 10.1007/978-3-031-27409-1_102 | USB-IDS-1 experience. Here USB-IDS-1 is a CICFlowMeter target from a different lab. |

## Related: uncertainty, calibration and drift

| Year | Title | Venue | DOI | Possible use |
|---|---|---|---|---|
| 2026 | Utilizing Uncertainty Measures to Improve the Performance of Intrusion Detection Systems | SN Computer Science | 10.1007/s42979-026-04923-8 | Uncertainty-gated pseudo-labelling on the target (candidate RIFT extension). |
| 2024 | Kernel Methods for Conformal Prediction to Detect Botnets | Lecture Notes in Networks and Systems | 10.1007/978-981-99-8476-3_3 | Conformal / calibrated decision rules under shift. |
| 2026 | ST-FedXIDS: A Spatiotemporal Federated Explainable IDS With Drift-Adaptive Graph Learning for High-Density Public WiFi Environments | Int. J. Advances in Signal and Image Sciences | 10.29284/zcrk2p65 | Drift adaptation; temporal shift is the within-network counterpart of cross-dataset shift. |

## Other IDS work (background citations)

- Fuzzy Embedding to Detect Intrusion in Software-Defined Networks (2024), 10.1007/978-3-031-67195-1_78
- Utilizing LLM Embedding Technique for Intrusion Detection (2025), 10.1007/978-981-97-7717-4_50
- Using Transformer Technique for Intrusion Detection (2023), 10.1007/978-981-99-8296-7_13
- Intrusion Detection in Internet of Medical Things (2024), 10.1007/978-981-96-0434-0_28
- Detecting Intrusion in WiFi Network Using Graph Neural Networks (2023), 10.1007/978-981-19-7753-4_48
- Parallel Optimization Technique to Improve the Performance of Lightweight Intrusion Detection Systems (2024), 10.1007/978-981-99-6586-1_26
- Detecting IoT Malware Using Federated Learning (2024), 10.1007/978-981-99-7862-5_6
- Detecting the Attacks to DNS (2022), 10.1007/978-3-030-85799-8_15

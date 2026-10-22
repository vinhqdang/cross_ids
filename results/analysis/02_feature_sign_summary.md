# Feature sign consistency

ROC-AUC is computed on the cached 100k-row samples; `sign` is `+` when every available dataset has AUC > 0.5, `-` when every one has AUC < 0.5, otherwise `mixed`. CTU-13 only supports the minimal tier, so core-only features are absent there.

| feature | datasets_available | datasets_total | n_auc_above_0.5 | n_auc_below_0.5 | min_auc | max_auc | sign | consistent_sign_across_datasets |
|---|---|---|---|---|---|---|---|---|
| duration | 8 | 8 | 5 | 3 | 0.279 | 0.793 | mixed | False |
| fwd_pkts | 7 | 8 | 2 | 5 | 0.057 | 0.808 | mixed | False |
| bwd_pkts | 7 | 8 | 3 | 4 | 0.093 | 0.868 | mixed | False |
| fwd_bytes | 8 | 8 | 2 | 6 | 0.269 | 0.749 | mixed | False |
| bwd_bytes | 8 | 8 | 2 | 6 | 0.028 | 0.750 | mixed | False |
| total_pkts | 8 | 8 | 2 | 6 | 0.065 | 0.804 | mixed | False |
| total_bytes | 8 | 8 | 2 | 6 | 0.025 | 0.760 | mixed | False |
| bytes_per_s | 8 | 8 | 2 | 6 | 0.231 | 0.601 | mixed | False |
| pkts_per_s | 8 | 8 | 3 | 5 | 0.183 | 0.681 | mixed | False |
| fwd_bytes_per_pkt | 7 | 8 | 3 | 4 | 0.291 | 0.955 | mixed | False |
| bwd_bytes_per_pkt | 7 | 8 | 2 | 5 | 0.068 | 0.778 | mixed | False |
| fwd_bwd_pkt_ratio | 7 | 8 | 4 | 3 | 0.333 | 0.791 | mixed | False |
| fwd_bwd_byte_ratio | 8 | 8 | 3 | 5 | 0.288 | 0.979 | mixed | False |
| bytes_per_pkt | 8 | 8 | 2 | 6 | 0.063 | 0.789 | mixed | False |

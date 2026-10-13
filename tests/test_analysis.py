"""Analysis module: taxonomy, figure-free helpers and an end-to-end run on small synthetic results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crossids import analysis
from crossids.analysis import ATTACK_FAMILY_ORDER, attack_family, defence_variant, run_analysis
from crossids.cli import build_parser

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    "name, family",
    [
        ("DoS Hulk", "dos"),
        ("Hulk-Evasive", "dos"),
        ("DDoS", "ddos"),
        ("DDOS attack-HOIC", "ddos"),
        ("PortScan", "scan"),
        ("PartOfAHorizontalPortScan", "scan"),
        ("SSH-Bruteforce", "bruteforce"),
        ("FTP-Patator", "bruteforce"),
        ("Brute Force -Web", "web"),
        ("Web Attack � Brute Force", "web"),
        ("Sql Injection", "web"),
        ("xss", "web"),
        ("C&C-HeartBeat", "botnet"),
        ("botnet-s42", "botnet"),
        ("Okiru", "botnet"),
        ("Infilteration", "exploit_malware"),
        ("Backdoor", "exploit_malware"),
        ("Analysis", "other"),
        ("Fuzzers", "other"),
        ("BENIGN", "benign"),
        ("normal", "benign"),
    ],
)
def test_attack_family_rules(name, family):
    assert attack_family(name) == family


def test_defence_variant_suffixes():
    assert defence_variant("Hulk-NoDefense") == "NoDefense"
    assert defence_variant("TCPFlood-Security2") == "Security2"
    assert defence_variant("PortScan") is None


def test_every_cached_attack_type_has_a_family():
    folder = DATA / "processed"
    if not folder.exists():
        pytest.skip("no processed data available")
    names = set()
    for path in folder.glob("*.parquet"):
        names |= set(pd.read_parquet(path, columns=["attack_type"])["attack_type"].astype(str))
    explicit = {n for n in names if n.lower() in analysis.EXPLICIT_OTHER}
    other = sorted(n for n in names if attack_family(n) == "other")
    print("attack types mapped to 'other':", other)
    assert set(other) <= explicit
    assert all(attack_family(n) in ATTACK_FAMILY_ORDER + ("benign",) for n in names)


def test_cli_exposes_analyze():
    args = build_parser().parse_args(["analyze", "--results", "r", "--data-dir", "d", "--out", "o"])
    assert (args.command, args.results, args.data_dir, args.out) == ("analyze", "r", "d", "o")


def _synthetic(tmp_path: Path, with_fewshot: bool) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    results = tmp_path / "results"
    results.mkdir()
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    features = list(analysis.CORE_FEATURES)
    attacks = {
        "cicids2017": ["BENIGN", "DoS Hulk", "PortScan"],
        "usbids1": ["BENIGN", "Hulk-NoDefense", "Hulk-Evasive"],
        "ctu13": ["normal", "botnet-s42"],
    }
    for ds, names in attacks.items():
        n = 300
        label = rng.integers(0, 2, n)
        frame = pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)
        frame["label"] = label
        frame["attack_type"] = [names[1 if (l and len(names) > 2) else (len(names) - 1 if l else 0)] if False else
                                (names[int(rng.integers(1, len(names)))] if l else names[0]) for l in label]
        frame["dataset"] = ds
        frame.to_parquet(data / f"{ds}.parquet", index=False)

    rows, pa_rows = [], []
    scenarios = {
        "indomain": [("cicids2017", "cicids2017"), ("usbids1", "usbids1"), ("ctu13", "ctu13")],
        "pairwise": [("cicids2017", "usbids1"), ("usbids1", "cicids2017"), ("cicids2017", "ctu13")],
        "lodo": [("leave-usbids1-out", "usbids1")],
    }
    for method in ["hgb", "rift", "dann"]:
        for mode in ["transductive"]:
            for exp, pairs in scenarios.items():
                for scen, tgt in pairs:
                    if exp == "pairwise":
                        scen = f"{scen}->{tgt}"
                    base = {"method": method, "mode": mode, "experiment": exp, "scenario": scen, "seed": 0}
                    auc = float(rng.uniform(0.5, 0.95))
                    est = float(rng.uniform(0.1, 0.9)) if method == "rift" else np.nan
                    true = float(rng.uniform(0.1, 0.9))
                    rows.append({**base, "transfer_type": "in-domain" if exp == "indomain" else (
                        "lodo" if exp == "lodo" else "cross-extractor"), "roc_auc": auc, "pr_auc": auc,
                        "balanced_accuracy": auc, "f1": auc, "mcc": auc - 0.5, "fpr": 0.1,
                        "est_prior": est, "true_prior": true})
                    names = {"usbids1": ["Hulk-NoDefense", "Hulk-Evasive", "BENIGN"],
                             "cicids2017": ["DoS Hulk", "PortScan"], "ctu13": ["botnet-s42"]}[tgt]
                    for at in names:
                        if at == "BENIGN":
                            continue
                        pa_rows.append({**base, "attack_type": at, "n": 50, "detected": int(rng.integers(0, 50))})
    if with_fewshot:
        for k in (1, 5):
            for scen in ("cicids2017->usbids1", "usbids1->cicids2017"):
                rows.append({"method": "fewshot-catboost@" + str(k), "mode": "transductive",
                             "experiment": "fewshot", "scenario": scen, "seed": 0, "transfer_type": "cross-extractor",
                             "roc_auc": 0.8 + 0.01 * k, "balanced_accuracy": 0.7, "fewshot_k": float(k),
                             "shift_score": float(rng.uniform(0, 1))})
    pd.DataFrame(rows).to_csv(results / "runs.csv", index=False)
    pd.DataFrame(pa_rows).to_csv(results / "per_attack.csv", index=False)
    return results, tmp_path / "data"


def test_run_analysis_on_partial_results(tmp_path):
    results, data = _synthetic(tmp_path, with_fewshot=False)
    out = tmp_path / "analysis"
    result = run_analysis(results, data, out, methods=("hgb", "rift", "dann"))
    names = {p for p, _ in result["files"]}
    assert (out / "README.md").exists()
    assert (out / "01_attack_taxonomy.csv").exists()
    assert (out / "04_summary.md").exists()
    assert (out / "fig02_feature_sign_heatmap.png").exists()
    assert (out / "fig02_feature_sign_heatmap.pdf").exists()
    assert (out / "fig03_pairwise_matrix_transductive.png").exists()
    assert any("few-shot" in sec for sec, _ in result["skipped"])
    assert names
    # the input files are only read
    assert len(pd.read_csv(results / "runs.csv")) > 0


def test_run_analysis_with_fewshot_and_shift(tmp_path):
    results, data = _synthetic(tmp_path, with_fewshot=True)
    out = tmp_path / "analysis"
    result = run_analysis(results, data, out, methods=("hgb", "rift", "dann"))
    assert (out / "fig07_fewshot_curve_transductive.png").exists()
    assert (out / "07_fewshot_curve.csv").exists()
    assert not any("few-shot curve" == sec for sec, _ in result["skipped"])


def test_spearman_handles_constant_input():
    assert np.isnan(analysis._spearman(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.1, 0.2, 0.3])))


def test_paired_cells_compare_sources_with_and_without_family():
    pair = pd.DataFrame(
        {
            "method": ["hgb"] * 4,
            "mode": ["transductive"] * 4,
            "target": ["t", "t", "t", "t"],
            "source": ["a", "b", "c", "d"],
            "family": ["dos"] * 4,
            "seen": [True, True, False, False],
            "informedness": [0.6, 0.8, 0.2, 0.4],
            "recall": [0.6, 0.8, 0.2, 0.4],
        }
    )
    cells = analysis.paired_family_cells(pair)
    assert len(cells) == 1
    assert cells["paired_difference"].iloc[0] == pytest.approx(0.4)
    stats_df = analysis.paired_family_stats(cells)
    assert set(stats_df["family"]) == {"dos", "all"}


def test_sign_rule_requires_two_datasets():
    auc = pd.DataFrame(
        {"dataset": ["a", "b", "a"], "feature": ["duration", "duration", "bytes_per_pkt"],
         "roc_auc": [0.7, 0.8, 0.3], "n": [1, 1, 1]}
    )
    summary = analysis.feature_sign_summary(auc, n_datasets=2).set_index("feature")
    assert bool(summary.loc["duration", "consistent_sign_across_datasets"]) is True
    assert summary.loc["bytes_per_pkt", "sign"] == "single dataset"
    assert bool(summary.loc["bytes_per_pkt", "consistent_sign_across_datasets"]) is False

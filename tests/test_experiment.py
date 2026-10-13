"""Experiment runner, resumable results, report, data preparation, download and CLI on synthetic data."""

import copy
import functools
import http.server
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest

from crossids import datasets
from crossids.cli import main
from crossids.datasets import real_info, synthetic_tables
from crossids.experiment import DEFAULT_CONFIG, RUN_COLUMNS, build_scenarios, load_config, run_experiments, split_tables
from crossids.methods import rift as rift_module
from crossids.report import summarise, write_report
from crossids.schema import SPELLINGS, dataset_features, features_for


def _cfg(tmp_path, **overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        source="synthetic",
        synthetic={"n_domains": 3, "n": 400, "seed": 0},
        seeds=[0],
        methods=["lr", "hgb", "rift"],
        experiments=["indomain", "lodo"],
        results_dir=str(tmp_path / "results"),
        threads=1,
    )
    cfg.update(overrides)
    return cfg


def test_runner_writes_results_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    runs_path = run_experiments(cfg, data_dir=tmp_path)
    runs = pd.read_csv(runs_path)
    assert list(runs.columns) == RUN_COLUMNS
    # 3 in-domain + 3 LODO scenarios, 3 methods, 2 modes, 1 seed
    assert len(runs) == 6 * 3 * 2
    assert set(runs["experiment"]) == {"indomain", "lodo"}
    attack = pd.read_csv(tmp_path / "results" / "per_attack.csv")
    assert {"attack_type", "recall"} <= set(attack.columns)

    run_experiments(cfg, data_dir=tmp_path)
    again = pd.read_csv(runs_path)
    assert len(again) == len(runs)
    assert pd.read_csv(tmp_path / "results" / "per_attack.csv").shape[0] == attack.shape[0]


def test_runner_resumes_after_partial_results(tmp_path):
    cfg = _cfg(tmp_path, methods=["lr"], experiments=["indomain"])
    runs_path = run_experiments(cfg, data_dir=tmp_path)
    full = pd.read_csv(runs_path)
    partial = full.iloc[:1]
    partial.to_csv(runs_path, index=False)
    run_experiments(cfg, data_dir=tmp_path)
    resumed = pd.read_csv(runs_path)
    assert len(resumed) == len(full)
    assert sorted(resumed["scenario"]) == sorted(full["scenario"])


def test_pool_equals_target_train_features_and_is_disjoint_from_test(tmp_path, monkeypatch):
    seen = []
    original = rift_module.Rift.fit

    def spy(self, sources, target_pool=None):
        seen.append(None if target_pool is None else np.array(target_pool, copy=True))
        return original(self, sources, target_pool)

    monkeypatch.setattr(rift_module.Rift, "fit", spy)
    cfg = _cfg(tmp_path, methods=["rift"], experiments=["lodo"], modes=["transductive", "inductive"])
    run_experiments(cfg, data_dir=tmp_path)
    features = list(features_for("core"))
    tables = synthetic_tables(**cfg["synthetic"])
    splits = split_tables(tables, cfg["test_size"], 0)
    pools = [p for p in seen if p is not None]
    assert len(pools) == 3 and sum(p is None for p in seen) == 3
    for pool in pools:
        matches = [k for k, (train, test) in splits.items()
                   if pool.shape == (len(train), len(features)) and np.array_equal(pool, train[features].to_numpy(float))]
        assert len(matches) == 1
        test_rows = {row.tobytes() for row in splits[matches[0]][1][features].to_numpy(float)}
        assert not any(row.tobytes() in test_rows for row in pool)


def test_scenarios_and_transfer_types():
    real = {k: real_info(k) for k in ("cicids2017", "cicids2018", "unswnb15")}
    keys = list(real)
    pairs = build_scenarios(["pairwise"], keys, real)
    assert len(pairs) == 6
    types = {p.name: p.transfer_type for p in pairs}
    assert types["cicids2017->cicids2018"] == "same-family"
    assert types["cicids2017->unswnb15"] == "cross-extractor"
    lodo = build_scenarios(["lodo"], keys, real)
    assert len(lodo) == 3 and all(s.transfer_type == "lodo" for s in lodo)


def test_transfer_types_for_zeek_argus_and_ucd_pairs():
    keys = ["toniot", "iot23", "unswnb15", "ctu13", "insdn", "usbids1", "cicids2018"]
    infos = {k: real_info(k) for k in keys}
    types = {p.name: p.transfer_type for p in build_scenarios(["pairwise"], keys, infos)}
    assert types["toniot->unswnb15"] == "same-family"
    assert types["toniot->iot23"] == "same-extractor"
    assert types["iot23->toniot"] == "same-extractor"
    assert types["iot23->insdn"] == "cross-extractor"
    assert types["insdn->usbids1"] == "same-extractor"
    assert types["insdn->cicids2018"] == "same-extractor"
    assert types["ctu13->unswnb15"] == "same-extractor"
    assert types["unswnb15->insdn"] == "cross-extractor"
    assert types["ctu13->cicids2018"] == "cross-extractor"


def test_resume_does_not_skip_a_different_mode_or_schema(tmp_path):
    cfg = _cfg(tmp_path, methods=["lr"], experiments=["indomain"], modes=["transductive"])
    runs_path = run_experiments(cfg, data_dir=tmp_path)
    assert len(pd.read_csv(runs_path)) == 3
    run_experiments(_cfg(tmp_path, methods=["lr"], experiments=["indomain"], modes=["inductive"]), data_dir=tmp_path)
    runs = pd.read_csv(runs_path)
    assert len(runs) == 6 and set(runs["mode"]) == {"transductive", "inductive"}
    run_experiments(_cfg(tmp_path, methods=["lr"], experiments=["indomain"], modes=["inductive"], schema="minimal"),
                    data_dir=tmp_path)
    runs = pd.read_csv(runs_path)
    assert len(runs) == 9 and set(runs["schema"]) == {"core", "minimal"}
    assert runs["config_hash"].notna().all()


def test_resume_with_a_different_header_fails_with_a_clear_message(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "runs.csv").write_text("method,roc_auc\nlr,0.9\n")
    cfg = _cfg(tmp_path, methods=["lr"], experiments=["indomain"])
    with pytest.raises(ValueError, match="different header"):
        run_experiments(cfg, data_dir=tmp_path)


def test_fewshot_rows_carry_k_and_shift_score(tmp_path):
    cfg = _cfg(tmp_path, methods=[], experiments=["fewshot"], modes=["transductive"],
               synthetic={"n_domains": 2, "n": 300, "seed": 0}, fewshot_k=[1, 5])
    runs_path = run_experiments(cfg, data_dir=tmp_path)
    runs = pd.read_csv(runs_path)
    assert len(runs) == 2 * 2
    assert set(runs["mode"]) == {"fewshot"}
    assert set(runs["fewshot_k"]) == {1.0, 5.0}
    assert runs["shift_score"].notna().all()
    assert runs["method"].str.startswith("fewshot-catboost@").all()


def test_device_and_mode_values_are_validated(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("device: tpu\n")
    with pytest.raises(ValueError, match="device"):
        load_config(path)
    path.write_text("modes: [sideways]\n")
    with pytest.raises(ValueError, match="modes"):
        load_config(path)


def test_summary_reports_prior_error_and_distinct_scenario_counts():
    rows = []
    for scenario, est in (("a->b", 0.30), ("a->c", 0.10)):
        rows.append({"method": "rift", "mode": "transductive", "experiment": "pairwise", "scenario": scenario,
                     "transfer_type": "same-family", "seed": 0, "roc_auc": 0.9, "pr_auc": 0.5, "f1": 0.5,
                     "balanced_accuracy": 0.5, "mcc": 0.0, "detection_rate": 0.5, "fpr": 0.1,
                     "est_prior": est, "true_prior": 0.2})
    summary = summarise(pd.DataFrame(rows)).iloc[0]
    assert summary["n_scenarios"] == 2
    assert np.isclose(summary["prior_abs_error_mean"], np.mean([0.10, 0.10]))


def test_report_summary_averages_scenarios_then_seeds():
    rows = []
    for seed in (0, 1):
        for scenario, roc in (("a->b", 0.8 + 0.1 * seed), ("a->c", 0.6 + 0.1 * seed)):
            rows.append(
                {
                    "method": "lr", "experiment": "pairwise", "scenario": scenario,
                    "transfer_type": "same-family", "seed": seed, "roc_auc": roc,
                    "pr_auc": 0.5, "f1": 0.5, "balanced_accuracy": 0.5, "mcc": 0.0,
                    "detection_rate": 0.5, "fpr": 0.1,
                }
            )
    summary = summarise(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["n_seeds"] == 2
    assert np.isclose(row["roc_auc_mean"], 0.75)
    assert np.isclose(row["roc_auc_std"], np.std([0.7, 0.8], ddof=1))


def test_write_report_creates_csv_and_markdown(tmp_path):
    cfg = _cfg(tmp_path, methods=["lr", "hgb"], experiments=["indomain"])
    run_experiments(cfg, data_dir=tmp_path)
    csv_path, md_path = write_report(tmp_path / "results")
    assert csv_path.exists() and md_path.exists()
    text = md_path.read_text()
    assert "indomain" in text and "ROC-AUC" in text


def test_prepare_samples_a_raw_file_into_parquet(tmp_path, monkeypatch):
    names = SPELLINGS["cic2018"]
    header = [names[f][0] for f in ("duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
                                    "fwd_iat_mean", "bwd_iat_mean", "fwd_init_win", "bwd_init_win")]
    rng = np.random.default_rng(0)
    lines = [",".join([*header, "Label"])]
    for i in range(3000):
        values = [str(float(v)) for v in rng.integers(1, 10_000, size=9)]
        if i % 500 == 0:
            lines.append(",".join([*header, "Label"]))
        lines.append(",".join([*values, "Benign" if i % 3 else "DDOS attack-HOIC"]))
    raw = tmp_path / "raw" / "cicids2018"
    raw.mkdir(parents=True)
    (raw / "day.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setitem(datasets.RAW_FILES, "cicids2018", (datasets.RawFile("unused", "day.csv"),))
    out = datasets.prepare("cicids2018", tmp_path, max_rows=500, seed=0, chunk_rows=700)
    table = pd.read_parquet(out)
    assert len(table) == 500
    assert set(table["label"].unique()) == {0, 1}
    assert set(dataset_features("cicids2018")) <= set(table.columns)
    assert not table[list(dataset_features("cicids2018"))].isna().any().any()


def test_insdn_zip_is_read_through_its_csv_members(tmp_path, monkeypatch):
    names = SPELLINGS["cic2018"]
    header = [names[f][0] for f in ("duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
                                    "fwd_iat_mean", "bwd_iat_mean", "fwd_init_win", "bwd_init_win")]
    body = "\n".join(
        ",".join([*[str(v + i) for v in range(9)], lab])
        for i, lab in enumerate(["Normal", "DoS", "Normal", "DDoS"])
    )
    csv_text = ",".join([*header, "Label"]) + "\n" + body + "\n"
    zip_path = tmp_path / "raw" / "insdn" / "InSDN_DatasetCSV.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("InSDN_DatasetCSV/Normal_data.csv", csv_text)
    monkeypatch.setitem(datasets.RAW_FILES, "insdn", (datasets.RawFile("unused", zip_path.name),))
    out = datasets.prepare("insdn", tmp_path, max_rows=100)
    table = pd.read_parquet(out)
    assert len(table) == 4
    assert table["label"].sum() == 2


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        return


def test_download_skips_existing_and_completes_partial(tmp_path):
    payload = b"0123456789" * 50
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "f.bin").write_bytes(payload)
    handler = functools.partial(_Quiet, directory=str(tmp_path / "srv"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/f.bin"
        dest = tmp_path / "raw" / "k" / "f.bin"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x")
        datasets.download_file("http://127.0.0.1:1/never", dest)
        assert dest.read_bytes() == b"x"
        dest.unlink()
        (dest.with_name(dest.name + ".part")).write_bytes(payload[:20])
        datasets.download_file(url, dest)
        assert dest.read_bytes() == payload
        assert not dest.with_name(dest.name + ".part").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_cli_synth_and_report(tmp_path):
    out = tmp_path / "synthetic"
    assert main(["synth", "--out", str(out), "--n-domains", "2", "--n", "120"]) == 0
    assert sorted(p.name for p in out.glob("*.parquet")) == ["syn0.parquet", "syn1.parquet"]
    cfg = _cfg(tmp_path, methods=["lr"], experiments=["indomain"])
    run_experiments(cfg, data_dir=tmp_path)
    assert main(["report", "--results", str(tmp_path / "results")]) == 0
    assert (tmp_path / "results" / "summary.md").exists()

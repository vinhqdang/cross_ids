"""Experiment runner, resumable results, report, data preparation, download and CLI on synthetic data."""

import copy
import functools
import http.server
import threading
import zipfile

import numpy as np
import pandas as pd

from crossids import datasets
from crossids.cli import main
from crossids.datasets import real_info
from crossids.experiment import DEFAULT_CONFIG, RUN_COLUMNS, build_scenarios, run_experiments
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
    # 3 in-domain + 3 LODO scenarios, 3 methods, 1 seed
    assert len(runs) == 6 * 3
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


def test_pool_contains_only_target_features(tmp_path, monkeypatch):
    seen = []
    original = rift_module.Rift.fit

    def spy(self, sources, target_pool=None):
        seen.append(None if target_pool is None else np.asarray(target_pool).shape)
        return original(self, sources, target_pool)

    monkeypatch.setattr(rift_module.Rift, "fit", spy)
    cfg = _cfg(tmp_path, methods=["rift"], experiments=["lodo"])
    run_experiments(cfg, data_dir=tmp_path)
    assert seen and all(shape is not None and len(shape) == 2 for shape in seen)
    assert all(shape[1] == len(features_for("core")) for shape in seen)


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

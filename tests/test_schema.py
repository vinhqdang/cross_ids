"""Schema harmonisation on tiny hand-made frames for each extractor family."""

import numpy as np
import pandas as pd
import pytest

from crossids.schema import (
    CORE_FEATURES,
    EXTENDED_FEATURES,
    MINIMAL_FEATURES,
    SOURCES,
    SPELLINGS,
    dataset_features,
    features_for,
    harmonise,
    supports,
)


def _cic_frame(dataset: str, labels: list[str]) -> pd.DataFrame:
    names = SPELLINGS[SOURCES[dataset].spelling]
    n = len(labels)
    data = {
        names["duration"][0]: [2_000_000.0] * n,
        names["fwd_pkts"][0]: [3.0] * n,
        names["bwd_pkts"][0]: [1.0] * n,
        names["fwd_bytes"][0]: [100.0] * n,
        names["bwd_bytes"][0]: [50.0] * n,
        names["fwd_iat_mean"][0]: [1_000_000.0] * n,
        names["bwd_iat_mean"][0]: [2_000_000.0] * n,
        names["fwd_init_win"][0]: [8192.0] * n,
        names["bwd_init_win"][0]: [-1.0] * n,
        "Label": labels,
    }
    return pd.DataFrame(data)


def test_cic2017_family_units_and_labels():
    frame = _cic_frame("cicids2017", ["BENIGN", "DDoS"])
    out = harmonise(frame, "cicids2017")
    assert list(out["label"]) == [0, 1]
    assert list(out["attack_type"]) == ["BENIGN", "DDoS"]
    assert set(EXTENDED_FEATURES) <= set(out.columns)
    assert np.isclose(np.expm1(out["duration"].iloc[0]), 2.0)
    assert np.isclose(np.expm1(out["fwd_iat_mean"].iloc[0]), 1.0)
    assert np.isclose(np.expm1(out["bwd_init_win"].iloc[0]), 0.0)


def test_cic2018_drops_repeated_header_rows():
    frame = _cic_frame("cicids2018", ["Benign", "Label", "DDOS attack-HOIC"])
    out = harmonise(frame, "cicids2018")
    assert list(out["label"]) == [0, 1]
    assert "Label" not in set(out["attack_type"])


def test_usbids1_sannio_spelling():
    frame = _cic_frame("usbids1", ["BENIGN", "Hulk-NoDefense"])
    out = harmonise(frame, "usbids1")
    assert list(out["label"]) == [0, 1]
    assert np.isclose(np.expm1(out["fwd_bytes"].iloc[0]), 100.0)
    assert supports("usbids1", "extended")


def test_insdn_normal_is_benign():
    frame = _cic_frame("insdn", ["Normal", "DoS"])
    out = harmonise(frame, "insdn")
    assert list(out["label"]) == [0, 1]
    assert out["dataset"].iloc[0] == "insdn"


def test_argus_units_and_label_text():
    frame = pd.DataFrame(
        {
            "dur": [2.0, 2.0],
            "spkts": [3, 3],
            "dpkts": [1, 1],
            "sbytes": [100, 100],
            "dbytes": [50, 50],
            "sinpkt": [1000.0, 1000.0],
            "dinpkt": [2000.0, 2000.0],
            "swin": [255, 255],
            "dwin": [0, 0],
            "attack_cat": ["Normal", " Exploits "],
            "label": [0, 1],
        }
    )
    out = harmonise(frame, "unswnb15")
    assert list(out["label"]) == [0, 1]
    assert list(out["attack_type"]) == ["Normal", "Exploits"]
    assert np.isclose(np.expm1(out["fwd_iat_mean"].iloc[0]), 1.0)
    assert np.isclose(np.expm1(out["bwd_iat_mean"].iloc[0]), 2.0)


def test_zeek_ton_dash_is_zero_and_type_column():
    frame = pd.DataFrame(
        {
            "duration": ["-", "290.5"],
            "src_pkts": [3, 4],
            "dst_pkts": [1, 2],
            "src_bytes": ["-", "100"],
            "dst_bytes": ["50", "-"],
            "label": [0, 1],
            "type": ["normal", "backdoor"],
        }
    )
    out = harmonise(frame, "toniot")
    assert list(out["label"]) == [0, 1]
    assert list(out["attack_type"]) == ["normal", "backdoor"]
    assert np.isclose(np.expm1(out["duration"].iloc[0]), 0.0)
    assert np.isclose(np.expm1(out["fwd_bytes"].iloc[0]), 0.0)
    assert set(out.columns) >= set(MINIMAL_FEATURES)


def test_zeek_iot23_benign_prefix_and_orig_names():
    frame = pd.DataFrame(
        {
            "duration": [0.0, 1.5],
            "orig_pkts": [1, 2],
            "resp_pkts": [0, 3],
            "orig_bytes": [0, 40],
            "resp_bytes": [0, 60],
            "label": ["Benign", "PartOfAHorizontalPortScan"],
        }
    )
    out = harmonise(frame, "iot23")
    assert list(out["label"]) == [0, 1]
    assert list(out["attack_type"]) == ["Benign", "PartOfAHorizontalPortScan"]
    assert "fwd_iat_mean" not in out.columns


def test_ctu13_background_dropped_and_bwd_from_total():
    frame = pd.DataFrame(
        {
            "Dur": [2.0, 2.0, 2.0, 2.0],
            "TotPkts": [4, 4, 4, 4],
            "SrcBytes": [100, 100, 300, 10],
            "TotBytes": [150, 150, 200, 30],
            "Label": [
                "flow=Background-UDP-Established",
                "flow=From-Botnet-V42-TCP-Attempt",
                "flow=To-Normal-V42-Grill",
                "flow=From-Normal-V42-Jist",
            ],
        }
    )
    out = harmonise(frame, "ctu13")
    assert len(out) == 3
    assert list(out["label"]) == [1, 0, 0]
    assert np.isclose(np.expm1(out["bwd_bytes"].iloc[0]), 50.0)
    assert np.isclose(np.expm1(out["bwd_bytes"].iloc[1]), 0.0)
    assert list(out.columns[: len(MINIMAL_FEATURES)]) == list(MINIMAL_FEATURES)


def test_derived_minimal_and_core_values():
    frame = _cic_frame("cicids2017", ["BENIGN"])
    out = harmonise(frame, "cicids2017")
    expected_bytes_per_pkt = 150.0 / 4.0
    assert np.isclose(np.expm1(out["bytes_per_pkt"].iloc[0]), expected_bytes_per_pkt)
    assert np.isclose(np.expm1(out["total_pkts"].iloc[0]), 4.0)
    assert np.isclose(np.expm1(out["fwd_bwd_pkt_ratio"].iloc[0]), 3.0 / 2.0)


def test_tiers_and_support():
    assert len(MINIMAL_FEATURES) == 9
    assert len(CORE_FEATURES) == 13
    assert len(EXTENDED_FEATURES) == 17
    assert features_for("core") == CORE_FEATURES
    assert supports("ctu13", "minimal") and not supports("ctu13", "core")
    assert set(dataset_features("ctu13")) == set(MINIMAL_FEATURES)
    assert set(CORE_FEATURES) <= set(dataset_features("cicids2017"))
    assert not supports("toniot", "extended")
    with pytest.raises(ValueError):
        features_for("bogus")


def test_missing_required_column_raises():
    frame = pd.DataFrame({"Flow Duration": [1.0], "Label": ["BENIGN"]})
    with pytest.raises(KeyError):
        harmonise(frame, "cicids2017")

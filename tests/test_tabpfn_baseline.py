"""Versioned TabPFN baselines: registration, token lookup and the status check, without downloading weights."""
import os

import pytest

from crossids.flare import evaluate as ev


def test_versioned_tabpfn_methods_are_registered_and_the_unversioned_name_is_gone():
    for name in ("tabpfn-v2", "tabpfn-v3", "tabpfn-v3.5"):
        assert name in ev.ALL_METHODS
    assert "tabpfn" not in ev.ALL_METHODS
    assert ev.TABPFN_LOCAL == {"tabpfn-v2": "v2"}  # run on this machine
    assert ev.TABPFN_API == {"tabpfn-v3": "v3_default", "tabpfn-v3.5": "v3.5_default"}  # run through the Prior Labs API


def test_token_is_read_from_the_environment_and_never_overrides_an_existing_one(monkeypatch, tmp_path):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    f = tmp_path / "token.txt"
    f.write_text("  abc123\n")
    assert ev.load_tabpfn_token(f) is True
    assert os.environ["TABPFN_TOKEN"] == "abc123"
    f.write_text("other")
    ev.load_tabpfn_token(f)
    assert os.environ["TABPFN_TOKEN"] == "abc123"


def test_missing_token_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    assert ev.load_tabpfn_token(tmp_path / "absent.txt") is False
    assert "TABPFN_TOKEN" not in os.environ


def test_unknown_version_is_rejected():
    with pytest.raises(KeyError):
        ev.tabpfn_status("tabpfn-v9")

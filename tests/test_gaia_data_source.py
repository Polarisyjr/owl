from __future__ import annotations

import sys
from pathlib import Path

import pytest

OWL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWL_ROOT))

from utils.gaia import GAIABenchmark


def test_missing_upstream_data_never_falls_back_to_curated(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GAIA_USE_CURATED", raising=False)
    benchmark = GAIABenchmark(
        data_dir=str(tmp_path / "gaia"),
        save_to=str(tmp_path / "results.json"),
    )

    def require_real_download():
        raise RuntimeError("real HF download required")

    monkeypatch.setattr(benchmark, "download", require_real_download)

    with pytest.raises(RuntimeError, match="real HF download required"):
        benchmark.load()


def test_curated_data_remains_an_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("GAIA_USE_CURATED", "1")
    benchmark = GAIABenchmark(
        data_dir=str(tmp_path / "gaia"),
        save_to=str(tmp_path / "results.json"),
    )

    def unexpected_download():
        raise AssertionError("explicit curated mode must not download HF data")

    monkeypatch.setattr(benchmark, "download", unexpected_download)
    tasks = benchmark._load_tasks("valid", "all", randomize=False)

    assert len(tasks) == 165
    assert all(not task["file_name"] for task in tasks)

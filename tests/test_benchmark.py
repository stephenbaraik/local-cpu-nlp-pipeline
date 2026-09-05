from __future__ import annotations

import pytest

from pipeline import artifacts, benchmark
from pipeline.config import Config

pytestmark = pytest.mark.slow


def test_benchmark_never_reads_a_cached_artifact(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("benchmark must never touch the artifact cache")

    monkeypatch.setattr(artifacts, "load", _boom)
    monkeypatch.setattr(artifacts, "is_cached", _boom)
    monkeypatch.setattr(artifacts, "save", _boom)

    result = benchmark.run_benchmark(Config())

    for stage in ("keywords", "classify", "summarize"):
        block = result[stage]
        assert block["init_s"] >= 0
        assert len(block["inference_s_runs"]) == 3
        assert block["inference_s_median"] >= 0
        assert block["peak_rss_mb"] > 0

    env = result["environment"]
    assert env["cores_physical"] >= 1
    assert env["versions"]["transformers"]

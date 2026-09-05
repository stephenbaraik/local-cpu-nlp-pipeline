from __future__ import annotations

import pytest

from pipeline import artifacts, benchmark, runner

pytestmark = pytest.mark.slow

# Full-scale grid (real thread/worker counts, real corpus) is a
# target-server question the design doc leaves open (core count unknown --
# see AGENTS.md/system_design.md Constraints table). This proves the
# mechanism -- one small cell each -- not the production grid.


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    import textwrap

    import pymupdf

    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    text = (
        "Investigators traced the intrusion back to a compromised vendor account "
        "that had standing access for months without triggering any alert."
    )
    doc = pymupdf.open()
    doc.insert_page(-1, text=textwrap.fill(text, width=90))
    doc.save(pdfs_dir / "doc.pdf")
    doc.close()

    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)


def test_thread_worker_grid_reports_peak_rss_per_cell(sandbox):
    cells = benchmark.thread_worker_grid(thread_counts=(1,), worker_counts=(1,))

    assert len(cells) == 1
    cell = cells[0]
    assert cell == {"threads": 1, "workers": 1, "peak_rss_mb": cell["peak_rss_mb"]}
    assert cell["peak_rss_mb"] > 0

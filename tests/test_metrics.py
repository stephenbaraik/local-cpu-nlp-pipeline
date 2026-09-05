from __future__ import annotations

import csv
import json

import pymupdf
import pytest

from pipeline import artifacts, metrics, runner

# Deliberately short (under the 120-word validate floor) so the doc is
# always rejected before the model-backed stages -- keeps this test fast
# and model-free, same trick as test_cache.py.
SHORT_TEXT = "Too short to pass validation."


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    doc = pymupdf.open()
    doc.insert_page(-1, text=SHORT_TEXT)
    doc.save(pdfs_dir / "doc.pdf")
    doc.close()

    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    (reports_dir / "cpu").mkdir(parents=True)
    (reports_dir / "gpu").mkdir(parents=True)

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(metrics, "RUNS_DIR", runs_dir)

    return {"runs_dir": runs_dir, "reports_dir": reports_dir}


def test_write_metrics_produces_three_layer_json_and_two_csvs(sandbox):
    result = runner.run()
    paths = metrics.write_metrics(
        result["run_id"], runs_dir=sandbox["runs_dir"], reports_dir=sandbox["reports_dir"]
    )

    data = json.loads(paths["json"].read_text())
    assert set(data) == {"run", "documents", "stages"}
    assert data["run"]["run_id"] == result["run_id"]
    assert "parallel_info" in data["run"]

    assert len(data["documents"]) == 1
    doc_row = data["documents"][0]
    assert doc_row["rejected"] is True
    assert doc_row["reject_reason"] == "too_short"
    # rejected before any model-backed stage runs -- those stages never fire
    assert doc_row["keyword_time_sec"] == 0.0
    assert doc_row["inference_time_sec"] == 0.0

    # extract, clean, validate ran; keywords/classify/context/summarize skipped
    assert len(data["stages"]) == 3
    assert {r["stage"] for r in data["stages"]} == {"extract", "clean", "validate"}
    assert all(r["peak_ram_mb"] is not None for r in data["stages"])

    with paths["documents_csv"].open(newline="") as f:
        doc_rows = list(csv.DictReader(f))
    assert len(doc_rows) == 1
    assert doc_rows[0]["pdf_id"] == doc_row["pdf_id"]

    with paths["stages_csv"].open(newline="") as f:
        stage_rows = list(csv.DictReader(f))
    assert len(stage_rows) == 3

    # default Config().device == "cpu" -- reports/cpu/ gets the copy, reports/gpu/ stays empty
    assert paths["report_documents_csv"].parent == sandbox["reports_dir"] / "cpu"
    assert paths["report_documents_csv"].exists()
    assert paths["report_stages_csv"].exists()
    assert list((sandbox["reports_dir"] / "gpu").iterdir()) == []

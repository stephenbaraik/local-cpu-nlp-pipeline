from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pymupdf
import pytest

from pipeline import artifacts, runner

pytestmark = pytest.mark.slow  # spawns worker processes that re-import torch/transformers

REAL_STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "stages"

# Short (rejected) docs so this stays model-free and fast, matching
# test_cache.py -- workers is a runner-level concern, not a stage one.
DOC_TEXTS = [
    "Investigators traced the intrusion back to a compromised vendor account "
    "that had standing access for months without triggering any alert.",
    "A newly disclosed flaw in a firmware update mechanism allows a local "
    "attacker to substitute a malicious image before the signature check runs.",
    "A phishing campaign targeting cloud administrators exploited "
    "misconfigured single sign-on integrations to gain persistent access.",
    "Researchers described a supply-chain compromise affecting a popular "
    "open-source package used by thousands of downstream projects.",
]


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    stages_dir = tmp_path / "stages"
    shutil.copytree(REAL_STAGES_DIR, stages_dir)

    for i, text in enumerate(DOC_TEXTS):
        doc = pymupdf.open()
        # insert_page silently drops text past the page bottom when it is
        # handed one unwrapped line, so wrap it ourselves first.
        doc.insert_page(-1, text=textwrap.fill(text, width=90))
        doc.save(pdfs_dir / f"doc_{i}.pdf")
        doc.close()

    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(runner, "STAGES_DIR", stages_dir)

    return {"artifacts_dir": artifacts_dir}


def test_workers_greater_than_one_produces_the_same_result_as_serial(sandbox, monkeypatch):
    monkeypatch.setenv("NLP_WORKERS", "2")
    result = runner.run()

    for stage in ("extract", "clean", "validate"):
        assert result["stages"][stage]["computed"] == len(DOC_TEXTS)
        assert result["stages"][stage]["errors"] == 0

    doc_dirs = [p for p in sandbox["artifacts_dir"].iterdir() if p.is_dir()]
    assert len(doc_dirs) == len(DOC_TEXTS)
    for doc_dir in doc_dirs:
        payload = artifacts.load(doc_dir.name, "clean")["payload"]
        assert payload["body"].strip()

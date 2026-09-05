from __future__ import annotations

import textwrap

import pymupdf
import pytest

from pipeline import artifacts, runner

pytestmark = pytest.mark.slow

# Model-backed counterpart to test_cache.py: proves the same fingerprint
# mechanism holds for stages 4-7, using one real accepted document.
ACCEPTED_TEXT = (
    "Investigators traced the intrusion back to a compromised vendor account that had "
    "standing access to production systems for nearly eighteen months without "
    "triggering any automated alert. The incident review found that credential "
    "rotation policies existed on paper but were rarely enforced in practice, and that "
    "monitoring dashboards were tuned to suppress exactly the kind of anomalous access "
    "pattern the attacker relied on. Analysts recommended tightening least privilege "
    "boundaries, shortening credential lifetimes, and reviewing every dashboard "
    "suppression rule quarterly rather than leaving them untouched indefinitely once "
    "configured. A follow-up audit across sibling business units uncovered three "
    "additional vendor accounts with similarly broad standing access, prompting a "
    "wider remediation effort that is still underway across the organization's cloud "
    "environment and internal ticketing systems."
)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    doc = pymupdf.open()
    doc.insert_page(-1, text=textwrap.fill(ACCEPTED_TEXT, width=90))
    doc.save(pdfs_dir / "doc.pdf")
    doc.close()

    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)


def test_config_key_change_leaves_sibling_and_upstream_stages_cached(sandbox, monkeypatch):
    runner.run()
    runner.run()

    # classify has no downstream dependents (context and summarize depend on
    # keywords/clean, not classify) -- the sibling-isolation case.
    monkeypatch.setenv("NLP_ZEROSHOT_MAX_CHUNKS", "1")
    result = runner.run()

    assert result["stages"]["classify"]["computed"] == 1
    assert result["stages"]["classify"]["cached"] == 0
    for stage in ("extract", "clean", "validate", "keywords", "context", "summarize"):
        assert result["stages"][stage]["cached"] == 1
        assert result["stages"][stage]["computed"] == 0

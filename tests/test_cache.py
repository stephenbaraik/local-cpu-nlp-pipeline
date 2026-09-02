from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pymupdf
import pytest

from pipeline import artifacts, runner

REAL_STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "stages"

STAGE_NAMES = ("extract", "clean", "validate", "keywords", "classify", "context", "summarize")

DOC_A_TEXT = (
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

DOC_B_TEXT = (
    "A newly disclosed flaw in a widely used firmware update mechanism allows an "
    "attacker on the local network to substitute a malicious image before the "
    "signature check runs, according to the researchers who reported it. The vendor "
    "confirmed the finding and shipped a patch within two weeks, though the advisory "
    "notes that devices behind consumer routers without automatic updates could remain "
    "exposed for months or years. Independent testing across a sample of deployed "
    "devices found that fewer than half had applied the patch a month after release, "
    "prompting renewed calls for mandatory automatic firmware updates on consumer "
    "network hardware sold going forward. The disclosure follows a similar case last "
    "year involving a different vendor, suggesting the underlying weakness may be "
    "common across implementations that reuse the same reference update library "
    "without independently reviewing its signature verification logic before shipping."
)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"
    stages_dir = tmp_path / "stages"
    shutil.copytree(REAL_STAGES_DIR, stages_dir)

    for name, text in (("doc_a.pdf", DOC_A_TEXT), ("doc_b.pdf", DOC_B_TEXT)):
        doc = pymupdf.open()
        # insert_page silently drops text past the page bottom when it is
        # handed one unwrapped line, so wrap it ourselves first.
        doc.insert_page(-1, text=textwrap.fill(text, width=90))
        doc.save(pdfs_dir / name)
        doc.close()

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(runner, "STAGES_DIR", stages_dir)

    return {"stages_dir": stages_dir}


def test_first_run_computes_every_stage_for_every_doc(sandbox):
    result = runner.run()
    for stage in STAGE_NAMES:
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0
        assert result["stages"][stage]["errors"] == 0


def test_second_run_is_fully_cached(sandbox):
    runner.run()
    result = runner.run()
    for stage in STAGE_NAMES:
        assert result["stages"][stage]["cached"] == 2
        assert result["stages"][stage]["computed"] == 0


def test_declared_config_key_change_invalidates_stage_and_downstream(sandbox, monkeypatch):
    runner.run()
    runner.run()

    # MAX_PAGES is extract's only declared config key.
    monkeypatch.setenv("NLP_MAX_PAGES", "1")
    result = runner.run()

    for stage in STAGE_NAMES:
        assert result["stages"][stage]["computed"] == 2, f"{stage} should have recomputed"
        assert result["stages"][stage]["cached"] == 0


def test_config_key_change_leaves_unrelated_and_upstream_stages_cached(sandbox, monkeypatch):
    runner.run()
    runner.run()

    # classify has no downstream dependents (context and summarize depend on
    # keywords/clean, not classify), so this is the sibling-isolation case.
    monkeypatch.setenv("NLP_ZEROSHOT_MAX_CHUNKS", "9")
    result = runner.run()

    assert result["stages"]["classify"]["computed"] == 2
    assert result["stages"]["classify"]["cached"] == 0
    for stage in ("extract", "clean", "validate", "keywords", "context", "summarize"):
        assert result["stages"][stage]["cached"] == 2
        assert result["stages"][stage]["computed"] == 0


def test_editing_stage_source_invalidates_stage_and_downstream_leaves_upstream(sandbox):
    runner.run()
    runner.run()

    clean_path = sandbox["stages_dir"] / "clean.py"
    clean_path.write_text(clean_path.read_text() + "\n# cache-test touch\n")

    result = runner.run()

    assert result["stages"]["extract"]["cached"] == 2
    assert result["stages"]["extract"]["computed"] == 0
    for stage in ("clean", "validate", "keywords", "classify", "context", "summarize"):
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0


def test_force_recomputes_everything_regardless_of_fingerprints(sandbox):
    runner.run()
    runner.run()

    result = runner.run(force=True)

    for stage in STAGE_NAMES:
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0

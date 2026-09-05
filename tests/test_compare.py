from __future__ import annotations

import textwrap

import pymupdf
import pytest

from pipeline import artifacts, compare, report, runner

pytestmark = pytest.mark.slow

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
    monkeypatch.setattr(report, "RUNS_DIR", runs_dir)


def test_compare_guard_on_vs_guard_off_produces_a_diff_table(sandbox):
    result = compare.run_compare(["guard-on", "guard-off"])

    assert result["modes"] == ["guard-on", "guard-off"]
    assert len(result["pairs"]) == 1

    pair = result["pairs"][0]
    assert pair["left_mode"] == "guard-on"
    assert pair["right_mode"] == "guard-off"
    assert isinstance(pair["timing_delta_s"], float)
    assert isinstance(pair["accept_reject_changes"], list)
    assert isinstance(pair["label_changes"], list)
    assert isinstance(pair["keyphrase_jaccard"], list)
    # one document went through both modes, so there is something to compare
    assert len(pair["keyphrase_jaccard"]) == 1

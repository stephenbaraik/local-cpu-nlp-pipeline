from __future__ import annotations

import shutil
from pathlib import Path

import pymupdf
import pytest

from pipeline import artifacts, runner

REAL_STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "stages"

STAGE_NAMES = ("extract", "clean", "validate", "keywords", "classify", "context", "summarize")
MODEL_FREE_STAGES = ("extract", "clean", "validate")
MODEL_BACKED_STAGES = ("keywords", "classify", "context", "summarize")

# Deliberately short (well under the 120-word validate floor) so these docs
# are always rejected and stages 4-7 are always skipped, never computed.
# That keeps this suite model-free and fast -- the model-backed stages'
# participation in the same cache mechanism is covered separately, marked
# slow, in test_cache_models.py.
DOC_A_TEXT = (
    "Investigators traced the intrusion back to a compromised vendor account "
    "that had standing access to production systems for months without "
    "triggering any alert."
)

DOC_B_TEXT = (
    "A newly disclosed flaw in a firmware update mechanism allows a local "
    "attacker to substitute a malicious image before the signature check runs."
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
        doc.insert_page(-1, text=text)
        doc.save(pdfs_dir / name)
        doc.close()

    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(artifacts, "INDEX_PATH", artifacts_dir / "_index.json")
    monkeypatch.setattr(runner, "PDFS_DIR", pdfs_dir)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(runner, "STAGES_DIR", stages_dir)

    return {"stages_dir": stages_dir}


def _assert_all_skipped(result: dict) -> None:
    for stage in MODEL_BACKED_STAGES:
        assert result["stages"][stage]["skipped"] == 2
        assert result["stages"][stage]["computed"] == 0
        assert result["stages"][stage]["cached"] == 0


def test_first_run_computes_model_free_stages_and_skips_the_rest(sandbox):
    result = runner.run()
    for stage in MODEL_FREE_STAGES:
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0
        assert result["stages"][stage]["errors"] == 0
    _assert_all_skipped(result)


def test_second_run_is_fully_cached(sandbox):
    runner.run()
    result = runner.run()
    for stage in MODEL_FREE_STAGES:
        assert result["stages"][stage]["cached"] == 2
        assert result["stages"][stage]["computed"] == 0
    _assert_all_skipped(result)


def test_declared_config_key_change_invalidates_stage_and_downstream(sandbox, monkeypatch):
    runner.run()
    runner.run()

    # MAX_PAGES is extract's only declared config key; clean and validate
    # are downstream of it.
    monkeypatch.setenv("NLP_MAX_PAGES", "1")
    result = runner.run()

    for stage in MODEL_FREE_STAGES:
        assert result["stages"][stage]["computed"] == 2, f"{stage} should have recomputed"
        assert result["stages"][stage]["cached"] == 0


def test_editing_stage_source_invalidates_stage_and_downstream_leaves_upstream(sandbox):
    runner.run()
    runner.run()

    clean_path = sandbox["stages_dir"] / "clean.py"
    clean_path.write_text(clean_path.read_text() + "\n# cache-test touch\n")

    result = runner.run()

    assert result["stages"]["extract"]["cached"] == 2
    assert result["stages"]["extract"]["computed"] == 0
    for stage in ("clean", "validate"):
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0


def test_force_recomputes_everything_regardless_of_fingerprints(sandbox):
    runner.run()
    runner.run()

    result = runner.run(force=True)

    for stage in MODEL_FREE_STAGES:
        assert result["stages"][stage]["computed"] == 2
        assert result["stages"][stage]["cached"] == 0

from __future__ import annotations

import textwrap

import pymupdf
import pytest
from keybert import KeyBERT
from keybert.backend import BaseEmbedder

from pipeline import artifacts, runner
from pipeline.config import load_config
from pipeline.stages.classify import CANDIDATE_LABELS
from pipeline.stages.keywords import SecureBertEmbedder

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

    return {"artifacts_dir": artifacts_dir}


def test_keybert_backend_is_securebert_not_the_minilm_fallback():
    # KeyBERT(model=some_hf_model) does not raise -- select_backend silently
    # downloads MiniLM instead unless the object is a real BaseEmbedder.
    embedder = SecureBertEmbedder(load_config())
    assert isinstance(embedder, BaseEmbedder)
    kw_model = KeyBERT(model=embedder)
    assert kw_model.model is embedder


def test_one_accepted_document_end_to_end(sandbox):
    result = runner.run()

    for stage in ("extract", "clean", "validate", "keywords", "classify", "context", "summarize"):
        assert result["stages"][stage]["errors"] == 0
        assert result["stages"][stage]["computed"] == 1

    doc_dirs = [p for p in sandbox["artifacts_dir"].iterdir() if p.is_dir()]
    assert len(doc_dirs) == 1
    doc_id = doc_dirs[0].name

    assert artifacts.load(doc_id, "validate")["payload"]["status"] == "accepted"

    # shape and non-emptiness only -- quantization moves the exact values.
    keywords = artifacts.load(doc_id, "keywords")["payload"]["keywords"]
    assert 0 < len(keywords) <= 5
    for phrase, score in keywords:
        assert isinstance(phrase, str) and phrase.strip()
        assert isinstance(score, float)

    classify = artifacts.load(doc_id, "classify")["payload"]
    assert classify["predicted_label"] in CANDIDATE_LABELS
    assert set(classify["label_scores"]) == set(CANDIDATE_LABELS)
    assert classify["n_windows"] >= 1

    context = artifacts.load(doc_id, "context")["payload"]
    assert context["sentences_kept"] > 0
    assert context["reduced_context"].strip()

    summarize = artifacts.load(doc_id, "summarize")["payload"]
    assert summarize["summary"].strip()
    assert summarize["generated_tokens"] > 0
    assert summarize["prompt_tokens"] > 0

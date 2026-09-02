from __future__ import annotations

from dataclasses import replace

from pipeline.config import Config
from pipeline.stages import DocContext
from pipeline.stages.context import ContextStage

BODY = (
    "The committee said the proposal was rejected. "
    "Researchers are exploring how AI systems can detect anomalies. "
    "The weather was mild that day. "
    "Officials raided the building last night. "
    "AI adoption is accelerating across most industries."
)


def _run(keywords: list[str], config: Config = Config()) -> dict:
    doc = DocContext(
        doc_id="test",
        pdf_bytes=b"",
        config=config,
        payloads={
            "clean": {"body": BODY},
            "keywords": {"keywords": [[k, 1.0] for k in keywords]},
        },
    )
    return ContextStage().run(doc)


def test_word_boundary_ai_does_not_match_said():
    result = _run(["ai"])
    kept = result["reduced_context"]
    assert "committee said" not in kept
    assert "AI systems can detect anomalies" in kept


def test_kept_sentences_stay_in_original_order():
    result = _run(["ai"])
    kept = result["reduced_context"]
    assert kept.index("AI systems can detect anomalies") < kept.index("AI adoption is accelerating")


def test_no_match_falls_back_to_lead_sentences_not_whole_document():
    result = _run(["xyz_nonexistent_keyword"])
    assert result["used_fallback"] is True
    assert result["reduced_context"].startswith("The committee said the proposal was rejected.")


def test_token_budget_is_respected():
    tight_config = replace(Config(), reduced_context_chars=40)
    result = _run(["ai"], config=tight_config)
    # at least one sentence is always kept even if it alone exceeds budget,
    # but a second sentence is never added once the budget is already spent
    assert result["sentences_kept"] >= 1
    if result["sentences_kept"] > 1:
        assert result["reduced_context_chars"] <= 40

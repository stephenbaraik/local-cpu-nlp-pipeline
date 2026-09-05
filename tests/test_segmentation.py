from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer

from pipeline import segmentation
from pipeline.segmentation import build_candidates, segment_text


def test_pattern_compiles():
    # this is the regression test for the fixed-width lookbehind trap: a
    # unified alternation of differing-length abbreviations raises at
    # import time, so successfully importing the module is itself a check.
    assert segmentation._BOUNDARY is not None


def test_abbreviations_and_decimals_are_not_split():
    text = "Dr. Smith published a report. The dataset included 3.14 percent noise, according to Prof. Lee."
    segments = segment_text(text)
    assert segments == [
        "Dr. Smith published a report.",
        "The dataset included 3.14 percent noise,",
        "according to Prof. Lee.",
    ]


def test_clause_boundaries_split_too():
    text = "The attacker gained access; the team responded within an hour, containing the breach."
    segments = segment_text(text)
    assert segments == [
        "The attacker gained access;",
        "the team responded within an hour,",
        "containing the breach.",
    ]


def test_bridging_bigram_absent_real_bigram_present():
    text = (
        "Officials confirmed the incident occurred in the county. "
        "Alameda authorities are investigating further details of the breach. "
        "Jailbreaking paper released today describes new attack vectors."
    )
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    candidates = build_candidates(text, vectorizer)

    assert "county alameda" not in candidates
    assert "jailbreaking paper" in candidates

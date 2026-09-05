from __future__ import annotations

import re

from sklearn.feature_extraction.text import CountVectorizer

# Abbreviations that must not be treated as sentence-ending periods. Each is
# its own lookbehind, individually fixed-width; merging them into one
# alternation -- (?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|Inc|Corp|Ltd))
# -- raises "look-behind requires fixed-width pattern" at compile time
# because the branches differ in length. Confirmed: that exact merged form
# was tried and does raise on this Python. Superset of the requested list
# (adds St/Rev/eg/ie/US/UK/Co/No/Vol/Fig/month abbreviations); nothing in
# the requested list was dropped.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Rev", "vs",
    "etc", "eg", "ie", "US", "UK", "Inc", "Corp", "Ltd", "Co", "No", "Vol",
    "Fig", "approx", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
    "Sep", "Sept", "Oct", "Nov", "Dec",
)

_ABBREV_LOOKBEHINDS = "".join(rf"(?<!\b{a})" for a in _ABBREVIATIONS)

# A boundary is either an abbreviation-guarded period, or an unguarded
# clause mark (; , ! ?) -- the abbreviation guard only ever applied to the
# period in the original spec, so it stays out of the second branch. A
# decimal like "3.14" never matches because the period there is followed by
# a digit, not whitespace -- no separate digit lookbehind is needed.
_BOUNDARY = re.compile(rf"(?:{_ABBREV_LOOKBEHINDS}\.|[;,!?])(?:(?=\s)|$)")


def segment_text(text: str) -> list[str]:
    """Split cleaned body text into sentence- and clause-level segments
    (on . ; , ! ?). Used so a CountVectorizer built over these segments
    never forms an n-gram that bridges two unrelated segments -- it would
    if given one raw string, since CountVectorizer treats newlines as
    whitespace too."""
    segments: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        start = end
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def build_candidates(text: str, vectorizer: CountVectorizer) -> list[str]:
    """Candidate n-grams for KeyBERT, built from segments as separate
    documents so a bigram never bridges a sentence or clause boundary.
    Residual limit: stop-words are removed before n-gramming, so a bigram
    like "jailbreaking paper" can still form inside one segment even after
    an intervening stop-word phrase -- that needs MMR, not segmentation."""
    segments = segment_text(text)
    if not segments:
        return []
    vectorizer.fit(segments)
    return list(vectorizer.get_feature_names_out())

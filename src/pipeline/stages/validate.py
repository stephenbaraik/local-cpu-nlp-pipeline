from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.stages import DocContext, register

# Generic bot-check / interstitial language, never a site name or domain.
FAILURE_SIGNAL_PHRASES = (
    "checking your browser before accessing",
    "please enable cookies and reload the page",
    "attention required",
    "verify you are a human",
    "enable javascript and cookies to continue",
    "ray id",
    "one more step",
    "sorry, you have been blocked",
)

# The team's pipeline reads only page one; 120 words is the line below which
# a first page is legitimately too short to summarize (see phase 3 gate).
MIN_CONTENT_WORDS = 120
MIN_UNIQUE_WORD_RATIO = 0.3


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _failure_signal(text: str) -> str | None:
    low = text.lower()
    for phrase in FAILURE_SIGNAL_PHRASES:
        if phrase in low:
            return phrase
    return None


@dataclass
class ValidateStage:
    name: str = "validate"
    version: str = "1"
    depends_on: tuple[str, ...] = ("clean",)
    config_keys: tuple[str, ...] = ()

    def run(self, doc: DocContext) -> dict:
        body = doc.payloads["clean"]["body"]
        words = _words(body)
        content_words = len(words)
        unique_word_ratio = (len(set(words)) / content_words) if content_words else 0.0
        failure_signal = _failure_signal(body)

        signals = {
            "content_words": content_words,
            "failure_signal": failure_signal,
            "unique_word_ratio": round(unique_word_ratio, 4),
        }

        # Rule 1: known interstitial/bot-check language.
        if failure_signal is not None:
            return {"status": "rejected", "reason": "failure_signal", "signals": signals}
        # Rule 2: repetitive filler padded past the word-count floor -- a
        # naive length check alone is easy to evade, low lexical diversity is not.
        if content_words > 0 and unique_word_ratio < MIN_UNIQUE_WORD_RATIO:
            return {"status": "rejected", "reason": "low_unique_word_ratio", "signals": signals}
        # Rule 3: genuinely too short to be real content.
        if content_words < MIN_CONTENT_WORDS:
            return {"status": "rejected", "reason": "too_short", "signals": signals}

        return {"status": "accepted", "reason": None, "signals": signals}


register(ValidateStage())

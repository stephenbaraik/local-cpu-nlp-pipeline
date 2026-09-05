from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.chunking import sentences as _sentences
from pipeline.stages import DocContext, register


def _keyword_pattern(keywords: list[str]) -> re.Pattern | None:
    escaped = [re.escape(k) for k in keywords if k.strip()]
    if not escaped:
        return None
    # \b...\b: a substring test would let "ai" match inside "said".
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


@dataclass
class ContextStage:
    name: str = "context"
    version: str = "1"
    depends_on: tuple[str, ...] = ("keywords", "clean")
    config_keys: tuple[str, ...] = ("REDUCED_CONTEXT_CHARS",)

    def run(self, doc: DocContext) -> dict:
        body = doc.payloads["clean"]["body"]
        keywords = [k for k, _score in doc.payloads["keywords"]["keywords"]]
        budget = doc.config.reduced_context_chars

        sentences = _sentences(body)
        pattern = _keyword_pattern(keywords)
        matched = [s for s in sentences if pattern and pattern.search(s)]

        used_fallback = not matched
        selected = matched if matched else sentences  # never the whole document unbounded

        kept: list[str] = []
        total_chars = 0
        for s in selected:
            if kept and total_chars + len(s) > budget:
                break
            kept.append(s)
            total_chars += len(s)

        return {
            "sentences_kept": len(kept),
            "reduced_context_chars": total_chars,
            "reduced_context": " ".join(kept),
            "used_fallback": used_fallback,
        }


register(ContextStage())

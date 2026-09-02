from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from pipeline.stages import DocContext, register

# Generic structural boilerplate signals -- phrases a nav/cookie/newsletter/
# related-links block tends to contain, never a site name or domain.
BOILERPLATE_PHRASES = (
    "cookie",
    "subscribe",
    "newsletter",
    "sign up",
    "sign in",
    "log in",
    "log out",
    "related articles",
    "related posts",
    "you might also like",
    "read more",
    "share this",
    "follow us",
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "terms of use",
    "advertisement",
    "skip to content",
    "skip to main content",
    "back to top",
)

BODY_SCORE_THRESHOLD = 15.0
TITLE_MAX_WORDS = 15


def _remove_repeated_lines(pages: list[str]) -> tuple[list[str], list[str]]:
    """Lines that recur across most pages are headers/footers, not body text."""
    if len(pages) <= 1:
        return pages, []
    page_lines = [p.splitlines() for p in pages]
    counts: Counter[str] = Counter()
    for lines in page_lines:
        for line in {l.strip() for l in lines if l.strip()}:
            counts[line] += 1
    threshold = len(pages) // 2 + 1
    repeated = {line for line, c in counts.items() if c >= threshold}
    cleaned = ["\n".join(l for l in lines if l.strip() not in repeated) for lines in page_lines]
    return cleaned, sorted(repeated)


def _split_blocks(page_text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", page_text) if b.strip()]


def _line_stats(block: str) -> tuple[int, float]:
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not lines:
        return 0, 0.0
    words = sum(len(l.split()) for l in lines)
    short_line_ratio = sum(1 for l in lines if len(l.split()) <= 4) / len(lines)
    return words, short_line_ratio


def _boilerplate_hits(block: str) -> int:
    low = block.lower()
    return sum(1 for phrase in BOILERPLATE_PHRASES if phrase in low)


def _score_block(block: str) -> float:
    words, short_line_ratio = _line_stats(block)
    return float(words) - short_line_ratio * 50 - _boilerplate_hits(block) * 30


def _extract_title(blocks: list[str]) -> tuple[str | None, list[str]]:
    if not blocks:
        return None, blocks
    first = blocks[0]
    if "\n" not in first and len(first.split()) <= TITLE_MAX_WORDS:
        return first, blocks[1:]
    return None, blocks


@dataclass
class CleanStage:
    name: str = "clean"
    version: str = "1"
    depends_on: tuple[str, ...] = ("extract",)
    config_keys: tuple[str, ...] = ()

    def run(self, doc: DocContext) -> dict:
        pages = doc.payloads["extract"]["pages"]
        deduped_pages, removed_lines = _remove_repeated_lines(pages)

        all_blocks: list[str] = []
        for page in deduped_pages:
            all_blocks.extend(_split_blocks(page))

        title, rest = _extract_title(all_blocks)
        body_blocks = [b for b in rest if _score_block(b) >= BODY_SCORE_THRESHOLD]

        return {
            "title": title,
            "body": "\n\n".join(body_blocks),
            "block_count": len(body_blocks),
            "removed_lines": removed_lines,
        }


register(CleanStage())

from __future__ import annotations

"""Shared token-aware, sentence-boundary chunker. Any stage that needs to
window a long document for a model with a token cap uses this instead of
its own ad hoc splitting -- one place that never cuts a sentence in half.
"""

import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
    return _NLP


def sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    doc = _get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def chunk_by_tokens(text: str, tokenizer, max_tokens: int, overlap_ratio: float = 0.125) -> list[str]:
    """Packs whole sentences into windows of at most max_tokens (counted by
    `tokenizer`, so it matches whatever model actually consumes the chunk),
    carrying ~overlap_ratio of the previous window's trailing sentences into
    the next so a phrase spanning a window boundary still appears whole in
    at least one window. A single sentence longer than max_tokens is the
    only case that gets a mid-sentence (token-level) split -- there is no
    other way to fit it.
    """
    sents = sentences(text)
    if not sents:
        return []

    sent_tokens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents]
    overlap_budget = max(0, int(max_tokens * overlap_ratio))
    n = len(sents)

    windows: list[str] = []
    i = 0
    while i < n:
        window_sents: list[str] = []
        count = 0
        j = i
        if sent_tokens[j] > max_tokens:
            ids = tokenizer(sents[j], add_special_tokens=False)["input_ids"]
            window_sents = [tokenizer.decode(ids[:max_tokens])]
            j += 1
        else:
            while j < n and count + sent_tokens[j] <= max_tokens:
                window_sents.append(sents[j])
                count += sent_tokens[j]
                j += 1

        windows.append(" ".join(window_sents))
        if j >= n:
            break

        # Step back from j to include ~overlap_budget tokens of trailing
        # sentences in the next window, but always advance past i.
        back = j
        carried = 0
        while back > i + 1 and carried < overlap_budget:
            back -= 1
            carried += sent_tokens[back]
        i = back

    return windows

from __future__ import annotations

from pipeline.chunking import chunk_by_tokens, sentences


class _FakeTokenizer:
    """Word-count tokenizer -- exercises the packing/overlap/fallback logic
    without paying for a real HF tokenizer in a fast test."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


TEXT = (
    "One two three. Four five six seven. Eight nine. "
    "Ten eleven twelve thirteen fourteen. Fifteen sixteen."
)


def test_sentences_splits_on_boundaries():
    assert sentences(TEXT) == [
        "One two three.",
        "Four five six seven.",
        "Eight nine.",
        "Ten eleven twelve thirteen fourteen.",
        "Fifteen sixteen.",
    ]


def test_chunk_by_tokens_packs_sentences_and_overlaps():
    windows = chunk_by_tokens(TEXT, _FakeTokenizer(), max_tokens=8, overlap_ratio=0.25)

    assert windows == [
        "One two three. Four five six seven.",
        "Four five six seven. Eight nine.",
        "Eight nine. Ten eleven twelve thirteen fourteen.",
        "Ten eleven twelve thirteen fourteen. Fifteen sixteen.",
    ]
    # every window respects the token budget
    tok = _FakeTokenizer()
    for w in windows:
        assert len(tok(w)["input_ids"]) <= 8
    # a sentence at a window boundary survives whole in at least one window
    assert any("Four five six seven." in w for w in windows)


def test_chunk_by_tokens_never_infinite_loops_on_oversized_sentence():
    huge = " ".join(f"w{i}" for i in range(20)) + "."
    windows = chunk_by_tokens(huge, _FakeTokenizer(), max_tokens=5, overlap_ratio=0.2)
    assert windows == ["w0 w1 w2 w3 w4"]


def test_chunk_by_tokens_empty_text_returns_no_windows():
    assert chunk_by_tokens("   ", _FakeTokenizer(), max_tokens=8) == []

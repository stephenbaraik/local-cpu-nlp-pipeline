from __future__ import annotations

from dataclasses import dataclass

from pipeline.stages import DocContext, register

# Phase 1 stub: fixed, plausible keyphrases so downstream stages (context,
# report) have real shapes to run against. Real config_keys declared now so
# phase 2's SecureBERT swap does not silently serve phase-1 cached results.
_STUB_KEYWORDS = [
    ["placeholder keyphrase one", 1.0],
    ["placeholder keyphrase two", 0.9],
    ["placeholder keyphrase three", 0.8],
    ["placeholder keyphrase four", 0.7],
    ["placeholder keyphrase five", 0.6],
]


@dataclass
class KeywordsStage:
    name: str = "keywords"
    version: str = "1"
    depends_on: tuple[str, ...] = ("validate",)
    config_keys: tuple[str, ...] = ("BACKEND", "DEVICE", "ONNX_PROVIDER", "ATTN_IMPL")

    def run(self, doc: DocContext) -> dict:
        return {"keywords": [list(k) for k in _STUB_KEYWORDS]}


register(KeywordsStage())

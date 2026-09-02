from __future__ import annotations

from dataclasses import dataclass

from pipeline.stages import DocContext, register

# Phase 1 stub: fixed, plausible summary output. Real config_keys declared
# now, matching the design doc's summarize example exactly.


@dataclass
class SummarizeStage:
    name: str = "summarize"
    version: str = "1"
    depends_on: tuple[str, ...] = ("context",)
    config_keys: tuple[str, ...] = (
        "SUMMARY_MAX_NEW_TOKENS",
        "REDUCED_CONTEXT_CHARS",
        "INJECTION_GUARD",
        "GEMMA_GGUF",
        "DEVICE",
    )

    def run(self, doc: DocContext) -> dict:
        return {
            "summary": "Placeholder one-sentence summary.",
            "output_guard_triggered": False,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "tokens_per_second": 0.0,
        }


register(SummarizeStage())

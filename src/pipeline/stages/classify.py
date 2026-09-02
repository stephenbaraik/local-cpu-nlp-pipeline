from __future__ import annotations

from dataclasses import dataclass

from pipeline.stages import DocContext, register

# Phase 1 stub: fixed, plausible label output. Real config_keys declared now.
_STUB_LABEL_SCORES = {"unknown": 0.5, "informational": 0.3, "advisory": 0.2}


@dataclass
class ClassifyStage:
    name: str = "classify"
    version: str = "1"
    depends_on: tuple[str, ...] = ("keywords",)
    config_keys: tuple[str, ...] = (
        "BACKEND",
        "DEVICE",
        "ONNX_PROVIDER",
        "ATTN_IMPL",
        "ZEROSHOT_MAX_CHUNKS",
    )

    def run(self, doc: DocContext) -> dict:
        return {
            "predicted_label": "unknown",
            "confidence": 0.5,
            "label_scores": dict(_STUB_LABEL_SCORES),
            "n_windows": 1,
        }


register(ClassifyStage())

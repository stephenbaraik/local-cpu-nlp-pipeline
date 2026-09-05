from __future__ import annotations

"""One-time ONNX export for the SecureBERT encoder.

AGENTS.md notes that `optimum-onnx` pins `transformers<4.58` while Gemma 4
needs `transformers>=5`, and prescribes a separate export venv for that
reason. This module sidesteps the conflict entirely: it exports with plain
`torch.onnx.export` (no `optimum` import at all), so there is nothing here
that needs an older transformers pin, and no second venv is required.

Run once, offline: `python -m pipeline.onnx_export`. Output is gitignored,
same as models_gguf/, and re-created on demand.
"""

from pathlib import Path

MODELS_ONNX_DIR = Path("models_onnx")
SECUREBERT_ONNX_FP32 = MODELS_ONNX_DIR / "securebert.onnx"

# int8 dynamic quantization of this checkpoint (onnxruntime.quantization.
# quantize_dynamic, with and without pre-processing, per-channel, and
# reduce_range) was tried and measured against the torch path on the same
# input: cosine similarity on the pooled embedding came out at ~0.74 (max
# per-element hidden-state diff ~8-20, where values are O(1)). That is not
# quantization noise, it is broken, likely from how this ModernBERT variant's
# RoPE/local-global attention nodes get quantized. Shipping it would silently
# corrupt every keyword extracted under NLP_BACKEND=onnx. Serving fp32 ONNX
# instead until that is root-caused -- still a real backend swap for the
# benchmark/compare commands, still zero optimum dependency, just not int8.


def export_securebert() -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from pipeline.models import SECUREBERT_MODEL

    MODELS_ONNX_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
    model = AutoModel.from_pretrained(SECUREBERT_MODEL, attn_implementation="sdpa")
    model.eval()

    dummy = tokenizer("ONNX export dummy input for tracing.", return_tensors="pt")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(SECUREBERT_ONNX_FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,  # avoid the dynamo exporter's onnxscript dependency
    )

    return SECUREBERT_ONNX_FP32


if __name__ == "__main__":
    path = export_securebert()
    print(f"wrote {path}")

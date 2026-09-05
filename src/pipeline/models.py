from __future__ import annotations

import gc
import logging
import time
from typing import Any

from pipeline.config import Config

logger = logging.getLogger("pipeline")

SECUREBERT_MODEL = "cisco-ai/SecureBERT2.0-base"
MODERNBERT_ZEROSHOT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"

_CACHE: dict[str, Any] = {}

# Model load time, per cache key, for the metrics run header -- separate
# from inference time, per CLAUDE.md. Not cleared by release_all(): a run's
# header wants every model this run ever loaded, not just what's currently
# resident. reset_load_times() is what runner.run() calls at the start of
# a run to get a clean header. Only reliable in-process (NLP_WORKERS<=1);
# a worker process's loads happen in that child and never reach this dict.
LOAD_TIMES: dict[str, float] = {}


def reset_load_times() -> None:
    LOAD_TIMES.clear()


def _get_or_create(key: str, factory) -> Any:
    if key not in _CACHE:
        logger.info("loading model: %s", key)
        start = time.monotonic()
        _CACHE[key] = factory()
        LOAD_TIMES[key] = round(time.monotonic() - start, 4)
    return _CACHE[key]


def release_all() -> None:
    """Drops every cached model. The runner calls this between stages so
    only one model is resident at a time (stage-major execution)."""
    if not _CACHE:
        return
    logger.info("releasing models: %s", sorted(_CACHE))
    _CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_securebert(config: Config) -> tuple:
    """Returns the (tokenizer, model) pair for SecureBERT2.0-base. This is
    a ModernBERT masked-LM checkpoint with no trained pooler -- callers must
    do attention-masked mean pooling over last_hidden_state themselves."""
    key = f"securebert:{config.backend}:{config.device}:{config.attn_impl}"
    return _get_or_create(key, lambda: _load_securebert(config))


def _torch_dtype(config: Config):
    """fp16 only makes sense on GPU (CLAUDE.md: TU117 has no bf16, fp16 is
    the real win there via its dedicated FP16 cores). Never fp16 on CPU --
    no speed benefit on this hardware and it just risks NaNs for nothing."""
    import torch

    if config.device == "cuda" and config.dtype == "fp16":
        return torch.float16
    return torch.float32


def _load_securebert(config: Config) -> tuple:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
    model = AutoModel.from_pretrained(
        SECUREBERT_MODEL, attn_implementation=config.attn_impl, dtype=_torch_dtype(config)
    )
    model.to(config.device)
    model.eval()
    return tokenizer, model


def get_zeroshot_classifier(config: Config):
    """Returns a transformers zero-shot-classification pipeline over
    ModernBERT-large-zeroshot-v2.0 (an entailment/not_entailment NLI head)."""
    key = f"modernbert:{config.backend}:{config.device}:{config.attn_impl}"
    return _get_or_create(key, lambda: _load_zeroshot_classifier(config))


def _load_zeroshot_classifier(config: Config):
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=MODERNBERT_ZEROSHOT_MODEL,
        device=config.device,
        model_kwargs={"attn_implementation": config.attn_impl, "dtype": _torch_dtype(config)},
    )


def get_zeroshot_max_tokens(cap: int = 4096) -> int:
    """model.config.max_position_embeddings for the zero-shot checkpoint,
    per CLAUDE.md ("check max_position_embeddings rather than hardcoding
    limits") -- ModernBERT-large-zeroshot-v2.0 supports 8192, so most
    documents need no windowing at all. Capped well below that: the NLI
    pair also carries the hypothesis, and an 8192-token single forward pass
    times out any latency budget for a document that could just be windowed
    instead. Config-only lookup (AutoConfig), not a full model load."""
    return _get_or_create(
        "modernbert_max_tokens",
        lambda: min(_zeroshot_config().max_position_embeddings, cap),
    )


def _zeroshot_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODERNBERT_ZEROSHOT_MODEL)


def get_securebert_onnx(config: Config) -> tuple:
    """Raw onnxruntime.InferenceSession over the fp32 ONNX export (see
    onnx_export.py for why this is fp32, not int8) plus the shared
    tokenizer. No optimum import."""
    key = f"securebert_onnx:{config.onnx_provider}"
    return _get_or_create(key, lambda: _load_securebert_onnx(config))


def _load_securebert_onnx(config: Config) -> tuple:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from pipeline.onnx_export import SECUREBERT_ONNX_FP32

    if not SECUREBERT_ONNX_FP32.exists():
        raise FileNotFoundError(
            f"{SECUREBERT_ONNX_FP32} is missing -- run `python -m pipeline.onnx_export` first"
        )
    tokenizer = AutoTokenizer.from_pretrained(SECUREBERT_MODEL)
    session = ort.InferenceSession(str(SECUREBERT_ONNX_FP32), providers=[config.onnx_provider])
    return tokenizer, session


def get_zeroshot_onnx(config: Config) -> tuple:
    """Raw onnxruntime.InferenceSession over the int8 ONNX build already
    published in the MoritzLaurer repo -- no export or optimum needed here."""
    key = f"modernbert_onnx:{config.onnx_provider}"
    return _get_or_create(key, lambda: _load_zeroshot_onnx(config))


def _load_zeroshot_onnx(config: Config) -> tuple:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_ZEROSHOT_MODEL)
    onnx_path = hf_hub_download(MODERNBERT_ZEROSHOT_MODEL, "onnx/model_int8.onnx")
    session = ort.InferenceSession(onnx_path, providers=[config.onnx_provider])
    return tokenizer, session


def get_gemma_llm(config: Config):
    """Returns a llama_cpp.Llama loaded on the q4_0 QAT GGUF. This is the
    default summarization path; the transformers/AutoProcessor path is not
    implemented here (AGENTS.md: "exists only as a baseline")."""
    key = f"gemma:{config.gemma_gguf}"
    return _get_or_create(key, lambda: _load_gemma_llm(config))


def _load_gemma_llm(config: Config):
    import llama_cpp

    return llama_cpp.Llama(
        model_path=config.gemma_gguf,
        n_ctx=4096,
        n_threads=config.cpu_threads,
        verbose=False,
    )

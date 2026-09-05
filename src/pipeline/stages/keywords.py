from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sklearn.feature_extraction.text import CountVectorizer

from pipeline import models
from pipeline.chunking import chunk_by_tokens
from pipeline.config import Config
from pipeline.segmentation import build_candidates
from pipeline.stages import DocContext, register

TOP_N = 5
NGRAM_RANGE = (1, 3)
MAX_TOKENS = 512  # per-window budget; SecureBERT2.0-base is a ModernBERT-base finetune
WINDOW_OVERLAP_RATIO = 0.125  # ~10-15% overlap so a boundary-spanning phrase survives in one window


class SecureBertEmbedder(BaseEmbedder):
    """KeyBERT backend for SecureBERT2.0-base. Passing a bare HF model to
    KeyBERT does not raise -- select_backend silently downloads MiniLM
    instead. Subclassing BaseEmbedder is what makes SecureBERT actually
    used. This checkpoint has no trained pooler: pooling is attention-masked
    mean over last_hidden_state, then L2 normalised."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.tokenizer, self.model = models.get_securebert(config)
        self.device = next(self.model.parameters()).device

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        short_docs: list[str] = []
        short_idx: list[int] = []
        results: list[np.ndarray | None] = [None] * len(documents)

        for i, doc in enumerate(documents):
            token_count = len(self.tokenizer(doc, add_special_tokens=False)["input_ids"])
            if token_count <= MAX_TOKENS:
                short_docs.append(doc)
                short_idx.append(i)
            else:
                results[i] = self._embed_long(doc)

        if short_docs:
            # one shared batch forward pass, not one string at a time
            for idx, vector in zip(short_idx, self._encode_batch(short_docs)):
                results[idx] = vector

        return np.vstack(results)

    def _embed_long(self, text: str) -> np.ndarray:
        # Sentence-packed windows, not raw token slices -- a raw slice can
        # (and did) cut a sentence in half at the window boundary.
        windows = chunk_by_tokens(text, self.tokenizer, MAX_TOKENS, overlap_ratio=WINDOW_OVERLAP_RATIO)
        pooled = self._encode_batch(windows).mean(axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return pooled.cpu().numpy()


class SecureBertONNXEmbedder(BaseEmbedder):
    """ONNX counterpart of SecureBertEmbedder: same pooling, raw
    onnxruntime.InferenceSession instead of torch. Does not window long
    documents the way the torch path does (truncates at MAX_TOKENS instead)
    -- a simplification, not a parity guarantee, for documents that exceed it."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.tokenizer, self.session = models.get_securebert_onnx(config)

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        # KeyBERT may hand this a numpy array of strings, which the
        # tokenizer's batch path rejects (it only accepts a plain list).
        encoded = self.tokenizer(
            list(documents), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np"
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        (last_hidden_state,) = self.session.run(
            ["last_hidden_state"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        mask = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(1)
        counts = np.clip(mask.sum(1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        return pooled / norms


@dataclass
class KeywordsStage:
    name: str = "keywords"
    version: str = "3"  # bumped: segmentation.py (shared helper) now also splits on clause marks
    depends_on: tuple[str, ...] = ("validate",)
    config_keys: tuple[str, ...] = ("BACKEND", "DEVICE", "DTYPE", "ONNX_PROVIDER", "ATTN_IMPL")

    def run(self, doc: DocContext) -> dict:
        body = doc.payloads["clean"]["body"]

        if doc.config.backend == "onnx":
            embedder = SecureBertONNXEmbedder(doc.config)
        else:
            embedder = SecureBertEmbedder(doc.config)
        kw_model = KeyBERT(model=embedder)
        assert isinstance(kw_model.model, BaseEmbedder), "KeyBERT silently fell back off SecureBERT"

        vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words="english")
        candidates = build_candidates(body, vectorizer)
        if not candidates:
            return {"keywords": []}

        keywords = kw_model.extract_keywords(body, candidates=candidates, top_n=TOP_N)
        return {"keywords": [[phrase, float(score)] for phrase, score in keywords]}


register(KeywordsStage())

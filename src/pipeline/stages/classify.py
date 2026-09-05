from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline import models
from pipeline.chunking import chunk_by_tokens
from pipeline.stages import DocContext, register

# Exact taxonomy from the assessment brief -- graded against a private
# reference set, so these strings must match verbatim. Single-label
# (multi_label=False): one-of-three.
CANDIDATE_LABELS = (
    "Threats, Attacks & Vulnerabilities",
    "Security Technology & Industry",
    "AI Governance & Policy",
)

# Denzel-aligned flags. Independent (multi_label=True): a document can be
# both CII and standards at once, so their pipeline emits independent 1/0
# flags per label rather than picking one.
DENZEL_LABELS = (
    "critical information infrastructure security",
    "cybersecurity standards, law or policy",
    "security of a specific technology",
)
CII_LABEL = DENZEL_LABELS[0]

# Denzel's exact 11-sector list is not confirmed against their code --
# BUILD_GUIDE.md only says "eleven sector names," with no enumeration
# available here. This is a standard critical-infrastructure sector set
# (CISA's 16 sectors trimmed to the commonly cited core 11). Do not publish
# a sector comparison against Denzel until their exact list is confirmed --
# same rule as the E4B-vs-LFM2.5-1.2B model-identity question in CLAUDE.md.
SECTOR_LABELS = (
    "energy",
    "water and wastewater systems",
    "healthcare and public health",
    "financial services",
    "transportation systems",
    "communications",
    "government facilities",
    "food and agriculture",
    "information technology",
    "emergency services",
    "defense industrial base",
)

MULTI_LABEL_THRESHOLD = 0.5


def onnx_zero_shot(
    tokenizer, session, text: str, labels: tuple[str, ...], hypothesis_template: str, max_length: int = 512
) -> dict:
    """Single-label NLI scoring: entailment logit per candidate label
    (id2label[0] == "entailment" for this checkpoint), softmax ACROSS
    labels so scores sum to 1 and compete against each other -- correct
    only when labels are mutually exclusive. No optimum import.

    max_length defaults to 512 for standalone callers (e.g. the benchmark's
    single-sample timing), but classify.py always passes the real model
    max (models.get_zeroshot_max_tokens()) -- the exported graph accepts a
    dynamic sequence length (confirmed by direct test up to 1200 tokens),
    so a caller that chunked windows up to ~4000 tokens and then encoded
    them here with a hardcoded 512 would silently truncate the window right
    back down after already paying to chunk it correctly."""
    entailment_logits = []
    for label in labels:
        hypothesis = hypothesis_template.format(label)
        encoded = tokenizer(text, hypothesis, truncation=True, max_length=max_length, return_tensors="np")
        (logits,) = session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )
        entailment_logits.append(float(logits[0][0]))

    scores = np.exp(entailment_logits)
    scores /= scores.sum()
    order = np.argsort(-scores)
    return {"labels": [labels[i] for i in order], "scores": [float(scores[i]) for i in order]}


def onnx_zero_shot_multi_label(
    tokenizer, session, text: str, labels: tuple[str, ...], hypothesis_template: str, max_length: int = 512
) -> dict:
    """Multi-label NLI scoring: each label's score is softmax over this
    checkpoint's own two classes (entailment vs not_entailment) for that
    label alone -- never normalized against the other labels. That
    independence is what multi_label=True means: labels are not mutually
    exclusive, so one label scoring high must not suppress another's score
    the way single-label's cross-label softmax would. See onnx_zero_shot's
    docstring for why max_length must match the real chunk size."""
    labels_out, scores_out = [], []
    for label in labels:
        hypothesis = hypothesis_template.format(label)
        encoded = tokenizer(text, hypothesis, truncation=True, max_length=max_length, return_tensors="np")
        (logits,) = session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )
        probs = np.exp(logits[0]) / np.exp(logits[0]).sum()
        labels_out.append(label)
        scores_out.append(float(probs[0]))  # index 0 = entailment
    return {"labels": labels_out, "scores": scores_out}


def _subsample_evenly(items: list[str], n: int) -> list[str]:
    """When chunk_by_tokens produces more windows than the cost budget
    allows, keep n of them spread across the document rather than just the
    front -- classification cost scales with label count per window, so
    the window count is a real cost knob (CLAUDE.md), but truncating to
    the front would bias the label toward the introduction."""
    if len(items) <= n:
        return items
    if n <= 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def _aggregate(windows: list[str], labels: tuple[str, ...], classify_fn, uniform_default: bool) -> dict[str, float]:
    """Weighted average (by window word count) of each label's score across
    windows. uniform_default controls the empty-document fallback: True for
    single-label (1/N each -- "maximally uncertain" is the honest answer
    when there is nothing to classify), False for multi-label (0.0 each --
    no evidence means no flag, not "half-fired")."""
    totals = {label: 0.0 for label in labels}
    total_weight = 0
    for text in windows:
        result = classify_fn(text, labels)
        weight = len(text.split())
        for label, score in zip(result["labels"], result["scores"]):
            totals[label] += score * weight
        total_weight += weight

    if total_weight == 0:
        default = 1.0 / len(labels) if uniform_default else 0.0
        return {label: default for label in labels}
    return {label: totals[label] / total_weight for label in labels}


@dataclass
class ClassifyStage:
    name: str = "classify"
    version: str = "5"  # bumped: second (Denzel-aligned) taxonomy + conditional sector pass
    depends_on: tuple[str, ...] = ("keywords",)
    config_keys: tuple[str, ...] = (
        "BACKEND",
        "DEVICE",
        "DTYPE",
        "ONNX_PROVIDER",
        "ATTN_IMPL",
        "ZEROSHOT_MAX_CHUNKS",
        "HYPOTHESIS_TEMPLATE",
        "TAXONOMY",
    )

    def run(self, doc: DocContext) -> dict:
        body = doc.payloads["clean"]["body"]
        template = doc.config.hypothesis_template
        taxonomy = doc.config.taxonomy

        zeroshot_max_tokens = models.get_zeroshot_max_tokens()

        if doc.config.backend == "onnx":
            tokenizer, session = models.get_zeroshot_onnx(doc.config)

            def classify_single(text: str, labels: tuple[str, ...]) -> dict:
                return onnx_zero_shot(tokenizer, session, text, labels, template, zeroshot_max_tokens)

            def classify_multi(text: str, labels: tuple[str, ...]) -> dict:
                return onnx_zero_shot_multi_label(tokenizer, session, text, labels, template, zeroshot_max_tokens)
        else:
            classifier = models.get_zeroshot_classifier(doc.config)
            tokenizer = classifier.tokenizer

            def classify_single(text: str, labels: tuple[str, ...]) -> dict:
                return classifier(text, candidate_labels=list(labels), hypothesis_template=template)

            def classify_multi(text: str, labels: tuple[str, ...]) -> dict:
                return classifier(
                    text, candidate_labels=list(labels), hypothesis_template=template, multi_label=True
                )

        # model.config.max_position_embeddings, not a hardcoded limit
        # (CLAUDE.md) -- ModernBERT handles 8192, so a short document gets
        # exactly one window and pays for one forward pass per label, not
        # zeroshot_max_chunks of them. Budget leaves room for the hypothesis
        # (label text + template) that gets paired onto every window.
        text_budget = zeroshot_max_tokens - 64
        windows = chunk_by_tokens(body, tokenizer, text_budget)
        if len(windows) > doc.config.zeroshot_max_chunks:
            windows = _subsample_evenly(windows, doc.config.zeroshot_max_chunks)

        assessment_scores = _aggregate(windows, CANDIDATE_LABELS, classify_single, uniform_default=True)
        predicted_label = max(assessment_scores, key=assessment_scores.get)

        result: dict = {
            "predicted_label": predicted_label,
            "confidence": assessment_scores[predicted_label],
            "label_scores": assessment_scores,
            "n_windows": len(windows),
        }

        if taxonomy in ("denzel", "both"):
            denzel_scores = _aggregate(windows, DENZEL_LABELS, classify_multi, uniform_default=False)
            denzel_flags = {label: score >= MULTI_LABEL_THRESHOLD for label, score in denzel_scores.items()}
            result["denzel_scores"] = denzel_scores
            result["denzel_flags"] = denzel_flags

            if denzel_flags[CII_LABEL]:
                # Sectors only when CII fires -- Denzel's pipeline does the
                # same, so this saves 11 forward passes per window on every
                # document that isn't CII.
                sector_scores = _aggregate(windows, SECTOR_LABELS, classify_multi, uniform_default=False)
                result["sector_scores"] = sector_scores
                result["sectors"] = [s for s, sc in sector_scores.items() if sc >= MULTI_LABEL_THRESHOLD]
            else:
                result["sector_scores"] = None
                result["sectors"] = None

        return result


register(ClassifyStage())

from __future__ import annotations

import pytest

from pipeline.config import Config
from pipeline.stages.classify import CANDIDATE_LABELS, onnx_zero_shot

pytestmark = pytest.mark.slow

TEXT = (
    "A newly disclosed vulnerability affects widely used firmware update "
    "mechanisms, allowing a local attacker to substitute a malicious image "
    "before the signature check runs."
)


def test_onnx_zero_shot_agrees_with_torch_within_tolerance():
    from pipeline import models

    config = Config(backend="torch")
    torch_classifier = models.get_zeroshot_classifier(config)
    torch_result = torch_classifier(
        TEXT, candidate_labels=list(CANDIDATE_LABELS), hypothesis_template=config.hypothesis_template
    )
    torch_scores = dict(zip(torch_result["labels"], torch_result["scores"]))

    tokenizer, session = models.get_zeroshot_onnx(Config(backend="onnx"))
    onnx_result = onnx_zero_shot(tokenizer, session, TEXT, CANDIDATE_LABELS, config.hypothesis_template)
    onnx_scores = dict(zip(onnx_result["labels"], onnx_result["scores"]))

    models.release_all()

    # int8 quantization moves the exact probabilities; the top label and
    # rough confidence must still agree.
    assert torch_result["labels"][0] == onnx_result["labels"][0]
    for label in CANDIDATE_LABELS:
        assert abs(torch_scores[label] - onnx_scores[label]) < 0.15

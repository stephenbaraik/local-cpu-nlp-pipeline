from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pipeline import report, runner
from pipeline.config import new_run_id

RUNS_DIR = Path("runs")

MODE_ENV = {
    "full": {"NLP_MAX_PAGES": "0"},
    "page1": {"NLP_MAX_PAGES": "1"},
    "guard-on": {"NLP_INJECTION_GUARD": "1"},
    "guard-off": {"NLP_INJECTION_GUARD": "0"},
    "torch": {"NLP_BACKEND": "torch"},
    "onnx": {"NLP_BACKEND": "onnx"},
}


def _run_mode(mode: str) -> dict:
    if mode not in MODE_ENV:
        raise ValueError(f"unknown mode: {mode!r}, known: {sorted(MODE_ENV)}")

    env_overrides = MODE_ENV[mode]
    previous = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        start = time.monotonic()
        run_result = runner.run()
        duration = time.monotonic() - start
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    results = report.build_results(run_id=run_result["run_id"])
    return {"mode": mode, "duration_s": round(duration, 3), "results": results}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _diff_pair(left: dict, right: dict) -> dict:
    left_docs = {d["doc_id"]: d for d in left["results"]["documents"]}
    right_docs = {d["doc_id"]: d for d in right["results"]["documents"]}

    accept_reject_changes = []
    label_changes = []
    keyphrase_jaccard = []

    for doc_id in sorted(set(left_docs) & set(right_docs)):
        l, r = left_docs[doc_id], right_docs[doc_id]

        if l["status"] != r["status"]:
            accept_reject_changes.append({"doc_id": doc_id, "left": l["status"], "right": r["status"]})

        if l.get("predicted_label") != r.get("predicted_label"):
            label_changes.append(
                {"doc_id": doc_id, "left": l.get("predicted_label"), "right": r.get("predicted_label")}
            )

        l_kw = {phrase for phrase, _score in l.get("keywords", [])}
        r_kw = {phrase for phrase, _score in r.get("keywords", [])}
        if l_kw or r_kw:
            keyphrase_jaccard.append({"doc_id": doc_id, "jaccard": round(_jaccard(l_kw, r_kw), 3)})

    return {
        "left_mode": left["mode"],
        "right_mode": right["mode"],
        "timing_delta_s": round(right["duration_s"] - left["duration_s"], 3),
        "accept_reject_changes": accept_reject_changes,
        "label_changes": label_changes,
        "keyphrase_jaccard": keyphrase_jaccard,
    }


def run_compare(modes: list[str]) -> dict:
    if len(modes) < 2:
        raise ValueError("compare needs at least two modes")
    mode_results = [_run_mode(mode) for mode in modes]
    pairs = [_diff_pair(mode_results[i], mode_results[i + 1]) for i in range(len(mode_results) - 1)]
    return {"modes": modes, "pairs": pairs}


def write_compare(modes: list[str], runs_dir: Path = RUNS_DIR) -> Path:
    result = run_compare(modes)
    run_id = new_run_id()
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "compare.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return path

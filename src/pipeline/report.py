from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from pipeline import artifacts
from pipeline.config import Config
from pipeline.stages import ordered_stages

RUNS_DIR = Path("runs")

_PAYLOAD_STAGES = ("extract", "clean", "validate", "keywords", "classify", "context", "summarize")


def _latest_run_id() -> str:
    candidates = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("no runs found under runs/")
    return candidates[-1]


def _load_run_config(run_id: str) -> Config:
    data = json.loads((RUNS_DIR / run_id / "config.json").read_text())
    return Config(**data)


def _load_cached_stages(run_id: str, doc_id: str) -> list[str]:
    path = RUNS_DIR / run_id / "run_summary.json"
    if not path.exists():
        return []
    summary = json.loads(path.read_text())
    return summary.get("doc_cached_stages", {}).get(doc_id, [])


def _doc_entry(run_id: str, doc_id: str, index_entry: dict) -> dict:
    entry: dict = {
        "doc_id": doc_id,
        "file": Path(index_entry["source_path"]).name,
        "pages_total": index_entry.get("page_count", 0),
        "status": index_entry["status"],
        "cached_stages": _load_cached_stages(run_id, doc_id),
    }

    ok_payloads: dict[str, dict] = {}
    for stage_name in _PAYLOAD_STAGES:
        envelope = artifacts.load(doc_id, stage_name)
        if envelope and envelope.get("status") == "ok":
            ok_payloads[stage_name] = envelope["payload"]

    if "extract" in ok_payloads:
        entry["pages_total"] = ok_payloads["extract"]["pages_total"]
        entry["pages_read"] = ok_payloads["extract"]["pages_read"]

    if "validate" in ok_payloads:
        entry["reason"] = ok_payloads["validate"]["reason"]
        entry["signals"] = ok_payloads["validate"]["signals"]

    if "keywords" in ok_payloads:
        entry["keywords"] = ok_payloads["keywords"]["keywords"]

    if "classify" in ok_payloads:
        entry["predicted_label"] = ok_payloads["classify"]["predicted_label"]
        entry["confidence"] = ok_payloads["classify"]["confidence"]
        entry["label_scores"] = ok_payloads["classify"]["label_scores"]
        entry["n_windows"] = ok_payloads["classify"]["n_windows"]

    if "context" in ok_payloads:
        entry["reduced_context_chars"] = ok_payloads["context"]["reduced_context_chars"]
        entry["sentences_kept"] = ok_payloads["context"]["sentences_kept"]

    if "summarize" in ok_payloads:
        s = ok_payloads["summarize"]
        entry["summary"] = s["summary"]
        entry["output_guard_triggered"] = s["output_guard_triggered"]
        entry["prompt_tokens"] = s["prompt_tokens"]
        entry["generated_tokens"] = s["generated_tokens"]
        entry["tokens_per_second"] = s["tokens_per_second"]

    return entry


def build_results(run_id: str | None = None) -> dict:
    run_id = run_id or _latest_run_id()
    config = _load_run_config(run_id)

    index: dict[str, dict] = {}
    if artifacts.INDEX_PATH.exists():
        index = json.loads(artifacts.INDEX_PATH.read_text())

    documents = [_doc_entry(run_id, doc_id, entry) for doc_id, entry in sorted(index.items())]

    run_block = {
        "run_id": run_id,
        # placeholder label until phase 3 defines the real mode/compare taxonomy
        "mode": f"{config.backend}-{config.device}",
        "backend": config.backend,
        "device": config.device,
        "onnx_provider": config.onnx_provider,
        "max_pages": config.max_pages,
        "injection_guard": config.injection_guard,
        "cpu_threads": config.cpu_threads,
        "workers": config.workers,
        "gemma_gguf": config.gemma_gguf,
        "stage_versions": {s.name: s.version for s in ordered_stages()},
    }

    environment = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
    }

    return {"run": run_block, "environment": environment, "documents": documents}


def write_report(run_id: str | None = None) -> Path:
    results = build_results(run_id)
    resolved_run_id = results["run"]["run_id"]
    out_path = RUNS_DIR / resolved_run_id / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    return out_path

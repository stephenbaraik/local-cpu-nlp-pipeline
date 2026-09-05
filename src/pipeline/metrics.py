from __future__ import annotations

"""Three-layer metrics schema (run / documents / stages), per CLAUDE.md.
One JSON file is the source of truth; documents.csv and stages.csv are
derived from it, never hand-built separately.

Sourced entirely from what runner.run() already wrote for a run_id:
config.json, run_summary.json, and the per-(doc, stage) artifact envelopes
(each already carries duration_s, peak_ram_mb, peak_vram_mb -- see
artifacts.save()). This module does no timing of its own; it only reads
and reshapes what the real pipeline run already measured.
"""

import csv
import json
import platform
import sys
from pathlib import Path

from pipeline import artifacts, benchmark
from pipeline.config import Config
from pipeline.stages import ordered_stages

RUNS_DIR = Path("runs")
REPORTS_DIR = Path("reports")
DEVICE_REPORT_DIR = {"cpu": "cpu", "cuda": "gpu"}

# keyword + classify + summarize: model compute only, matching Denzel's
# single_turn_time_sec definition. No extraction, cleaning, parsing, or I/O.
INFERENCE_STAGES = ("keywords", "classify", "summarize")
ALL_STAGES = tuple(s.name for s in ordered_stages())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _run_block(run_id: str, config: Config, run_summary: dict, documents: list[dict]) -> dict:
    total_time = sum(d["total_time_sec"] for d in documents)
    inference_time = sum(d["inference_time_sec"] for d in documents)
    return {
        "run_id": run_id,
        "protocol": config.protocol,
        "environment": benchmark.environment_block(),
        "config": {
            "backend": config.backend,
            "device": config.device,
            "dtype": config.dtype,
            "onnx_provider": config.onnx_provider,
            "attn_impl": config.attn_impl,
            "threads": config.cpu_threads,
            "workers": config.workers,
            "batch_size": config.batch_size,
            "max_pages": config.max_pages,
            "injection_check": config.injection_guard,
            "hypothesis_template": config.hypothesis_template,
            "taxonomy": config.taxonomy,
        },
        "parallel_info": run_summary.get("parallel_info"),
        "model_load_times_s": run_summary.get("model_load_times_s", {}),
        "gpu_samples": run_summary.get("gpu_samples", []),
        "totals": {
            "documents": len(documents),
            "total_time_sec": round(total_time, 4),
            "inference_time_sec": round(inference_time, 4),
        },
        "python": sys.version.split()[0],
        "os": platform.platform(),
    }


def _stage_rows_for_doc(doc_id: str, config: Config) -> list[dict]:
    rows = []
    for stage_name in ALL_STAGES:
        envelope = artifacts.load(doc_id, stage_name)
        if envelope is None:
            continue
        payload = envelope.get("payload", {})
        rows.append(
            {
                "pdf_id": doc_id,
                "stage": stage_name,
                "status": envelope["status"],
                "device": config.device,
                "dtype": config.dtype,
                "batch_size": config.batch_size,
                "wall_time_sec": envelope["duration_s"],
                "compute_time_sec": envelope["duration_s"],  # no transfer/compute split measured yet
                "transfer_time_sec": None,
                "tokens_in": payload.get("prompt_tokens"),
                "tokens_out": payload.get("generated_tokens"),
                "peak_ram_mb": envelope.get("peak_ram_mb"),
                "peak_vram_mb": envelope.get("peak_vram_mb"),
            }
        )
    return rows


def _document_row(doc_id: str, index_entry: dict, stage_rows: list[dict]) -> dict:
    by_stage = {r["stage"]: r for r in stage_rows}

    def duration(stage: str) -> float:
        r = by_stage.get(stage)
        return r["wall_time_sec"] if r else 0.0

    def payload_of(stage: str) -> dict:
        envelope = artifacts.load(doc_id, stage)
        return envelope["payload"] if envelope and envelope["status"] == "ok" else {}

    extract_p = payload_of("extract")
    clean_p = payload_of("clean")
    validate_p = payload_of("validate")
    classify_p = payload_of("classify")
    summarize_p = payload_of("summarize")
    keywords_p = payload_of("keywords")

    raw_chars = sum(len(p) for p in extract_p.get("pages", [])) if extract_p else None
    raw_words = sum(len(p.split()) for p in extract_p.get("pages", [])) if extract_p else None
    clean_chars = len(clean_p.get("body", "")) if clean_p else None
    reduction_pct = (
        round(100 * (1 - clean_chars / raw_chars), 2) if raw_chars and clean_chars is not None else None
    )

    inference_time_sec = sum(duration(s) for s in INFERENCE_STAGES)
    total_time_sec = sum(r["wall_time_sec"] for r in stage_rows)
    peak_ram_values = [r["peak_ram_mb"] for r in stage_rows if r["peak_ram_mb"] is not None]
    peak_vram_values = [r["peak_vram_mb"] for r in stage_rows if r["peak_vram_mb"] is not None]

    return {
        "pdf_id": doc_id,
        "pdf_name": Path(index_entry["source_path"]).name,
        "rejected": index_entry["status"] == "rejected",
        "reject_reason": validate_p.get("reason"),
        "pages_read": extract_p.get("pages_read"),
        "extraction_time_sec": duration("extract"),
        "raw_chars": raw_chars,
        "raw_words": raw_words,
        "clean_chars": clean_chars,
        "reduction_pct": reduction_pct,
        # No dedicated chunker yet (migration item 7): classify's
        # window count stands in as chunk_count; tokens_total is not
        # measured anywhere, left None rather than guessed.
        "chunk_count": classify_p.get("n_windows"),
        "tokens_total": None,
        "keyword_time_sec": duration("keywords"),
        "classify_time_sec": duration("classify"),
        "summarize_time_sec": duration("summarize"),
        "inference_time_sec": round(inference_time_sec, 4),
        "total_time_sec": round(total_time_sec, 4),
        "peak_ram_mb": max(peak_ram_values) if peak_ram_values else None,
        "peak_vram_mb": max(peak_vram_values) if peak_vram_values else None,
        "prompt_tokens": summarize_p.get("prompt_tokens"),
        "generated_tokens": summarize_p.get("generated_tokens"),
        "tokens_per_sec": summarize_p.get("tokens_per_second"),
        "predicted_label": classify_p.get("predicted_label"),
        "confidence": classify_p.get("confidence"),
        "keywords": keywords_p.get("keywords"),
        "summary": summarize_p.get("summary"),
    }


def _latest_run_id(runs_dir: Path) -> str:
    candidates = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no runs found under {runs_dir}/")
    return candidates[-1]


def build_metrics(run_id: str | None, runs_dir: Path = RUNS_DIR) -> dict:
    run_id = run_id or _latest_run_id(runs_dir)
    run_dir = runs_dir / run_id
    config = Config(**_load_json(run_dir / "config.json"))
    run_summary = _load_json(run_dir / "run_summary.json")
    index = _load_json(artifacts.INDEX_PATH)

    stages_all: list[dict] = []
    documents: list[dict] = []
    for doc_id, index_entry in sorted(index.items()):
        stage_rows = _stage_rows_for_doc(doc_id, config)
        stages_all.extend(stage_rows)
        documents.append(_document_row(doc_id, index_entry, stage_rows))

    return {
        "run": _run_block(run_id, config, run_summary, documents),
        "documents": documents,
        "stages": stages_all,
    }


def _write_csv(rows: list[dict], fieldnames: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


DOCUMENT_FIELDS = [
    "pdf_id", "pdf_name", "rejected", "reject_reason", "pages_read",
    "extraction_time_sec", "raw_chars", "raw_words", "clean_chars", "reduction_pct",
    "chunk_count", "tokens_total", "keyword_time_sec", "classify_time_sec",
    "summarize_time_sec", "inference_time_sec", "total_time_sec",
    "peak_ram_mb", "peak_vram_mb", "prompt_tokens", "generated_tokens",
    "tokens_per_sec", "predicted_label", "confidence", "keywords", "summary",
]

STAGE_FIELDS = [
    "pdf_id", "stage", "status", "device", "dtype", "batch_size",
    "wall_time_sec", "compute_time_sec", "transfer_time_sec",
    "tokens_in", "tokens_out", "peak_ram_mb", "peak_vram_mb",
]


def write_metrics(
    run_id: str | None = None, runs_dir: Path = RUNS_DIR, reports_dir: Path = REPORTS_DIR
) -> dict[str, Path]:
    result = build_metrics(run_id, runs_dir)
    run_id = result["run"]["run_id"]
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))

    documents_path = out_dir / "documents.csv"
    _write_csv(result["documents"], DOCUMENT_FIELDS, documents_path)

    stages_path = out_dir / "stages.csv"
    _write_csv(result["stages"], STAGE_FIELDS, stages_path)

    paths = {"json": json_path, "documents_csv": documents_path, "stages_csv": stages_path}

    # reports/cpu/ or reports/gpu/, keyed by device -- a second, flat copy
    # alongside the runs/<run_id>/ source of truth, for pointing an
    # analysis tool (or a person) straight at "every CPU run" without
    # walking every runs/ subdirectory.
    device = result["run"]["config"]["device"]
    device_dir_name = DEVICE_REPORT_DIR.get(device, device)
    device_dir = reports_dir / device_dir_name
    if device_dir.is_dir():
        report_documents_path = device_dir / f"documents_{run_id}.csv"
        _write_csv(result["documents"], DOCUMENT_FIELDS, report_documents_path)
        report_stages_path = device_dir / f"stages_{run_id}.csv"
        _write_csv(result["stages"], STAGE_FIELDS, report_stages_path)
        paths["report_documents_csv"] = report_documents_path
        paths["report_stages_csv"] = report_stages_path

    return paths

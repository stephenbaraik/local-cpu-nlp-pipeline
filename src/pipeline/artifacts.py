from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "_index.json"

# Fixed on purpose: the seven stages plus source are the whole shape of this
# pipeline (see AGENTS.md). An eighth stage is a design decision, not a dict edit.
STAGE_FILE_PREFIX = {
    "source": "00",
    "extract": "01",
    "clean": "02",
    "validate": "03",
    "keywords": "04",
    "classify": "05",
    "context": "06",
    "summarize": "07",
}


def doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _doc_dir(doc_id: str) -> Path:
    return ARTIFACTS_DIR / doc_id


def artifact_path(doc_id: str, stage: str) -> Path:
    prefix = STAGE_FILE_PREFIX[stage]
    return _doc_dir(doc_id) / f"{prefix}_{stage}.json"


def save_source(doc_id: str, pdf_path: Path, pdf_bytes: bytes, page_count: int) -> dict:
    payload = {
        "path": str(pdf_path),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "bytes": len(pdf_bytes),
        "page_count": page_count,
    }
    out = artifact_path(doc_id, "source")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_source(doc_id: str) -> dict | None:
    path = artifact_path(doc_id, "source")
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_code_fingerprint(stage_module_path: Path, stage_version: str) -> str:
    """Hashes the stage module's own source plus its declared version. Does
    not follow imports, so an edit to a shared helper (segmentation.py) will
    not invalidate a stage that calls it -- that is what stage_version is for."""
    source = stage_module_path.read_text(encoding="utf-8")
    blob = f"{stage_version}:{source}"
    return hashlib.sha256(blob.encode()).hexdigest()


def compute_input_fingerprint(source_sha256: str, dependency_identities: list[str]) -> str:
    """`source_sha256` anchors every stage to the source PDF even through a
    chain of unrelated dependencies; `dependency_identities` are the upstream
    artifacts' own identities (see artifact_identity)."""
    parts = [source_sha256, *sorted(dependency_identities)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def artifact_identity(envelope: dict) -> str:
    """The fingerprint downstream stages depend on: this artifact's full
    content identity, combining all three of its own fingerprints."""
    blob = "|".join(
        [envelope["input_fingerprint"], envelope["config_fingerprint"], envelope["code_fingerprint"]]
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def load(doc_id: str, stage: str) -> dict | None:
    path = artifact_path(doc_id, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def is_cached(existing: dict | None, input_fp: str, config_fp: str, code_fp: str) -> bool:
    if existing is None or existing.get("status") == "error":
        return False
    return (
        existing["input_fingerprint"] == input_fp
        and existing["config_fingerprint"] == config_fp
        and existing["code_fingerprint"] == code_fp
    )


def save(
    doc_id: str,
    stage: str,
    stage_version: str,
    payload: dict,
    input_fingerprint: str,
    config_fingerprint: str,
    code_fingerprint: str,
    duration_s: float,
    status: str = "ok",
    error: str | None = None,
    peak_ram_mb: float | None = None,
    peak_vram_mb: float | None = None,
) -> dict:
    envelope: dict[str, Any] = {
        "stage": stage,
        "stage_version": stage_version,
        "doc_id": doc_id,
        "status": status,
        "input_fingerprint": input_fingerprint,
        "config_fingerprint": config_fingerprint,
        "code_fingerprint": code_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
        "payload": payload,
    }
    if error is not None:
        envelope["error"] = error
    path = artifact_path(doc_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
    return envelope


def snapshot_timings(doc_ids: list[str], stage_names: list[str]) -> dict[tuple[str, str], tuple]:
    """(duration_s, peak_ram_mb, peak_vram_mb) per (doc_id, stage) as currently
    on disk. Protocol B calls this right after each timed pass, before the
    next pass overwrites these same files."""
    out: dict[tuple[str, str], tuple] = {}
    for doc_id in doc_ids:
        for stage in stage_names:
            envelope = load(doc_id, stage)
            if envelope is None:
                continue
            out[(doc_id, stage)] = (
                envelope["duration_s"],
                envelope.get("peak_ram_mb"),
                envelope.get("peak_vram_mb"),
            )
    return out


def apply_timing(doc_id: str, stage: str, duration_s: float, peak_ram_mb: float | None, peak_vram_mb: float | None) -> None:
    """Patches only the timing fields of an already-saved envelope in place.
    Protocol B uses this to replace a single pass's numbers with the
    3-run median, without touching the payload or fingerprints."""
    envelope = load(doc_id, stage)
    if envelope is None:
        return
    envelope["duration_s"] = duration_s
    envelope["peak_ram_mb"] = peak_ram_mb
    envelope["peak_vram_mb"] = peak_vram_mb
    artifact_path(doc_id, stage).write_text(json.dumps(envelope, indent=2, sort_keys=True))


def update_index(doc_id: str, source_path: Path, page_count: int, status: str) -> None:
    index: dict[str, dict] = {}
    if INDEX_PATH.exists():
        index = json.loads(INDEX_PATH.read_text())
    index[doc_id] = {"source_path": str(source_path), "page_count": page_count, "status": status}
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True))

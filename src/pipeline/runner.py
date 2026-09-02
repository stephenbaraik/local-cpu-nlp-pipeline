from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from pipeline import artifacts
from pipeline.config import config_fingerprint, load_config, new_run_id, write_config
from pipeline.stages import DocContext, Stage, downstream_closure, ordered_stages

PDFS_DIR = Path("pdfs")
RUNS_DIR = Path("runs")
STAGES_DIR = Path(__file__).parent / "stages"

logger = logging.getLogger("pipeline")


def _setup_logging(run_id: str) -> None:
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    log_dir = RUNS_DIR / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)


@dataclass(slots=True)
class Doc:
    doc_id: str
    path: Path
    bytes: bytes


def _discover_docs() -> list[Doc]:
    docs = []
    for path in sorted(PDFS_DIR.glob("*.pdf")):
        data = path.read_bytes()
        docs.append(Doc(artifacts.doc_id(data), path, data))
    # doc_id order, not filename order, so a partial run resumes predictably
    # regardless of how the source directory happens to sort.
    docs.sort(key=lambda d: d.doc_id)
    return docs


def _stage_module_path(stage_name: str) -> Path:
    return STAGES_DIR / f"{stage_name}.py"


def _select_stages(
    all_stages: list[Stage], only: list[str] | None, through: str | None
) -> tuple[list[Stage], set[str]]:
    """Returns (stages to execute this invocation, stage names to force-recompute).

    --through restricts execution to a prefix of the dependency order.
    --only does not restrict execution: it force-recomputes the named
    stage(s) and everything downstream of them, while upstream and unrelated
    stages still use the normal cache check. This is what the phase 1 gate
    exercises: `--only clean` recomputes clean and everything downstream,
    and leaves extract cached.
    """
    names = [s.name for s in all_stages]
    if through is not None:
        if through not in names:
            raise ValueError(f"unknown stage: {through}")
        cutoff = names.index(through)
        allowed = set(names[: cutoff + 1])
        return [s for s in all_stages if s.name in allowed], set()

    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(f"unknown stage(s): {unknown}")
        return all_stages, downstream_closure(set(only))

    return all_stages, set()


def run(
    force: bool = False,
    only: list[str] | None = None,
    through: str | None = None,
    doc_filter: str | None = None,
) -> dict:
    run_id = new_run_id()
    config = load_config()
    _setup_logging(run_id)
    write_config(config, run_id, runs_dir=RUNS_DIR)

    all_stages = ordered_stages()
    stages_to_run, force_stage_names = _select_stages(all_stages, only, through)

    docs = _discover_docs()
    if doc_filter:
        docs = [d for d in docs if d.doc_id.startswith(doc_filter) or doc_filter in d.path.name]
        if not docs:
            raise ValueError(f"no document matches --doc {doc_filter!r}")

    sources = {
        d.doc_id: artifacts.load_source(d.doc_id) or artifacts.save_source(d.doc_id, d.path, d.bytes, 0)
        for d in docs
    }

    doc_status: dict[str, str] = {d.doc_id: "ok" for d in docs}
    doc_payloads: dict[str, dict[str, dict]] = {d.doc_id: {} for d in docs}
    doc_cached_stages: dict[str, list[str]] = {d.doc_id: [] for d in docs}
    identities: dict[str, dict[str, str]] = {d.doc_id: {} for d in docs}

    stage_summary: dict[str, dict[str, int]] = {}

    for stage in stages_to_run:
        stage_force = force or stage.name in force_stage_names
        code_fp = artifacts.compute_code_fingerprint(_stage_module_path(stage.name), stage.version)
        cfg_fp = config_fingerprint(config, stage.config_keys)

        computed = cached = skipped = errors = 0

        for d in docs:
            if doc_status[d.doc_id] != "ok":
                skipped += 1
                continue

            source = sources[d.doc_id]
            dep_identities = [identities[d.doc_id][dep] for dep in stage.depends_on]
            input_fp = artifacts.compute_input_fingerprint(source["sha256"], dep_identities)

            existing = artifacts.load(d.doc_id, stage.name)
            if not stage_force and artifacts.is_cached(existing, input_fp, cfg_fp, code_fp):
                envelope = existing
                cached += 1
                doc_cached_stages[d.doc_id].append(stage.name)
            else:
                doc_ctx = DocContext(
                    doc_id=d.doc_id,
                    pdf_bytes=d.bytes,
                    config=config,
                    payloads=doc_payloads[d.doc_id],
                )
                start = time.monotonic()
                try:
                    payload = stage.run(doc_ctx)
                except Exception:  # noqa: BLE001 -- one bad document must not kill the run
                    duration = time.monotonic() - start
                    tb = traceback.format_exc()
                    envelope = artifacts.save(
                        d.doc_id,
                        stage.name,
                        stage.version,
                        {},
                        input_fp,
                        cfg_fp,
                        code_fp,
                        duration,
                        status="error",
                        error=tb,
                    )
                    logger.debug("stage %s doc %s failed:\n%s", stage.name, d.doc_id, tb)
                    logger.info("  %s: error in %s", d.doc_id, stage.name)
                    doc_status[d.doc_id] = "error"
                    errors += 1
                    continue
                duration = time.monotonic() - start
                envelope = artifacts.save(
                    d.doc_id, stage.name, stage.version, payload, input_fp, cfg_fp, code_fp, duration
                )
                computed += 1

            doc_payloads[d.doc_id][stage.name] = envelope["payload"]
            identities[d.doc_id][stage.name] = artifacts.artifact_identity(envelope)

            if stage.name == "extract":
                sources[d.doc_id] = artifacts.save_source(
                    d.doc_id, d.path, d.bytes, envelope["payload"]["pages_total"]
                )
            if stage.name == "validate":
                status = envelope["payload"]["status"]
                artifacts.update_index(d.doc_id, d.path, sources[d.doc_id]["page_count"], status)
                if status == "rejected":
                    doc_status[d.doc_id] = "rejected"

        stage_summary[stage.name] = {
            "computed": computed,
            "cached": cached,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info(
            "%-10s computed=%d cached=%d skipped=%d errors=%d",
            stage.name,
            computed,
            cached,
            skipped,
            errors,
        )

    summary_path = RUNS_DIR / run_id / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {"run_id": run_id, "stages": stage_summary, "doc_cached_stages": doc_cached_stages},
            indent=2,
            sort_keys=True,
        )
    )

    return {"run_id": run_id, "stages": stage_summary}

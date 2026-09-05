from __future__ import annotations

import json
import logging
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from pipeline import artifacts, hardware, models
from pipeline.config import Config, config_fingerprint, load_config, new_run_id, write_config
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


def _run_stage_on_docs(stage_name: str, config: Config, doc_specs: list[tuple]) -> list[tuple]:
    """Runs stage.run() over doc_specs, either in-process (called directly
    when NLP_WORKERS<=1) or inside a worker process (submitted to a
    ProcessPoolExecutor when NLP_WORKERS>1). Must stay a module-level
    function so it can be pickled for the worker-process case.

    doc_specs: (doc_id, pdf_bytes, payloads) tuples.
    Returns (doc_id, payload_or_None, error_traceback_or_None, duration_s,
    peak_ram_mb, peak_vram_mb_or_None, is_oom) -- the per-(doc, stage) cell
    of the metrics "stages" table.
    """
    # Re-applied here (not just once in run()) because on Windows/spawn a
    # worker process is a fresh interpreter -- it has none of the parent's
    # already-imported modules, so torch has not been imported yet in this
    # process either, and this still runs before that first import.
    hardware.configure(config.cpu_threads)
    stage = ordered_stages_by_name()[stage_name]
    results = []
    for doc_id, pdf_bytes, payloads in doc_specs:
        doc_ctx = DocContext(doc_id=doc_id, pdf_bytes=pdf_bytes, config=config, payloads=payloads)
        start = time.monotonic()
        error: str | None = None
        is_oom = False
        payload: dict = {}
        with hardware.RSSSampler() as sampler:
            try:
                payload = stage.run(doc_ctx)
            except Exception as exc:  # noqa: BLE001 -- one bad document must not kill the run
                error = traceback.format_exc()
                if hardware.is_cuda_oom(exc):
                    # The OOM boundary is a headline GPU result (CLAUDE.md),
                    # not an error to hide -- record it distinctly and clear
                    # the allocator so the next document gets a clean slate.
                    # No batch-size retry: nothing in this pipeline batches
                    # across documents yet (see Config.batch_size), so
                    # "retry at half" has nothing to halve.
                    is_oom = True
                    hardware.empty_cuda_cache()
            hardware.sync(config.device)  # non-negotiable: see hardware.sync docstring
        duration = time.monotonic() - start
        peak_ram_mb = round(sampler.peak_rss / (1024 * 1024), 1)
        results.append(
            (doc_id, None if error else payload, error, duration, peak_ram_mb, sampler.peak_vram_mb, is_oom)
        )
    return results


def ordered_stages_by_name() -> dict[str, Stage]:
    return {s.name: s for s in ordered_stages()}


def _compute_stage(stage: Stage, config: Config, to_compute: list[tuple]) -> list[tuple]:
    """to_compute: (doc, payloads_snapshot) pairs. Dispatches to worker
    processes when config.workers>1, splitting NLP_CPU_THREADS across them
    (memory multiplies by worker count; this is the lever for that
    trade-off, off by default)."""
    if not to_compute:
        return []

    specs = [(d.doc_id, d.bytes, payloads) for d, payloads in to_compute]

    if config.workers <= 1:
        return _run_stage_on_docs(stage.name, config, specs)

    worker_threads = max(1, config.cpu_threads // config.workers)
    worker_config = replace(config, cpu_threads=worker_threads)
    chunks = [specs[i :: config.workers] for i in range(config.workers)]
    chunks = [c for c in chunks if c]

    results: list[tuple] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_run_stage_on_docs, stage.name, worker_config, chunk) for chunk in chunks]
        for future in futures:
            results.extend(future.result())
    return results


def _apply_side_effects(stage: Stage, d: Doc, envelope: dict, sources: dict, doc_status: dict) -> None:
    if stage.name == "extract":
        sources[d.doc_id] = artifacts.save_source(d.doc_id, d.path, d.bytes, envelope["payload"]["pages_total"])
    if stage.name == "validate":
        status = envelope["payload"]["status"]
        artifacts.update_index(d.doc_id, d.path, sources[d.doc_id]["page_count"], status)
        if status == "rejected":
            doc_status[d.doc_id] = "rejected"


def _execute_pass(
    config: Config,
    stages_to_run: list[Stage],
    force_stage_names: set[str],
    docs: list[Doc],
    sources: dict[str, dict],
    force: bool,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    """One full pass over stages_to_run x docs: cache-check, dispatch to
    _compute_stage, save artifacts. This is the entire single-pass pipeline
    -- Protocol A calls it once, Protocol B calls it 4 times (1 warm-up +
    3 timed) against the same docs/sources, each with force=True."""
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
        to_compute: list[tuple] = []  # (Doc, payloads_snapshot)
        input_fps: dict[str, str] = {}

        for d in docs:
            if doc_status[d.doc_id] != "ok":
                skipped += 1
                continue

            source = sources[d.doc_id]
            dep_identities = [identities[d.doc_id][dep] for dep in stage.depends_on]
            input_fp = artifacts.compute_input_fingerprint(source["sha256"], dep_identities)
            input_fps[d.doc_id] = input_fp

            existing = artifacts.load(d.doc_id, stage.name)
            if not stage_force and artifacts.is_cached(existing, input_fp, cfg_fp, code_fp):
                envelope = existing
                cached += 1
                doc_cached_stages[d.doc_id].append(stage.name)
                doc_payloads[d.doc_id][stage.name] = envelope["payload"]
                identities[d.doc_id][stage.name] = artifacts.artifact_identity(envelope)
                _apply_side_effects(stage, d, envelope, sources, doc_status)
            else:
                to_compute.append((d, dict(doc_payloads[d.doc_id])))

        for doc_id, payload, error, duration, peak_ram_mb, peak_vram_mb, is_oom in _compute_stage(
            stage, config, to_compute
        ):
            d = next(d for d, _ in to_compute if d.doc_id == doc_id)
            input_fp = input_fps[doc_id]

            if error is not None:
                envelope = artifacts.save(
                    doc_id, stage.name, stage.version, {}, input_fp, cfg_fp, code_fp, duration,
                    status="oom" if is_oom else "error", error=error,
                    peak_ram_mb=peak_ram_mb, peak_vram_mb=peak_vram_mb,
                )
                logger.debug("stage %s doc %s failed:\n%s", stage.name, doc_id, error)
                logger.info("  %s: %s in %s", doc_id, "oom" if is_oom else "error", stage.name)
                doc_status[doc_id] = "error"
                errors += 1
                continue

            envelope = artifacts.save(
                doc_id, stage.name, stage.version, payload, input_fp, cfg_fp, code_fp, duration,
                peak_ram_mb=peak_ram_mb, peak_vram_mb=peak_vram_mb,
            )
            computed += 1
            doc_payloads[doc_id][stage.name] = envelope["payload"]
            identities[doc_id][stage.name] = artifacts.artifact_identity(envelope)
            _apply_side_effects(stage, d, envelope, sources, doc_status)

        # stage-major: release this stage's model before the next stage
        # loads its own, so peak memory is the largest single model, not
        # the sum of all of them. LOAD_TIMES survives this -- only the
        # cache clears -- so the run header still has every model's load
        # time even though only one is resident at a time.
        models.release_all()

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

    return stage_summary, doc_cached_stages


def _median_timings(snapshots: list[dict[tuple[str, str], tuple]]) -> dict[tuple[str, str], tuple]:
    keys: set[tuple[str, str]] = set()
    for snap in snapshots:
        keys |= snap.keys()

    medians: dict[tuple[str, str], tuple] = {}
    for key in keys:
        durations = [s[key][0] for s in snapshots if key in s]
        rams = [s[key][1] for s in snapshots if key in s and s[key][1] is not None]
        vrams = [s[key][2] for s in snapshots if key in s and s[key][2] is not None]
        medians[key] = (
            round(statistics.median(durations), 4) if durations else 0.0,
            round(statistics.median(rams), 1) if rams else None,
            round(statistics.median(vrams), 1) if vrams else None,
        )
    return medians


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
    parallel_info = hardware.configure(config.cpu_threads)
    logger.debug("torch parallel_info:\n%s", parallel_info)
    models.reset_load_times()  # only in-process (workers<=1) loads land here; see models.LOAD_TIMES
    gpu_start = hardware.gpu_clock_temp()  # None off this hardware / no nvidia-smi -- never faked

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

    if config.protocol == "B":
        # 1 untimed warm-up (discarded), then 3 timed passes -- all force=True,
        # since Protocol B measures real compute, never a cache hit. The
        # final timed pass's payloads stay on disk; only its timing fields
        # get overwritten below with the 3-run median.
        _execute_pass(config, stages_to_run, force_stage_names, docs, sources, force=True)

        doc_ids = [d.doc_id for d in docs]
        stage_names = [s.name for s in stages_to_run]
        snapshots = []
        stage_summary: dict[str, dict[str, int]] = {}
        doc_cached_stages: dict[str, list[str]] = {}
        for _ in range(3):
            stage_summary, doc_cached_stages = _execute_pass(
                config, stages_to_run, force_stage_names, docs, sources, force=True
            )
            snapshots.append(artifacts.snapshot_timings(doc_ids, stage_names))

        for (doc_id, stage_name), (duration_s, peak_ram_mb, peak_vram_mb) in _median_timings(snapshots).items():
            artifacts.apply_timing(doc_id, stage_name, duration_s, peak_ram_mb, peak_vram_mb)
    else:
        stage_summary, doc_cached_stages = _execute_pass(
            config, stages_to_run, force_stage_names, docs, sources, force=force
        )

    gpu_end = hardware.gpu_clock_temp()

    summary_path = RUNS_DIR / run_id / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "protocol": config.protocol,
                "stages": stage_summary,
                "doc_cached_stages": doc_cached_stages,
                "parallel_info": parallel_info,
                "model_load_times_s": dict(models.LOAD_TIMES),
                # Start/end samples, not continuous -- a background sampler
                # like RSSSampler would need real GPU load to validate
                # against, which this dev box's CPU-only torch build can't
                # produce. Good enough to catch "did clocks drop" though.
                "gpu_samples": [s for s in (gpu_start, gpu_end) if s is not None],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return {"run_id": run_id, "stages": stage_summary}

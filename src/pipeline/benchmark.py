from __future__ import annotations

import importlib.metadata
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil

from pipeline import hardware, models
from pipeline.config import Config, load_config
from pipeline.hardware import RSSSampler
from pipeline.segmentation import build_candidates

# Caching is disabled unconditionally, with no flag. Timing a cache hit is
# the one way to publish a badly wrong number, so nothing in this module
# ever imports pipeline.artifacts or touches the artifact store.

WARMUP_RUNS = 1
TIMED_RUNS = 3

SAMPLE_TEXT = (
    "A newly disclosed vulnerability affects widely used firmware update "
    "mechanisms, allowing a local attacker to substitute a malicious image "
    "before the signature check runs. The vendor confirmed the finding and "
    "shipped a patch within two weeks, though the advisory notes that "
    "devices behind consumer routers without automatic updates could "
    "remain exposed for months or years. Independent testing across a "
    "sample of deployed devices found that fewer than half had applied the "
    "patch a month after release, prompting renewed calls for mandatory "
    "automatic firmware updates on consumer network hardware sold going "
    "forward across the industry."
)


def _timed_runs(fn, device: str, warmup: int = WARMUP_RUNS, timed: int = TIMED_RUNS) -> list[float]:
    for _ in range(warmup):
        fn()
        hardware.sync(device)
    durations = []
    for _ in range(timed):
        start = time.monotonic()
        fn()
        hardware.sync(device)  # non-negotiable: see hardware.sync docstring
        durations.append(time.monotonic() - start)
    return durations


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_block() -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False) or os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {
            pkg: _package_version(pkg)
            for pkg in (
                "transformers",
                "torch",
                "keybert",
                "spacy",
                "pymupdf",
                "llama-cpp-python",
                "onnxruntime",
                "scikit-learn",
            )
        },
    }


def _benchmark_keywords(config: Config) -> dict:
    from keybert import KeyBERT
    from sklearn.feature_extraction.text import CountVectorizer

    with RSSSampler() as sampler:
        init_start = time.monotonic()
        if config.backend == "onnx":
            from pipeline.stages.keywords import SecureBertONNXEmbedder

            embedder = SecureBertONNXEmbedder(config)
        else:
            from pipeline.stages.keywords import SecureBertEmbedder

            embedder = SecureBertEmbedder(config)
        kw_model = KeyBERT(model=embedder)
        init_s = time.monotonic() - init_start

        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english")
        candidates = build_candidates(SAMPLE_TEXT, vectorizer)

        def _run() -> None:
            kw_model.extract_keywords(SAMPLE_TEXT, candidates=candidates, top_n=5)

        durations = _timed_runs(_run, config.device)

    models.release_all()
    return _result_block(init_s, durations, sampler)


def _benchmark_classify(config: Config) -> dict:
    from pipeline.stages.classify import CANDIDATE_LABELS, onnx_zero_shot

    with RSSSampler() as sampler:
        init_start = time.monotonic()
        if config.backend == "onnx":
            tokenizer, session = models.get_zeroshot_onnx(config)

            def _classify() -> None:
                onnx_zero_shot(tokenizer, session, SAMPLE_TEXT, CANDIDATE_LABELS, config.hypothesis_template)
        else:
            classifier = models.get_zeroshot_classifier(config)

            def _classify() -> None:
                classifier(
                    SAMPLE_TEXT,
                    candidate_labels=list(CANDIDATE_LABELS),
                    hypothesis_template=config.hypothesis_template,
                )
        init_s = time.monotonic() - init_start

        durations = _timed_runs(_classify, config.device)

    models.release_all()
    return _result_block(init_s, durations, sampler)


def _benchmark_summarize(config: Config) -> dict:
    from pipeline.stages.summarize import _build_fenced_prompt, _INSTRUCTION

    with RSSSampler() as sampler:
        init_start = time.monotonic()
        llm = models.get_gemma_llm(config)
        init_s = time.monotonic() - init_start

        prompt = _build_fenced_prompt(SAMPLE_TEXT, _INSTRUCTION)

        def _run() -> None:
            # create_chat_completion, not the raw completion API -- see
            # summarize._generate's docstring for why the raw API can
            # return zero tokens on this model at temperature=0.0.
            llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.summary_max_new_tokens,
                temperature=0.0,
            )

        durations = _timed_runs(_run, config.device)

    models.release_all()
    return _result_block(init_s, durations, sampler)


def _result_block(init_s: float, durations: list[float], sampler: "RSSSampler") -> dict:
    return {
        "init_s": round(init_s, 4),
        "inference_s_median": round(statistics.median(durations), 4),
        "inference_s_runs": [round(d, 4) for d in durations],
        "peak_rss_mb": round(sampler.peak_rss / (1024 * 1024), 1),
        "peak_vram_mb": sampler.peak_vram_mb,
    }


def run_benchmark(config: Config | None = None) -> dict:
    config = config or load_config()
    return {
        "config": {"backend": config.backend, "device": config.device},
        "keywords": _benchmark_keywords(config),
        "classify": _benchmark_classify(config),
        "summarize": _benchmark_summarize(config),
        "environment": environment_block(),
    }


def thread_worker_grid(
    thread_counts: tuple[int, ...] = (1, 2),
    worker_counts: tuple[int, ...] = (1, 2),
) -> list[dict]:
    """Peak RSS per (threads, workers) cell over a real pipeline run --
    reports memory, not a throughput line, because at this design's ~4.8GB
    single-model peak, worker count hits memory before it hits cores (see
    AGENTS.md). Which axis (threads vs workers) actually pays off is a
    target-server question the design doc leaves open (core count unknown);
    this grid is the tool to answer it once that hardware exists, not a
    substitute for running it there.
    """
    import os

    from pipeline import runner

    cells = []
    for threads in thread_counts:
        for workers in worker_counts:
            previous = {
                "NLP_CPU_THREADS": os.environ.get("NLP_CPU_THREADS"),
                "NLP_WORKERS": os.environ.get("NLP_WORKERS"),
            }
            os.environ["NLP_CPU_THREADS"] = str(threads)
            os.environ["NLP_WORKERS"] = str(workers)
            try:
                with RSSSampler() as sampler:
                    runner.run(force=True)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            cells.append(
                {"threads": threads, "workers": workers, "peak_rss_mb": round(sampler.peak_rss / (1024 * 1024), 1)}
            )
    return cells


def write_benchmark(run_id: str, config: Config | None = None, runs_dir: Path = Path("runs")) -> Path:
    import json

    result = run_benchmark(config)
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return path

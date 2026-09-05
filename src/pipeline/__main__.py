from __future__ import annotations

import argparse
import os
import sys

from pipeline import benchmark as benchmark_module
from pipeline import compare as compare_module
from pipeline import metrics as metrics_module
from pipeline import report as report_module
from pipeline import runner
from pipeline.config import new_run_id

# --match-denzel reproduces their effective first-page behaviour exactly
# (see BUILD_GUIDE.md Step 4 / Step 6).
MATCH_DENZEL_ENV = {"NLP_MAX_PAGES": "1", "NLP_MIN_CHARS": "100", "NLP_MIN_WORDS": "20"}


def _add_hardware_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--dtype", choices=["fp32", "fp16"], default=None)
    p.add_argument("--threads", type=int, default=None, help="maps to NLP_CPU_THREADS")
    p.add_argument("--batch-size", type=int, default=None, help="maps to NLP_BATCH_SIZE")
    p.add_argument(
        "--protocol",
        choices=["A", "B"],
        default=None,
        help="A: single pass, no warm-up (matches Denzel). B: 1 warm-up + 3 timed, median.",
    )
    p.add_argument(
        "--injection-check",
        action="store_true",
        help="enable the prompt-injection guard on the summarize stage (default: off)",
    )
    p.add_argument(
        "--taxonomy",
        choices=["assessment", "denzel", "both"],
        default=None,
        help="which classification taxonomy(ies) to run (default: both)",
    )
    p.add_argument(
        "--match-denzel",
        action="store_true",
        help="preset: max_pages=1, min_chars=100, min_words=20 (their effective first-page gate)",
    )


def _apply_hardware_flags(args: argparse.Namespace) -> None:
    """Translates CLI flags to NLP_* env vars before Config is loaded.
    --match-denzel applies first so an explicit flag alongside it still wins."""
    if args.match_denzel:
        os.environ.update(MATCH_DENZEL_ENV)
    if args.device is not None:
        os.environ["NLP_DEVICE"] = args.device
    if args.dtype is not None:
        os.environ["NLP_DTYPE"] = args.dtype
    if args.threads is not None:
        os.environ["NLP_CPU_THREADS"] = str(args.threads)
    if args.batch_size is not None:
        os.environ["NLP_BATCH_SIZE"] = str(args.batch_size)
    if args.protocol is not None:
        os.environ["NLP_PROTOCOL"] = args.protocol
    if args.taxonomy is not None:
        os.environ["NLP_TAXONOMY"] = args.taxonomy
    if args.injection_check:
        os.environ["NLP_INJECTION_GUARD"] = "1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run stages over all documents, cached")
    run_p.add_argument("--force", action="store_true", help="ignore cache, recompute everything")
    run_p.add_argument("--only", type=str, default=None, help="comma-separated stage names to force-recompute (plus everything downstream of them)")
    run_p.add_argument("--through", type=str, default=None, help="run only stages up to and including this one")
    run_p.add_argument("--doc", type=str, default=None, help="restrict to one document: doc_id prefix or filename substring")
    _add_hardware_flags(run_p)

    report_p = sub.add_parser("report", help="assemble runs/<run_id>/results.json from artifacts")
    report_p.add_argument("--run-id", type=str, default=None, help="defaults to the most recent run")

    bench_p = sub.add_parser("bench", help="timing benchmark, cache always disabled")
    bench_p.add_argument(
        "--grid", action="store_true", help="thread x worker peak-RSS grid over the real corpus instead"
    )
    _add_hardware_flags(bench_p)

    compare_p = sub.add_parser("compare", help="diff tables across flag combinations")
    compare_p.add_argument("--modes", type=str, required=True, help="comma-separated mode names")

    metrics_p = sub.add_parser(
        "metrics", help="write the three-layer run/documents/stages metrics.json plus two CSVs"
    )
    metrics_p.add_argument("--run-id", type=str, default=None, help="defaults to the most recent run")

    args = parser.parse_args(argv)

    if args.command == "run":
        _apply_hardware_flags(args)
        only = args.only.split(",") if args.only else None
        result = runner.run(force=args.force, only=only, through=args.through, doc_filter=args.doc)
        print(f"run_id: {result['run_id']}")
        for name, counts in result["stages"].items():
            print(f"  {name:10s} {counts}")
        return 0

    if args.command == "report":
        path = report_module.write_report(run_id=args.run_id)
        print(f"wrote {path}")
        return 0

    if args.command == "bench":
        _apply_hardware_flags(args)
        run_id = new_run_id()
        if args.grid:
            import json
            from pathlib import Path

            grid = benchmark_module.thread_worker_grid()
            out_dir = Path("runs") / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "thread_worker_grid.json"
            path.write_text(json.dumps(grid, indent=2))
        else:
            path = benchmark_module.write_benchmark(run_id)
        print(f"wrote {path}")
        return 0

    if args.command == "compare":
        modes = args.modes.split(",")
        path = compare_module.write_compare(modes)
        print(f"wrote {path}")
        return 0

    if args.command == "metrics":
        paths = metrics_module.write_metrics(run_id=args.run_id)
        for name, path in paths.items():
            print(f"wrote {name}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

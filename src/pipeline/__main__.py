from __future__ import annotations

import argparse
import sys

from pipeline import report as report_module
from pipeline import runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run stages over all documents, cached")
    run_p.add_argument("--force", action="store_true", help="ignore cache, recompute everything")
    run_p.add_argument("--only", type=str, default=None, help="comma-separated stage names to force-recompute (plus everything downstream of them)")
    run_p.add_argument("--through", type=str, default=None, help="run only stages up to and including this one")
    run_p.add_argument("--doc", type=str, default=None, help="restrict to one document: doc_id prefix or filename substring")

    report_p = sub.add_parser("report", help="assemble runs/<run_id>/results.json from artifacts")
    report_p.add_argument("--run-id", type=str, default=None, help="defaults to the most recent run")

    args = parser.parse_args(argv)

    if args.command == "run":
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

    return 1


if __name__ == "__main__":
    sys.exit(main())

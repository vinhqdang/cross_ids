"""Command line interface: crossids download | prepare | run | report | synth."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import datasets
from .experiment import load_config, run_experiments
from .methods import METHOD_NAMES
from .report import write_report
from .schema import SOURCES


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data", help="root of raw and processed data (default: data)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crossids", description="Cross-dataset intrusion detection benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download raw dataset files into data/raw/{key}/")
    p.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    _add_common(p)

    p = sub.add_parser("prepare", help="harmonise, sample and cache datasets as parquet")
    p.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-rows", type=int, default=None)
    _add_common(p)

    p = sub.add_parser("run", help="run experiments from a config (resumable)")
    p.add_argument("--config", required=True)
    p.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=None)
    p.add_argument("--methods", nargs="+", choices=list(METHOD_NAMES), default=None)
    p.add_argument("--experiments", nargs="+", choices=["indomain", "pairwise", "lodo"], default=None)
    p.add_argument("--results-dir", default=None)
    _add_common(p)

    p = sub.add_parser("report", help="write summary.csv and summary.md from runs.csv")
    p.add_argument("--results", default="results")

    p = sub.add_parser("synth", help="write synthetic domains as parquet")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-domains", type=int, default=3)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``crossids`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    args = build_parser().parse_args(argv)
    if args.command == "download":
        datasets.download(args.datasets, Path(args.data_dir))
    elif args.command == "prepare":
        cfg = load_config(args.config, {"max_rows": args.max_rows})
        keys = args.datasets or cfg["datasets"]
        for key in keys:
            datasets.prepare(key, Path(args.data_dir), int(cfg["max_rows"]))
    elif args.command == "run":
        cfg = load_config(
            args.config,
            {
                "datasets": args.datasets,
                "methods": args.methods,
                "experiments": args.experiments,
                "results_dir": args.results_dir,
                "data_dir": args.data_dir,
            },
        )
        run_experiments(cfg)
    elif args.command == "report":
        csv_path, md_path = write_report(Path(args.results))
        logging.getLogger("crossids").info("wrote %s and %s", csv_path, md_path)
    elif args.command == "synth":
        from .datasets import synthetic_tables

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in synthetic_tables(args.n_domains, args.n, args.seed).items():
            frame.to_parquet(out / f"{name}.parquet", index=False)
            logging.getLogger("crossids").info("wrote %s (%d rows)", out / f"{name}.parquet", len(frame))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

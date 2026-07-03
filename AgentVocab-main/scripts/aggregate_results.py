#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.evaluation import aggregate_overall_results
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate tidy evaluation results into overall metrics.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX with benchmark, model, and metric columns.")
    parser.add_argument("--output", required=True, help="Output CSV/XLSX.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Aggregating results %s -> %s", args.input, args.output)
    result = aggregate_overall_results(args.input, args.output)
    print(result)
    LOGGER.info("Wrote aggregated results to %s", args.output)


if __name__ == "__main__":
    main()

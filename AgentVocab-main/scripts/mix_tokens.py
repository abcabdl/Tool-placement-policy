#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.io import get_logger
from agentvocab.reachability import merge_token_files


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple token-list JSON files with order-preserving deduplication.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input token-list JSON files.")
    parser.add_argument("--output", required=True, help="Output merged token-list JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_token_files(args.inputs, args.output)
    LOGGER.info("Wrote %d merged tokens to %s", len(merged), args.output)


if __name__ == "__main__":
    main()

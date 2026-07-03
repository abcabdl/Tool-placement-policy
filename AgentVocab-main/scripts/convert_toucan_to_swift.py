#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.data import convert_toucan_parquet_to_swift
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Toucan parquet files into SWIFT agent-format JSONL.")
    parser.add_argument("--input-dir", required=True, help="Directory containing Toucan parquet files.")
    parser.add_argument("--output", required=True, help="Output SWIFT-format JSONL file.")
    parser.add_argument(
        "--keep-non-final-assistant",
        action="store_true",
        help="Do not require the final message to be a non-empty assistant response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting Toucan -> SWIFT conversion")
    count = convert_toucan_parquet_to_swift(
        args.input_dir,
        args.output,
        require_final_assistant=not args.keep_non_final_assistant,
    )
    LOGGER.info("Finished conversion: wrote %d SWIFT-format records to %s", count, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.data import filter_valid_rendered_data
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rendered actual-content records and recover valid SWIFT data.")
    parser.add_argument("--input", required=True, help="Rendered JSONL from render_swift_actual_content.py.")
    parser.add_argument("--output", required=True, help="Output valid SWIFT-format JSONL.")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting valid-record filtering")
    count = filter_valid_rendered_data(args.input, args.output, max_input_tokens=args.max_input_tokens)
    LOGGER.info("Wrote %d valid SWIFT records to %s", count, args.output)


if __name__ == "__main__":
    main()

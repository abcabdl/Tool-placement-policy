#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.io import get_logger
from agentvocab.mining import mine_structural_tokens


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine structure-aware tokens from extracted structural spans.")
    parser.add_argument("--input", required=True, help="Text file with one structural span per line.")
    parser.add_argument("--tokenizer", required=True, help="Base tokenizer path.")
    parser.add_argument("--output", required=True, help="Output scored-token JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    parser.add_argument("--min-frequency", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting structural token mining")
    selected = mine_structural_tokens(
        args.input,
        args.tokenizer,
        args.output,
        max_new_tokens=args.max_new_tokens,
        min_frequency=args.min_frequency,
    )
    LOGGER.info("Wrote %d structural tokens to %s", len(selected), args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.data import split_structural_and_content_text
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split rendered actual_input text into structural and content streams for token mining."
    )
    parser.add_argument("--input", required=True, help="Rendered JSONL from render_swift_actual_content.py.")
    parser.add_argument("--structural-output", required=True, help="Output text file for structural spans.")
    parser.add_argument("--content-output", required=True, help="Output text file for content spans.")
    parser.add_argument("--text-field", default="actual_input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting structural/content split")
    n_structural, n_content = split_structural_and_content_text(
        args.input,
        args.structural_output,
        args.content_output,
        text_field=args.text_field,
    )
    LOGGER.info("Wrote %d structural spans to %s", n_structural, args.structural_output)
    LOGGER.info("Wrote %d content spans to %s", n_content, args.content_output)


if __name__ == "__main__":
    main()

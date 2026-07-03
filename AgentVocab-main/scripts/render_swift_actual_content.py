#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.data import render_swift_actual_content
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SWIFT-format agent data into actual model inputs.")
    parser.add_argument("--input", required=True, help="Input SWIFT-format JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL with actual_input and actual_label fields.")
    parser.add_argument("--model", required=True, help="Model/tokenizer path used by SWIFT.")
    parser.add_argument("--agent-template", default="hermes", help="SWIFT agent template name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting SWIFT actual-content rendering")
    render_swift_actual_content(
        args.input,
        args.output,
        args.model,
        agent_template=args.agent_template,
        limit=args.limit,
    )
    LOGGER.info("Finished rendering to %s", args.output)


if __name__ == "__main__":
    main()

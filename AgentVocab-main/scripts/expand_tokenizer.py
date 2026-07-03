#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.io import get_logger
from agentvocab.tokenizer_utils import expand_vocab_with_mean_initialization


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a model tokenizer and initialize new embeddings by subtoken means.")
    parser.add_argument("--base-model", required=True, help="Base model or first-stage checkpoint.")
    parser.add_argument("--tokens", required=True, help="JSON token list to add.")
    parser.add_argument("--output", required=True, help="Output model/tokenizer directory.")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--torch-dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting tokenizer expansion")
    added = expand_vocab_with_mean_initialization(
        args.base_model,
        args.tokens,
        args.output,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    LOGGER.info("Added %d new tokens. Saved to %s", added, args.output)


if __name__ == "__main__":
    main()

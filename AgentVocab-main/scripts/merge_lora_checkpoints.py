#!/usr/bin/env python
from __future__ import annotations

import argparse

from agentvocab.export import merge_lora_checkpoints
from agentvocab.io import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoints into full model directories for evaluation.")
    parser.add_argument("--series-dir", required=True, help="Training series directory containing train-*/checkpoint-* folders.")
    parser.add_argument("--base-model", required=True, help="Base Step0 model used for LoRA training.")
    parser.add_argument("--output-dir", required=True, help="Directory where merged models are written.")
    parser.add_argument("--output-prefix", required=True, help="Merged model name prefix; step number is appended.")
    parser.add_argument("--device-map", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Starting LoRA checkpoint merge")
    outputs = merge_lora_checkpoints(
        args.series_dir,
        args.base_model,
        args.output_dir,
        args.output_prefix,
        device_map=args.device_map,
    )
    LOGGER.info("Merged or found %d output checkpoints", len(outputs))
    for path in outputs:
        LOGGER.info("Output: %s", path)


if __name__ == "__main__":
    main()

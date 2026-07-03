#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from agentvocab.io import get_logger
from agentvocab.reachability import select_reachable_tokens


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter mined tokens by real tokenizer reachability.")
    parser.add_argument("--scored-tokens", required=True, help="Scored candidate-token JSON file.")
    parser.add_argument("--corpus", required=True, help="JSONL corpus with actual_input text.")
    parser.add_argument("--tokenizer", required=True, help="Base tokenizer path.")
    parser.add_argument("--output-dir", required=True, help="Directory for top-N token lists.")
    parser.add_argument("--token-type", required=True, help="Name used in output filenames, e.g. structural or content.")
    parser.add_argument("--targets", type=int, nargs="+", required=True, help="Target token counts.")
    parser.add_argument("--text-field", default="actual_input")
    parser.add_argument("--pool-multiplier", type=int, default=3)
    parser.add_argument("--min-hit-ratio", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting reachability filtering for targets: %s", ", ".join(map(str, args.targets)))
    for target in args.targets:
        output_file = output_dir / f"top_{target}_reachable_{args.token_type}_tokens.json"
        selected = select_reachable_tokens(
            args.scored_tokens,
            args.corpus,
            args.tokenizer,
            output_file,
            target_number=target,
            text_field=args.text_field,
            pool_multiplier=args.pool_multiplier,
            min_hit_ratio=args.min_hit_ratio,
        )
        LOGGER.info("Target %d: selected %d tokens -> %s", target, len(selected), output_file)


if __name__ == "__main__":
    main()

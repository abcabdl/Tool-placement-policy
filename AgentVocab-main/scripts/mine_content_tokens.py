#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from agentvocab.io import get_logger
from agentvocab.mining import mine_content_candidates_tlbpe, score_content_candidates_by_gradient


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine content tokens with TL-BPE candidate mining followed by VEGAD-style gradient ranking."
    )
    parser.add_argument("--input", required=True, help="Text file used for content-token candidate mining.")
    parser.add_argument("--tokenizer", required=True, help="Base tokenizer path.")
    parser.add_argument("--corpus", required=True, help="Rendered JSONL corpus with actual_input text for gradient ranking.")
    parser.add_argument("--model", required=True, help="Model path used for VEGAD-style gradient ranking.")
    parser.add_argument("--output", required=True, help="Output final gradient-ranked content-token JSON file.")
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    parser.add_argument("--min-frequency", type=int, default=10)
    parser.add_argument("--max-subwords", type=int, default=4)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--text-field", default="actual_input")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir or Path(args.output).parent)
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = work_dir / "content_tlbpe_candidates.json"

    LOGGER.info("Stage 1/2: mining TL-BPE content candidates")
    candidates = mine_content_candidates_tlbpe(
        args.input,
        args.tokenizer,
        candidate_file,
        max_new_tokens=args.max_new_tokens,
        min_frequency=args.min_frequency,
        max_subwords=args.max_subwords,
        work_dir=work_dir,
    )
    LOGGER.info("Stage 1/2 complete: wrote %d candidates to %s", len(candidates), candidate_file)

    LOGGER.info("Stage 2/2: VEGAD-style gradient ranking")
    ranked = score_content_candidates_by_gradient(
        candidate_file,
        args.corpus,
        args.model,
        args.output,
        text_field=args.text_field,
        sample_size=args.sample_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
        random_seed=args.random_seed,
    )
    LOGGER.info("Stage 2/2 complete: wrote %d ranked content tokens to %s", len(ranked), args.output)


if __name__ == "__main__":
    main()

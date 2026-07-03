from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import dump_json, get_logger, iter_text_field, load_json


LOGGER = get_logger(__name__)


def normalize_scored_tokens(data: list[Any]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, str):
            tokens.append({"token": item})
        elif isinstance(item, dict) and isinstance(item.get("token"), str):
            tokens.append(dict(item))
    return tokens


def select_reachable_tokens(
    scored_tokens_file: str | Path,
    corpus_jsonl: str | Path,
    tokenizer_path: str,
    output_file: str | Path,
    *,
    target_number: int,
    text_field: str = "actual_input",
    pool_multiplier: int = 3,
    min_hit_ratio: float = 0.001,
) -> list[str]:
    """Select top-ranked tokens that are actually used after tokenizer insertion."""
    from transformers import AutoTokenizer

    scored_tokens = normalize_scored_tokens(load_json(scored_tokens_file))
    pool_size = min(target_number * pool_multiplier, len(scored_tokens))
    candidate_pool = scored_tokens[:pool_size]
    candidate_strings = [item["token"] for item in candidate_pool]
    LOGGER.info(
        "Selecting %d reachable tokens from a pool of %d candidates (%s)",
        target_number,
        pool_size,
        scored_tokens_file,
    )

    LOGGER.info("Loading tokenizer from %s and inserting candidate tokens", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.add_tokens(candidate_strings)
    token_to_id = {token: tokenizer.convert_tokens_to_ids(token) for token in candidate_strings}
    candidate_ids = set(token_to_id.values())

    texts = list(iter_text_field(corpus_jsonl, text_field))
    min_hit_count = math.ceil(len(texts) * min_hit_ratio) if min_hit_ratio > 0 else 1
    LOGGER.info("Checking reachability on %d texts; minimum hit count is %d", len(texts), min_hit_count)

    hit_counter: Counter[int] = Counter()
    for text in tqdm(texts, desc="check token reachability"):
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            if token_id in candidate_ids:
                hit_counter[token_id] += 1

    reachable: list[dict[str, Any]] = []
    for item in candidate_pool:
        token = item["token"]
        hits = hit_counter.get(token_to_id[token], 0)
        if hits > min_hit_count:
            item = dict(item)
            item["actual_hits"] = hits
            reachable.append(item)

    selected = reachable[:target_number]
    final_tokens = [item["token"] for item in selected]
    dump_json(output_file, final_tokens)
    LOGGER.info("Wrote %d reachable tokens to %s", len(final_tokens), output_file)
    return final_tokens


def merge_token_files(input_files: list[str | Path], output_file: str | Path) -> list[str]:
    """Merge token lists while preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    LOGGER.info("Merging %d token files", len(input_files))
    for input_file in input_files:
        for item in load_json(input_file):
            token = item if isinstance(item, str) else item.get("token") if isinstance(item, dict) else None
            if isinstance(token, str) and token not in seen:
                merged.append(token)
                seen.add(token)
    dump_json(output_file, merged)
    LOGGER.info("Wrote %d merged tokens to %s", len(merged), output_file)
    return merged

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from tqdm import tqdm


T = TypeVar("T")


def get_logger(name: str = "agentvocab") -> logging.Logger:
    """Return a lightweight console logger used by command-line scripts."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[AgentVocab] %(message)s",
            stream=sys.stderr,
        )
    return logger


def count_text_lines(path: str | Path) -> int:
    """Count lines in a text file for progress bars."""
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, unit: str = "item") -> Iterator[T]:
    """Wrap an iterable with a consistent tqdm progress bar."""
    yield from tqdm(iterable, desc=desc, total=total, unit=unit)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def read_jsonl_progress(path: str | Path, *, desc: str) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows with a progress bar."""
    yield from progress(read_jsonl(path), desc=desc, total=count_text_lines(path), unit="row")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_token_list(path: str | Path) -> list[str]:
    """Load tokens from either a JSON list of strings or scored token dicts."""
    data = load_json(path)
    tokens: list[str] = []
    for item in data:
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, dict) and isinstance(item.get("token"), str):
            tokens.append(item["token"])
    return tokens


def iter_text_field(path: str | Path, field: str = "actual_input") -> Iterator[str]:
    for row in read_jsonl(path):
        value = row.get(field)
        if isinstance(value, str) and value:
            yield value

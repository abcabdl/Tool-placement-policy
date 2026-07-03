from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from root_dir_path import DATA_ROOT_DIR


STB_ALIASES = {"stb", "stabletoolbench", "toolbench"}


@dataclass(frozen=True)
class TrainingBlock:
    example_index: int
    source_id: Optional[str]
    tid: int
    tool_schema: Dict[str, Any]
    training_data: List[Dict[str, Any]]
    function_tools: List[Dict[str, Any]]


def normalize_training_dataset(dataset: str) -> str:
    ds = str(dataset or "").strip().lower()
    if ds in STB_ALIASES:
        return "stb"
    if ds == "bfcl":
        return "bfcl"
    raise ValueError(f"Unsupported training dataset: {dataset!r}")


def category_slug(category: str) -> str:
    cat = str(category or "").strip()
    if not cat:
        raise ValueError("Category must be a non-empty string.")
    if cat.endswith(".json"):
        raise ValueError(
            f"JSON array files are no longer supported for BFCL/STB training data: {cat!r}. "
            "Use a bare category name or a .jsonl filename."
        )
    if cat.endswith(".jsonl"):
        cat = cat[: -len(".jsonl")]
    if not cat:
        raise ValueError("Category must not resolve to an empty name.")
    return cat


def data_filename(category: str) -> str:
    return f"{category_slug(category)}.jsonl"


def resolve_training_jsonl(dataset: str, category: str) -> str:
    ds = normalize_training_dataset(dataset)
    subdir = "BFCL" if ds == "bfcl" else "STB"
    path = os.path.join(DATA_ROOT_DIR, subdir, data_filename(category))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training JSONL file not found: {path}. "
            f"Expected current {subdir} data at data/{subdir}/{data_filename(category)}."
        )
    return path


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    if not expanded.endswith(".jsonl"):
        raise ValueError(f"Only .jsonl training data is supported, got: {expanded}")
    records: List[Dict[str, Any]] = []
    with open(expanded, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in JSONL file {expanded}:{line_no}")
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Each JSONL line must be an object, got {type(obj).__name__} "
                    f"at {expanded}:{line_no}"
                )
            records.append(obj)
    return records


def pure_tool_schema(function: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(function, dict):
        return {}
    return {k: v for k, v in function.items() if k != "training_data"}


def pure_tool_list(functions: Any) -> List[Dict[str, Any]]:
    if not isinstance(functions, list):
        return []
    return [pure_tool_schema(fn) for fn in functions if isinstance(fn, dict)]


def candidate_tools_from_qa(qa: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = qa.get("candidate_tools") if isinstance(qa, dict) else None
    if not isinstance(tools, list):
        return []
    return [t for t in tools if isinstance(t, dict)]


def iter_training_blocks(records: List[Dict[str, Any]]) -> Iterator[TrainingBlock]:
    for ex_idx, sample in enumerate(records):
        if not isinstance(sample, dict):
            continue
        functions = sample.get("function")
        if not isinstance(functions, list):
            continue
        function_tools = pure_tool_list(functions)
        source_id = (
            str(sample.get("id")).strip() if sample.get("id") is not None else None
        )
        for tid, fn in enumerate(functions):
            if not isinstance(fn, dict):
                continue
            training_data = fn.get("training_data") or []
            if not isinstance(training_data, list):
                continue
            qa_list = [qa for qa in training_data if isinstance(qa, dict)]
            yield TrainingBlock(
                example_index=int(ex_idx),
                source_id=source_id,
                tid=int(tid),
                tool_schema=pure_tool_schema(fn),
                training_data=qa_list,
                function_tools=function_tools,
            )


__all__ = [
    "TrainingBlock",
    "candidate_tools_from_qa",
    "category_slug",
    "data_filename",
    "iter_training_blocks",
    "load_jsonl_records",
    "normalize_training_dataset",
    "pure_tool_list",
    "pure_tool_schema",
    "resolve_training_jsonl",
]

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils import _normalize_question_to_text, parse_python_tool_calls
from dataset.data_io import (
    candidate_tools_from_qa,
    iter_training_blocks,
    load_jsonl_records,
    resolve_training_jsonl,
)


@dataclass
class SoftToolSelectionSample:
    example_index: int
    question: str
    tools: List[Dict[str, Any]]
    answer_idx: Optional[int]
    answer_text: str

def _name_variants(name: str) -> List[str]:
    n = str(name or "").strip()
    if not n:
        return []
    return [n, n.replace(".", "_"), n.replace("_", ".")]

def _dict_to_python_call(func_name: str, params_dict: Any) -> str:
    if not isinstance(params_dict, dict):
        return f"{func_name}({params_dict!r})"
    if not params_dict:
        return f"{func_name}()"
    parts = [f"{k}={v!r}" for k, v in params_dict.items()]
    return f"{func_name}({', '.join(parts)})"

def _infer_answer_name_and_text(qa: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    ans_str = qa.get("answer")
    if isinstance(ans_str, str) and ans_str.strip():
        try:
            calls = parse_python_tool_calls(ans_str)
            if calls:
                fn = list(calls[0].keys())[0]
                return str(fn).strip(), ans_str
        except Exception:
            pass

    action_name = qa.get("action")
    if not isinstance(action_name, str) or not action_name.strip():
        return None, None

    fn_std = action_name.strip()
    input_raw = qa.get("input")
    params_dict: Any = {}
    if isinstance(input_raw, str) and input_raw.strip():
        try:
            parsed = json.loads(input_raw)
            if isinstance(parsed, dict):
                params_dict = parsed
        except Exception:
            params_dict = {}

    return fn_std, f"[{_dict_to_python_call(fn_std, params_dict)}]"

def _find_answer_idx(tools: List[Dict[str, Any]], answer_name: str) -> Optional[int]:
    variants = set(_name_variants(answer_name))
    if not variants:
        return None

    for tid, schema in enumerate(tools):
        if not isinstance(schema, dict):
            continue
        name = str(schema.get("name", "")).strip()
        if name in variants:
            return tid
    return None

def load_soft_tool_selection_samples(
    *,
    dataset: str,
    category: str,
    data_path: Optional[str] = None,
) -> List[SoftToolSelectionSample]:
    if data_path is None:
        data_path = resolve_training_jsonl(dataset, category)

    data = load_jsonl_records(data_path)

    samples: List[SoftToolSelectionSample] = []

    for block in iter_training_blocks(data):
        for qa in block.training_data:
            if not isinstance(qa, dict):
                continue
            q_text = _normalize_question_to_text(qa.get("question"))
            tools = candidate_tools_from_qa(qa)
            if not isinstance(q_text, str) or not q_text.strip():
                continue
            if not tools:
                continue

            answer_name, answer_text = _infer_answer_name_and_text(qa)
            if not answer_name or not answer_text:
                continue

            answer_idx = _find_answer_idx(tools, answer_name)
            if answer_idx is None:
                continue

            samples.append(
                SoftToolSelectionSample(
                    example_index=block.example_index,
                    question=q_text,
                    tools=tools,
                    answer_idx=answer_idx,
                    answer_text=answer_text,
                )
            )

    return samples

__all__ = ["SoftToolSelectionSample", "load_soft_tool_selection_samples"]

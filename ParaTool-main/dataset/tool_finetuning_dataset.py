from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils import _normalize_question_to_text, parse_python_tool_calls

from .data_io import (
    candidate_tools_from_qa,
    iter_training_blocks,
    load_jsonl_records,
    normalize_training_dataset,
    resolve_training_jsonl,
)
from .profiles import build_stb_react_answer_from_qa
from .tool_keys import tool_key_for_schema, tool_key_hash


@dataclass
class ToolFinetuningQA:
    question: str
    tools: List[Dict[str, Any]]
    answer_text: str
    answer_idx: Optional[int]
    answer_tool_key: Optional[str] = None

@dataclass
class ToolFinetuningEntry:
    example_index: int
    tid: int
    tool_schema: Dict[str, Any]
    tool_key: str
    function_tools: List[Dict[str, Any]]
    qas: List[ToolFinetuningQA]
    source_question: Any = None
    source_id: Optional[str] = None

def _name_variants(name: str) -> List[str]:
    n = str(name or "").strip()
    if not n:
        return []
    return [n, n.replace(".", "_"), n.replace("_", ".")]

def _find_answer_idx(tools_seq: List[Dict[str, Any]], tool_name: str) -> Optional[int]:
    variants = set(_name_variants(tool_name))
    if not variants:
        return None
    for tid, schema in enumerate(tools_seq):
        if not isinstance(schema, dict):
            continue
        name = str(schema.get("name", "")).strip()
        if name in variants:
            return tid
    return None

def _find_answer_idx_by_tool_key(
    tools_seq: List[Dict[str, Any]],
    answer_tool_key: str,
    *,
    dataset: str,
) -> Optional[int]:
    key = str(answer_tool_key or "").strip()
    if not key:
        return None
    for tid, schema in enumerate(tools_seq):
        if not isinstance(schema, dict):
            continue
        if tool_key_for_schema(schema, dataset=dataset) == key:
            return tid
    return None

def _extract_bfcl_answer_tool_name(answer_text: str) -> Optional[str]:
    try:
        cleaned = (
            str(answer_text)
            .replace("<CALL_CONT>", "")
            .replace("<CALL_END>", "")
            .strip()
        )
        calls = parse_python_tool_calls(cleaned)
        if not calls:
            return None
        fn = list(calls[0].keys())[0]
        return str(fn).strip() if fn is not None else None
    except Exception:
        return None

def load_tool_finetuning_entries(
    *,
    dataset: str,
    category: str,
    data_path: Optional[str] = None,
) -> List[ToolFinetuningEntry]:
    ds = normalize_training_dataset(dataset)
    if data_path is None:
        data_path = resolve_training_jsonl(ds, category)

    data = load_jsonl_records(data_path)

    entries: List[ToolFinetuningEntry] = []

    for block in iter_training_blocks(data):
        tool_schema = block.tool_schema
        if not tool_schema:
            continue

        tool_key = tool_key_for_schema(tool_schema, dataset=ds)
        if not block.training_data:
            continue

        qas: List[ToolFinetuningQA] = []
        for qa in block.training_data:
            q_text = _normalize_question_to_text(qa.get("question"))
            tools_filtered = candidate_tools_from_qa(qa)
            if not isinstance(q_text, str) or not q_text.strip():
                continue
            if not tools_filtered:
                continue

            answer_text: Optional[str]
            answer_name: Optional[str]
            if ds == "bfcl":
                ans = qa.get("answer")
                if not isinstance(ans, str) or not ans.strip():
                    continue
                answer_text = ans.strip()
                answer_name = _extract_bfcl_answer_tool_name(answer_text)
            else:
                answer_text = build_stb_react_answer_from_qa(qa)
                answer_name = (
                    str(qa.get("action") or "").strip()
                    if qa.get("action") is not None
                    else None
                )

            if not answer_text:
                continue
            if not answer_name:
                continue

            answer_tool_key: Optional[str] = None
            if ds == "bfcl":
                target_key = tool_key_for_schema(tool_schema, dataset=ds)
                answer_idx = _find_answer_idx_by_tool_key(
                    tools_filtered,
                    target_key,
                    dataset=ds,
                )
                if answer_idx is not None:
                    answer_tool_key = target_key
                else:
                    answer_idx = _find_answer_idx(tools_filtered, answer_name)
                    if answer_idx is not None:
                        answer_tool_key = tool_key_for_schema(
                            tools_filtered[answer_idx],
                            dataset=ds,
                        )
            else:
                answer_idx = _find_answer_idx(tools_filtered, answer_name)
                if answer_idx is not None:
                    answer_tool_key = tool_key_for_schema(
                        tools_filtered[answer_idx],
                        dataset=ds,
                    )
            if answer_idx is None:
                continue

            qas.append(
                ToolFinetuningQA(
                    question=q_text,
                    tools=tools_filtered,
                    answer_text=answer_text,
                    answer_idx=answer_idx,
                    answer_tool_key=answer_tool_key,
                )
            )

        if qas:
            entries.append(
                ToolFinetuningEntry(
                    example_index=block.example_index,
                    tid=block.tid,
                    tool_schema=tool_schema,
                    tool_key=tool_key,
                    function_tools=block.function_tools,
                    qas=qas,
                    source_question=None,
                    source_id=block.source_id,
                )
            )

    return entries

@dataclass
class ToolFinetuningIndexRecord:
    tool_key: str
    adapter_dir: str
    status: str
    updated_at: float
    meta: Dict[str, Any]

class ToolFinetuningIndex:
    def __init__(self, path: str, *, data: Optional[Dict[str, Any]] = None) -> None:
        self.path = str(path)
        self.data: Dict[str, Any] = data if isinstance(data, dict) else {"version": 1, "tools": {}}
        self.data.setdefault("version", 1)
        self.data.setdefault("tools", {})

    @classmethod
    def load(cls, path: str) -> "ToolFinetuningIndex":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(path, data=data)
            except Exception:
                return cls(path)
        return cls(path)

    def get(self, tool_key: str) -> Optional[ToolFinetuningIndexRecord]:
        h = tool_key_hash(tool_key)
        bucket = self.data.get("tools", {}).get(h)
        if isinstance(bucket, dict):
            if bucket.get("tool_key") == tool_key:
                return _dict_to_record(bucket)
            return None
        if isinstance(bucket, list):
            for entry in bucket:
                if isinstance(entry, dict) and entry.get("tool_key") == tool_key:
                    return _dict_to_record(entry)
        return None

    def upsert(
        self,
        tool_key: str,
        *,
        adapter_dir: str,
        status: str = "done",
        meta: Optional[Dict[str, Any]] = None,
    ) -> ToolFinetuningIndexRecord:
        record = ToolFinetuningIndexRecord(
            tool_key=str(tool_key),
            adapter_dir=str(adapter_dir),
            status=str(status),
            updated_at=float(time.time()),
            meta=dict(meta or {}),
        )
        h = tool_key_hash(record.tool_key)
        tools = self.data.setdefault("tools", {})
        existing = tools.get(h)
        record_dict = _record_to_dict(record)

        if existing is None:
            tools[h] = record_dict
            return record
        if isinstance(existing, dict):
            if existing.get("tool_key") == record.tool_key:
                existing.update(record_dict)
                tools[h] = existing
                return _dict_to_record(existing)
            tools[h] = [existing, record_dict]
            return record
        if isinstance(existing, list):
            for i, entry in enumerate(existing):
                if isinstance(entry, dict) and entry.get("tool_key") == record.tool_key:
                    entry.update(record_dict)
                    existing[i] = entry
                    tools[h] = existing
                    return _dict_to_record(entry)
            existing.append(record_dict)
            tools[h] = existing
            return record

        tools[h] = record_dict
        return record

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

def _record_to_dict(record: ToolFinetuningIndexRecord) -> Dict[str, Any]:
    return {
        "tool_key": record.tool_key,
        "adapter_dir": record.adapter_dir,
        "status": record.status,
        "updated_at": record.updated_at,
        "meta": record.meta,
    }

def _dict_to_record(d: Dict[str, Any]) -> ToolFinetuningIndexRecord:
    return ToolFinetuningIndexRecord(
        tool_key=str(d.get("tool_key") or ""),
        adapter_dir=str(d.get("adapter_dir") or ""),
        status=str(d.get("status") or ""),
        updated_at=float(d.get("updated_at") or 0.0),
        meta=dict(d.get("meta") or {}),
    )

__all__ = [
    "ToolFinetuningQA",
    "ToolFinetuningEntry",
    "load_tool_finetuning_entries",
    "ToolFinetuningIndex",
    "ToolFinetuningIndexRecord",
]

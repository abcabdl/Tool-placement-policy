from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple


_STB_DATASET_NAMES = {"stb", "stabletoolbench", "toolbench"}


def canonical_schema(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)

def stb_triplet(schema: Dict[str, Any]) -> Tuple[str, str, str]:
    category_name = str(schema.get("category_name") or "").strip()
    tool_name = str(schema.get("tool_name") or "").strip()
    api_name = str(schema.get("api_name") or "").strip()
    return (category_name, tool_name, api_name)

def tool_key_for_schema(schema: Any, *, dataset: str) -> str:
    ds = str(dataset or "").strip().lower() or "unknown"
    if not isinstance(schema, dict):
        return f"{ds}:non_dict:{str(schema)}"

    if ds in _STB_DATASET_NAMES:
        cat, tool, api = stb_triplet(schema)
        if cat or tool or api:
            return f"stb:{cat}|{tool}|{api}"

    return f"{ds}:{canonical_schema(schema)}"

def tool_key_hash(tool_key: str, *, n: int = 32) -> str:
    return hashlib.md5(tool_key.encode("utf-8")).hexdigest()[: int(n)]

__all__ = ["canonical_schema", "stb_triplet", "tool_key_for_schema", "tool_key_hash"]

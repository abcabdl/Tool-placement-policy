from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prompts.common import MULTISTEP_TOOLCALL_HINT
from prompts.with_docs import (
    BFCL_SYSTEM_PROMPT_TEMPLATE as TOOL_CALL_PROMPT_PYTHON_SYSTEM,
    STB_SYSTEM_PROMPT as STB_TOOL_CALL_PROMPT_PYTHON_SYSTEM,
)
from prompts.without_docs import (
    BFCL_SYSTEM_PROMPT_TEMPLATE as TOOL_CALL_PROMPT_PYTHON_WITHOUT_SYSTEM,
    STB_SYSTEM_PROMPT as STB_TOOL_CALL_PROMPT_WITHOUT_SYSTEM,
)
from root_dir_path import ROOT_DIR
from utils import (
    _json_dumps_safe,
    _normalize_question_to_text,
    parse_python_tool_calls,
)


@dataclass
class TrainEncoding:
    input_ids: List[int]
    prompt_len: int
    debug_question: Optional[str]
    debug_answer: Optional[str]

class ToolCallProfile:
    name: str

    def build_messages(
        self,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> List[Dict[str, str]]:
        raise NotImplementedError

    def encode_train_sample(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        gold_answer: Any,
        *,
        multi_step: bool = False,
    ) -> TrainEncoding:
        raise NotImplementedError

    def build_inference_prompt_ids(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> Tuple[List[int], int]:
        raise NotImplementedError

    def parse_inference_output(
        self,
        text: str,
        tools: Any,
    ) -> Any:
        raise NotImplementedError

def _contains_bfcl_call_status_tokens(text: str) -> bool:
    return "<CALL_CONT>" in text or "<CALL_END>" in text

def _bfcl_parse_call_status(text: str) -> Optional[str]:
    if "<CALL_END>" in text:
        return "END"
    if "<CALL_CONT>" in text:
        return "CONT"
    return None

def _bfcl_strip_call_status_tokens(text: str) -> str:
    return text.replace("<CALL_CONT>", "").replace("<CALL_END>", "").strip()

def _coerce_tools_to_list(tools: Any) -> List[Dict[str, Any]]:
    if tools is None:
        return []
    if isinstance(tools, dict):
        return [tools]
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    return []

def _bfcl_format_available_tool_names(tools: Any) -> str:
    schemas = _coerce_tools_to_list(tools)

    sigs: List[str] = []
    for t in schemas:
        name = str(t.get("name", "")).strip()

        if not name:
            continue

        sigs.append(name)

    if not sigs:
        return "(no tool names provided)"

    return "\n".join(f"- {s}" for s in sigs)

def _bfcl_system_with_tools(tools: Any, *, multi_step: bool) -> str:
    tools_json = json.dumps(_coerce_tools_to_list(tools), indent=2, ensure_ascii=False)
    system_content = TOOL_CALL_PROMPT_PYTHON_SYSTEM.format(available_tools=tools_json)
    if multi_step:
        system_content = system_content.rstrip() + "\n\n" + MULTISTEP_TOOLCALL_HINT
    return system_content

def _bfcl_system_without_tools(tools: Any, *, multi_step: bool) -> str:
    system_content = TOOL_CALL_PROMPT_PYTHON_WITHOUT_SYSTEM.format(
        available_tool_names=_bfcl_format_available_tool_names(tools)
    )
    if multi_step:
        system_content = system_content.rstrip() + "\n\n" + MULTISTEP_TOOLCALL_HINT
    return system_content

def _bfcl_build_messages(
    question: Any,
    tools: Any,
    *,
    variant: str,
    multi_step: bool = False,
) -> List[Dict[str, str]]:
    if variant == "with":
        system_content = _bfcl_system_with_tools(tools, multi_step=multi_step)
    elif variant == "without":
        system_content = _bfcl_system_without_tools(tools, multi_step=multi_step)
    else:
        raise ValueError(f"Unknown BFCL variant: {variant}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _normalize_question_to_text(question)},
    ]

def _bfcl_parse_python_tool_calls_robust(text: str) -> List[Dict[str, Any]]:
    cleaned = _bfcl_strip_call_status_tokens(text)
    if not cleaned:
        return []
    try:
        calls = parse_python_tool_calls(cleaned)
        if isinstance(calls, list):
            return calls
    except Exception:
        pass

    all_calls: List[Dict[str, Any]] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sub = parse_python_tool_calls(line)
        except Exception:
            continue
        if isinstance(sub, list):
            all_calls.extend(sub)
    return all_calls

class BfclPythonProfile(ToolCallProfile):
    def __init__(self, variant: str):
        if variant not in {"with", "without"}:
            raise ValueError(f"Unknown BFCL python variant: {variant}")
        self.variant = variant
        self.name = f"bfcl_python_{self.variant}"

    def build_messages(
        self,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> List[Dict[str, str]]:
        return _bfcl_build_messages(
            question,
            tools,
            variant=self.variant,
            multi_step=multi_step,
        )

    def encode_train_sample(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        gold_answer: Any,
        *,
        multi_step: bool = False,
    ) -> TrainEncoding:
        assistant_text = (
            gold_answer
            if isinstance(gold_answer, str)
            else _json_dumps_safe(gold_answer)
        )
        effective_multi_step = bool(multi_step)
        messages = self.build_messages(
            question,
            tools,
            multi_step=effective_multi_step,
        )
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_len = len(input_ids)
        input_ids += tokenizer.encode(assistant_text, add_special_tokens=False)
        return TrainEncoding(
            input_ids=list(input_ids),
            prompt_len=prompt_len,
            debug_question=_normalize_question_to_text(question),
            debug_answer=assistant_text,
        )

    def build_inference_prompt_ids(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> Tuple[List[int], int]:
        messages = self.build_messages(
            question,
            tools,
            multi_step=multi_step,
        )
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return list(input_ids), len(input_ids)

    def parse_inference_output(self, text: str, tools: Any) -> Any:
        status = _bfcl_parse_call_status(text)
        calls = _bfcl_parse_python_tool_calls_robust(text)
        if not calls:
            return {"raw_text": text, "parsed": None, "call_status": status}

        return {
            "raw_text": text,
            "parsed": calls,
            "call_status": status,
        }

def _ensure_stabletoolbench_import_path() -> None:
    stb_root = os.path.join(ROOT_DIR, "StableToolBench")
    if stb_root not in sys.path:
        sys.path.insert(0, stb_root)

def _stb_format_available_tool_names(functions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for fn in functions or []:
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        lines.append(f"- {name.strip()}")

    if not lines:
        return "(no APIs provided)"
    return "\n".join(lines)

def _stb_build_messages(
    question: Any,
    tools: Any,
    *,
    variant: str,
) -> List[Dict[str, str]]:
    _ensure_stabletoolbench_import_path()
    functions = _coerce_tools_to_list(tools)
    variant_norm = (variant or "").lower()

    if variant_norm == "without":
        system_content = (
            STB_TOOL_CALL_PROMPT_WITHOUT_SYSTEM
            + _stb_format_available_tool_names(functions)
        )
    elif variant_norm == "with":
        tools_json = json.dumps(functions, ensure_ascii=False)
        system_content = (
            STB_TOOL_CALL_PROMPT_PYTHON_SYSTEM
            + "\nSpecifically, you have access to the following APIs: "
            + tools_json
        )
    else:
        raise ValueError(f"Unknown STB ReAct variant: {variant}")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": _normalize_question_to_text(question)},
    ]

def _fallback_react_parser(text: str) -> Tuple[str, str, str]:
    raw = str(text or "")

    m = re.search(
        r"Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)\nEnd Action",
        raw,
        flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    action = ""
    action_input = ""
    thought = ""

    m_action = re.search(r"\n?Action:\s*(.+)", raw)
    if m_action:
        action = m_action.group(1).strip().splitlines()[0].strip()

    m_input = re.search(r"Action Input:\s*(.*?)\nEnd Action", raw, flags=re.DOTALL)
    if m_input:
        action_input = m_input.group(1).strip()

    m_thought = re.search(r"\n?Thought:\s*(.*?)\nAction:", raw, flags=re.DOTALL)
    if m_thought:
        thought = m_thought.group(1).strip()

    return thought, action, action_input

class StbReactProfile(ToolCallProfile):
    def __init__(self, variant: str = "with"):
        self.variant = (variant or "with").lower()
        if self.variant not in {"with", "without"}:
            raise ValueError(f"Unknown STB ReAct variant: {variant}")
        self.name = f"stb_react_{self.variant}"

    def build_messages(
        self,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> List[Dict[str, str]]:
        return _stb_build_messages(
            question,
            tools,
            variant=self.variant,
        )

    def encode_train_sample(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        gold_answer: Any,
        *,
        multi_step: bool = False,
    ) -> TrainEncoding:
        messages = self.build_messages(
            question,
            tools,
            multi_step=multi_step,
        )
        input_ids = list(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        )
        prompt_len = len(input_ids)

        assistant_text = str(gold_answer).strip()
        if assistant_text and not assistant_text.endswith("End Action"):
            assistant_text = assistant_text + "\nEnd Action"
        if assistant_text:
            input_ids += tokenizer.encode(assistant_text, add_special_tokens=False)

            eos_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_id, int) and (not input_ids or input_ids[-1] != eos_id):
                input_ids.append(eos_id)

        return TrainEncoding(
            input_ids=list(input_ids),
            prompt_len=prompt_len,
            debug_question=_normalize_question_to_text(question),
            debug_answer=assistant_text,
        )

    def build_inference_prompt_ids(
        self,
        tokenizer: Any,
        question: Any,
        tools: Any,
        *,
        multi_step: bool = False,
    ) -> Tuple[List[int], int]:
        messages = self.build_messages(
            question,
            tools,
            multi_step=multi_step,
        )
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return list(input_ids), len(input_ids)

    def parse_inference_output(self, text: str, tools: Any) -> Any:
        _ensure_stabletoolbench_import_path()
        try:
            from toolbench.inference.utils import react_parser  # type: ignore

            thought, action, action_input = react_parser(text)
        except Exception:
            thought, action, action_input = _fallback_react_parser(text)

        return {
            "raw_text": text,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

def build_stb_react_answer_from_qa(qa: Dict[str, Any]) -> Optional[str]:
    raw = qa.get("raw_react_full") or qa.get("raw_react")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    thought = qa.get("thought")
    action = qa.get("action")
    action_input = qa.get("input")

    action_str = str(action).strip() if action is not None else ""
    if not action_str:
        return None

    thought_str = str(thought).strip() if thought is not None else ""

    if isinstance(action_input, str):
        action_input_json = action_input.strip()
    else:
        try:
            action_input_json = json.dumps(action_input or {}, ensure_ascii=False)
        except Exception:
            action_input_json = _json_dumps_safe(action_input or {})

    return (
        f"Thought: {thought_str}\n"
        f"Action: {action_str}\n"
        f"Action Input: {action_input_json}\n"
        "End Action"
    )

def make_tool_call_profile(
    *,
    dataset: str,
    tool_call_ways: str,
) -> ToolCallProfile:
    ds = (dataset or "").lower()
    ways = (tool_call_ways or "").lower()

    if ds == "bfcl":
        if ways == "with":
            return BfclPythonProfile("with")
        if ways == "without":
            return BfclPythonProfile("without")
        raise ValueError(f"Unknown BFCL tool profile: {tool_call_ways!r}")

    if ds in {"stabletoolbench", "toolbench", "stb"}:
        if ways in {"with", "without"}:
            return StbReactProfile(ways)
        raise ValueError(f"Unknown STB tool profile: {tool_call_ways!r}")

    raise ValueError(
        f"Unknown dataset/tool_call_ways combination: dataset={dataset!r}, "
        f"tool_call_ways={tool_call_ways!r}"
    )

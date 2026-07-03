from __future__ import annotations

import re
import string
import ast
import glob
import itertools
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import count_text_lines, get_logger, progress, read_jsonl, read_jsonl_progress, write_jsonl


IGNORE_INDEX = -100
LOGGER = get_logger(__name__)

STRUCTURAL_KEYWORDS = {
    "type", "properties", "required", "description", "title", "default",
    "patternProperties", "parameters", "enum", "items", "format", "pattern",
    "minimum", "maximum", "minLength", "maxLength", "additionalProperties",
    "$schema", "$id", "$ref", "$comment", "definitions", "allOf", "anyOf",
    "oneOf", "not", "const", "examples", "multipleOf", "exclusiveMinimum",
    "exclusiveMaximum", "minItems", "maxItems", "uniqueItems", "contains",
    "maxProperties", "minProperties", "dependencies", "propertyNames", "if",
    "then", "else", "readOnly", "writeOnly", "additionalItems",
    "contentMediaType", "contentEncoding", "string", "number", "integer",
    "object", "array", "boolean", "null", "true", "false", "tools",
    "tool_call", "tool_response", "function", "name", "arguments",
    "function-name", "args-json-object",
}

STRUCTURAL_PUNCTUATION = set(string.punctuation) | set("｛｝【】（）！：；，“”‘’《》？。、·")
ESCAPED_LINEBREAK_PATTERN = re.compile(r"(?:\\+[rR]\\+[nN]|\\+[rRnN])")


def _build_split_pattern(keywords: set[str]) -> re.Pattern[str]:
    keyword_pattern = "|".join(
        rf"(?<![a-zA-Z]){re.escape(keyword)}(?![a-zA-Z])"
        for keyword in sorted(keywords, key=len, reverse=True)
    )
    return re.compile(
        rf"(<[^>]+>"
        rf"|{keyword_pattern}"
        rf"|(?:[-_]+)?[A-Za-z]+(?:'[A-Za-z]+)*"
        rf"|[^\W\d_]+"
        rf"|[0-9]+(?:\.[0-9]+)?"
        rf"|[ \t]+"
        rf"|.)"
    )


TOKENIZE_PATTERN = _build_split_pattern(STRUCTURAL_KEYWORDS)


def normalize_escaped_linebreaks(text: str) -> str:
    """Turn nested escaped line breaks into real line breaks."""
    previous = None
    while previous != text:
        previous = text
        text = ESCAPED_LINEBREAK_PATTERN.sub("\n", text)
    return text


def classify_fragment(fragment: str, structural_keywords: set[str] = STRUCTURAL_KEYWORDS) -> str:
    """Classify a lexical fragment as structural, content, numeric, or space."""
    if fragment.isspace():
        return "SPACE"
    if fragment.startswith("<") and fragment.endswith(">"):
        tag = fragment[1:-1].lstrip("/")
        return "S_STRONG" if tag in structural_keywords else "N"
    if fragment in structural_keywords:
        return "S_KW"
    if re.match(r"^(?:[-_]+)?[A-Za-z]+(?:'[A-Za-z]+)*$", fragment) or fragment.isalpha():
        return "C"
    if re.match(r"^[0-9]+(?:\.[0-9]+)?$", fragment):
        return "N"
    if all(char in STRUCTURAL_PUNCTUATION for char in fragment):
        return "S_PUNCT"
    return "C"


def safe_decode(tokenizer: Any, ids: list[int] | None) -> str:
    """Decode token ids after removing ignored labels."""
    if not ids:
        return "[EMPTY_OR_NONE]"
    valid_ids = [token_id for token_id in ids if token_id is not None and token_id != IGNORE_INDEX]
    if not valid_ids:
        return "[NO_VALID_TOKENS]"
    return tokenizer.decode(
        valid_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        errors="replace",
    )


def validate_message_record(record: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(record, dict):
        return False, "record is not a JSON object"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "missing non-empty messages"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return False, f"messages[{index}] is not an object"
        if "role" not in message or "content" not in message:
            return False, f"messages[{index}] must contain role and content"
    return True, ""


def render_swift_actual_content(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str,
    *,
    agent_template: str = "hermes",
    limit: int | None = None,
) -> None:
    """Render SWIFT-format agent data into decoded model inputs and labels.

    This mirrors the data-inspection step used in AgentVocab token mining while
    keeping paths and model names configurable.
    """
    from swift.llm import get_model_tokenizer, get_template

    LOGGER.info("Loading SWIFT tokenizer/template from %s", model_path)
    _, tokenizer = get_model_tokenizer(model_path, load_model=False, trust_remote_code=True)
    template = get_template(tokenizer.model_meta.template, tokenizer, agent_template=agent_template)
    template.set_mode("train")
    LOGGER.info("Rendering %s -> %s", input_path, output_path)

    rows: list[dict[str, Any]] = []
    total = count_text_lines(input_path)
    record_iter = read_jsonl(input_path)
    if limit is not None:
        total = min(total, limit)
        record_iter = itertools.islice(record_iter, limit)
    for line_number, record in enumerate(
        progress(record_iter, desc="render SWIFT records", total=total, unit="row"),
        1,
    ):
        valid, error = validate_message_record(record)
        if not valid:
            rows.append(
                {
                    "line_number": line_number,
                    "original_data": record,
                    "actual_input": "[FORMAT_ERROR]",
                    "actual_label": "[FORMAT_ERROR]",
                    "has_none_ids": True,
                    "error_msg": error,
                    "token_stats": {
                        "total_input_tokens": 0,
                        "total_labels_tokens": 0,
                        "valid_labels_tokens": 0,
                        "valid_label_ratio": 0.0,
                    },
                }
            )
            continue

        try:
            encoded = template.encode(record)
            if not isinstance(encoded, dict):
                raise ValueError(f"template.encode returned {type(encoded)}")
            input_ids = encoded.get("input_ids")
            labels = encoded.get("labels")
            total_input_tokens = len(input_ids) if isinstance(input_ids, list) else 0
            total_labels_tokens = len(labels) if isinstance(labels, list) else 0
            valid_labels_tokens = sum(1 for token_id in labels or [] if token_id != IGNORE_INDEX)
            valid_label_ratio = (valid_labels_tokens / total_labels_tokens * 100) if total_labels_tokens else 0.0
            rows.append(
                {
                    "line_number": line_number,
                    "original_data": record,
                    "actual_input": safe_decode(tokenizer, input_ids),
                    "actual_label": safe_decode(tokenizer, labels),
                    "has_none_ids": input_ids is None or labels is None,
                    "error_msg": "",
                    "token_stats": {
                        "total_input_tokens": total_input_tokens,
                        "total_labels_tokens": total_labels_tokens,
                        "valid_labels_tokens": valid_labels_tokens,
                        "valid_label_ratio": round(valid_label_ratio, 2),
                    },
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "line_number": line_number,
                    "original_data": record,
                    "actual_input": "[PROCESS_ERROR]",
                    "actual_label": "[PROCESS_ERROR]",
                    "has_none_ids": True,
                    "error_msg": f"processing error: {exc}",
                    "token_stats": {
                        "total_input_tokens": 0,
                        "total_labels_tokens": 0,
                        "valid_labels_tokens": 0,
                        "valid_label_ratio": 0.0,
                    },
                }
            )

    write_jsonl(output_path, rows)
    LOGGER.info("Wrote %d rendered records to %s", len(rows), output_path)


def _parse_python_dict_str(value: str) -> dict[str, Any] | None:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def convert_toucan_parquet_to_swift(
    input_dir: str | Path,
    output_file: str | Path,
    *,
    require_final_assistant: bool = True,
) -> int:
    """Convert Toucan parquet files into SWIFT agent-format JSONL.

    This parameterized version follows the original conversion script: tools
    are preserved as JSON strings, tool calls are normalized into JSON content,
    and conversations whose final message is not a non-empty assistant response
    can be filtered out.
    """
    import pandas as pd

    parquet_files = sorted(glob.glob(str(Path(input_dir) / "*.parquet")))
    LOGGER.info("Converting %d parquet files from %s", len(parquet_files), input_dir)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for parquet_file in progress(parquet_files, desc="convert parquet files", unit="file"):
        frame = pd.read_parquet(parquet_file)
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"rows: {Path(parquet_file).name}", unit="row", leave=False):
            try:
                tools_str = row.get("tools", "[]")
                if not (tools_str and len(str(tools_str)) > 5):
                    tools_json_str = "[]"
                else:
                    try:
                        json.loads(tools_str)
                        tools_json_str = tools_str
                    except json.JSONDecodeError:
                        tools_json_str = "[]"

                original_messages = json.loads(row["messages"])
                swift_messages: list[dict[str, str]] = []
                for index, message in enumerate(original_messages):
                    role = message.get("role")
                    content = message.get("content")
                    if role == "user":
                        swift_messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        swift_messages.append({"role": "assistant", "content": content})
                    elif role == "tool_call":
                        call_dict = _parse_python_dict_str(content)
                        if call_dict is None:
                            continue
                        arguments_str = call_dict.get("arguments", "{}")
                        try:
                            call_dict["arguments"] = json.loads(arguments_str)
                        except (json.JSONDecodeError, TypeError):
                            call_dict["arguments"] = {}
                        swift_messages.append({"role": "tool_call", "content": json.dumps(call_dict, ensure_ascii=False)})
                    elif role == "tool_response":
                        try:
                            content_obj = json.loads(content)
                            content_str = json.dumps(content_obj, ensure_ascii=False)
                        except (json.JSONDecodeError, TypeError):
                            content_str = str(content)
                        swift_messages.append({"role": "tool_response", "content": content_str})

                if require_final_assistant:
                    if not swift_messages or swift_messages[-1].get("role") != "assistant":
                        continue
                    if not str(swift_messages[-1].get("content", "")).strip():
                        continue

                rows.append({"tools": tools_json_str, "messages": swift_messages})
            except Exception:
                skipped += 1
                continue

    write_jsonl(output_file, rows)
    LOGGER.info("Wrote %d SWIFT records to %s; skipped %d rows", len(rows), output_file, skipped)
    return len(rows)


def filter_valid_rendered_data(input_path: str | Path, output_path: str | Path, *, max_input_tokens: int | None = None) -> int:
    """Filter rendered actual-content records and write original SWIFT records.

    Keeps records with no encoding errors, no None ids, and optionally bounded
    input length. This follows the original `filter_valid_data` step.
    """
    LOGGER.info("Filtering rendered data %s -> %s", input_path, output_path)
    rows: list[dict[str, Any]] = []
    skipped_error = 0
    skipped_length = 0
    for record in read_jsonl_progress(input_path, desc="filter valid records"):
        has_none_ids = record.get("has_none_ids", True)
        error_msg = record.get("error_msg", "")
        token_stats = record.get("token_stats", {})
        total_input_tokens = token_stats.get("total_input_tokens", 0)
        if has_none_ids or error_msg:
            skipped_error += 1
            continue
        if max_input_tokens is not None and total_input_tokens >= max_input_tokens:
            skipped_length += 1
            continue
        original_data = record.get("original_data")
        if isinstance(original_data, dict):
            rows.append(original_data)
    write_jsonl(output_path, rows)
    LOGGER.info(
        "Wrote %d valid records to %s; skipped %d invalid/error records and %d length-filtered records",
        len(rows),
        output_path,
        skipped_error,
        skipped_length,
    )
    return len(rows)


def split_structural_and_content_text(
    input_path: str | Path,
    structural_output: str | Path,
    content_output: str | Path,
    *,
    text_field: str = "actual_input",
    structural_keywords: set[str] = STRUCTURAL_KEYWORDS,
) -> tuple[int, int]:
    """Split rendered agent inputs into structural spans and content spans.

    The structural stream is used by structure-aware token mining. The content
    stream is used by TL-BPE + VEGAD-style content-token mining.
    """
    LOGGER.info("Splitting structural/content streams from %s", input_path)
    tokenize_pattern = _build_split_pattern(structural_keywords)
    structural_spans: list[str] = []
    content_spans: list[str] = []

    for record in read_jsonl_progress(input_path, desc="split rendered records"):
        text = record.get(text_field, "")
        if not isinstance(text, str) or not text:
            continue
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = normalize_escaped_linebreaks(text)

        for single_line in re.split(r"[\r\n]+", text):
            if not single_line.strip():
                continue

            fragments = [fragment for fragment in tokenize_pattern.findall(single_line) if fragment]
            typed = [[fragment, classify_fragment(fragment, structural_keywords)] for fragment in fragments]

            for index, (_, fragment_type) in enumerate(typed):
                if fragment_type != "S_KW":
                    continue
                left_type = next((typed[j][1] for j in range(index - 1, -1, -1) if typed[j][1] != "SPACE"), None)
                right_type = next((typed[j][1] for j in range(index + 1, len(typed)) if typed[j][1] != "SPACE"), None)
                if left_type not in {"S_STRONG", "S_PUNCT"} and right_type not in {"S_STRONG", "S_PUNCT"}:
                    typed[index][1] = "C"

            current_structural: list[str] = []
            current_content: list[str] = []
            pending_space = ""

            def flush_structural() -> None:
                if current_structural:
                    span = "".join(current_structural)
                    if len(span.strip()) >= 2:
                        structural_spans.append(span)
                    current_structural.clear()

            def flush_content() -> None:
                if current_content:
                    span = "".join(current_content).rstrip()
                    if span.strip():
                        content_spans.append(span)
                    current_content.clear()

            for index, (fragment, fragment_type) in enumerate(typed):
                if fragment_type == "SPACE":
                    pending_space += fragment
                    next_type = next((typed[j][1] for j in range(index + 1, len(typed)) if typed[j][1] != "SPACE"), None)
                    if next_type == "C":
                        flush_structural()
                    elif next_type in {"S_STRONG", "S_PUNCT", "S_KW"}:
                        flush_content()
                    else:
                        flush_structural()
                        flush_content()
                        pending_space = ""
                    continue

                if fragment_type == "C":
                    if re.match(r"^[-_]", fragment) and not pending_space:
                        flush_content()
                    flush_structural()
                    if pending_space:
                        current_content.append(pending_space)
                    current_content.append(fragment)
                    pending_space = ""
                elif fragment_type in {"S_STRONG", "S_PUNCT", "S_KW"}:
                    flush_content()
                    if pending_space:
                        current_structural.append(pending_space)
                    current_structural.append(fragment)
                    pending_space = ""
                elif fragment_type == "N":
                    flush_structural()
                    flush_content()
                    pending_space = ""

            flush_structural()
            flush_content()

    structural_path = Path(structural_output)
    content_path = Path(content_output)
    structural_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    structural_path.write_text("\n".join(structural_spans) + ("\n" if structural_spans else ""), encoding="utf-8")
    content_path.write_text("\n".join(content_spans) + ("\n" if content_spans else ""), encoding="utf-8")
    LOGGER.info("Wrote %d structural spans to %s", len(structural_spans), structural_output)
    LOGGER.info("Wrote %d content spans to %s", len(content_spans), content_output)
    return len(structural_spans), len(content_spans)

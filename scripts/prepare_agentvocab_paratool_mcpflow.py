import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ADAPTER_CANDIDATES = [
    "brightdata-mcp_search_engine",
    "brightdata-mcp_scraping_browser_screenshot",
    "brightdata-mcp_scraping_browser_scroll",
    "brightdata-mcp_scraping_browser_navigate",
    "brightdata-mcp_extract",
    "brightdata-mcp_session_stats",
    "brightdata-mcp_scraping_browser_get_text",
    "brightdata-mcp_web_data_linkedin_people_search",
    "brightdata-mcp_scraping_browser_links",
    "brightdata-mcp_scraping_browser_click",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def placeholder(value):
    typ = value_type(value)
    if typ == "str":
        return "<string>"
    if typ == "int":
        return "<integer>"
    if typ == "float":
        return "<number>"
    if typ == "bool":
        return "<boolean>"
    if typ == "list":
        return []
    if typ == "dict":
        return {}
    return "<value>"


def call_template(task):
    args = task.get("gold_arguments") or {}
    templated_args = {
        key: placeholder(args[key])
        for key in sorted(args)
    }
    return {
        "name": task["gold_tool"],
        "arguments": templated_args,
    }


def structural_spans(tasks, tools_by_id, only_server=None):
    spans = []
    for task in tasks:
        if only_server and task.get("server") != only_server:
            continue
        tool_id = task["gold_tool"]
        args = task.get("gold_arguments") or {}
        arg_keys = sorted(args)

        spans.append(tool_id)
        spans.append(tool_id.replace("-", "_"))
        if arg_keys:
            spans.append(f"{tool_id}(" + ",".join(arg_keys) + ")")
            spans.append("arguments:" + ",".join(arg_keys))

        spans.append(compact_json({"name": tool_id, "arguments": args}))
        spans.append(compact_json(call_template(task)))

        tool = tools_by_id.get(tool_id)
        if tool:
            params = tool.get("parameters") or {}
            props = params.get("properties") or {}
            required = params.get("required") or []
            if required:
                spans.append(f"{tool_id}:required:" + ",".join(sorted(required)))
            if props:
                type_sig = ",".join(
                    f"{name}:{(spec or {}).get('type', 'any')}"
                    for name, spec in sorted(props.items())
                )
                spans.append(f"{tool_id}:schema:{type_sig}")
    return spans


def write_structural_spans(path, spans):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for span in spans:
            span = str(span).strip()
            if span:
                f.write(span + "\n")


def render_actual_input(task, tool):
    call = {
        "name": task["gold_tool"],
        "arguments": task.get("gold_arguments") or {},
    }
    tool_card = tool.get("compact_card", "") if tool else ""
    return "\n".join(
        [
            "<|system|>",
            "You are a tool-calling agent. Return JSON only.",
            "<|user|>",
            "Task:",
            task.get("instruction", ""),
            "Available tool:",
            tool_card,
            "<|assistant|>",
            compact_json(call),
        ]
    )


def actual_content_rows(tasks, tools_by_id, only_server=None):
    rows = []
    for task in tasks:
        if only_server and task.get("server") != only_server:
            continue
        tool = tools_by_id.get(task["gold_tool"])
        rows.append(
            {
                "task_id": task["task_id"],
                "server": task.get("server", ""),
                "gold_tool": task["gold_tool"],
                "actual_input": render_actual_input(task, tool),
            }
        )
    return rows


def to_openai_function(tool):
    return {
        "name": tool["tool_id"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters") or {},
    }


def paratool_record(tasks, tools_by_id, candidate_ids, max_tasks, max_per_tool):
    candidate_tools = [
        to_openai_function(tools_by_id[tool_id])
        for tool_id in candidate_ids
        if tool_id in tools_by_id
    ]
    training_by_tool = defaultdict(list)
    selected_tasks = []
    per_tool_counts = defaultdict(int)
    candidate_set = set(candidate_ids)
    for task in tasks:
        gold_tool = task.get("gold_tool")
        if task.get("server") != "Bright Data" or gold_tool not in candidate_set:
            continue
        if max_per_tool and per_tool_counts[gold_tool] >= max_per_tool:
            continue
        selected_tasks.append(task)
        per_tool_counts[gold_tool] += 1
        if max_tasks and len(selected_tasks) >= max_tasks:
            break

    for task in selected_tasks:
        call = {
            "name": task["gold_tool"],
            "arguments": task.get("gold_arguments") or {},
        }
        training_by_tool[task["gold_tool"]].append(
            {
                "question": task["instruction"],
                "candidate_tools": candidate_tools,
                "answer": compact_json(call),
                "gold_tool": task["gold_tool"],
                "gold_arguments": task.get("gold_arguments") or {},
                "source_task_id": task["task_id"],
            }
        )

    functions = []
    for tool_id in candidate_ids:
        if tool_id not in tools_by_id:
            continue
        function = to_openai_function(tools_by_id[tool_id])
        function["training_data"] = training_by_tool.get(tool_id, [])
        functions.append(function)

    return {
        "id": "mcpflow_brightdata_adapter_candidates",
        "source": "MCP-Flow local sample",
        "function": functions,
    }


def resolve_candidate_tools(tasks, tools_by_id, mode, server, explicit_candidates):
    if mode == "explicit":
        return explicit_candidates
    observed = []
    seen = set()
    for task in tasks:
        if mode == "observed-server" and task.get("server") != server:
            continue
        tool_id = task.get("gold_tool")
        if tool_id in tools_by_id and tool_id not in seen:
            seen.add(tool_id)
            observed.append(tool_id)
    if mode == "adapter-default":
        return [tool_id for tool_id in DEFAULT_ADAPTER_CANDIDATES if tool_id in tools_by_id]
    if mode in {"observed-server", "all-observed"}:
        return observed
    raise ValueError(f"unknown candidate mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\data")
    parser.add_argument("--agentvocab-out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main\outputs\data\mcpflow_structural_spans.txt")
    parser.add_argument("--agentvocab-actual-out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main\outputs\data\mcpflow_actual_content.jsonl")
    parser.add_argument("--paratool-out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\ParaTool-main\data\MCPFLOW\brightdata_adapter_candidates.jsonl")
    parser.add_argument("--max-paratool-tasks", type=int, default=300)
    parser.add_argument("--max-paratool-per-tool", type=int, default=30)
    parser.add_argument("--server", default="Bright Data")
    parser.add_argument(
        "--candidate-mode",
        choices=["adapter-default", "observed-server", "all-observed", "explicit"],
        default="adapter-default",
    )
    parser.add_argument("--candidate-tools", nargs="+", default=DEFAULT_ADAPTER_CANDIDATES)
    args = parser.parse_args()

    data_dir = Path(args.data)
    tools = load_jsonl(data_dir / "tools.jsonl")
    tasks = load_jsonl(data_dir / "tasks.jsonl")
    tools_by_id = {tool["tool_id"]: tool for tool in tools}

    spans = structural_spans(tasks, tools_by_id, only_server=args.server)
    write_structural_spans(Path(args.agentvocab_out), spans)
    actual_rows = actual_content_rows(tasks, tools_by_id, only_server=args.server)
    write_jsonl(Path(args.agentvocab_actual_out), actual_rows)

    candidate_tools = resolve_candidate_tools(
        tasks,
        tools_by_id,
        args.candidate_mode,
        args.server,
        args.candidate_tools,
    )
    record = paratool_record(
        tasks,
        tools_by_id,
        candidate_tools,
        args.max_paratool_tasks,
        args.max_paratool_per_tool,
    )
    write_jsonl(Path(args.paratool_out), [record])

    training_count = sum(len(fn.get("training_data", [])) for fn in record["function"])
    print(json.dumps(
        {
            "structural_spans": len(spans),
            "agentvocab_out": args.agentvocab_out,
            "actual_content_rows": len(actual_rows),
            "agentvocab_actual_out": args.agentvocab_actual_out,
            "candidate_mode": args.candidate_mode,
            "paratool_functions": len(record["function"]),
            "paratool_training_examples": training_count,
            "paratool_out": args.paratool_out,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

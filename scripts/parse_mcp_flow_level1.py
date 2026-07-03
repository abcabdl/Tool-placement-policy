import argparse
import json
import re
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_server_name(path):
    return Path(path).stem


def compact_tool_card(tool):
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    required_args = []
    optional_args = []
    for name, spec in props.items():
        typ = spec.get("type", "any")
        desc = spec.get("description") or spec.get("title") or ""
        item = f"{name}: {typ}"
        if desc:
            item += f" ({desc})"
        if name in required:
            required_args.append(item)
        else:
            optional_args.append(item)

    description = " ".join(str(tool.get("description", "")).split())
    if len(description) > 240:
        description = description[:237].rstrip() + "..."

    parts = [
        f"Tool: {tool.get('name', '')}",
        f"Use: {description}" if description else "Use: N/A",
        "Required args: " + ("; ".join(required_args) if required_args else "none"),
        "Optional args: " + ("; ".join(optional_args) if optional_args else "none"),
    ]
    return "\n".join(parts)


def full_tool_card(tool):
    return json.dumps(tool, ensure_ascii=False, indent=2)


def parse_tools(repo_root):
    tools = []
    tool_dirs = [
        repo_root / "data" / "tools" / "smithery",
        repo_root / "data" / "tools" / "deepnlp",
    ]
    seen = set()
    for tool_dir in tool_dirs:
        if not tool_dir.exists():
            continue
        for path in sorted(tool_dir.glob("*.json")):
            try:
                records = load_json(path)
            except Exception as exc:
                print(f"[WARN] skip tool file {path}: {exc}")
                continue
            if isinstance(records, dict):
                records = [records]
            server = normalize_server_name(path)
            marketplace = path.parent.name
            for tool in records:
                if not isinstance(tool, dict) or "name" not in tool:
                    continue
                tool_id = str(tool["name"])
                key = (marketplace, server, tool_id)
                if key in seen:
                    continue
                seen.add(key)
                tools.append(
                    {
                        "tool_id": tool_id,
                        "server": server,
                        "marketplace": marketplace,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "full_card": full_tool_card(tool),
                        "compact_card": compact_tool_card(tool),
                        "source_file": str(path),
                    }
                )
    return tools


def parse_tasks(repo_root):
    tasks = []
    task_dir = repo_root / "data" / "function_call" / "Smithery"
    if not task_dir.exists():
        return tasks
    for path in sorted(task_dir.glob("*.json")):
        try:
            records = load_json(path)
        except Exception as exc:
            print(f"[WARN] skip task file {path}: {exc}")
            continue
        if isinstance(records, dict):
            records = [records]
        server = normalize_server_name(path)
        for idx, row in enumerate(records):
            if not isinstance(row, dict):
                continue
            fc = row.get("function_call") or {}
            gold_tool = row.get("tool") or fc.get("name")
            instruction = row.get("source_instruction", "")
            if not gold_tool or not instruction:
                continue
            tasks.append(
                {
                    "task_id": f"{server}::{idx}",
                    "server": server,
                    "instruction": instruction,
                    "gold_tool": gold_tool,
                    "gold_arguments": fc.get("arguments", {}),
                    "source_file": str(path),
                }
            )
    return tasks


def summarize(tools, tasks):
    tool_ids = {t["tool_id"] for t in tools}
    covered = [t for t in tasks if t["gold_tool"] in tool_ids]
    servers = sorted({t["server"] for t in tasks})
    print(f"tools: {len(tools)}")
    print(f"tasks: {len(tasks)}")
    print(f"tasks with local tool schema: {len(covered)} / {len(tasks)}")
    print(f"task servers: {len(servers)}")
    for s in servers[:20]:
        count = sum(1 for t in tasks if t["server"] == s)
        print(f"  - {s}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"C:\Users\zrz20\Desktop\vscode\Tool\MCP-Flow")
    parser.add_argument("--out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\data")
    args = parser.parse_args()

    repo_root = Path(args.repo)
    out_dir = Path(args.out)
    tools = parse_tools(repo_root)
    tasks = parse_tasks(repo_root)

    write_jsonl(out_dir / "tools.jsonl", tools)
    write_jsonl(out_dir / "tasks.jsonl", tasks)
    write_jsonl(
        out_dir / "tool_cards_compact.jsonl",
        [{"tool_id": t["tool_id"], "card": t["compact_card"], "server": t["server"]} for t in tools],
    )
    write_jsonl(
        out_dir / "tool_cards_full.jsonl",
        [{"tool_id": t["tool_id"], "card": t["full_card"], "server": t["server"]} for t in tools],
    )
    summarize(tools, tasks)


if __name__ == "__main__":
    main()

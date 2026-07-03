import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from run_retrieval_baseline import BM25, build_doc, load_jsonl, write_jsonl


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def chat_completion(messages, model, api_key, base_url, temperature=0.0, timeout=120):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text), None
    except Exception:
        pass
    match = JSON_RE.search(text)
    if not match:
        return None, "no_json"
    try:
        return json.loads(match.group(0)), None
    except Exception as exc:
        return None, f"json_parse_error: {exc}"


def make_prompt(task, tool_cards):
    tools_text = "\n\n".join(tool_cards)
    system = (
        "You are a tool-calling agent. Choose exactly one tool from the provided tools "
        "and fill its arguments. Return JSON only, with this schema: "
        "{\"name\": \"tool_name\", \"arguments\": { ... }}. "
        "Do not explain."
    )
    user = (
        f"Task:\n{task['instruction']}\n\n"
        f"Available tools:\n{tools_text}\n\n"
        "Return the function call JSON now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def placeholder_for_type(type_name):
    return {
        "string": "<string>",
        "int": "<integer>",
        "float": "<number>",
        "bool": "<boolean>",
        "array": [],
        "object": {},
        "null": None,
    }.get(type_name, "<value>")


def build_template_macros(tasks):
    signatures = defaultdict(lambda: defaultdict(int))
    for task in tasks:
        args = task.get("gold_arguments") or {}
        if not isinstance(args, dict):
            continue
        signature = tuple(sorted((key, value_type(value)) for key, value in args.items()))
        signatures[task["gold_tool"]][signature] += 1

    macros = {}
    for tool_id, counts in signatures.items():
        if not counts:
            continue
        signature, count = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
        arguments = {key: placeholder_for_type(type_name) for key, type_name in signature}
        macros[tool_id] = {
            "support": count,
            "template": {
                "name": tool_id,
                "arguments": arguments,
            },
        }
    return macros


def add_template_macro(card, tool_id, template_macros):
    macro = template_macros.get(tool_id)
    if not macro:
        return card
    template = json.dumps(macro["template"], ensure_ascii=False)
    return (
        f"{card}\n"
        f"Common call template observed in traces (support={macro['support']}): {template}"
    )


def group_tools(tools):
    by_server = defaultdict(list)
    by_id = {}
    for tool in tools:
        by_server[tool["server"]].append(tool)
        by_id[tool["tool_id"]] = tool
    return by_server, by_id


def bm25_rank(tools, query, doc_mode):
    docs = [build_doc(t, doc_mode) for t in tools]
    scores = BM25(docs).score(query)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def select_tools(task, tools, by_id, placement, max_candidate_tools, retrieval_topk, template_macros=None, retrieval_doc_mode="compact"):
    template_macros = template_macros or {}
    server_tools = tools
    if placement == "full":
        doc_mode = "full"
        ranked = bm25_rank(server_tools, task["instruction"], "full")
        selected = [server_tools[i] for i in ranked[:max_candidate_tools]]
        # Full/compact candidates are a placement comparison, not a retriever test:
        # include gold if it belongs to this server so the model has a fair chance.
        if task["gold_tool"] in by_id and all(t["tool_id"] != task["gold_tool"] for t in selected):
            selected = selected[:-1] + [by_id[task["gold_tool"]]]
        cards = [t["full_card"] for t in selected]
        return selected, cards
    if placement == "compact":
        ranked = bm25_rank(server_tools, task["instruction"], "compact")
        selected = [server_tools[i] for i in ranked[:max_candidate_tools]]
        if task["gold_tool"] in by_id and all(t["tool_id"] != task["gold_tool"] for t in selected):
            selected = selected[:-1] + [by_id[task["gold_tool"]]]
        cards = [t["compact_card"] for t in selected]
        return selected, cards
    if placement == "retrieval":
        ranked = bm25_rank(server_tools, task["instruction"], retrieval_doc_mode)
        selected = [server_tools[i] for i in ranked[:retrieval_topk]]
        cards = [t["compact_card"] for t in selected]
        return selected, cards
    if placement == "template_macro":
        ranked = bm25_rank(server_tools, task["instruction"], retrieval_doc_mode)
        selected = [server_tools[i] for i in ranked[:retrieval_topk]]
        cards = [
            add_template_macro(t["compact_card"], t["tool_id"], template_macros)
            for t in selected
        ]
        return selected, cards
    raise ValueError(f"unknown placement {placement}")


def exact_args(pred_args, gold_args):
    return pred_args == gold_args


def run(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = args.model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    tools = load_jsonl(data_dir / "tools.jsonl")
    tasks = load_jsonl(data_dir / "tasks.jsonl")
    by_server, by_id = group_tools(tools)
    template_macros = build_template_macros(tasks)

    tasks = [t for t in tasks if t["server"] in by_server and t["gold_tool"] in by_id]
    if args.server:
        tasks = [t for t in tasks if t["server"] == args.server]
    if args.gold_tools:
        gold_tools = set(args.gold_tools)
        tasks = [t for t in tasks if t["gold_tool"] in gold_tools]
    if args.task_id_file:
        task_ids = {
            line.strip()
            for line in Path(args.task_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    placements = args.placements
    rows = []
    for task_idx, task in enumerate(tasks, 1):
        server_tools = by_server[task["server"]]
        for placement in placements:
            selected, cards = select_tools(
                task,
                server_tools,
                by_id,
                placement,
                args.max_candidate_tools,
                args.retrieval_topk,
                template_macros,
                args.retrieval_doc_mode,
            )
            messages = make_prompt(task, cards)
            prompt_chars = sum(len(m["content"]) for m in messages)
            selected_ids = [t["tool_id"] for t in selected]
            started = time.time()
            raw = ""
            error = None
            parsed = None
            try:
                raw = chat_completion(messages, model, api_key, base_url, args.temperature)
                parsed, error = extract_json(raw)
            except Exception as exc:
                error = str(exc)
            latency = time.time() - started
            pred_name = parsed.get("name") if isinstance(parsed, dict) else None
            pred_args = parsed.get("arguments") if isinstance(parsed, dict) else None
            valid_json = isinstance(parsed, dict) and isinstance(pred_name, str) and isinstance(pred_args, dict)
            tool_exact = pred_name == task["gold_tool"]
            args_exact = exact_args(pred_args, task["gold_arguments"]) if valid_json else False
            rows.append(
                {
                    "task_id": task["task_id"],
                    "server": task["server"],
                    "placement": placement,
                    "model": model,
                    "instruction": task["instruction"],
                    "gold_tool": task["gold_tool"],
                    "gold_arguments": task["gold_arguments"],
                    "selected_tools": selected_ids,
                    "gold_in_prompt": task["gold_tool"] in selected_ids,
                    "prompt_chars": prompt_chars,
                    "raw_response": raw,
                    "parsed": parsed,
                    "error": error,
                    "valid_json": valid_json,
                    "tool_exact": tool_exact,
                    "args_exact": args_exact,
                    "both_exact": tool_exact and args_exact,
                    "latency_sec": latency,
                }
            )
            print(
                f"[{task_idx}/{len(tasks)}] {placement} "
                f"gold_in_prompt={task['gold_tool'] in selected_ids} "
                f"tool={tool_exact} args={args_exact} valid={valid_json}"
            )
            if args.sleep:
                time.sleep(args.sleep)

    write_jsonl(out_dir / "placement_predictions.jsonl", rows)
    summary = []
    for placement in placements:
        subset = [r for r in rows if r["placement"] == placement]
        n = len(subset) or 1
        summary.append(
            {
                "placement": placement,
                "runs": len(subset),
                "gold_in_prompt_rate": sum(r["gold_in_prompt"] for r in subset) / n,
                "valid_json_rate": sum(r["valid_json"] for r in subset) / n,
                "tool_accuracy": sum(r["tool_exact"] for r in subset) / n,
                "argument_exact": sum(r["args_exact"] for r in subset) / n,
                "both_exact": sum(r["both_exact"] for r in subset) / n,
                "avg_prompt_chars": sum(r["prompt_chars"] for r in subset) / n,
                "avg_latency_sec": sum(r["latency_sec"] for r in subset) / n,
            }
        )
    with open(out_dir / "placement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\data")
    parser.add_argument("--out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\results")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--placements", nargs="+", default=["full", "compact", "retrieval"])
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--server", default=None)
    parser.add_argument("--gold-tools", nargs="+", default=None)
    parser.add_argument("--task-id-file", default=None)
    parser.add_argument("--max-candidate-tools", type=int, default=30)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-doc-mode", default="compact", choices=["name_desc", "compact", "full", "weighted_compact"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

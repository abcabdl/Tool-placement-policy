import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def json_dumps_compact(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_stats(parameters):
    props = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    enum_count = 0
    nested_objects = 0
    array_count = 0
    property_types = Counter()

    def walk(spec, depth=0):
        nonlocal enum_count, nested_objects, array_count
        if not isinstance(spec, dict):
            return depth
        if "enum" in spec and isinstance(spec["enum"], list):
            enum_count += len(spec["enum"])
        typ = spec.get("type")
        if typ:
            property_types[str(typ)] += 1
        max_depth = depth
        if typ == "object" or "properties" in spec:
            if depth > 0:
                nested_objects += 1
            for child in (spec.get("properties") or {}).values():
                max_depth = max(max_depth, walk(child, depth + 1))
        if typ == "array":
            array_count += 1
            max_depth = max(max_depth, walk(spec.get("items") or {}, depth + 1))
        return max_depth

    max_depth = 0
    for spec in props.values():
        max_depth = max(max_depth, walk(spec, 1))

    return {
        "arg_count": len(props),
        "required_arg_count": len(required),
        "optional_arg_count": max(0, len(props) - len(required)),
        "enum_count": enum_count,
        "nested_object_count": nested_objects,
        "array_count": array_count,
        "schema_depth": max_depth,
        "property_type_count": len(property_types),
    }


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


def template_signature(tool_id, arguments):
    if not isinstance(arguments, dict):
        return f"{tool_id}()"
    parts = []
    for key in sorted(arguments):
        parts.append(f"{key}:{value_type(arguments[key])}")
    return f"{tool_id}(" + ",".join(parts) + ")"


def key_signature(arguments):
    if not isinstance(arguments, dict):
        return ""
    return ",".join(sorted(arguments.keys()))


def shannon_entropy(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def clamp01(x):
    return max(0.0, min(1.0, x))


def minmax(values, value):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def retrieval_by_tool(predictions):
    by_tool = defaultdict(list)
    for row in predictions:
        if row.get("doc_mode") not in (None, "compact"):
            continue
        gold = row.get("gold_tool")
        if not gold:
            continue
        rank = row.get("rank")
        by_tool[gold].append(rank if isinstance(rank, int) else None)
    result = {}
    for tool_id, ranks in by_tool.items():
        valid = [r for r in ranks if isinstance(r, int)]
        n = len(ranks)
        result[tool_id] = {
            "retrieval_eval_count": n,
            "compact_recall_at_1": sum(1 for r in valid if r <= 1) / max(1, n),
            "compact_recall_at_5": sum(1 for r in valid if r <= 5) / max(1, n),
            "compact_recall_at_10": sum(1 for r in valid if r <= 10) / max(1, n),
            "compact_mean_rank": sum(valid) / max(1, len(valid)) if valid else "",
        }
    return result


def classify_tool(row):
    if row["call_count"] == 0:
        return "catalog_only/context_retrieval_candidate"
    recommendations = []
    if row["context_retrieval_score"] >= 0.58:
        recommendations.append("context/retrieval")
    if row["vocab_template_score"] >= 0.58:
        recommendations.append("vocab/template")
    if row["parameter_adapter_score"] >= 0.58:
        recommendations.append("parameter/adapter")
    if not recommendations:
        recommendations.append("uncertain")
    return ";".join(recommendations)


def analyze(tools, tasks, retrieval_predictions):
    tasks_by_tool = defaultdict(list)
    for task in tasks:
        tasks_by_tool[task["gold_tool"]].append(task)

    total_tasks = len(tasks)
    retrieval = retrieval_by_tool(retrieval_predictions)
    full_lengths = [len(t.get("full_card", "")) for t in tools]
    compact_lengths = [len(t.get("compact_card", "")) for t in tools]
    call_counts = [len(tasks_by_tool[t["tool_id"]]) for t in tools]

    rows = []
    for tool in tools:
        tool_id = tool["tool_id"]
        tool_tasks = tasks_by_tool[tool_id]
        call_count = len(tool_tasks)
        arg_key_counter = Counter()
        template_counter = Counter()
        arg_value_type_counter = Counter()
        call_json_lengths = []
        for task in tool_tasks:
            args = task.get("gold_arguments") or {}
            arg_key_counter[key_signature(args)] += 1
            template_counter[template_signature(tool_id, args)] += 1
            for value in args.values():
                arg_value_type_counter[value_type(value)] += 1
            call_json_lengths.append(len(json_dumps_compact({"name": tool_id, "arguments": args})))

        top_template_count = template_counter.most_common(1)[0][1] if template_counter else 0
        top_key_count = arg_key_counter.most_common(1)[0][1] if arg_key_counter else 0
        full_chars = len(tool.get("full_card", ""))
        compact_chars = len(tool.get("compact_card", ""))
        description_tokens = len(tokenize(tool.get("description", "")))
        stats = schema_stats(tool.get("parameters") or {})
        ret = retrieval.get(tool_id, {})

        call_norm = minmax(call_counts, call_count)
        full_len_norm = minmax(full_lengths, full_chars)
        compact_len_norm = minmax(compact_lengths, compact_chars)
        template_stability = top_template_count / max(1, call_count)
        key_stability = top_key_count / max(1, call_count)
        value_type_entropy = shannon_entropy(arg_value_type_counter)
        low_frequency = 1.0 - call_norm
        compact_recall_10 = ret.get("compact_recall_at_10", 0.0)
        retrieval_gap = 1.0 - compact_recall_10
        schema_complexity = clamp01(
            0.35 * min(stats["arg_count"] / 8, 1)
            + 0.25 * min(stats["schema_depth"] / 4, 1)
            + 0.20 * min(stats["enum_count"] / 12, 1)
            + 0.20 * full_len_norm
        )

        vocab_template_score = clamp01(
            0.34 * call_norm
            + 0.28 * template_stability
            + 0.18 * key_stability
            + 0.12 * min((sum(call_json_lengths) / max(1, len(call_json_lengths))) / 180, 1)
            + 0.08 * (1.0 - min(value_type_entropy / 3, 1))
        )
        parameter_adapter_score = clamp01(
            0.38 * call_norm
            + 0.25 * template_stability
            + 0.18 * schema_complexity
            + 0.11 * retrieval_gap
            + 0.08 * full_len_norm
        )
        vocab_non_frequency_score = clamp01(
            0.43 * template_stability
            + 0.27 * key_stability
            + 0.18 * min((sum(call_json_lengths) / max(1, len(call_json_lengths))) / 180, 1)
            + 0.12 * (1.0 - min(value_type_entropy / 3, 1))
        )
        adapter_non_frequency_score = clamp01(
            0.40 * template_stability
            + 0.30 * schema_complexity
            + 0.18 * retrieval_gap
            + 0.12 * full_len_norm
        )
        context_retrieval_score = clamp01(
            0.32 * low_frequency
            + 0.23 * (1.0 - template_stability)
            + 0.20 * compact_recall_10
            + 0.15 * compact_len_norm
            + 0.10 * (1.0 - schema_complexity)
        )

        row = {
            "tool_id": tool_id,
            "server": tool.get("server", ""),
            "marketplace": tool.get("marketplace", ""),
            "call_count": call_count,
            "call_share": call_count / max(1, total_tasks),
            "description_tokens": description_tokens,
            "full_card_chars": full_chars,
            "compact_card_chars": compact_chars,
            "compression_ratio": compact_chars / max(1, full_chars),
            "unique_arg_key_signatures": len(arg_key_counter),
            "top_arg_key_signature_share": key_stability,
            "unique_call_templates": len(template_counter),
            "top_call_template_share": template_stability,
            "avg_call_json_chars": sum(call_json_lengths) / max(1, len(call_json_lengths)),
            "arg_value_type_entropy": value_type_entropy,
            **stats,
            "compact_recall_at_1": ret.get("compact_recall_at_1", ""),
            "compact_recall_at_5": ret.get("compact_recall_at_5", ""),
            "compact_recall_at_10": ret.get("compact_recall_at_10", ""),
            "compact_mean_rank": ret.get("compact_mean_rank", ""),
            "frequency_score": call_norm,
            "context_retrieval_score": context_retrieval_score,
            "vocab_template_score": vocab_template_score,
            "vocab_non_frequency_score": vocab_non_frequency_score,
            "parameter_adapter_score": parameter_adapter_score,
            "adapter_non_frequency_score": adapter_non_frequency_score,
        }
        row["recommendation"] = classify_tool(row)
        rows.append(row)

    rows.sort(
        key=lambda r: (
            max(r["context_retrieval_score"], r["vocab_template_score"], r["parameter_adapter_score"]),
            r["call_count"],
        ),
        reverse=True,
    )
    return rows


def aggregate(rows):
    observed = [r for r in rows if r["call_count"] > 0]
    catalog_only = [r for r in rows if r["call_count"] == 0]

    def count_bins(subset):
        bins = Counter()
        for row in subset:
            for label in row["recommendation"].split(";"):
                bins[label] += 1
        return dict(bins)

    bins = Counter()
    for row in observed:
        for label in row["recommendation"].split(";"):
            bins[label] += 1
    high_vocab = [r for r in observed if r["vocab_template_score"] >= 0.58]
    high_param = [r for r in observed if r["parameter_adapter_score"] >= 0.58]
    high_context = [r for r in observed if r["context_retrieval_score"] >= 0.58]
    server_counts = Counter(r["server"] for r in observed)
    return {
        "tool_count": len(rows),
        "observed_tool_count": len(observed),
        "catalog_only_tool_count": len(catalog_only),
        "observed_recommendation_counts": dict(bins),
        "all_recommendation_counts": count_bins(rows),
        "observed_server_counts": dict(server_counts.most_common()),
        "avg_full_card_chars": sum(r["full_card_chars"] for r in rows) / max(1, len(rows)),
        "avg_compact_card_chars": sum(r["compact_card_chars"] for r in rows) / max(1, len(rows)),
        "avg_compression_ratio": sum(r["compression_ratio"] for r in rows) / max(1, len(rows)),
        "tools_with_tasks": sum(1 for r in rows if r["call_count"] > 0),
        "high_context_retrieval": len(high_context),
        "high_vocab_template": len(high_vocab),
        "high_parameter_adapter": len(high_param),
        "top_context_retrieval": [r["tool_id"] for r in sorted(high_context, key=lambda x: x["context_retrieval_score"], reverse=True)[:10]],
        "top_vocab_template": [r["tool_id"] for r in sorted(high_vocab, key=lambda x: x["vocab_template_score"], reverse=True)[:10]],
        "top_parameter_adapter": [r["tool_id"] for r in sorted(high_param, key=lambda x: x["parameter_adapter_score"], reverse=True)[:10]],
    }


def fmt_float(value, digits=3):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def top_markdown(rows, score_key, title, n=12):
    subset = sorted(rows, key=lambda r: r[score_key], reverse=True)[:n]
    lines = [
        f"### {title}",
        "",
        "| Tool | Server | Calls | Score | Compact R@10 | Template Share | Full Chars | Recommendation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in subset:
        lines.append(
            "| {tool} | {server} | {calls} | {score} | {recall} | {template} | {chars} | {rec} |".format(
                tool=r["tool_id"],
                server=r["server"],
                calls=r["call_count"],
                score=fmt_float(r[score_key]),
                recall=fmt_float(r["compact_recall_at_10"]) if r["compact_recall_at_10"] != "" else "",
                template=fmt_float(r["top_call_template_share"]),
                chars=r["full_card_chars"],
                rec=r["recommendation"],
            )
        )
    return "\n".join(lines)


def write_report(path, rows, summary):
    lines = [
        "# Representation Signal Report",
        "",
        "This diagnostic asks where tool knowledge should live before training a placement policy.",
        "",
        "## Headline",
        "",
        f"- Tools analyzed: {summary['tool_count']}",
        f"- Tools with observed calls: {summary['observed_tool_count']}",
        f"- Catalog-only tools without observed calls: {summary['catalog_only_tool_count']}",
        f"- Average full card chars: {summary['avg_full_card_chars']:.1f}",
        f"- Average compact card chars: {summary['avg_compact_card_chars']:.1f}",
        f"- Average compact/full ratio: {summary['avg_compression_ratio']:.3f}",
        f"- High context/retrieval signal among observed tools: {summary['high_context_retrieval']}",
        f"- High vocab/template signal among observed tools: {summary['high_vocab_template']}",
        f"- High parameter/adapter signal among observed tools: {summary['high_parameter_adapter']}",
        f"- Observed server counts: {summary['observed_server_counts']}",
        "",
        "## How To Read The Scores",
        "",
        "- `context_retrieval_score`: stronger for long-tail tools, weaker repeated templates, compact-card retrievability, and moderate schema complexity.",
        "- `vocab_template_score`: stronger for frequent calls with stable argument-key/type templates and reusable JSON call skeletons.",
        "- `parameter_adapter_score`: stronger for frequent, stable, schema-heavy tools whose knowledge is costly to keep in prompts or hard to retrieve.",
        "- `catalog_only/context_retrieval_candidate` means a schema exists but no local calls were observed; treat it as a long-tail prior, not direct evidence.",
        "",
        top_markdown(rows, "context_retrieval_score", "Tools Suited To Context/Retrieval"),
        "",
        top_markdown(rows, "vocab_template_score", "Tools With Vocab/Template Compression Signal"),
        "",
        top_markdown(rows, "parameter_adapter_score", "Tools Worth Considering For Parameter/Adapter Placement"),
        "",
        "## Next Experimental Move",
        "",
        "1. Keep compact/retrieval as the low-cost long-tail baseline.",
        "2. Add a template-macro baseline for high `vocab_template_score` tools before attempting tokenizer changes.",
        "3. Treat high `parameter_adapter_score` tools as candidates for later ParaTool-style modules, not as a first-week baseline.",
        "4. Report observed-only numbers in the main text and catalog-only numbers as coverage analysis.",
        "5. Train any policy only after these scores predict oracle winners better than frequency-only heuristics.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\data")
    parser.add_argument("--retrieval", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\results\retrieval_predictions.jsonl")
    parser.add_argument("--out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\representation_signals")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    tools = load_jsonl(data_dir / "tools.jsonl")
    tasks = load_jsonl(data_dir / "tasks.jsonl")
    retrieval_predictions = load_jsonl(Path(args.retrieval))

    rows = analyze(tools, tasks, retrieval_predictions)
    summary = aggregate(rows)

    fields = [
        "tool_id",
        "server",
        "marketplace",
        "call_count",
        "call_share",
        "description_tokens",
        "full_card_chars",
        "compact_card_chars",
        "compression_ratio",
        "arg_count",
        "required_arg_count",
        "optional_arg_count",
        "enum_count",
        "nested_object_count",
        "array_count",
        "schema_depth",
        "unique_arg_key_signatures",
        "top_arg_key_signature_share",
        "unique_call_templates",
        "top_call_template_share",
        "avg_call_json_chars",
        "arg_value_type_entropy",
        "compact_recall_at_1",
        "compact_recall_at_5",
        "compact_recall_at_10",
        "compact_mean_rank",
        "frequency_score",
        "context_retrieval_score",
        "vocab_template_score",
        "vocab_non_frequency_score",
        "parameter_adapter_score",
        "adapter_non_frequency_score",
        "recommendation",
    ]
    write_csv(out_dir / "tool_representation_signals.csv", rows, fields)
    write_json(out_dir / "tool_representation_signals.json", rows)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "representation_signal_report.md", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

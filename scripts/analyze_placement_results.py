import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid JSONL line {lineno}: {exc}")
    return rows


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize(rows):
    by_placement = defaultdict(list)
    for row in rows:
        by_placement[row["placement"]].append(row)

    summary = []
    for placement, subset in sorted(by_placement.items()):
        n = len(subset) or 1
        summary.append(
            {
                "placement": placement,
                "runs": len(subset),
                "gold_in_prompt_rate": sum(bool(r.get("gold_in_prompt")) for r in subset) / n,
                "valid_json_rate": sum(bool(r.get("valid_json")) for r in subset) / n,
                "tool_accuracy": sum(bool(r.get("tool_exact")) for r in subset) / n,
                "argument_exact": sum(bool(r.get("args_exact")) for r in subset) / n,
                "both_exact": sum(bool(r.get("both_exact")) for r in subset) / n,
                "avg_prompt_chars": sum(float(r.get("prompt_chars") or 0) for r in subset) / n,
                "avg_latency_sec": sum(float(r.get("latency_sec") or 0) for r in subset) / n,
            }
        )
    return summary


def task_oracle(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    oracle = []
    for task_id, variants in sorted(by_task.items()):
        successful = [r for r in variants if r.get("both_exact")]
        if successful:
            winner = sorted(successful, key=lambda r: (r.get("prompt_chars") or 10**9, r.get("latency_sec") or 10**9))[0]
        else:
            valid = [r for r in variants if r.get("valid_json") and r.get("tool_exact")]
            winner = sorted(valid or variants, key=lambda r: (not r.get("tool_exact"), r.get("prompt_chars") or 10**9))[0]
        oracle.append(
            {
                "task_id": task_id,
                "server": winner.get("server"),
                "gold_tool": winner.get("gold_tool"),
                "oracle_placement": winner.get("placement"),
                "oracle_both_exact": bool(winner.get("both_exact")),
                "oracle_prompt_chars": winner.get("prompt_chars"),
                "placements": {
                    r["placement"]: {
                        "gold_in_prompt": r.get("gold_in_prompt"),
                        "valid_json": r.get("valid_json"),
                        "tool_exact": r.get("tool_exact"),
                        "args_exact": r.get("args_exact"),
                        "both_exact": r.get("both_exact"),
                        "prompt_chars": r.get("prompt_chars"),
                    }
                    for r in variants
                },
            }
        )
    return oracle


def write_markdown(path, summary, oracle):
    lines = [
        "# Placement Result Analysis",
        "",
        "## Summary",
        "",
        "| Placement | Runs | Gold In Prompt | Valid JSON | Tool Acc | Arg Exact | Both Exact | Avg Prompt Chars | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {placement} | {runs} | {gold:.3f} | {valid:.3f} | {tool:.3f} | {arg:.3f} | {both:.3f} | {chars:.1f} | {lat:.3f} |".format(
                placement=row["placement"],
                runs=row["runs"],
                gold=row["gold_in_prompt_rate"],
                valid=row["valid_json_rate"],
                tool=row["tool_accuracy"],
                arg=row["argument_exact"],
                both=row["both_exact"],
                chars=row["avg_prompt_chars"],
                lat=row["avg_latency_sec"],
            )
        )

    oracle_counts = defaultdict(int)
    for row in oracle:
        oracle_counts[row["oracle_placement"]] += 1
    lines.extend(["", "## Oracle Counts", ""])
    for placement, count in sorted(oracle_counts.items()):
        lines.append(f"- {placement}: {count}")

    hard = [row for row in oracle if not row["oracle_both_exact"]]
    lines.extend(["", "## Unsolved Or Partially Solved Tasks", ""])
    if not hard:
        lines.append("- none")
    else:
        for row in hard[:20]:
            lines.append(f"- {row['task_id']} | {row['gold_tool']} | oracle={row['oracle_placement']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Directory containing placement_predictions.jsonl")
    args = parser.parse_args()

    result_dir = Path(args.results)
    rows = load_jsonl(result_dir / "placement_predictions.jsonl")
    summary = summarize(rows)
    oracle = task_oracle(rows)
    write_json(result_dir / "placement_summary_recomputed.json", summary)
    write_json(result_dir / "placement_oracle_by_task.json", oracle)
    write_markdown(result_dir / "placement_analysis.md", summary, oracle)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {result_dir / 'placement_analysis.md'}")


if __name__ == "__main__":
    main()

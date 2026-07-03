import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


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


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(toks) for toks in self.doc_tokens]
        df = Counter()
        for toks in self.doc_tokens:
            for tok in set(toks):
                df[tok] += 1
        n = len(self.docs)
        self.idf = {
            tok: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def score(self, query):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_lens[i] or 1
            score = 0.0
            for tok in q:
                if tok not in tf:
                    continue
                idf = self.idf.get(tok, 0.0)
                freq = tf[tok]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def build_doc(tool, mode):
    if mode == "full":
        return tool.get("full_card", "")
    if mode == "compact":
        return tool.get("compact_card", "")
    if mode == "weighted_compact":
        params = tool.get("parameters") or {}
        props = params.get("properties") or {}
        arg_names = " ".join(props.keys())
        name = tool.get("tool_id", "")
        desc = tool.get("description", "")
        return "\n".join(
            [
                f"{name} {name} {name}",
                f"{desc} {desc}",
                f"arguments {arg_names}",
                tool.get("compact_card", ""),
            ]
        )
    if mode == "name_desc":
        return f"{tool.get('tool_id', '')}\n{tool.get('description', '')}"
    raise ValueError(f"unknown mode {mode}")


def eval_retrieval(tools, tasks, doc_mode, topks):
    docs = [build_doc(t, doc_mode) for t in tools]
    tool_ids = [t["tool_id"] for t in tools]
    bm25 = BM25(docs)
    rows = []
    hits = {k: 0 for k in topks}
    mrr_total = 0.0
    covered = 0
    for task in tasks:
        gold = task["gold_tool"]
        if gold not in tool_ids:
            continue
        covered += 1
        scores = bm25.score(task["instruction"])
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_tools = [tool_ids[i] for i in ranked]
        rank = ranked_tools.index(gold) + 1 if gold in ranked_tools else None
        if rank:
            mrr_total += 1.0 / rank
        for k in topks:
            if gold in ranked_tools[:k]:
                hits[k] += 1
        rows.append(
            {
                "task_id": task["task_id"],
                "instruction": task["instruction"],
                "gold_tool": gold,
                "rank": rank,
                "top10": ranked_tools[:10],
                "doc_mode": doc_mode,
            }
        )
    summary = {
        "doc_mode": doc_mode,
        "covered_tasks": covered,
        "mrr": mrr_total / max(1, covered),
    }
    for k in topks:
        summary[f"recall@{k}"] = hits[k] / max(1, covered)
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\data")
    parser.add_argument("--out", default=r"C:\Users\zrz20\Desktop\vscode\Tool\experiments\results")
    parser.add_argument("--doc-modes", nargs="+", default=["name_desc", "compact", "full"])
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    tools = load_jsonl(data_dir / "tools.jsonl")
    tasks = load_jsonl(data_dir / "tasks.jsonl")

    summaries = []
    all_rows = []
    for mode in args.doc_modes:
        summary, rows = eval_retrieval(tools, tasks, mode, args.topk)
        summaries.append(summary)
        all_rows.extend(rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_jsonl(out_dir / "retrieval_predictions.jsonl", all_rows)
    with open(out_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

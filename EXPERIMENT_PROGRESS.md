# Experiment Progress

Date: 2026-07-04
Workspace: `C:\Users\zrz20\Desktop\vscode\Tool`

## 1. Repository And Data Setup

Created an independent Git repository under:

```text
C:\Users\zrz20\Desktop\vscode\Tool
```

Pushed to:

```text
https://github.com/abcabdl/Tool-placement-policy.git
```

Included complete local experiment assets:

```text
MCP-Flow raw local checkout
AgentVocab-main
ParaTool-main
experiments/
scripts/
refine-logs/
command notes
```

## 2. MCP-Flow Parsing

Command:

```powershell
python scripts\parse_mcp_flow_level1.py
```

Result:

```text
tools: 1032
tasks: 1839
tasks with local tool schema: 1839 / 1839
task servers: 12
```

Server distribution:

```text
Bright Data: 1455
Google News and Trends: 100
Pokemon: 75
Weather MCP Server: 58
Lotus Wisdom: 37
Weather Query Server: 27
Calculator: 20
Remote Shell Server: 20
12306 MCP Server: 15
Human Messages Prompt Server: 15
Weather360 Server: 10
AAAAAA MCP Server: 7
```

## 3. Retrieval Baselines

Initial BM25 baseline showed compact card was best.

After tokenizer fix that splits `_` and `-` in tool names:

```text
name_desc:
  MRR: 0.375
  recall@5: 0.455
  recall@10: 0.533

compact:
  MRR: 0.414
  recall@5: 0.520
  recall@10: 0.594

full:
  MRR: 0.381
  recall@5: 0.477
  recall@10: 0.562
```

Conclusion:

```text
compact card > full schema > name/description
```

The tokenizer fix improved compact retrieval:

```text
recall@10: 0.568 -> 0.594
recall@5: 0.482 -> 0.520
MRR: 0.383 -> 0.414
```

## 4. Representation Signal Diagnosis

Command:

```powershell
python scripts\analyze_representation_signals.py `
  --retrieval experiments\results_split_tokens\retrieval_predictions.jsonl `
  --out experiments\representation_signals_split_tokens
```

Result:

```text
tool_count: 1032
observed_tool_count: 82
catalog_only_tool_count: 950
observed vocab/template signal: 81
observed parameter/adapter signal: 50
```

Important interpretation:

```text
Only 82 tools have observed calls.
The 950 catalog-only tools are long-tail retrieval/context candidates, not direct evidence for placement.
Observed tools are dominated by Bright Data.
```

Top parameter/adapter candidates:

```text
brightdata-mcp_search_engine
brightdata-mcp_scraping_browser_screenshot
brightdata-mcp_scraping_browser_scroll
brightdata-mcp_scraping_browser_navigate
brightdata-mcp_extract
brightdata-mcp_session_stats
brightdata-mcp_scraping_browser_get_text
brightdata-mcp_web_data_linkedin_people_search
brightdata-mcp_scraping_browser_links
brightdata-mcp_scraping_browser_click
```

## 5. Function-Call Placement API Experiments

### 5.1 Sanity Run

First 20 tasks:

```text
full:
  gold_in_prompt: 1.00
  both_exact: 0.75
  avg_prompt_chars: 1020

compact:
  gold_in_prompt: 1.00
  both_exact: 0.75
  avg_prompt_chars: 629

retrieval:
  gold_in_prompt: 1.00
  both_exact: 0.75
  avg_prompt_chars: 629
```

Conclusion:

```text
sanity passed, but too easy.
compact/retrieval save prompt without hurting success.
```

### 5.2 Bright Data Top10, Before Tokenizer Fix

```text
retrieval:
  gold_in_prompt: 0.73
  tool_accuracy: 0.63
  both_exact: 0.62
  avg_prompt_chars: 3301

template_macro:
  gold_in_prompt: 0.73
  tool_accuracy: 0.62
  both_exact: 0.61
  avg_prompt_chars: 4746
```

### 5.3 Bright Data Top10, After Tokenizer Fix

```text
retrieval:
  gold_in_prompt: 0.79
  tool_accuracy: 0.69
  argument_exact: 0.78
  both_exact: 0.68
  avg_prompt_chars: 3332

template_macro:
  gold_in_prompt: 0.79
  tool_accuracy: 0.68
  argument_exact: 0.79
  both_exact: 0.67
  avg_prompt_chars: 4788
```

Conclusion:

```text
Tokenizer fix gives a real gain.
Naive prompt-level template_macro is negative: more tokens, no success gain.
```

### 5.4 Bright Data Top20

```text
retrieval top20:
  gold_in_prompt: 0.89
  tool_accuracy: 0.70
  argument_exact: 0.79
  both_exact: 0.69
  avg_prompt_chars: 5676
```

Conclusion:

```text
Top20 improves gold recall but barely improves final success.
Candidate confusion and prompt cost dominate.
Need retrieve top20 -> rerank -> prompt top5/top10.
```

## 6. AgentVocab Reproduction Progress

Generated MCP-Flow structural spans and rendered content:

```text
AgentVocab-main\outputs\data\mcpflow_structural_spans.txt
AgentVocab-main\outputs\data\mcpflow_actual_content.jsonl
```

Counts:

```text
structural spans: 10884
actual content rows: 1455
```

Ran Qwen2.5-7B tokenizer structural mining:

```text
candidate structural tokens: 431
total marginal savings: 32275.82
```

Ran reachability filtering:

```text
target 100 -> 68 reachable structural tokens
target 200 -> 145 reachable structural tokens
target 400 -> 145 reachable structural tokens
```

Top reachable token examples:

```text
{"arguments":{},"name":"brightdata-mcp_scraping_browser_scroll"}
{"arguments":{},"name":"brightdata-mcp_scraping_browser_links"}
{"arguments":{},"name":"brightdata-mcp_scraping_browser_get_text"}
{"arguments":{},"name":"brightdata-mcp_session_stats"}
{"arguments":{"full_page":true},"name":"brightdata-mcp_scraping_browser_screenshot"}
```

Conclusion:

```text
AgentVocab structural-token mining and reachability are reproduced on MCP-Flow without training.
Next AgentVocab step requires tokenizer expansion and LoRA training.
```

## 7. ParaTool Reproduction Progress

Generated ParaTool-compatible MCPFLOW sample:

```text
ParaTool-main\data\MCPFLOW\brightdata_adapter_candidates.jsonl
```

Loader validation:

```text
blocks: 10
training_examples: 297
```

Per-tool counts:

```text
brightdata-mcp_search_engine: 30
brightdata-mcp_scraping_browser_screenshot: 29
brightdata-mcp_scraping_browser_scroll: 30
brightdata-mcp_scraping_browser_navigate: 30
brightdata-mcp_extract: 30
brightdata-mcp_session_stats: 30
brightdata-mcp_scraping_browser_get_text: 28
brightdata-mcp_web_data_linkedin_people_search: 30
brightdata-mcp_scraping_browser_links: 30
brightdata-mcp_scraping_browser_click: 30
```

Conclusion:

```text
ParaTool small-sample data conversion is ready.
Training has not been started.
Next step is adapting configs/loaders for dataset=MCPFLOW or using this file as a custom tool_pretraining_data_path.
```

## 8. Current Research Interpretation

The strongest evidence so far:

```text
1. compact cards are strong retrieval representations.
2. retrieval recall and reranking are the current bottleneck.
3. naive prompt-level template injection is not useful.
4. MCP-Flow has real AgentVocab-style structural token signal.
5. Bright Data has credible ParaTool-style adapter candidates.
```

Current recommendation:

```text
Do not train a placement policy yet.
First implement reranking and run one GPU reproduction path:
  - ParaTool small sample, or
  - AgentVocab tokenizer expansion + LoRA small run.
```

## 9. Next Concrete Work

CPU:

```text
Implement top20 -> rerank -> top5/top10 prompting.
```

GPU:

```text
Option A: ParaTool tool_pretraining on 10 Bright Data tools.
Option B: AgentVocab tokenizer expansion with top_200 reachable structural tokens.
```

Preferred first GPU run:

```text
ParaTool small sample, because current failure mode is tool confusion after retrieval.
```

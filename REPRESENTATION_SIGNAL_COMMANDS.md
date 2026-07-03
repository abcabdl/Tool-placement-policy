# Representation Signal Commands

Working directory:

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool
```

## Stage A: Local Diagnostics, No API Key

Rebuild MCP-Flow local data artifacts:

```powershell
python scripts\parse_mcp_flow_level1.py
```

Run BM25 retrieval sanity over name/compact/full cards:

```powershell
python scripts\run_retrieval_baseline.py
```

Generate representation signal tables and report:

```powershell
python scripts\analyze_representation_signals.py
```

Expected outputs:

```text
experiments\representation_signals\tool_representation_signals.csv
experiments\representation_signals\tool_representation_signals.json
experiments\representation_signals\summary.json
experiments\representation_signals\representation_signal_report.md
```

## Stage B: API Experiments, User Runs Only

Set your API configuration:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
# Optional if using a compatible proxy:
# $env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

Run a small placement sanity check:

```powershell
python scripts\run_function_call_placements.py `
  --max-tasks 20 `
  --placements full compact retrieval template_macro `
  --retrieval-topk 5 `
  --out experiments\results_api_sanity
```

Run a focused server check on the dominant Bright Data subset:

```powershell
python scripts\run_function_call_placements.py `
  --server "Bright Data" `
  --max-tasks 50 `
  --placements compact retrieval template_macro `
  --retrieval-topk 10 `
  --out experiments\results_brightdata_api_50
```

Run a high-template stress test for the AgentVocab proxy:

```powershell
python scripts\run_function_call_placements.py `
  --server "Bright Data" `
  --max-tasks 100 `
  --placements retrieval template_macro `
  --retrieval-topk 10 `
  --out experiments\results_template_macro_brightdata_100
```

After the tokenizer fix that splits MCP tool names on `_` / `-`, rerun the same setting under a fresh output directory:

```powershell
python scripts\run_function_call_placements.py `
  --server "Bright Data" `
  --max-tasks 100 `
  --placements retrieval template_macro `
  --retrieval-topk 10 `
  --out experiments\results_template_macro_brightdata_100_split_tokens
```

Then sweep retrieval depth without template macros to diagnose whether the bottleneck is retrieval recall or model confusion among many candidates:

```powershell
python scripts\run_function_call_placements.py `
  --server "Bright Data" `
  --max-tasks 100 `
  --placements retrieval `
  --retrieval-topk 20 `
  --out experiments\results_retrieval_brightdata_top20_100_split_tokens
```

Analyze any placement result directory:

```powershell
python scripts\analyze_placement_results.py `
  --results experiments\results_template_macro_brightdata_100_split_tokens
```

## Recommended Run Order

```text
R001 parse local data
R002 retrieval sanity
R003 representation signal report
R004 only then run API placement sanity
R005 only after enough API outputs, compare signal heuristic vs oracle
```

Do not start tokenizer adaptation, adapter training, or a learned policy until R003 shows interpretable non-frequency representation groups.

# AgentVocab / ParaTool MCP-Flow Reproduction Commands

## 1. Prepare MCP-Flow Artifacts

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool

python scripts\prepare_agentvocab_paratool_mcpflow.py
```

Outputs:

```text
AgentVocab-main\outputs\data\mcpflow_structural_spans.txt
ParaTool-main\data\MCPFLOW\brightdata_adapter_candidates.jsonl
```

Current expected counts:

```text
AgentVocab structural spans: 10884
ParaTool functions: 10
ParaTool training examples: 297
```

## 2. AgentVocab Structural Mining Smoke Test

This uses a local tokenizer only to test the mining pipeline. For paper-faithful reproduction, replace `--tokenizer` with a Qwen2.5 Instruct tokenizer path.

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main

$env:PYTHONPATH="C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main\src"

python scripts\mine_structural_tokens.py `
  --input outputs\data\mcpflow_structural_spans.txt `
  --tokenizer "C:\Users\zrz20\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41" `
  --output outputs\tokens\mcpflow_structural_scored_smoke.json `
  --max-new-tokens 200 `
  --min-frequency 5
```

Current smoke-test result:

```text
unique structural spans: 807
structural candidates: 431
selected tokens: 200
```

## 3. ParaTool JSONL Loader Smoke Test

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool\ParaTool-main

@'
from dataset.data_io import load_jsonl_records, iter_training_blocks
p=r"C:\Users\zrz20\Desktop\vscode\Tool\ParaTool-main\data\MCPFLOW\brightdata_adapter_candidates.jsonl"
blocks=list(iter_training_blocks(load_jsonl_records(p)))
print("blocks", len(blocks))
print("training_examples", sum(len(b.training_data) for b in blocks))
for b in blocks:
    print(b.tool_schema.get("name"), len(b.training_data))
'@ | python -
```

Expected result:

```text
blocks 10
training_examples 297
each candidate tool has 28-30 examples
```

## 4. Next Faithful Reproduction Steps

AgentVocab:

```text
Replace smoke-test tokenizer with Qwen2.5-7B-Instruct tokenizer.
Then run reachability filtering, tokenizer expansion, and stage1/stage2 LoRA on a GPU machine.
```

ParaTool:

```text
Either download the official BFCL/STB data from HuggingFace, or adapt ParaTool loaders to accept data\MCPFLOW.
Then run tool_pretraining.py -> soft_tool_selection_train.py -> tool_finetuning.py.
```

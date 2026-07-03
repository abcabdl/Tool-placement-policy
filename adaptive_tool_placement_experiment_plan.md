# Adaptive Tool-Knowledge Placement Experiment Plan

Date: 2026-07-04
Workspace: `C:\Users\zrz20\Desktop\vscode\Tool`
Repository: `https://github.com/abcabdl/Tool-placement-policy.git`

## 0. Current Thesis

Do not start with a learned placement policy.

The more useful research question is:

> Where should tool knowledge live: context, retrieval memory, reusable templates/vocabulary, or parameters?

The current project should be framed as **tool-knowledge representation selection**, with placement policy as a later decision layer.

The near-term goal is to identify which representation is justified by measurable signals:

```text
Context / compact card       -> cheap, explicit, good for long-tail tools
Retrieval memory             -> scalable tool-pool coverage
Template / vocabulary        -> repeated JSON/function-call structures
Parameter / adapter modules  -> frequent, stable, retrieval-ambiguous tools
```

## 1. Main Claims To Test

### Claim 1: Compact tool cards are a stronger retrieval representation than full schemas.

Evidence needed:

```text
compact BM25 > full-schema BM25 > name/description BM25
```

Primary metrics:

```text
MRR
recall@1 / recall@3 / recall@5 / recall@10
prompt chars
gold-in-prompt rate
```

### Claim 2: Retrieval is the current bottleneck before policy.

Evidence needed:

```text
top-k increases gold-in-prompt
but too-large top-k increases candidate confusion and prompt cost
```

Primary metrics:

```text
gold_in_prompt_rate
tool_accuracy
argument_exact
both_exact
avg_prompt_chars
```

### Claim 3: AgentVocab-style structural tokens exist in MCP-Flow traces.

Evidence needed:

```text
Qwen2.5 tokenizer mining finds repeated structural spans
reachability filtering keeps useful tokens
tokens correspond to real function-call skeletons, not noise
```

Primary metrics:

```text
candidate token count
reachable token count
estimated token savings
top token examples
```

### Claim 4: ParaTool-style adapter candidates can be selected before training.

Evidence needed:

```text
tools are frequent
schemas/call templates are stable
retrieval remains ambiguous
per-tool training data can be generated
```

Primary metrics:

```text
call_count
top_call_template_share
compact_recall@10
parameter_adapter_score
per-tool training examples
```

## 2. Experiment Roadmap

## Stage A: Data And Retrieval Baselines

Status: mostly done, keep as baseline.

Commands:

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool

python scripts\parse_mcp_flow_level1.py

python scripts\run_retrieval_baseline.py `
  --doc-modes name_desc compact full `
  --out experiments\results_split_tokens
```

Expected current result:

```text
tools: 1032
tasks: 1839
schema coverage: 1839 / 1839

compact MRR: 0.414
compact recall@5: 0.520
compact recall@10: 0.594
```

Decision:

```text
compact card remains the default retrieval representation.
```

## Stage B: Representation Signal Diagnosis

Status: done, rerun when data changes.

Command:

```powershell
python scripts\analyze_representation_signals.py `
  --retrieval experiments\results_split_tokens\retrieval_predictions.jsonl `
  --out experiments\representation_signals_split_tokens
```

Current interpretation:

```text
observed tools: 82
catalog-only tools: 950
vocab/template signal: 81 observed tools
parameter/adapter signal: 50 observed tools
```

Important caveat:

```text
The observed set is dominated by Bright Data.
Do not claim all 1032 tools have template/adapter evidence.
```

## Stage C: Retrieval Top-k And Template Macro Ablation

Status: done enough for diagnosis, not final.

Key result:

```text
top10 retrieval:
  gold_in_prompt = 0.79
  both_exact = 0.68
  avg_prompt_chars = 3332

top10 template_macro:
  gold_in_prompt = 0.79
  both_exact = 0.67
  avg_prompt_chars = 4788

top20 retrieval:
  gold_in_prompt = 0.89
  both_exact = 0.69
  avg_prompt_chars = 5676
```

Interpretation:

```text
Naive prompt-level template_macro is negative.
Increasing top-k improves recall but causes candidate confusion and high prompt cost.
Next retrieval work should be retrieve top20 -> rerank -> prompt top5/top10.
```

Next experiment to add:

```text
Stage C.1: Reranker before prompt
- candidate pool: BM25 compact top20
- rerank features: tool name subword match, schema arg match, description match, template similarity
- prompt only top5/top10
- compare against raw top10 and raw top20
```

## Stage D: AgentVocab Reproduction Without Training

Status: done up to GPU boundary.

Preparation command:

```powershell
python scripts\prepare_agentvocab_paratool_mcpflow.py
```

Generated files:

```text
AgentVocab-main\outputs\data\mcpflow_structural_spans.txt
AgentVocab-main\outputs\data\mcpflow_actual_content.jsonl
```

Structural mining command:

```powershell
cd C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main

$env:PYTHONPATH="C:\Users\zrz20\Desktop\vscode\Tool\AgentVocab-main\src"

python scripts\mine_structural_tokens.py `
  --input outputs\data\mcpflow_structural_spans.txt `
  --tokenizer Qwen/Qwen2.5-7B-Instruct `
  --output outputs\tokens\mcpflow_structural_scored_qwen25_7b.json `
  --max-new-tokens 1000 `
  --min-frequency 5
```

Reachability command:

```powershell
python scripts\select_reachable_tokens.py `
  --scored-tokens outputs\tokens\mcpflow_structural_scored_qwen25_7b.json `
  --corpus outputs\data\mcpflow_actual_content.jsonl `
  --tokenizer Qwen/Qwen2.5-7B-Instruct `
  --output-dir outputs\tokens `
  --token-type structural `
  --targets 100 200 400
```

Current result:

```text
structural spans: 10884
unique spans: 807
candidate structural tokens: 431
reachable tokens at target 100: 68
reachable tokens at target 200: 145
reachable tokens at target 400: 145
```

Conclusion:

```text
MCP-Flow has real AgentVocab-style structural token signal.
Prompt-level template_macro is not a faithful substitute for vocabulary adaptation.
```

Next GPU-bound step:

```text
expand tokenizer -> stage1 LoRA -> stage2 LoRA
```

Recommended first GPU run:

```text
Use top_200_reachable_structural_tokens.json.
Run a small Qwen2.5-7B stage2-style tokenizer expansion experiment.
Use reduced sequence length before attempting official AgentVocab scale.
```

## Stage E: ParaTool Reproduction Preparation

Status: data conversion done; training not started.

Generated files:

```text
ParaTool-main\data\MCPFLOW\brightdata_adapter_candidates.jsonl
ParaTool-main\data\MCPFLOW\brightdata_observed_tools.jsonl
```

Small adapter-candidate sample:

```text
10 candidate tools
297 training examples
28-30 examples per tool
```

Full Bright Data observed-tools sample:

```text
60 observed Bright Data tools
1455 training examples
8-30 examples per tool
all tools nonempty
```

Candidate tools:

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

Next reproduction work:

```text
Stage E.1: Adapt ParaTool loader/configs to accept dataset=MCPFLOW or pass custom data path.
Stage E.2: Run tool_pretraining.py on brightdata_observed_tools.jsonl, not only the 10-tool toy file.
Stage E.3: Run soft_tool_selection_train.py on 60-tool Bright Data.
Stage E.4: Run tool_finetuning.py with gating enabled.
Stage E.5: Compare parameter/gating against compact retrieval top10/top20.
```

GPU expectation:

```text
60-tool Bright Data Qwen2.5-7B LoRA: 2x A800 is enough to attempt a serious domain reproduction.
Qwen2.5-14B is possible but should be second priority.
Official BFCL/STB-scale ParaTool still needs official data and longer runtime.
```

## Stage F: Policy Only After Gates

Do not train AdaToolPlace yet.

A learned policy is justified only if all are true:

```text
1. reranker improves top20 -> top5/top10 selection
2. AgentVocab structural tokens reduce real token count after tokenizer expansion
3. ParaTool adapter/gating improves ambiguous high-frequency tools
4. oracle placement beats fixed compact retrieval by a meaningful margin
```

If these gates fail:

```text
Do not write a policy paper.
Pivot to retrieval-reranking or structural vocabulary adaptation.
```

## Stage G: Two-A800 Reproduction Plan

Assumed hardware:

```text
2 x NVIDIA A800, ideally 80GB each
Linux + CUDA + Python 3.10
```

With two A800s, do not stop at toy experiments. Run the largest feasible MCP-Flow/Bright-Data reproduction before moving to official BFCL/STB or tau-bench.

### G.1 AgentVocab on MCP-Flow/Bright Data

Input files already prepared:

```text
AgentVocab-main\outputs\data\mcpflow_actual_content.jsonl
AgentVocab-main\outputs\tokens\top_200_reachable_structural_tokens.json
```

Run order:

```text
1. Expand Qwen2.5-7B tokenizer with top_200 reachable structural tokens.
2. Run stage1 LoRA on MCP-Flow rendered data with original tokenizer.
3. Run stage2 LoRA on expanded tokenizer/model.
4. Evaluate token count reduction and function-call accuracy on held-out Bright Data tasks.
```

Recommended two-A800 modifications from official AgentVocab scripts:

```text
NPROC_PER_NODE=2 for both stage1 and stage2
CUDA_VISIBLE_DEVICES=0,1
stage1 max_length: start with 8192 or 16384, not 32768
stage2 max_length: 8192
LoRA rank: keep 64 if memory allows; otherwise 32
Use top_200 reachable structural tokens first; top_400 gives no additional reachable tokens in current corpus.
```

What counts as success:

```text
Input/output token count decreases versus Qwen2.5 baseline.
Function-call JSON validity does not drop.
Tool accuracy / argument exact do not degrade materially.
```

### G.2 ParaTool on 60-tool Bright Data

Input file:

```text
ParaTool-main\data\MCPFLOW\brightdata_observed_tools.jsonl
```

Run order:

```text
1. Add MCPFLOW dataset support or use tool_pretraining_data_path.
2. Stage 1: per-tool LoRA pretraining over 60 Bright Data tools.
3. Stage 2: soft tool selection / gating network.
4. Stage 3: gated tool finetuning.
5. Inference/eval on held-out Bright Data tasks.
```

Two-A800 target:

```text
Primary model: Qwen2.5-7B-Instruct
Secondary model if time permits: Qwen2.5-14B-Instruct
Candidate set: 60 Bright Data observed tools
Holdout: split 1455 tasks into train/validation/test, not train-on-all.
```

What counts as success:

```text
Parameter/gating improves over compact retrieval top10 at similar or lower prompt cost.
Gating reduces confusion for top20 candidate pools.
Adapter improves ambiguous tools such as search_engine, screenshot, scroll, get_text, linkedin_people_search.
```

### G.3 Fullness Criteria

The two-A800 run should be considered complete enough only if it includes:

```text
AgentVocab:
  tokenizer expansion
  stage1 LoRA
  stage2 LoRA
  token-saving measurement
  held-out function-call evaluation

ParaTool:
  60-tool Bright Data data
  tool pretraining
  soft tool selection
  tool finetuning
  held-out evaluation
```

If time is limited, priority order:

```text
1. ParaTool 60-tool Bright Data Qwen2.5-7B
2. AgentVocab top_200 structural-token Qwen2.5-7B
3. Official ParaTool BFCL/STB data
4. Official AgentVocab tau-bench/tau2-bench
```

## 3. Immediate Next Steps

### Next CPU task

Implement and evaluate a lightweight reranker:

```text
BM25 compact top20 -> feature rerank -> prompt top5/top10
```

This is the most urgent because top20 recall is high but prompt top20 has poor cost-performance.

### Next GPU task

Choose one:

```text
AgentVocab path:
  tokenizer expansion + stage1/stage2 LoRA with top_200 reachable structural tokens

ParaTool path:
  full 60-tool Bright Data observed-tools run
```

With two A800s, prefer the ParaTool 60-tool Bright Data run first because it directly targets the current retrieval/tool-confusion bottleneck.

## 4. Repository Notes

The repository has been pushed to:

```text
https://github.com/abcabdl/Tool-placement-policy.git
```

The pushed repository includes:

```text
MCP-Flow raw local checkout as normal files
parsed MCP-Flow data
retrieval/function-call experiment results
AgentVocab-main with MCP-Flow mining outputs
ParaTool-main with MCPFLOW small and 60-tool Bright Data samples
scripts and reproduction command notes
```

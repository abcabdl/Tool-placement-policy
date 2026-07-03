# Experiment Plan

**Problem**: Tool-use agents currently mix several kinds of tool knowledge placement without knowing which tools should live in context, retrieval memory, vocabulary/templates, or parameters.
**Method Thesis**: Before learning a placement policy, measure representation signals that predict where each tool's knowledge is best reused.
**Date**: 2026-07-03

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: Tool knowledge has heterogeneous representation signals. | If all tools prefer the same representation, a selection framework is not needed. | MCP-Flow tools separate into interpretable context/retrieval, vocab/template, and parameter/adapter candidate groups. | B1, B2 |
| C2: Lightweight diagnostics should precede heavy ParaTool/AgentVocab-style training. | This keeps the project from becoming an expensive toy policy or premature retraining pipeline. | Cheap statistics identify high-value subsets for template macros and later adapter modules. | B2, B3 |
| Anti-claim: The signal is only frequency. | Frequency alone would make the contribution shallow. | Scores must include and report schema complexity, template stability, retrieval ambiguity, and prompt cost. | B2, B4 |

## Paper Storyline

- Main paper must prove: representation choice is a measurable property of tools and tasks, not just a prompt-engineering switch.
- Appendix can support: extra datasets, alternate thresholds, larger MCP-Flow downloads, small-model sensitivity.
- Experiments intentionally cut for now: tokenizer expansion, LoRA training, ParaTool-style module training, RL/LLM controller policies.

## Experiment Blocks

### Block 1: Data and Retrieval Sanity

- Claim tested: the MCP-Flow subset is usable for representation diagnostics.
- Why this block exists: later claims are invalid if tool schemas, gold calls, or retrieval predictions are misaligned.
- Dataset / split / task: local MCP-Flow sample under `MCP-Flow/data/tools` and `MCP-Flow/data/function_call/Smithery`.
- Compared systems: BM25 over `name_desc`, `compact`, and `full` tool cards.
- Metrics: covered tasks, MRR, recall@1/3/5/10, compact/full card length.
- Setup details: reuse `scripts/parse_mcp_flow_level1.py` and `scripts/run_retrieval_baseline.py`.
- Success criterion: at least 95% of observed function-call tasks have local schemas; compact retrieval is nontrivial.
- Failure interpretation: local checkout is too partial; download or reconstruct more MCP-Flow data.
- Table / figure target: diagnostic table in method section.
- Priority: MUST-RUN.

### Block 2: Representation Signal Extraction

- Claim tested: tools expose measurable signals for context/retrieval, vocab/template compression, and parameter/adapter placement.
- Why this block exists: this is the new framing that connects compact/retrieval baselines with AgentVocab and ParaTool.
- Dataset / split / task: same MCP-Flow local subset.
- Compared systems: no learned systems; per-tool diagnostic scores and ranked candidate lists.
- Metrics: call frequency, schema complexity, full/compact prompt cost, argument-key stability, call-template stability, compact retrieval rank.
- Setup details: run `scripts/analyze_representation_signals.py`.
- Success criterion: report identifies nonempty and interpretable high-signal groups for at least two representation families.
- Failure interpretation: the local subset is too small or too homogeneous; expand to HuggingFace MCP-Flow or ToolBench.
- Table / figure target: top candidate tables and score-distribution histogram.
- Priority: MUST-RUN.

### Block 3: Template-Macro Baseline

- Claim tested: AgentVocab-like benefits can first be approximated without tokenizer training.
- Why this block exists: it isolates reusable call structure before paying the cost of vocabulary adaptation.
- Dataset / split / task: high `vocab_template_score` tools from Block 2.
- Compared systems: compact card prompting vs compact card plus reusable call templates/macros.
- Metrics: prompt chars, valid JSON rate, tool exact, argument exact, both exact.
- Setup details: extend `run_function_call_placements.py` with a `template_macro` placement that injects the dominant call skeleton.
- Success criterion: template macros reduce malformed calls or argument errors without increasing prompt cost substantially.
- Failure interpretation: repeated trace structure does not transfer to generation, so tokenizer adaptation is unlikely to be worth it yet.
- Table / figure target: main ablation table.
- Priority: NICE-TO-HAVE after B1-B2.

### Block 4: Policy Readiness Check

- Claim tested: representation signals predict oracle placement beyond frequency-only heuristics.
- Why this block exists: it gates whether a learned policy is scientifically justified.
- Dataset / split / task: tasks with completed full/compact/retrieval/model-call outputs.
- Compared systems: frequency-only heuristic, signal-score heuristic, oracle from measured success/cost.
- Metrics: oracle agreement, regret, success-cost Pareto, feature importance.
- Setup details: reuse placement predictions and add a small decision-tree or logistic regression only after enough model-call outputs exist.
- Success criterion: signal heuristic beats frequency-only and fixed placement on regret.
- Failure interpretation: do not train AdaToolPlace; pivot to schema drift or small-model setting.
- Table / figure target: go/no-go gate table.
- Priority: MUST-RUN before any learned policy.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Rebuild local data artifacts | `parse_mcp_flow_level1.py` | schemas cover most function-call tasks | CPU seconds | Windows checkout omitted some files |
| M1 | Retrieval sanity | `run_retrieval_baseline.py` | compact retrieval has useful recall | CPU seconds | BM25 may understate semantic retrieval |
| M2 | Signal report | `analyze_representation_signals.py` | ranked groups are nonempty and interpretable | CPU seconds | scores collapse to frequency |
| M3 | Template-macro prototype | add and run `template_macro` placement | improves JSON/arg accuracy on high-template tools | API budget | model-call variance |
| M4 | Policy readiness | compare signal heuristic vs oracle | signal beats frequency-only | API budget | not enough placement outputs |

## Compute and Data Budget

- Total estimated GPU-hours: 0 for the diagnostic stage.
- API calls: only needed for M3/M4 placement generation.
- Data preparation needs: local MCP-Flow subset now; later expand to full HuggingFace MCP-Flow if local signals are too sparse.
- Biggest bottleneck: enough task diversity per tool to distinguish template stability from frequency.

## Risks and Mitigations

- Risk: The local MCP-Flow sample overrepresents a few Smithery tools.
- Mitigation: report per-server counts and add a full MCP-Flow download only after diagnostics work locally.

- Risk: High vocab/template score is just call frequency.
- Mitigation: report template share, key signature share, and argument type entropy separately.

- Risk: Parameter/adaptor signal is speculative without training.
- Mitigation: label it as a candidate filter, not as evidence that adapters already work.

## Final Checklist

- [x] Main diagnostic blocks are defined.
- [x] Novelty is framed as representation selection, not only prompt placement.
- [x] Heavy training baselines are deferred.
- [ ] Template-macro baseline implemented.
- [ ] Policy readiness checked against oracle results.

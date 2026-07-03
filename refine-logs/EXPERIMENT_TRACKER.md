# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | rebuild data | MCP-Flow local parser | local sample | tools, tasks, schema coverage | MUST | DONE | Local CPU run: 1032 tools, 1839 tasks, 1839/1839 schema coverage. |
| R002 | M1 | retrieval sanity | BM25 name/compact/full | local sample | MRR, recall@k | MUST | DONE | Local CPU run: compact is strongest, MRR 0.383 and recall@10 0.568. |
| R003 | M2 | representation signals | signal diagnostics | local sample | per-tool scores and ranked groups | MUST | TODO | Uses `scripts/analyze_representation_signals.py`. |
| R004 | M3 | template proxy for AgentVocab | retrieval + template macro | high-template tools | valid JSON, tool exact, arg exact | NICE | READY | `template_macro` placement is implemented; API run is user-only. |
| R005 | M4 | policy readiness | signal heuristic vs oracle | tasks with placement outputs | oracle agreement, regret | MUST | TODO | Gate before any learned policy. |

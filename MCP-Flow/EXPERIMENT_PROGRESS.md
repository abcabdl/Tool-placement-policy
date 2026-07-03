# MCP-Flow Adaptive Tool Placement 实验进展

记录日期：2026-07-03  
仓库路径：`C:\Users\zrz20\Desktop\vscode\Tool\MCP-Flow`  
实验脚本路径：`C:\Users\zrz20\Desktop\vscode\Tool\scripts`

## 1. 当前目标

基于 MCP-Flow 的工具 schema 和 function-call 任务，比较不同工具知识放置方式：

```text
Full Context
Compact Context
Retrieval Memory
```

核心问题：

> 不同任务 / 工具是否真的适合不同 placement？

若存在明显 placement heterogeneity，则继续做 AdaToolPlace policy。

## 2. Clone 与数据状态

### 2.1 Clone 情况

GitHub 仓库：

```text
https://github.com/wwh0411/MCP-Flow
```

Windows 直接 checkout 会失败，因为仓库中有 6 个文件名包含 Windows 非法字符 `:` 或 `?`。这些文件是少数 MCP server 的 config / tool schema，不是代码。

当前已使用 `--no-checkout` 拉取安全路径：

```text
README.md
assets/
src/
wizard_utils/
data/context/
data/function_call/
data/tools/deepnlp/
部分 data/tools/smithery/*.json
```

### 2.2 本地样例解析

运行：

```powershell
python C:\Users\zrz20\Desktop\vscode\Tool\scripts\parse_mcp_flow_level1.py `
  --repo C:\Users\zrz20\Desktop\vscode\Tool\MCP-Flow `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\data
```

解析结果：

```text
tools: 1032
tasks: 1839
tasks with local tool schema: 1839 / 1839
task servers: 12
```

主要 server 分布：

| Server | Tasks | Tools |
|---|---:|---:|
| Bright Data | 1455 | 60 |
| Google News and Trends | 100 | 5 |
| Pokémon | 75 | 4 |
| Weather MCP Server | 58 | 3 |
| Calculator | 20 | 1 |

Bright Data 工具多、任务多，最适合作为 placement stress test。

## 3. 无 API 的 Retrieval Sanity

运行：

```powershell
python C:\Users\zrz20\Desktop\vscode\Tool\scripts\run_retrieval_baseline.py `
  --data C:\Users\zrz20\Desktop\vscode\Tool\experiments\data `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\results
```

结果：

| Retrieval Doc Mode | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| name_desc | 0.336 | 0.258 | 0.364 | 0.421 | 0.498 |
| compact | 0.383 | 0.292 | 0.424 | 0.482 | 0.568 |
| full | 0.343 | 0.256 | 0.373 | 0.431 | 0.524 |

观察：

- 用 compact tool card 建 BM25 文档，比 full JSON 和 name/description 更容易召回 gold tool。
- full JSON 可能包含太多结构噪声，不利于词面检索。

## 4. Function-call Placement 实验

模型：由 `$env:OPENAI_MODEL` 指定。  
评估：模型输出 JSON function call，并与 gold function_call 比较。

指标：

```text
gold_in_prompt_rate: gold tool 是否出现在 prompt 中
valid_json_rate: 模型输出是否能解析为 JSON
tool_accuracy: 工具名是否正确
argument_exact: arguments 是否 exact match
both_exact: 工具名和 arguments 是否都正确
avg_prompt_chars: prompt 字符数近似成本
avg_latency_sec: 延迟
```

注意：

- `argument_exact` 是严格 exact match，可能低估语义等价情况。
- 当前阶段主要看 `both_exact` 和 `avg_prompt_chars` 的 trade-off。

## 5. 初始 20 条 sanity

结果：

| Placement | Runs | Gold in Prompt | Tool Acc | Arg Exact | Both Exact | Avg Prompt Chars |
|---|---:|---:|---:|---:|---:|---:|
| full | 20 | 1.00 | 1.00 | 0.75 | 0.75 | 1020 |
| compact | 20 | 1.00 | 1.00 | 0.75 | 0.75 | 629 |
| retrieval | 20 | 1.00 | 1.00 | 0.75 | 0.75 | 629 |

解释：

- 前 20 条任务太简单，三种 placement 完全同分。
- 这批不适合作为 placement heterogeneity 判断。

## 6. Bright Data 50 条：Full / Compact / Retrieval Top-k

### 6.1 Full / Compact / Retrieval top5

| Placement | Runs | Gold in Prompt | Tool Acc | Arg Exact | Both Exact | Avg Prompt Chars |
|---|---:|---:|---:|---:|---:|---:|
| full | 50 | 1.00 | 0.70 | 0.74 | 0.68 | 15480 |
| compact | 50 | 1.00 | 0.60 | 0.74 | 0.60 | 7926 |
| retrieval top5 | 50 | 0.40 | 0.38 | 0.60 | 0.38 | 2068 |

解释：

- full 准确率最高，但 prompt 很长。
- compact 约省 49% prompt，both_exact 从 0.68 降到 0.60。
- retrieval top5 成本最低，但 gold_in_prompt 只有 0.40，召回不足是主要瓶颈。

### 6.2 Retrieval top-k 曲线

| Retrieval Top-k | Gold in Prompt | Tool Acc | Arg Exact | Both Exact | Avg Prompt Chars |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.40 | 0.38 | 0.60 | 0.38 | 2068 |
| 10 | 0.60 | 0.54 | 0.70 | 0.54 | 3297 |
| 20 | 0.84 | 0.68 | 0.80 | 0.68 | 5659 |
| 30 | 0.96 | 0.64 | 0.80 | 0.64 | 7926 |

关键观察：

1. top-k 增大能显著提高 gold tool recall。
2. top20 达到 `both_exact = 0.68`，与 full context 相同，但 prompt 只有 full 的约 36.6%。
3. top30 虽然 `gold_in_prompt_rate = 0.96`，但 `both_exact` 反而降到 0.64，说明候选工具过多会带来 tool overload / tool confusion。
4. 当前 Bright Data 上 retrieval 的 sweet spot 可能在 top20 附近。

## 7. 当前结论

Bright Data 50 条已经出现明确 trade-off：

| Placement | 优点 | 缺点 |
|---|---|---|
| Full Context | 准确最高，信息最完整 | prompt 极长 |
| Compact Context | 成本约减半，准确率尚可 | 相比 full 掉 8 个点 |
| Retrieval top5 | 最省 prompt | 漏工具严重 |
| Retrieval top20 | 达到 full 的 both_exact，成本低很多 | 仍依赖检索质量 |
| Retrieval top30 | 召回最高 | 候选过多导致工具混淆 |

目前可以初步支持：

> 没有单一 placement 永远最优。Retrieval 的 top-k 存在成本-召回-混淆 trade-off；Compact Context 是稳定中间方案；Full Context 是高成本上界。

## 8. 重要注意事项

### 8.1 输出目录被复用

用户运行 top10/top20/top30 时使用了同一个输出目录：

```text
experiments\results_brightdata_retrieval_top10
```

因此磁盘上的 `placement_summary.json` 和 `placement_predictions.jsonl` 可能被最后一次 top30 覆盖。

后续建议每次使用独立目录：

```text
results_brightdata_retrieval_top10
results_brightdata_retrieval_top20
results_brightdata_retrieval_top30
```

### 8.2 当前结果来自前 50 条 Bright Data

后续需要：

- 换随机 50 条；
- 增加到 100 / 200 条；
- 跑其他 server；
- 避免样本顺序偏差。

## 9. 下一步实验

### 9.1 复跑 top-k 曲线，避免覆盖

```powershell
python .\scripts\run_function_call_placements.py `
  --data .\experiments\data `
  --out .\experiments\results_brightdata_retrieval_top10 `
  --model $env:OPENAI_MODEL `
  --server "Bright Data" `
  --max-tasks 50 `
  --placements retrieval `
  --retrieval-topk 10
```

```powershell
python .\scripts\run_function_call_placements.py `
  --data .\experiments\data `
  --out .\experiments\results_brightdata_retrieval_top20 `
  --model $env:OPENAI_MODEL `
  --server "Bright Data" `
  --max-tasks 50 `
  --placements retrieval `
  --retrieval-topk 20
```

```powershell
python .\scripts\run_function_call_placements.py `
  --data .\experiments\data `
  --out .\experiments\results_brightdata_retrieval_top30 `
  --model $env:OPENAI_MODEL `
  --server "Bright Data" `
  --max-tasks 50 `
  --placements retrieval `
  --retrieval-topk 30
```

### 9.2 增加 aggregation / oracle 脚本

需要写：

```text
scripts\analyze_placement_results.py
```

功能：

- 合并 full / compact / retrieval top-k 结果；
- 计算 task-level oracle placement；
- 计算每类 placement 的 win count；
- 输出 placement heterogeneity。

### 9.3 错误分析

重点看：

1. top20 中 gold 在 prompt 但工具仍选错的任务；
2. compact 失败但 full 成功的任务；
3. top30 召回更高但准确率下降的任务；
4. arguments exact mismatch 是否只是语义等价但字符串不同。

### 9.4 扩展 server

建议继续跑：

```text
Google News and Trends
Pokémon
Weather MCP Server
```

这些 server 工具数较少，可检验 Bright Data 结果是否只来自大工具池。

## 10. 当前最有价值的发现

最重要发现不是 top30 recall 更高，而是：

> top20 在 Bright Data 50 条上达到 full context 的 both_exact，同时 prompt 只有 full 的约三分之一。

这说明：

- retrieval placement 不是简单越多越好；
- 存在 top-k sweet spot；
- adaptive placement 可以基于 retrieval confidence / tool ambiguity / cost budget 决定是否使用 retrieval、compact 或 full。

这已经是 Adaptive Tool-Knowledge Placement 的早期实验证据。

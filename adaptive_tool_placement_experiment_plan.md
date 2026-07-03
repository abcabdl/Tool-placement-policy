# Adaptive Tool-Knowledge Placement 实验计划

整理日期：2026-07-03  
工作目录：`C:\Users\zrz20\Desktop\vscode\Tool`

## 0. 总体策略

先不要复现 ParaTool / AgentVocab / ToolGen 这种重模型或重训练路线。

第一阶段目标不是立刻证明新方法，而是快速判断：

> **Adaptive Tool-Knowledge Placement 是否真的有信号？**

也就是：不同任务 / 工具是否真的适合不同 placement。

第一阶段只做低成本复现和诊断：

1. Retrieval Memory baseline；
2. Compact Context baseline；
3. Full Context upper/reference baseline；
4. Oracle Placement；
5. 如果有明显 placement heterogeneity，再做 Procedural Skill 和 AdaToolPlace policy。

## 1. 优先级 1：Retrieval Memory Baseline

### 为什么先做

Retrieval Memory 是最大工具池场景的基础，也是所有大规模 tool-use agent 的强 baseline。

如果 retrieval-only 已经很好，placement 的空间会变小。  
如果 retrieval 经常错工具，adaptive placement 就有意义。

### 数据

优先使用：

- MCP-Flow；
- ToolBench；
- ToolRet。

### 做法

1. 解析工具文档。
2. 为每个工具建立 tool card。
3. 建立三种 retriever：
   - BM25；
   - embedding；
   - BM25 + embedding hybrid。
4. 输入 task query。
5. 检索 top-k tools。
6. 把 top-k tool cards 放进 prompt。
7. 让 agent 生成 tool call。

### 第一阶段配置

```text
BM25
embedding
BM25 + embedding hybrid
top-k = 3 / 5 / 10
```

### 指标

- tool selection accuracy；
- argument accuracy；
- task success；
- prompt tokens；
- completion tokens；
- latency；
- success / 1K tokens。

## 2. 优先级 2：Compact Context Baseline

### 为什么第二个做

Compact Context 和 Retrieval Memory 对比最直接。它是低成本强 baseline，很可能在工具数量不大时比 retrieval 更稳。

### 做法

把每个工具压缩成统一 tool card：

```text
tool_name
one-line function
required args
optional args
return fields
one short example
common failure
```

实验两种设置：

1. 所有候选工具都用 compact card；
2. 只把 retriever 召回的 top-k 工具变成 compact card。

### 需要比较

```text
Full Context
Compact Context
Retrieval Memory
```

### 关键判断

如果 Compact Context 接近 Full Context，但 token 大幅降低，则 Compact Context 本身就是强 placement。

### 指标

- tool selection accuracy；
- argument accuracy；
- task success；
- token reduction；
- context overflow rate；
- malformed call rate。

## 3. 优先级 3：Oracle Placement

### 为什么关键

Oracle Placement 是 go / no-go gate。

在训练 adaptive policy 前，先离线计算每个 task 的最优 placement：

```text
Full Context
Compact Context
Retrieval Memory
```

如果结果是：

```text
80% 任务都是同一种 placement 最好
```

那 Adaptive Placement 没必要做。

如果结果是：

```text
不同任务 / 工具类型最佳 placement 明显不同
```

那这个方向成立。

### 做法

对每个 task 跑三套系统：

1. Full Context；
2. Compact Context；
3. Retrieval Memory。

记录每个 placement 的：

- success；
- tool accuracy；
- argument accuracy；
- token cost；
- latency。

然后定义 oracle：

```text
oracle(task) = 在满足成功优先、成本次优的条件下选择最佳 placement
```

建议 ranking：

1. 成功优先；
2. 成功相同选 token 更少；
3. token 相近选 latency 更低；
4. 都相近选更简单 placement。

### 输出

- 每个 task 的 oracle placement；
- 每类工具的最佳 placement 分布；
- oracle vs fixed placement 的上界差距；
- placement heterogeneity 分析。

## 4. 优先级 4：Procedural Skill Baseline

### 为什么第四个做

Procedural Skill 比前两个复杂，只有当任务里真的存在多步工具流程时才有价值。

太早做容易变成 another memory system。

### 适合任务

多步工具任务，例如：

```text
search -> filter -> open -> verify -> summarize
```

### 轻量做法

不用完整复现 Skill-Pro。先做 lightweight procedural skill：

1. 从成功轨迹中抽取 tool sequence；
2. 写成 skill card；
3. 根据 task similarity 检索 skill；
4. 注入 prompt。

### Skill card 格式

```text
activation condition
tool sequence
argument filling rule
verification step
failure recovery
termination condition
```

### 对比

```text
Retrieval Memory only
Retrieval Memory + Procedural Skill
Compact Context only
Compact Context + Procedural Skill
```

### 指标

- multi-step success；
- tool order accuracy；
- unnecessary tool-call reduction；
- failure recovery rate；
- stale skill error；
- token cost。

## 5. 优先级 5：AdaToolPlace Policy

### 什么时候做

只有当前四个实验跑出信号后再做。

尤其需要先确认：

```text
tool/task features 可以预测最佳 placement
```

### 输入特征

```text
tool frequency
doc length
argument complexity
schema stability
retrieval ambiguity
task type
model size
```

### 输出

```text
Compact Context
Retrieval Memory
Procedural Skill
```

### 模型

先用简单模型：

```text
decision tree
logistic regression
gradient boosting
```

不要一开始用：

- LLM controller；
- RL；
- 大模型 fine-tuning。

### 指标

- policy accuracy vs oracle；
- placement regret；
- success-cost Pareto；
- feature importance；
- generalization to unseen tools。

## 6. 暂时不要先复现的重方法

### 6.1 ParaTool

参数化工具知识，训练成本高。  
等 placement 方向有信号后，再作为 heavy baseline。

### 6.2 AgentVocab

Tokenizer / vocabulary adaptation 更重，且需要大量 tool-call traces。  
后期再加。

### 6.3 ToolGen

Tool token 路线也重，不适合作第一阶段。

### 6.4 FTRL / GOAT full training

它们适合提供数据/任务，不适合一开始完整复现训练 pipeline。

## 7. 推荐复现顺序

```text
R1. MCP-Flow / ToolBench 数据读取 + tool docs 解析
R2. Retrieval Memory baseline: BM25 / embedding / hybrid
R3. Compact Context baseline: compressed tool cards
R4. Full Context baseline: 上界和 token-cost 对照
R5. Oracle Placement: 每个 task 最优 placement 分布
R6. Procedural Skill lightweight baseline
R7. AdaToolPlace: decision tree / GBDT policy
R8. Schema drift split
R9. Small-model setting
R10. ParaTool / AgentVocab 轻量复现或 reported comparison
```

## 8. 第一周目标

第一周只做四件事：

1. 选 50-100 个 MCP-Flow 或 ToolBench tasks。
2. 跑 Full Context。
3. 跑 Compact Context。
4. 跑 Retrieval Memory。
5. 算 Oracle Placement。

第一周的核心问题只有一个：

> 是否存在明显的 placement heterogeneity？

也就是：

> 不同任务 / 工具是否真的适合不同 placement？

## 9. Go / No-Go Gate

### Go

继续做 AdaToolPlace，如果满足：

- oracle placement 明显优于任一固定 placement；
- 不同工具类型的最佳 placement 分布不同；
- compact / retrieval / full context 各自都有优势场景；
- token-cost 和 success 存在明显 trade-off。

### No-Go

考虑转向 schema drift 或 small-model setting，如果出现：

- 大多数任务同一种 placement 最好；
- retrieval-only 已经接近 oracle；
- compact context 始终不掉点且 token 最低；
- placement 特征无法预测 oracle。

## 10. 后续分支

### 分支 A：Schema Drift

如果普通任务上 heterogeneity 不够明显，构造 schema drift：

- 参数名变化；
- required field 增加；
- 返回字段变化；
- 工具拆分 / 合并；
- 新工具加入；
- 旧工具废弃。

检验：

```text
stable tools -> parameter/vocab/compact 更好？
unstable tools -> context/retrieval 更好？
```

### 分支 B：Small Model

如果大模型上 placement 差异不明显，换小模型：

- 1.5B；
- 3B；
- 7B；
- 8B。

假设：

> 小模型更依赖合适的 tool knowledge placement。

### 分支 C：Heavy Baselines

只有当方向确认后，再考虑：

- ParaTool；
- AgentVocab；
- ToolGen；
- FTRL / GOAT training。

## 11. 实验记录模板

| Run ID | Placement | Retriever | Top-k | Dataset | Model | Task Count | Success | Tool Acc | Arg Acc | Prompt Tokens | Latency | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R001 | Full Context | None | All | MCP-Flow | TBD | 50 | TBD | TBD | TBD | TBD | TBD | sanity |
| R002 | Compact Context | None | All | MCP-Flow | TBD | 50 | TBD | TBD | TBD | TBD | TBD | compact card |
| R003 | Retrieval Memory | BM25 | 3 | MCP-Flow | TBD | 50 | TBD | TBD | TBD | TBD | TBD | retrieval |
| R004 | Retrieval Memory | BM25 | 5 | MCP-Flow | TBD | 50 | TBD | TBD | TBD | TBD | TBD | retrieval |
| R005 | Retrieval Memory | Hybrid | 5 | MCP-Flow | TBD | 50 | TBD | TBD | TBD | TBD | TBD | retrieval |

## 12. 当前最重要判断

短期不要追求完整系统。

先回答：

```text
Does placement heterogeneity exist?
```

如果答案是 yes，再做 AdaToolPlace。

如果答案是 no，及时止损，转向 schema drift 或 small model。

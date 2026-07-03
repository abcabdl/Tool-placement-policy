# MCP-Flow Level 1 复现命令

整理日期：2026-07-03  
工作目录：`C:\Users\zrz20\Desktop\vscode\Tool`

## 0. Level 1 分成两层

### Level 1A：无 API key 的离线 sanity

目标：

- 解析 MCP-Flow 样例；
- 构造 `tools.jsonl` 和 `tasks.jsonl`；
- 跑 BM25 Retrieval Memory baseline；
- 看 gold tool 是否能被检索到 top-k。

不需要 API key。

### Level 1B：需要模型的 function-call 评测

目标：

- 对比 Full Context / Compact Context / Retrieval Memory；
- 让模型生成 function_call；
- 评估 tool name 和 arguments；
- 算 oracle placement。

需要：

- OpenAI / 兼容 API key；或
- 本地 Ollama / LMStudio 模型。

没有模型调用，就不能真正比较 Full / Compact / Retrieval 的最终 task success，只能比较检索覆盖率和 token 成本。

## 1. 当前已完成：Level 1A

### 1.1 解析 MCP-Flow 本地样例

命令：

```powershell
python C:\Users\zrz20\Desktop\vscode\Tool\scripts\parse_mcp_flow_level1.py `
  --repo C:\Users\zrz20\Desktop\vscode\Tool\MCP-Flow `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\data
```

当前输出：

```text
tools: 1032
tasks: 1839
tasks with local tool schema: 1839 / 1839
task servers: 12
```

生成文件：

```text
C:\Users\zrz20\Desktop\vscode\Tool\experiments\data\tools.jsonl
C:\Users\zrz20\Desktop\vscode\Tool\experiments\data\tasks.jsonl
C:\Users\zrz20\Desktop\vscode\Tool\experiments\data\tool_cards_full.jsonl
C:\Users\zrz20\Desktop\vscode\Tool\experiments\data\tool_cards_compact.jsonl
```

### 1.2 跑 BM25 Retrieval baseline

命令：

```powershell
python C:\Users\zrz20\Desktop\vscode\Tool\scripts\run_retrieval_baseline.py `
  --data C:\Users\zrz20\Desktop\vscode\Tool\experiments\data `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\results
```

当前结果：

| Retrieval Doc Mode | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| name_desc | 0.336 | 0.258 | 0.364 | 0.421 | 0.498 |
| compact | 0.383 | 0.292 | 0.424 | 0.482 | 0.568 |
| full | 0.343 | 0.256 | 0.373 | 0.431 | 0.524 |

初步观察：

> 用 compact tool card 建 BM25 文档，比 full JSON 和 name/description 更好。

这说明工具文档压缩本身可能有价值，至少在 retrieval 阶段更利于匹配。

生成文件：

```text
C:\Users\zrz20\Desktop\vscode\Tool\experiments\results\retrieval_summary.json
C:\Users\zrz20\Desktop\vscode\Tool\experiments\results\retrieval_predictions.jsonl
```

## 2. 怎么看 Full / Compact / Retrieval 的最优 placement 是否不同

必须跑 Level 1B。

对每个 task，需要让同一个模型分别在三种输入下生成 function_call：

```text
Full Context
Compact Context
Retrieval Memory
```

然后评估：

```text
tool name 是否等于 gold_tool
arguments 是否等于 gold_arguments 或语义等价
JSON 是否有效
prompt tokens
latency
```

每个 task 的 oracle placement：

```text
如果只有一个 placement 成功，选成功的。
如果多个 placement 都成功，选 token 更少的。
如果都失败，记录 none。
```

然后看：

```text
Full / Compact / Retrieval 分别成为 oracle 的比例
```

如果三者都有明显占比，说明 placement heterogeneity 存在。

如果 80% 以上都由同一个 placement 赢，Adaptive Placement 价值较弱。

## 3. Level 1B 是否需要 API key

需要模型生成 function_call，所以需要以下之一：

### 方案 A：OpenAI / 兼容 API

需要：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="..."   # 可选，如果用兼容 API
$env:OPENAI_MODEL="..."
```

优点：

- 最简单；
- 输出质量稳定；
- 适合快速比较 placement。

缺点：

- 有 API 成本。

### 方案 B：Ollama

需要本地启动：

```powershell
ollama serve
ollama pull qwen2.5:7b
```

优点：

- 不需要 API key；
- 适合低成本 sanity。

缺点：

- tool-call JSON 稳定性可能差；
- 速度慢；
- 需要本地模型能力足够。

### 方案 C：LMStudio

需要启动本地 OpenAI-compatible server。

优点：

- 可走 OpenAI-compatible API；
- 不需要云端 key。

缺点：

- 需要手动加载模型。

## 3.1 推荐先用什么模型

第一轮建议用便宜、稳定、函数调用/JSON 能力还可以的轻量模型。

如果是 OpenAI 官方 API：

```powershell
$env:OPENAI_MODEL="gpt-4o-mini"
```

如果是 OpenAI-compatible API：

```powershell
$env:OPENAI_MODEL="你的轻量强模型名"
```

建议顺序：

1. 先用便宜轻量模型跑 20-50 条 sanity。
2. 如果 JSON / argument 很不稳，再换强一点的模型。
3. 不要一开始跑全量 1839 条。

## 3.2 Level 1B 直接运行命令

### OpenAI 官方 API

```powershell
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_MODEL="gpt-4o-mini"

python C:\Users\zrz20\Desktop\vscode\Tool\scripts\run_function_call_placements.py `
  --data C:\Users\zrz20\Desktop\vscode\Tool\experiments\data `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\results `
  --model $env:OPENAI_MODEL `
  --max-tasks 20 `
  --placements full compact retrieval `
  --max-candidate-tools 30 `
  --retrieval-topk 5
```

### OpenAI-compatible API

如果你的 API 是兼容 OpenAI 格式的：

```powershell
$env:OPENAI_API_KEY="你的key"
$env:OPENAI_BASE_URL="https://你的服务地址/v1"
$env:OPENAI_MODEL="你的模型名"

python C:\Users\zrz20\Desktop\vscode\Tool\scripts\run_function_call_placements.py `
  --data C:\Users\zrz20\Desktop\vscode\Tool\experiments\data `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\results `
  --base-url $env:OPENAI_BASE_URL `
  --model $env:OPENAI_MODEL `
  --max-tasks 20 `
  --placements full compact retrieval `
  --max-candidate-tools 30 `
  --retrieval-topk 5
```

### 只跑某个 server 做 sanity

比如先跑 Calculator：

```powershell
python C:\Users\zrz20\Desktop\vscode\Tool\scripts\run_function_call_placements.py `
  --data C:\Users\zrz20\Desktop\vscode\Tool\experiments\data `
  --out C:\Users\zrz20\Desktop\vscode\Tool\experiments\results `
  --model $env:OPENAI_MODEL `
  --server "Calculator" `
  --max-tasks 10 `
  --placements full compact retrieval `
  --max-candidate-tools 30 `
  --retrieval-topk 5
```

## 3.3 输出文件

运行后生成：

```text
C:\Users\zrz20\Desktop\vscode\Tool\experiments\results\placement_predictions.jsonl
C:\Users\zrz20\Desktop\vscode\Tool\experiments\results\placement_summary.json
```

`placement_summary.json` 会给出：

```text
gold_in_prompt_rate
valid_json_rate
tool_accuracy
argument_exact
both_exact
avg_prompt_chars
avg_latency_sec
```

`placement_predictions.jsonl` 会保存每个 task、每个 placement 的原始模型输出和解析结果。

## 4. 下一步该写的脚本

下一步脚本：

```text
scripts/run_function_call_placements.py
```

功能：

1. 读取 `tasks.jsonl` 和 `tools.jsonl`。
2. 为每个 task 构造三种 prompt：
   - Full Context；
   - Compact Context；
   - Retrieval Memory top-k。
3. 调用模型。
4. 解析模型输出 JSON。
5. 评估 tool 和 arguments。
6. 输出 placement 对比表。

输出：

```text
experiments\results\placement_predictions.jsonl
experiments\results\placement_summary.json
experiments\results\oracle_placement.jsonl
```

## 5. 当前结论

不用 API key 能先完成：

- 数据解析；
- 工具 schema 检查；
- BM25 retrieval baseline；
- retrieval recall；
- compact/full/name_desc 文档形式对 retrieval 的影响。

需要 API key 或本地模型才能完成：

- Full Context function-call generation；
- Compact Context function-call generation；
- Retrieval Memory function-call generation；
- Oracle Placement；
- Adaptive Placement 是否有最终 task-level 信号。

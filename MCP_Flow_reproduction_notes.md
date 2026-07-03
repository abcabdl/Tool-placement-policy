# MCP-Flow 复现与使用笔记

整理日期：2026-07-03  
本地路径：`C:\Users\zrz20\Desktop\vscode\Tool\MCP-Flow`

## 1. Clone 状态

仓库：

```text
https://github.com/wwh0411/MCP-Flow
```

Windows 直接 clone 会失败 checkout，因为仓库里有 Windows 非法文件名，例如：

```text
data/mcp_config/manual/MCP Server: Mermaid Validator.json
data/mcp_config/smithery/Where's my train? MTA Guide.json
data/tools/manual/MCP Server: Mermaid Validator.json
data/tools/smithery/Where's my train? MTA Guide.json
```

当前采用的方式：

```text
git clone --no-checkout https://github.com/wwh0411/MCP-Flow.git MCP-Flow
git checkout HEAD -- README.md assets src wizard_utils data/context data/function_call data/gaia_103_qa_gt.json
```

然后额外 checkout 了安全的工具 schema：

```text
data/tools/deepnlp/
data/tools/smithery/Calculator.json
data/tools/smithery/Weather MCP Server.json
data/tools/smithery/Weather Query Server.json
data/tools/smithery/Weather360 Server.json
data/tools/smithery/Bright Data.json
...
```

## 2. README 关键信息

MCP-Flow 是 ACL 2026 Main Conference 论文：

```text
MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools
```

README 声称包含：

- 1,166 real-world servers；
- 11,536 tools；
- 68K+ instruction-function call pairs；
- function call / trajectory / testset 数据主要发布在 HuggingFace：

```text
https://huggingface.co/datasets/wwh0411/MCP-Flow
```

本 GitHub 仓库只包含样例数据和 MCP client 代码。

## 3. 本地数据结构

### 3.1 Function-call 样例

路径：

```text
data/function_call/Smithery/*.json
```

样例格式：

```json
{
  "function_call": {
    "name": "mcp-server-calculator_calculate",
    "arguments": {
      "expression": "7^4"
    }
  },
  "source_instruction": "Find the value of 7 raised to the power of 4.",
  "tool": "mcp-server-calculator_calculate"
}
```

可直接用于：

- tool selection accuracy；
- argument accuracy；
- full/compact/retrieval placement 对比。

### 3.2 Tool schema

路径：

```text
data/tools/smithery/*.json
data/tools/deepnlp/*.json
```

样例格式：

```json
[
  {
    "name": "mcp-server-calculator_calculate",
    "description": "Calculates/evaluates the given expression.",
    "parameters": {
      "properties": {
        "expression": {
          "title": "Expression",
          "type": "string"
        }
      },
      "required": ["expression"],
      "title": "calculateArguments",
      "type": "object"
    }
  }
]
```

可转成：

- Full Context；
- Compact Context；
- Retrieval Memory index；
- tool feature table。

## 4. 代码结构

`src` 下是 `dolphin-mcp` 包，主要是 MCP client：

```text
src/dolphin_mcp/client.py
src/dolphin_mcp/cli.py
src/dolphin_mcp/providers/openai.py
src/dolphin_mcp/providers/anthropic.py
src/dolphin_mcp/providers/ollama.py
src/dolphin_mcp/providers/lmstudio.py
```

依赖写在：

```text
src/dolphin_mcp.egg-info/requires.txt
```

包括：

```text
openai
mcp[cli]
python-dotenv
anthropic
ollama
jsonschema
PyYAML
```

仓库没有 `requirements.txt`，README 中的安装说明不完整。

对当前 Adaptive Tool-Knowledge Placement 实验来说，第一阶段不需要运行 dolphin-mcp client，只需要解析 JSON 数据。

## 5. 推荐复现路线

### Stage 0: 数据解析

目标：

从本地样例中构建：

```text
tools.jsonl
tasks.jsonl
tool_cards_full.jsonl
tool_cards_compact.jsonl
```

输入：

```text
data/function_call/Smithery/*.json
data/tools/smithery/*.json
data/tools/deepnlp/*.json
```

### Stage 1: Full Context baseline

对每个 task，把候选工具完整 schema 放进 prompt。

指标：

- tool selection accuracy；
- argument accuracy；
- prompt tokens。

### Stage 2: Compact Context baseline

把工具 schema 压缩成：

```text
tool_name
one-line description
required args
optional args
return fields
one short example
common failure
```

再跑相同任务。

### Stage 3: Retrieval Memory baseline

为工具 schema 建索引：

```text
BM25
embedding
hybrid
```

对每个 task 检索 top-k 工具：

```text
top-k = 3 / 5 / 10
```

再把 top-k 工具 schema 放进 prompt。

### Stage 4: Oracle Placement

对每个 task 比较：

```text
Full Context
Compact Context
Retrieval Memory
```

离线选择最佳 placement，判断是否存在 placement heterogeneity。

## 6. 第一阶段 Go / No-Go

### Go

继续做 AdaToolPlace，如果：

- 不同工具类型最佳 placement 不同；
- oracle placement 明显优于任一固定 placement；
- retrieval 和 compact context 各有优势场景；
- token-cost 与 success 存在明显 trade-off。

### No-Go

转向 schema drift 或 small-model setting，如果：

- 大多数任务同一种 placement 最好；
- retrieval-only 接近 oracle；
- compact context 几乎总是最优；
- placement 特征无法预测 oracle。

## 7. 下一步建议

优先写一个数据解析脚本：

```text
parse_mcp_flow.py
```

输出：

```text
experiments/data/tools.jsonl
experiments/data/tasks.jsonl
experiments/data/tool_cards_full.jsonl
experiments/data/tool_cards_compact.jsonl
```

然后先跑本地 Smithery 样例 sanity check，再决定是否下载 HuggingFace 全量数据。

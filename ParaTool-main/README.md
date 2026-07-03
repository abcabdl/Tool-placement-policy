# ParaTool: Shifting Tool Representations from Context to Parameters

Official implementation of **ParaTool** (ICML 2026) a framework that projects each tool into a dedicated, loadable set of parameters. By equipping a dynamic integration of these parameterized tools, the LLM can perform tool calling without relying on in-context documents or examples.

This repository covers the full training and evaluation pipeline on [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) and [StableToolBench](https://github.com/THUNLP-MT/StableToolBench).

## Overview

Mainstream tool-use paradigms predominantly rely on in-context learning (ICL), which typically integrates detailed tool documentation and usage examples directly into the input prompt. To address the resulting inference overhead and limited internalization of tool-specific details, we propose **ParaTool**, where each tool is projected into a dedicated set of parameters. By dynamically loading a weighted aggregation of these parameterized tools, the LLM can perform tool calling without relying on in-context documents or examples.

Specifically, our approach consists of three stages:

1. **Parametric tool pre-training** encapsulates the knowledge of different tools into independent parameter modules (`tool_pretraining.py`).
2. **Soft tool selection** employs a gating network to dynamically weigh and aggregate relevant tool parameters (`soft_tool_selection_train.py`).
3. **Parametric tool fine-tuning** jointly updates tool parameters to align the training and inference processes (`tool_finetuning.py`).

At inference (`bfcl_inference.py`, `stb_inference.py`), the model predicts actions with parameter-based tool representations—assigning aggregation weights to candidate tools and composing their parameters—rather than concatenating tool documents and examples into the prompt.

## Repository Layout

```text
.
├── tool_pretraining.py              # Stage 1: per-tool LoRA (tool pretraining)
├── soft_tool_selection_train.py     # Stage 2: GatingNetwork / soft tool selection
├── tool_finetuning.py               # Stage 3: gated LoRA expert finetuning
├── gating_network.py                # GatingNetwork, GatedLoraLinear, encoders
├── bfcl_inference.py                # BFCL parametric inference & evaluation
├── stb_inference.py                 # StableToolBench parametric inference
├── dataset/                         # JSONL loaders and training sample builders
├── prompts/                         # BFCL / STB prompt templates
├── configs/                         # YAML config templates
└── data/                            # BFCL / STB JSONL (see Data section)
```

## Installation

```bash
conda create -n paratool python=3.10 -y
conda activate paratool
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version before or after the command above, if the default wheel is not suitable for your GPU.

## External Benchmark Code

### BFCL

Clone Gorilla and expose the BFCL package at `berkeley-function-call-leaderboard/`:

```bash
git clone https://github.com/ShishirPatil/gorilla.git external/gorilla
ln -sfn external/gorilla/berkeley-function-call-leaderboard berkeley-function-call-leaderboard
```

### StableToolBench

```bash
git clone https://github.com/THUNLP-MT/StableToolBench.git StableToolBench
```

StableToolBench provides `toolbench`, `toolenv`, solvable-query files, and evaluator utilities used by `stb_inference.py`.

## Data

Place processed **JSONL** files under:

```text
data/BFCL/
data/STB/
```

Expected filenames (one JSON object per line):

```text
data/BFCL/multiple.jsonl
data/BFCL/parallel.jsonl
data/BFCL/parallel_multiple.jsonl
data/BFCL/live_multiple.jsonl
data/BFCL/live_parallel.jsonl
data/BFCL/live_parallel_multiple.jsonl
data/STB/G1_category.jsonl
data/STB/G1_instruction.jsonl
data/STB/G1_tool.jsonl
data/STB/G2_category.jsonl
data/STB/G2_instruction.jsonl
data/STB/G3_instruction.jsonl
```

Each record bundles tool schemas with `training_data` entries (question, candidate tools, and gold tool call). See `data/samples/` for minimal examples.

**Download (training data):**

```text
https://huggingface.co/datasets/nostaigia/ParaTool
```

## Models and Paths

Default Hugging Face model ids:


| Shortcut               | Hugging Face id                    |
| ---------------------- | ---------------------------------- |
| `Qwen2.5-7b-instruct`  | `Qwen/Qwen2.5-7B-Instruct`         |
| `Qwen2.5-14B-Instruct` | `Qwen/Qwen2.5-14B-Instruct`        |
| `Llama3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` |


Override with environment variables:

```bash
export QWEN25_7B_INSTRUCT_PATH=/path/to/Qwen2.5-7B-Instruct
export QWEN25_14B_INSTRUCT_PATH=/path/to/Qwen2.5-14B-Instruct
export LLAMA31_8B_INSTRUCT_PATH=/path/to/Llama-3.1-8B-Instruct
```

Artifact roots (defaults under `output/paratool/`):

```bash
export PARAM_ROOT_PATH=/path/to/paratool_outputs
```

For `q_encoder=bge`, the default encoder id is `BAAI/bge-small-en-v1.5`:

```bash
export BGE_QENC_MODEL_PATH=/path/to/bge-small-en-v1.5
```

## Training Pipeline

Edit copies of the YAML templates in `configs/`, or pass CLI flags. Run stages **in order** for a given `(dataset, category, model_name)` triple.

### Stage 1 — Tool pretraining

Trains one LoRA adapter per tool on synthetic / augmented tool-call QA.

```bash
python tool_pretraining.py \
  --config configs/tool_pretraining.yaml 
```

### Stage 2 — Soft tool selection (gating)

Trains `GatingNetwork` with cross-entropy over the gold tool index (plus optional entropy regularization).

```bash
python soft_tool_selection_train.py \
  --config configs/soft_tool_selection_train.yaml
```

Optional BGE question encoder (`q_encoder: bge`): a local sentence encoder embeds the question; set weights via `env.BGE_QENC_MODEL_PATH` in the YAML, or pass `--bge_model_path`. Checkpoints are tagged with `_qenc-bge` in the filename.

```bash
python soft_tool_selection_train.py \
  --config configs/soft_tool_selection_train.yaml \
  --q_encoder bge \
  --bge_model_path /path/to/bge-small-en-v1.5
```

### Stage 3 — Tool finetuning

Trains gated LoRA experts using routing weights; enable gating with the stage-2 checkpoint:

```bash
python tool_finetuning.py \
  --config configs/tool_finetuning.yaml
```

## Inference

Parametric inference composes stage-1 adapters, stage-3 experts, and (when `--gating true`) the stage-2 checkpoint. Outputs are written under `output/<dataset>/<model>/<category>/parameter(+gating)/`.

### BFCL

Offline JSONL eval on [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) categories (`multiple`, `parallel`, `live_multiple`, …). Scoring uses the bundled BFCL AST checker; use `--only_idx` to debug a single sample.

```bash
python bfcl_inference.py \
  --config configs/inference.yaml \
  --category multiple \
  --tool_pretraining_lr 1e-4 --tool_pretraining_epochs 3 \
  --tool_finetuning_lr 1e-4 --tool_finetuning_epochs 1
```

### StableToolBench

Live multi-step tool calling against a running MirrorAPI server (`STB_MIRRORAPI_URL`, default `http://127.0.0.1:8080/virtual`). Categories are STB splits (`G1_category`, `G1_instruction`, …). Optional tooleval scoring needs OpenAI credentials unless you pass `--disable_eval`.

```bash
python stb_inference.py \
  --config configs/inference.yaml \
  --category G1_category \
  --tool_pretraining_lr 1e-4 --tool_pretraining_epochs 3 \
  --tool_finetuning_lr 1e-4 --tool_finetuning_epochs 1 \
  --max_steps 30
```

## Citation

If you use this code or **ParaTool** in your work, please cite:

```bibtex
@inproceedings{paratool2026icml,
  title     = {ParaTool: Shifting Tool Representations from Context to Parameters},
  author    = {Zekai Yu and Qi Meng and Qizhi Chu and Yu Hao and Chuan Shi and Cheng Yang},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
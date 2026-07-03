import os
import sys
import json
import gc
import argparse
import glob
import uuid
from datetime import datetime

import sitecustomize  # noqa: F401  -- forces third_party / BFCL paths to load

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from dataset import get_dataset
from dataset.tool_finetuning_dataset import ToolFinetuningIndex
from dataset.tool_keys import tool_key_for_schema

from gating_network import (
    GatingNetwork,
    GatingQuestionEncoder,
    make_gating_question_encoder,
    normalize_gating_q_encoder_mode,
    map_gating_alphas_with_finish_to_experts,
    resolve_tool_ids,
    _replace_with_gated_lora_linear,
    _load_lora_experts_for_paths,
    _normalize_module_name,
)
from root_dir_path import (
    ROOT_DIR,
    TOOL_FINETUNING_ROOT_PATH,
)
from utils import (
    STB_DATASET_NAMES,
    _safe_filename_component,
    adapter_exists,
    get_model,
    normalize_dataset_name,
    train_filename,
)
from dotenv import load_dotenv

load_dotenv()


def _require_tool_finetuning_locators(args: argparse.Namespace) -> "tuple[float, int]":
    tool_finetuning_lr = getattr(args, "tool_finetuning_lr", None)
    tool_finetuning_epochs = getattr(args, "tool_finetuning_epochs", None)
    missing = []
    if tool_finetuning_lr is None:
        missing.append("tool_finetuning_lr")
    if tool_finetuning_epochs is None:
        missing.append("tool_finetuning_epochs")
    if missing:
        raise ValueError(
            "Parametric inference requires tool_finetuning locators. Missing: "
            + ", ".join(missing)
            + ". Provide `--tool_finetuning_lr` and `--tool_finetuning_epochs` (or set them in your YAML config)."
        )
    assert tool_finetuning_lr is not None
    assert tool_finetuning_epochs is not None
    return float(tool_finetuning_lr), int(tool_finetuning_epochs)

def _require_tool_pretraining_locators(args: argparse.Namespace) -> "tuple[float, int]":
    tool_pretraining_lr = getattr(args, "tool_pretraining_lr", None)
    tool_pretraining_epochs = getattr(args, "tool_pretraining_epochs", None)
    missing = []
    if tool_pretraining_lr is None:
        missing.append("tool_pretraining_lr")
    if tool_pretraining_epochs is None:
        missing.append("tool_pretraining_epochs")
    if missing:
        raise ValueError(
            "Parametric inference requires tool pretraining adapter locators. Missing: "
            + ", ".join(missing)
            + ". Provide `--tool_pretraining_lr` and `--tool_pretraining_epochs` (or set them in your YAML config)."
        )
    assert tool_pretraining_lr is not None
    assert tool_pretraining_epochs is not None
    return float(tool_pretraining_lr), int(tool_pretraining_epochs)

def _parse_lr_epoch_segment(seg: str) -> Optional[Tuple[float, int]]:
    s = str(seg or "").strip()
    if not s.startswith("lr=") or "_epoch=" not in s:
        return None
    try:
        lr_str = s[len("lr=") : s.index("_epoch=")]
        ep_str = s[s.index("_epoch=") + len("_epoch=") :]
        return float(lr_str), int(ep_str)
    except Exception:
        return None

def _infer_tool_pretraining_locators_from_tool_finetuning(
    *,
    dataset_train: str,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    tool_finetuning_lr: float,
    tool_finetuning_epochs: int,
    category_file: str,
    run_tag: Optional[str] = None,
) -> "tuple[float, int]":
    rank_dir = f"rank={int(lora_rank)}_alpha={int(lora_alpha)}"
    ds_dir = str(dataset_train or "").strip().lower() or "bfcl"
    cat = str(category_file or "").strip()
    if not cat.endswith(".jsonl"):
        cat = train_filename(cat)
    run_segments = []
    tag = str(run_tag or "").strip()
    if tag:
        run_segments.append(f"run={_safe_filename_component(tag)}")

    root = os.path.join(TOOL_FINETUNING_ROOT_PATH, model_name, rank_dir, ds_dir)
    pattern = os.path.join(
        root,
        "lr=*_epoch=*",
        f"tool_finetuning_lr={float(tool_finetuning_lr)}_epoch={int(tool_finetuning_epochs)}",
        *run_segments,
        cat,
        "tool_index.json",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Tool finetuning run not found for category={cat}. Tried: {pattern}"
        )

    pairs = {}
    for m in matches:
        parts = os.path.normpath(m).split(os.sep)
        parsed = None
        for p in parts:
            parsed = _parse_lr_epoch_segment(p)
            if parsed is not None:
                break
        if parsed is not None:
            pairs.setdefault(parsed, []).append(m)

    if not pairs:
        raise RuntimeError(
            f"Found tool_finetuning runs but failed to parse base lr/epoch from paths: {matches[:3]}"
        )
    if len(pairs) > 1:
        desc = []
        for (lr, ep), paths in sorted(pairs.items(), key=lambda x: (x[0][0], x[0][1])):
            desc.append(f"lr={lr}_epoch={ep} (n={len(paths)})")
        raise ValueError(
            "Multiple tool pretraining adapter locators match this tool_finetuning run. "
            "Please specify `--tool_pretraining_lr` and `--tool_pretraining_epochs`. Candidates: "
            + ", ".join(desc)
        )
    (lr, ep), _ = next(iter(pairs.items()))
    return float(lr), int(ep)

def _normalize_dataset_name_for_output(dataset: str) -> str:
    ds = str(dataset or "").strip().lower()
    if ds in STB_DATASET_NAMES:
        return "stable_toolbench"
    return ds or "bfcl"

def _make_output_stamp() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"

def _normalize_category_for_output(category: str) -> str:
    cat = str(category or "").strip().lower().replace("-", "_").replace(" ", "_")
    return cat or "all"

def _infer_experiment_variant_dir(args: argparse.Namespace) -> str:
    return "parameter+gating" if bool(getattr(args, "gating", False)) else "parameter"

def _build_output_root_dir(args: argparse.Namespace) -> str:
    dataset_out = _normalize_dataset_name_for_output(getattr(args, "dataset", ""))
    category_out = _normalize_category_for_output(getattr(args, "category", "") or "all")
    variant_out = _infer_experiment_variant_dir(args)
    stamp = _make_output_stamp()

    base = os.path.join(ROOT_DIR, "output", dataset_out, category_out, variant_out)

    return os.path.join(base, stamp)

def _inject_gated_lora_experts(
    *,
    model: nn.Module,
    gated_modules: Dict[str, Any],
    adapter_files: List[str],
    target_modules: List[str],
    lora_alpha: int,
    lora_rank: int,
) -> None:
    base_dtype = next(model.parameters()).dtype
    experts_per_module = _load_lora_experts_for_paths(
        adapter_files,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        r=lora_rank,
        device=model.device,
        dtype=base_dtype,
    )
    for full_name, module in gated_modules.items():
        matched_experts = []
        norm_full = _normalize_module_name(full_name)
        for mod_name, exps in experts_per_module.items():
            short = _normalize_module_name(mod_name)
            if norm_full == short:
                matched_experts.extend(exps)
        module.set_experts(matched_experts)
        module.set_alphas(None)

def _encode_tools_for_gating(
    tools: list,
    *,
    model: nn.Module,
    tokenizer: Any,
    tool_embedding: Optional[nn.Embedding],
    tool_vocab: Dict[str, int],
    tool_vocab_type: Optional[str],
    unk_tool_id: Optional[int],
) -> torch.Tensor:
    if tool_embedding is not None and tool_vocab and isinstance(unk_tool_id, int):
        ids = resolve_tool_ids(
            tools,
            tool_vocab=tool_vocab,
            tool_vocab_type=tool_vocab_type,
            unk_tool_id=unk_tool_id,
        )
        if not ids:
            ids = [int(unk_tool_id)]
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=model.device)
        return tool_embedding(ids_tensor).unsqueeze(0)

    texts = []
    for schema in tools:
        if not isinstance(schema, dict):
            continue
        name = schema.get("name", "")
        desc = schema.get("description", "")
        texts.append(f"{name}: {desc}")
    if not texts:
        texts = [""]
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
    return hidden.mean(dim=1).unsqueeze(0)

def _clear_gated_modules(gated_modules: Dict[str, Any]) -> None:
    if not gated_modules:
        return
    for module in gated_modules.values():
        module.set_experts([])
        module.set_alphas(None)

def _set_all_alphas(
    gated_modules: Dict[str, Any],
    alphas: Optional[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if not gated_modules:
        return
    if alphas is not None and (device is not None or dtype is not None):
        alphas = alphas.to(
            device=device if device is not None else alphas.device,
            dtype=dtype if dtype is not None else alphas.dtype,
        )
    for module in gated_modules.values():
        module.set_alphas(alphas)

def _compute_gating_alphas(
    *,
    gating_net: Optional[nn.Module],
    model: nn.Module,
    q_emb: torch.Tensor,
    tool_embs: torch.Tensor,
    use_gating: bool,
    num_tools: int,
) -> torch.Tensor:
    alphas = None
    if use_gating and gating_net is not None:
        B, M, _ = tool_embs.shape
        tool_mask = torch.ones(B, M, device=model.device, dtype=torch.bool)
        dtype = next(gating_net.parameters()).dtype
        raw_alphas, _ = gating_net(
            q_emb.to(dtype=dtype),
            tool_embs.to(dtype=dtype),
            tool_mask,
            return_scores=False,
        )
        alphas = raw_alphas[0]

    if alphas is None:
        base_dtype = next(model.parameters()).dtype
        if num_tools > 0:
            alphas = torch.full(
                (num_tools,), 1.0 / float(num_tools),
                device=model.device, dtype=base_dtype,
            )
        else:
            alphas = torch.zeros(0, device=model.device, dtype=base_dtype)

    return alphas

def _load_gating_checkpoint(
    ckpt_path: str,
    device: torch.device,
    model_config: Any,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_size = int(ckpt.get("hidden_size", model_config.hidden_size))
    ckpt_hidden_dim = int(ckpt.get("hidden_dim", 512))
    gating_net = GatingNetwork(dim=hidden_size, hidden_dim=ckpt_hidden_dim).to(device)
    gating_net.load_state_dict(ckpt["gating_state_dict"])
    gating_net.eval()

    tool_embedding = None
    tool_vocab: Dict[str, int] = {}
    tool_vocab_type: Optional[str] = None
    unk_tool_id: Optional[int] = None

    emb_state = ckpt.get("tool_embedding_state_dict")
    if isinstance(emb_state, dict) and isinstance(emb_state.get("weight"), torch.Tensor):
        weight = emb_state["weight"]
        tool_embedding = nn.Embedding(
            num_embeddings=int(weight.shape[0]),
            embedding_dim=int(weight.shape[1]),
            sparse=True,
        ).to(device=device, dtype=weight.dtype)
        tool_embedding.load_state_dict(emb_state)
        tool_embedding.eval()
        vocab = ckpt.get("tool_vocab")
        if isinstance(vocab, dict):
            tool_vocab = {str(k): int(v) for k, v in vocab.items() if isinstance(v, int)}
        tool_vocab_type = str(ckpt.get("tool_vocab_type") or "").strip() or None
        if "unk_tool_id" in ckpt:
            try:
                unk_tool_id = int(ckpt["unk_tool_id"])
            except Exception:
                unk_tool_id = None

    q_encoder = normalize_gating_q_encoder_mode(ckpt.get("q_encoder", "llm"))

    return {
        "gating_net": gating_net,
        "tool_embedding": tool_embedding,
        "tool_vocab": tool_vocab,
        "tool_vocab_type": tool_vocab_type,
        "unk_tool_id": unk_tool_id,
        "q_encoder": q_encoder,
    }

def run_toolbench_inference(args):
    from toolbench_runtime import (
        ToolbenchEnv,
        run_single_query,
        prepare_tools_from_sample,
        find_meta_by_function_name,
    )

    from toolbench.tooleval.evaluators import load_registered_automatic_evaluator  # type: ignore
    from toolbench.tooleval.evaluators.registered_cls.rtl import AnswerStatus  # type: ignore
    from toolbench.tooleval.evaluation import ExecutionGraph, ExecutionNode  # type: ignore
    from toolbench.tooleval.utils import generate_init_message_node  # type: ignore

    print("[ToolBench] Loading model...")
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )

    _default_gating_question_encoder = make_gating_question_encoder(
        "llm",
        device=model.device,
        base_model=model,
        tokenizer=tokenizer,
    )
    gating_question_encoder: GatingQuestionEncoder = _default_gating_question_encoder

    evaluator = None
    if getattr(args, "disable_eval", False):
        print("[ToolBench] Evaluator disabled (--disable_eval).")
    else:
        try:
            evaluator_name = getattr(args, "evaluator_name", None)
            eval_model = getattr(args, "eval_model", None)
            if not evaluator_name:
                if eval_model:
                    evaluator_name = f"tooleval_{eval_model}_default"
                else:
                    evaluator_name = "tooleval_gpt-3.5-turbo_default"

            evaluators_dir = os.path.join(
                ROOT_DIR, "StableToolBench", "toolbench", "tooleval", "evaluators"
            )
            evaluator = load_registered_automatic_evaluator(
                evaluator_name=evaluator_name,
                evaluators_cfg_path=evaluators_dir,
            )
        except Exception as e:
            print(
                f"[ToolBench] Warning: failed to init evaluator, skip per-sample scoring: {e}"
            )

    print("[ToolBench] Initializing environment...")
    env = ToolbenchEnv(
        service_url=os.environ.get(
            "STB_MIRRORAPI_URL", "http://127.0.0.1:8080/virtual"
        ),
        toolbench_key=os.environ.get("STB_TOOLBENCH_KEY", "EMPTY"),
        strip="truncate",
        timeout=30,
    )

    tool_root_dir = os.path.join(ROOT_DIR, "StableToolBench", "toolenv")
    if not os.path.exists(tool_root_dir):
        raise FileNotFoundError(
            f"ToolBench toolenv directory not found: {tool_root_dir}"
        )

    print("[ToolBench] Loading dataset...")
    try:
        dataset_obj = get_dataset(args.dataset)

        if not args.category:
            raise ValueError("--category argument is required for StableToolBench")
        data_list = dataset_obj.load_data(args.category)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load StableToolBench dataset: {e}. "
            f"Make sure dataset.py supports '{args.dataset}' dataset."
        )

    use_gating_flag = bool(getattr(args, "gating", False))

    tool_finetuning_lr, tool_finetuning_epochs = _require_tool_finetuning_locators(args)
    try:
        tool_pretraining_lr, tool_pretraining_epochs = _require_tool_pretraining_locators(args)
    except ValueError:
        ds_train_infer = normalize_dataset_name(getattr(args, "dataset", ""))
        category_file_infer = train_filename(getattr(args, "category", ""))
        tool_pretraining_lr, tool_pretraining_epochs = _infer_tool_pretraining_locators_from_tool_finetuning(
            dataset_train=ds_train_infer,
            model_name=args.model_name,
            lora_rank=int(getattr(args, "lora_rank", 0)),
            lora_alpha=int(getattr(args, "lora_alpha", 0)),
            tool_finetuning_lr=tool_finetuning_lr,
            tool_finetuning_epochs=tool_finetuning_epochs,
            category_file=category_file_infer,
            run_tag=getattr(args, "run_tag", None),
        )
        args.tool_pretraining_lr = tool_pretraining_lr
        args.tool_pretraining_epochs = tool_pretraining_epochs
        args.tool_finetuning_lr = tool_finetuning_lr
        args.tool_finetuning_epochs = tool_finetuning_epochs
        print(
            f"[ToolBench][info] inferred tool pretraining adapters from tool_finetuning: "
            f"tool_pretraining_lr={tool_pretraining_lr}, tool_pretraining_epochs={tool_pretraining_epochs}"
        )

    gating_target_modules = ["down_proj", "up_proj", "gate_proj"]
    gated_modules = _replace_with_gated_lora_linear(model, gating_target_modules)
    if not gated_modules:
        raise RuntimeError(
            f"No Linear modules wrapped by GatedLoraLinear for target_modules={gating_target_modules}. "
            "Parametric inference requires these modules to exist in the base model."
        )

    gating_net = None
    tool_embedding = None
    tool_vocab: Dict[str, int] = {}
    tool_vocab_type: Optional[str] = None
    unk_tool_id = None

    if use_gating_flag:
        explicit_path = getattr(args, "gating_network_ckpt_path", None)
        if not isinstance(explicit_path, str) or not explicit_path.strip():
            raise ValueError(
                "GatingNetwork is enabled (`gating=True`) but `gating_network_ckpt_path` is not set. "
                "Please pass an absolute checkpoint path."
            )
        template = explicit_path.strip()
        rendered = template
        if "{category}" in template:
            rendered = template.format(category=str(args.category or "").strip())
        if not os.path.exists(rendered):
            print(
                "[ToolBench][gating] checkpoint not found via --gating_network_ckpt_path, disable gating:\n"
                f"  - {rendered}"
            )
        else:
            ckpt_path = rendered
            try:
                result = _load_gating_checkpoint(ckpt_path, model.device, model.config)
                gating_net = result["gating_net"]
                tool_embedding = result["tool_embedding"]
                tool_vocab = result["tool_vocab"]
                tool_vocab_type = result["tool_vocab_type"]
                unk_tool_id = result["unk_tool_id"]
                gating_question_encoder = make_gating_question_encoder(
                    result["q_encoder"],
                    device=model.device,
                    base_model=model,
                    tokenizer=tokenizer,
                )
                print(f"[ToolBench][gating] loaded gating network from: {ckpt_path}")
            except Exception as e:
                print(f"[ToolBench][gating] failed to load gating network ({ckpt_path}): {e}")
                gating_net = None
                tool_embedding = None
                tool_vocab = {}
                tool_vocab_type = None
                unk_tool_id = None
                gating_question_encoder = _default_gating_question_encoder

    ds_train = normalize_dataset_name(getattr(args, "dataset", ""))
    category_file = train_filename(getattr(args, "category", ""))
    run_segments = []
    run_tag = str(getattr(args, "run_tag", "") or "").strip()
    if run_tag:
        run_segments.append(f"run={_safe_filename_component(run_tag)}")
    tool_finetuning_root = os.path.join(
        TOOL_FINETUNING_ROOT_PATH,
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        ds_train,
        f"lr={tool_pretraining_lr}_epoch={tool_pretraining_epochs}",
        f"tool_finetuning_lr={tool_finetuning_lr}_epoch={tool_finetuning_epochs}",
        *run_segments,
        category_file,
    )
    index_path = os.path.join(tool_finetuning_root, "tool_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[ToolBench][tool_finetuning] tool_index.json not found: {index_path}. "
            f"Make sure tool_finetuning adapters were trained/exported for dataset={ds_train}, category={category_file}."
        )
    tool_finetuning_index = ToolFinetuningIndex.load(index_path)

    output_root_dir = _build_output_root_dir(args)
    os.makedirs(output_root_dir, exist_ok=True)

    config_file = os.path.join(output_root_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=4)

    result_file = os.path.join(output_root_dir, "result.json")

    results: List[Dict[str, Any]] = []

    def _write_results_file() -> None:
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e_res:
            print(f"[ToolBench][warn] failed to write result: {e_res}")

    def build_tooleval_example(raw_item, result, tool_root_dir):
        functions, meta_list = prepare_tools_from_sample(raw_item, tool_root_dir)

        eg = ExecutionGraph()
        last = generate_init_message_node(
            eg,
            functions,
            raw_item.get("query", ""),
        )

        for step in result.get("history_steps", []):
            msg = {
                "name": step.get("action", ""),
                "arguments": step.get("input", ""),
                "response": step.get("observation", ""),
            }
            node = ExecutionNode(role="tool", message=msg)
            eg.add_node(node)
            eg[last, node] = None
            last = node

        final_answer = result.get("final_answer") or ""
        if final_answer:
            ans_node = ExecutionNode(role="assistant", message=final_answer)
            eg.add_node(ans_node)
            eg[last, ans_node] = None
            last = ans_node

        answer = {
            "method": "parametric_prompt",
            "total_steps": eg.node_count,
            "final_answer": final_answer,
            "answer_details": eg.convert_to_dict(),
        }

        example = {
            "query": raw_item.get("query", ""),
            "available_tools": functions,
            "answer": answer,
        }
        return example

    total_samples = len(data_list)
    print(f"[ToolBench] Starting inference on {total_samples} samples...")

    for idx, raw_item in enumerate(tqdm(data_list, desc="Processing samples")):
        query_id = raw_item.get("query_id", str(idx))

        print(f"\n[ToolBench] Sample {idx}/{total_samples}: {query_id}")
        print(f"Query: {raw_item.get('query', '')[:100]}...")

        try:
            functions, meta_list = prepare_tools_from_sample(raw_item, tool_root_dir)

            parametric_tools_experts: List[Dict[str, Any]] = []
            finish_tool_schema: Optional[Dict[str, Any]] = None
            for schema in functions:
                if not isinstance(schema, dict):
                    continue
                fn = schema.get("function")
                if not isinstance(fn, dict):
                    continue
                if str(fn.get("name") or "").strip() == "Finish":
                    finish_tool_schema = dict(fn)
                    finish_tool_schema.update(
                        {
                            "category_name": "__special__",
                            "tool_name": "__finish__",
                            "api_name": "Finish",
                            "name": "Finish",
                        }
                    )
                    continue
                meta = find_meta_by_function_name(str(fn.get("name") or ""), meta_list)
                if meta:
                    fn = dict(fn)
                    fn.update(
                        {
                            "category_name": str(
                                meta.get("category_name") or ""
                            ).strip(),
                            "tool_name": str(
                                meta.get("original_tool_name") or ""
                            ).strip(),
                            "api_name": str(
                                meta.get("original_api_name") or ""
                            ).strip(),
                            "standard_tool_name": str(
                                meta.get("standard_tool_name") or ""
                            ).strip(),
                            "pure_api_name": str(
                                meta.get("pure_api_name") or ""
                            ).strip(),
                        }
                    )
                parametric_tools_experts.append(fn)

            parametric_tools_gating: List[Dict[str, Any]] = list(
                parametric_tools_experts
            )
            if finish_tool_schema is None:
                finish_tool_schema = {
                    "category_name": "__special__",
                    "tool_name": "__finish__",
                    "api_name": "Finish",
                    "name": "Finish",
                    "description": "End the tool-calling loop and output the final answer.",
                    "parameters": {"type": "dict", "properties": {}, "required": []},
                }
            parametric_tools_gating.append(finish_tool_schema)

            if gated_modules and tool_finetuning_index is not None:
                ds_train = normalize_dataset_name(
                    getattr(args, "dataset", "")
                )

                adapter_files: List[str] = []
                missing: List[str] = []
                for tool_schema in parametric_tools_experts:
                    tool_key = tool_key_for_schema(tool_schema, dataset=ds_train)
                    rec = tool_finetuning_index.get(tool_key)
                    if rec is None or not adapter_exists(rec.adapter_dir):
                        missing.append(
                            f"{tool_key} -> {rec.adapter_dir if rec else 'None'}"
                        )
                        continue
                    adapter_files.append(
                        os.path.join(rec.adapter_dir, "adapter_model.safetensors")
                    )

                if missing:
                    msg = (
                        "[ToolBench][tool_finetuning] FATAL: missing adapters for some tools. "
                        + "; ".join(missing[:20])
                        + (" ..." if len(missing) > 20 else "")
                    )
                    print(msg)
                    sys.exit(1)

                try:
                    _clear_gated_modules(gated_modules)
                    _inject_gated_lora_experts(
                        model=model,
                        gated_modules=gated_modules,
                        adapter_files=adapter_files,
                        target_modules=gating_target_modules,
                        lora_alpha=int(getattr(args, "lora_alpha", 0)),
                        lora_rank=int(getattr(args, "lora_rank", 0)),
                    )

                    _set_all_alphas(gated_modules, None)
                    tool_embs_cached = _encode_tools_for_gating(
                        parametric_tools_gating,
                        model=model, tokenizer=tokenizer,
                        tool_embedding=tool_embedding, tool_vocab=tool_vocab,
                        tool_vocab_type=tool_vocab_type, unk_tool_id=unk_tool_id,
                    )
                    expert_indices = list(range(len(parametric_tools_experts)))

                    def _pre_generate_hook(
                        question_text: str,
                        _tools_converted: list,
                        _step: int,
                        _history_steps: list,
                    ):
                        # Recompute routing before each ReAct step because the prompt includes history.
                        with torch.no_grad():
                            _set_all_alphas(gated_modules, None)
                            q_emb = gating_question_encoder.encode(question_text)

                            alphas_gating = _compute_gating_alphas(
                                gating_net=gating_net, model=model,
                                q_emb=q_emb, tool_embs=tool_embs_cached,
                                use_gating=use_gating_flag,
                                num_tools=len(parametric_tools_gating),
                            )
                            applied_alphas, _finish_selected, _finish_idx = (
                                map_gating_alphas_with_finish_to_experts(
                                    alphas_gating,
                                    tools_gating=parametric_tools_gating,
                                    expert_indices=expert_indices,
                                )
                            )

                            _set_all_alphas(gated_modules, applied_alphas)

                    result = run_single_query(
                        raw_item=raw_item,
                        env=env,
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=generation_config,
                        tool_root_dir=tool_root_dir,
                        max_steps=int(getattr(args, "max_steps", 30) or 30),
                        functions=functions,
                        meta_list=meta_list,
                        pre_generate_hook=_pre_generate_hook,
                        verbose=bool(getattr(args, "verbose", False)),
                    )
                finally:
                    _clear_gated_modules(gated_modules)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            else:
                result = run_single_query(
                    raw_item=raw_item,
                    env=env,
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    tool_root_dir=tool_root_dir,
                    max_steps=int(getattr(args, "max_steps", 30) or 30),
                    functions=functions,
                    meta_list=meta_list,
                    verbose=bool(getattr(args, "verbose", False)),
                )

            example = build_tooleval_example(raw_item, result, tool_root_dir)

            eval_status = None
            eval_reason = ""
            if bool(result.get("hit_max_steps")):
                eval_status = "Unsolved"
                eval_reason = "Max steps reached"
            elif evaluator is not None:
                try:
                    task_desc = {
                        "query": example["query"],
                        "available_tools": example["available_tools"],
                    }
                    ans = example["answer"]
                    status, reason = evaluator.check_is_solved(
                        task_desc, ans, return_reason=True
                    )
                    eval_status = (
                        status.name if isinstance(status, AnswerStatus) else str(status)
                    )
                    eval_reason = reason or ""
                    print(
                        f"[ToolBench][Eval] query_id={query_id}, status={eval_status}"
                    )
                except Exception as e_eval:
                    eval_status = "EvalError"
                    eval_reason = str(e_eval)
                    print(f"[ToolBench][Eval] failed for query_id={query_id}: {e_eval}")
            valid = eval_status in ("Solved", "AnswerStatus.Solved")
            err_list: list[str] = []
            run_err = str(result.get("error") or "").strip()
            if run_err:
                err_list.append(run_err)
            if (not valid) and eval_reason:
                err_list.append(str(eval_reason))
            if eval_status is None and (not evaluator) and (not err_list):
                err_list.append("No evaluation result")

            check_result = {
                "valid": bool(valid),
                "error": err_list,
            }

            result_entry = {
                "query_id": query_id,
                "query": example["query"],
                "available_tools": example["available_tools"],
                "answer": example["answer"],
                "check_result": check_result,
                "eval": {
                    "answer_status": eval_status,
                    "reason": eval_reason,
                },
            }
            results.append(result_entry)
            _write_results_file()

        except Exception as e:
            print(f"[ToolBench] Error processing sample {idx}: {e}")

            check_result = {
                "valid": False,
                "error": [str(e)],
            }

            empty_answer = {
                "method": "parametric_prompt",
                "total_steps": 0,
                "final_answer": "",
                "answer_details": [],
            }
            result_entry = {
                "query_id": query_id,
                "query": raw_item.get("query", ""),
                "available_tools": [],
                "answer": empty_answer,
                "check_result": check_result,
                "eval": {
                    "answer_status": "EvalError",
                    "reason": str(e),
                },
            }
            results.append(result_entry)
            _write_results_file()

    print(f"\n[ToolBench] Inference completed!")
    print(f"Total samples: {total_samples}")
    print(f"Processed: {len(results)}")
    print(f"Results saved to: {result_file}")

    solved = 0
    total_eval = 0
    for ex in results:
        status = ex.get("eval", {}).get("answer_status")
        if status in ("Solved", "AnswerStatus.Solved"):
            solved += 1
        if status is not None:
            total_eval += 1

    solve_rate = solved / total_eval if total_eval > 0 else 0.0

    if total_eval > 0:
        print(f"\n[ToolBench][Eval] Solve rate: {solved}/{total_eval} = {solve_rate:.3%}")
    else:
        print(f"\n[ToolBench][Eval] No samples evaluated.")

    statistics = {
        "total_samples": total_samples,
        "processed_samples": len(results),
        "solved_samples": solved,
        "evaluated_samples": total_eval,
        "solve_rate": solve_rate,
    }
    stat_path = os.path.join(output_root_dir, "statistic.json")
    try:
        with open(stat_path, "w") as fout:
            json.dump(statistics, fout, indent=4)
        print(f"Statistics saved to: {stat_path}")
    except Exception as e:
        print(f"[warn] failed to save statistics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (keys match CLI args). CLI args override YAML.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="G1_category",
        help="STB test category (e.g., G1_category, G1_instruction, G1_tool, G2_category, G2_instruction, G3_instruction)",
    )
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7b-instruct")
    parser.add_argument(
        "--eval_model",
        type=str,
        default=None,
        help=(
            "Evaluation model shortcut for ToolBench (maps to evaluator folder "
            "`tooleval_{eval_model}_default`). Example: gpt-3.5-turbo, gpt-5.1."
        ),
    )
    parser.add_argument(
        "--evaluator_name",
        type=str,
        default=None,
        help=(
            "Exact evaluator folder name under `StableToolBench/toolbench/tooleval/evaluators/`, "
            "e.g. `tooleval_gpt-3.5-turbo_default`."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--tool_pretraining_lr",
        type=float,
        default=None,
        help="Tool pretraining(single-tool) learning rate used to locate adapters.",
    )
    parser.add_argument(
        "--tool_pretraining_epochs",
        type=int,
        default=None,
        help="Tool pretraining(single-tool) epochs used to locate adapters.",
    )
    parser.add_argument(
        "--tool_finetuning_lr",
        type=float,
        default=None,
        help="Tool finetuning learning rate identifier used to locate tool_finetuning outputs.",
    )
    parser.add_argument(
        "--tool_finetuning_epochs",
        type=int,
        default=None,
        help="Tool finetuning epochs identifier used to locate tool_finetuning outputs.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Max tool-calling steps per sample; hitting the limit is treated as Unsolved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step ToolBench inference progress (generation/tool calls).",
    )
    parser.add_argument(
        "--disable_eval",
        action="store_true",
        help="Disable ToolBench tooleval per-sample scoring (avoids OpenAI calls).",
    )
    parser.add_argument(
        "--gating_network_ckpt_path",
        type=str,
        default=None,
        help="Explicit GatingNetwork checkpoint path; supports `{category}` placeholder.",
    )
    parser.add_argument(
        "--gating",
        action="store_true",
        help="Use GatingNetwork network to predict per-tool weights for LoRA expert mixing.",
    )
    pre_args, _ = parser.parse_known_args()
    if getattr(pre_args, "config", None):
        from utils import (
            apply_env_from_config,
            load_yaml_file,
            set_parser_defaults_from_config,
        )

        cfg = load_yaml_file(pre_args.config)
        apply_env_from_config(cfg, config_path=pre_args.config)
        set_parser_defaults_from_config(parser, cfg)

    args = parser.parse_args()
    args.dataset = "stb"
    print(args)
    run_toolbench_inference(args)

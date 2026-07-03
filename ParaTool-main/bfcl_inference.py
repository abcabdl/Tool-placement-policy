import os
import json
import gc
import argparse
import uuid
from datetime import datetime

import sitecustomize  # noqa: F401  -- forces third_party / BFCL paths to load

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from bfcl_eval.utils import load_dataset_entry  # type: ignore
from dataset import get_dataset
from dataset.tool_finetuning_dataset import ToolFinetuningIndex
from dataset.tool_keys import tool_key_for_schema

from gating_network import (
    GatingNetwork,
    GatingQuestionEncoder,
    make_gating_question_encoder,
    normalize_gating_q_encoder_mode,
    resolve_tool_ids,
    _replace_with_gated_lora_linear,
    _load_lora_experts_for_paths,
    _normalize_module_name,
)
from root_dir_path import (
    TOOL_PRETRAINING_ROOT_PATH,
    ROOT_DIR,
    TOOL_FINETUNING_ROOT_PATH,
)
from utils import (
    STB_DATASET_NAMES,
    _safe_filename_component,
    _normalize_question_to_text,
    adapter_exists,
    get_model,
    load_data,
    normalize_dataset_name,
    category_slug,
    train_filename,
)
from dataset.profiles import BfclPythonProfile
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

    model_out = str(getattr(args, "model_name", "") or "").strip().lower()
    base = os.path.join(ROOT_DIR, "output", dataset_out, model_out, category_out, variant_out)

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

_EMPTY_GATING_RESULT: Dict[str, Any] = {
    "gating_net": None,
    "tool_embedding": None,
    "tool_vocab": {},
    "tool_vocab_type": None,
    "unk_tool_id": None,
    "q_encoder": "llm",
    "gating_question_encoder": None,
}


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
        B, M, d = tool_embs.shape
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

def run_bfcl_inference(args):
    bfcl_prompt_cache: Dict[str, List[Dict[str, Any]]] = {}

    data_list = load_data(args.dataset, args.category)

    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    use_gating_flag = bool(getattr(args, "gating", False))

    _default_gating_question_encoder = make_gating_question_encoder(
        "llm",
        device=model.device,
        base_model=model,
        tokenizer=tokenizer,
    )
    gating_question_encoder: GatingQuestionEncoder = _default_gating_question_encoder

    use_tool_finetuning_adapters = bool(use_gating_flag)

    tool_pretraining_lr, tool_pretraining_epochs = _require_tool_pretraining_locators(args)
    if use_tool_finetuning_adapters:
        tool_finetuning_lr, tool_finetuning_epochs = _require_tool_finetuning_locators(args)
        args.tool_finetuning_lr = tool_finetuning_lr
        args.tool_finetuning_epochs = tool_finetuning_epochs

    profile = BfclPythonProfile("without")

    def _generate_with_profile(
        q_text: Any,
        available_tools: List[Dict[str, Any]],
        *,
        multi_step: bool = False,
        gen_cfg: Any = None,
    ) -> str:
        cfg = gen_cfg or generation_config
        input_ids, input_len = profile.build_inference_prompt_ids(
            tokenizer,
            q_text,
            available_tools,
            multi_step=multi_step,
        )
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids_tensor,
                attention_mask=torch.ones(input_ids_tensor.shape).to(model.device),
                generation_config=cfg,
            )
        if hasattr(output, "sequences"):
            output_ids = output.sequences[0][input_len:]
        else:
            output_ids = output[0][input_len:]
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    output_root_dir = _build_output_root_dir(args)

    gating_net = None
    tool_embedding = None
    tool_vocab: Dict[str, int] = {}
    tool_vocab_type: Optional[str] = None
    unk_tool_id = None
    gating_target_modules = ["down_proj", "up_proj", "gate_proj"]
    gated_modules = _replace_with_gated_lora_linear(model, gating_target_modules)
    if not gated_modules:
        raise RuntimeError(
            f"No Linear modules wrapped by GatedLoraLinear for target_modules={gating_target_modules}. "
            "Parametric inference requires these modules to exist in the base model."
        )

    gating_cache: Dict[str, Dict[str, Any]] = {}

    def _load_gating_network_for_category(real_category: str) -> None:
        nonlocal gating_net, tool_embedding, tool_vocab, tool_vocab_type, unk_tool_id, gating_question_encoder
        raw = str(real_category or "").strip()
        candidates: List[str] = []
        for c in (raw, raw.lower()):
            c = str(c or "").strip()
            if c and c not in candidates:
                candidates.append(c)
        if not candidates:
            candidates = ["unknown"]

        for c in candidates:
            if c in gating_cache:
                cached = gating_cache[c]
                gating_net = cached.get("gating_net")
                tool_embedding = cached.get("tool_embedding")
                tool_vocab = dict(cached.get("tool_vocab") or {})
                tool_vocab_type = cached.get("tool_vocab_type")
                unk_tool_id = cached.get("unk_tool_id")
                gating_question_encoder = cached.get("gating_question_encoder") or _default_gating_question_encoder
                return

        gating_net = None
        tool_embedding = None
        tool_vocab = {}
        tool_vocab_type = None
        unk_tool_id = None
        gating_question_encoder = _default_gating_question_encoder

        explicit_path = getattr(args, "gating_network_ckpt_path", None)
        if not isinstance(explicit_path, str) or not explicit_path.strip():
            raise ValueError(
                "GatingNetwork is enabled (`gating=True`) but `gating_network_ckpt_path` is not set. "
                "Please pass an absolute checkpoint path (optionally with `{category}` placeholder)."
            )

        template = explicit_path.strip()
        tried_all: List[str] = []
        ckpt_path: Optional[str] = None
        for c in candidates:
            rendered = template
            if "{category}" in template:
                rendered = template.format(category=c)
            tried_all.append(rendered)
            if os.path.exists(rendered):
                ckpt_path = rendered
                break

        if ckpt_path is None:
            print(
                "[gating] checkpoint not found via --gating_network_ckpt_path, disable gating for category="
                f"{raw or candidates[0]}:\n  - " + "\n  - ".join(tried_all)
            )
            gating_question_encoder = _default_gating_question_encoder
            for c in candidates:
                row = dict(_EMPTY_GATING_RESULT)
                row["gating_question_encoder"] = _default_gating_question_encoder
                gating_cache[c] = row
            return

        result: Optional[Dict[str, Any]] = None
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
            print(f"[gating] loaded gating network from: {ckpt_path}")
        except Exception as e:
            print(f"[gating] failed to load gating network ({ckpt_path}): {e}")
            result = None
            gating_net = None
            tool_embedding = None
            tool_vocab = {}
            tool_vocab_type = None
            unk_tool_id = None
            gating_question_encoder = _default_gating_question_encoder

        for c in candidates:
            qe_cached = (
                normalize_gating_q_encoder_mode(result.get("q_encoder", "llm"))
                if result is not None and gating_net is not None
                else "llm"
            )
            gating_cache[c] = {
                "gating_net": gating_net,
                "tool_embedding": tool_embedding,
                "tool_vocab": dict(tool_vocab),
                "tool_vocab_type": tool_vocab_type,
                "unk_tool_id": unk_tool_id,
                "q_encoder": qe_cached,
                "gating_question_encoder": gating_question_encoder,
            }

    def _dict_to_python_call_local(func_name, params_dict):
        if not isinstance(params_dict, dict):
            return f"{func_name}({repr(params_dict)})"
        if len(params_dict) == 0:
            return f"{func_name}()"
        parts = [f"{k}={repr(v)}" for k, v in params_dict.items()]
        return f"{func_name}({', '.join(parts)})"

    def build_multistep_question(original_q, history_calls):
        base = _normalize_question_to_text(original_q)
        if not history_calls:
            return base
        history_py = (
            "["
            + ", ".join(
                _dict_to_python_call_local(list(c.keys())[0], list(c.values())[0])
                for c in history_calls
            )
            + "]"
        )
        return f"{base}\n\n[TOOLCALL_HISTORY]\n{history_py}"

    def parse_call_status(text: str):
        if "<CALL_END>" in text:
            return "END"
        if "<CALL_CONT>" in text:
            return "CONT"
        return None

    try:
        dataset_obj = get_dataset(args.dataset, model_name=args.model_name)
    except Exception:
        dataset_obj = None

    aggregate_ret: List[Dict[str, Any]] = []

    for filename, full_dataset in data_list:
        real_category = category_slug(filename)
        real_category_key = str(real_category or "").strip()
        filename_json = train_filename(filename)

        if use_gating_flag:
            _load_gating_network_for_category(real_category=real_category_key)
            if gating_net is None:
                raise RuntimeError(
                    "gating=True requires a GatingNetwork checkpoint, but gating_net failed to load. "
                    "Check your `--gating_network_ckpt_path` and ensure the file exists."
                )
        else:
            gating_net = None
            tool_embedding = None
            tool_vocab = {}
            tool_vocab_type = None
            unk_tool_id = None

        total_samples = len(full_dataset)
        target_idx = getattr(args, "only_idx", None)
        write_outputs = target_idx is None
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        if write_outputs:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as fout:
                json.dump(vars(args), fout, indent=4)
            result_file = os.path.join(output_dir, "result.json")
            ret = []
        else:
            result_file = None
            ret = []

        if target_idx is not None:
            if target_idx < 0 or target_idx >= total_samples:
                print(
                    f"[warn] only_idx {target_idx} out of range for {filename}; skip this file."
                )
                continue
            indices_to_run = [target_idx]
            print(
                f"[info] only_idx set, running sample {target_idx} of {total_samples}"
            )
        else:
            indices_to_run = list(range(0, total_samples))

        rank_dir = f"rank={args.lora_rank}_alpha={args.lora_alpha}"
        lr_dir = f"lr={float(args.tool_pretraining_lr)}_epoch={int(args.tool_pretraining_epochs)}"
        run_segments = []
        run_tag = str(getattr(args, "run_tag", "") or "").strip()
        if run_tag:
            run_segments.append(f"run={_safe_filename_component(run_tag)}")

        tft_index: Optional[ToolFinetuningIndex] = None
        tft_tools_root = ""
        ds_train_bfcl = ""
        if use_tool_finetuning_adapters:
            ds_train_bfcl = normalize_dataset_name(getattr(args, "dataset", ""))
            tft_lr, tft_ep = _require_tool_finetuning_locators(args)
            tft_dir = f"tool_finetuning_lr={float(tft_lr)}_epoch={int(tft_ep)}"
            tft_root = os.path.join(
                TOOL_FINETUNING_ROOT_PATH,
                args.model_name,
                rank_dir,
                ds_train_bfcl,
                lr_dir,
                tft_dir,
                *run_segments,
                filename_json,
            )
            tft_index_path = os.path.join(tft_root, "tool_index.json")
            if not os.path.exists(tft_index_path):
                raise FileNotFoundError(
                    f"No global joint tool_finetuning tool_index.json for {filename}: {tft_index_path}"
                )
            tft_index = ToolFinetuningIndex.load(tft_index_path)
            tft_tools_root = os.path.abspath(os.path.join(tft_root, "tools"))

        def _choose_existing_adapter_dir(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                if adapter_exists(cand):
                    return cand
            return None

        for test_id in tqdm(indices_to_run, total=len(indices_to_run)):
            data = full_dataset[test_id]
            print(f"test_id: {test_id}")
            if target_idx is None:
                assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"
            question = data["question"]
            available_tools = data["function"]
            print(f"question: {question}")

            def _predict(model, test_id):
                output = None
                gen_cfg = generation_config
                sample_gating_records = []
                try:
                    multi_turn_categories = {"parallel_multiple", "live_parallel_multiple"}
                    if (
                        str(real_category_key or "").strip().lower() in multi_turn_categories
                    ):
                        def _single_step_generate(q_text: str) -> str:
                            try:
                                local_cfg = gen_cfg.clone()
                            except Exception:
                                from transformers import GenerationConfig

                                local_cfg = GenerationConfig.from_model_config(
                                    model.config
                                )
                                local_cfg.__dict__.update(gen_cfg.__dict__)

                            extra_eos = []
                            try:
                                enc_cont = tokenizer(
                                    "<CALL_CONT>", add_special_tokens=False
                                )
                                enc_end = tokenizer(
                                    "<CALL_END>", add_special_tokens=False
                                )
                                cont_ids = enc_cont.get("input_ids", [])
                                end_ids = enc_end.get("input_ids", [])

                                if (
                                    isinstance(cont_ids, (list, tuple))
                                    and cont_ids
                                    and isinstance(cont_ids[0], (list, tuple))
                                ):
                                    cont_ids = cont_ids[0]
                                if (
                                    isinstance(end_ids, (list, tuple))
                                    and end_ids
                                    and isinstance(end_ids[0], (list, tuple))
                                ):
                                    end_ids = end_ids[0]
                                if (
                                    isinstance(cont_ids, (list, tuple))
                                    and len(cont_ids) > 0
                                ):
                                    extra_eos.append(int(cont_ids[-1]))
                                if (
                                    isinstance(end_ids, (list, tuple))
                                    and len(end_ids) > 0
                                ):
                                    extra_eos.append(int(end_ids[-1]))
                            except Exception:
                                extra_eos = []

                            if extra_eos:
                                base_eos = getattr(local_cfg, "eos_token_id", None)
                                if base_eos is None:
                                    eos_list = list(extra_eos)
                                elif isinstance(base_eos, int):
                                    eos_list = [base_eos] + list(extra_eos)
                                else:
                                    eos_list = list(
                                        set(list(base_eos)) | set(extra_eos)
                                    )
                                local_cfg.eos_token_id = (
                                    eos_list[0] if len(eos_list) == 1 else eos_list
                                )

                            return _generate_with_profile(
                                q_text,
                                available_tools,
                                multi_step=True,
                                gen_cfg=local_cfg,
                            )

                        history_calls = []
                        max_steps = len(available_tools) + 2
                        for step in range(max_steps):
                            q_step = build_multistep_question(question, history_calls)

                            if gated_modules:
                                with torch.no_grad():
                                    q_emb = gating_question_encoder.encode(q_step)
                                    tool_embs = _encode_tools_for_gating(
                                        available_tools,
                                        model=model, tokenizer=tokenizer,
                                        tool_embedding=tool_embedding, tool_vocab=tool_vocab,
                                        tool_vocab_type=tool_vocab_type, unk_tool_id=unk_tool_id,
                                    )

                                    alphas = _compute_gating_alphas(
                                        gating_net=gating_net, model=model,
                                        q_emb=q_emb, tool_embs=tool_embs,
                                        use_gating=use_gating_flag,
                                        num_tools=len(available_tools),
                                    )

                                try:
                                    alpha_vals = (
                                        alphas.detach().cpu().tolist()
                                        if isinstance(alphas, torch.Tensor)
                                        else []
                                    )
                                    tool_names = []
                                    for schema in available_tools:
                                        if isinstance(schema, dict):
                                            tool_names.append(schema.get("name", ""))
                                        else:
                                            tool_names.append(str(schema))
                                    print(
                                        "[gating][infer] "
                                        f"step={step}, "
                                        "tools_weights="
                                        + ", ".join(
                                            f"{name}:{val:.4f}"
                                            for name, val in zip(tool_names, alpha_vals)
                                        )
                                    )
                                except Exception:
                                    pass

                                if alphas is not None and len(available_tools) > 0:
                                    try:
                                        alpha_vals = (
                                            alphas.detach().cpu().tolist()
                                            if isinstance(alphas, torch.Tensor)
                                            else []
                                        )
                                        tool_names = []
                                        for schema in available_tools:
                                            if isinstance(schema, dict):
                                                tool_names.append(schema.get("name", ""))
                                            else:
                                                tool_names.append(str(schema))

                                        if len(alpha_vals) > 0 and len(tool_names) > 0:
                                            top1_idx = alpha_vals.index(max(alpha_vals))
                                            top1_tool = tool_names[top1_idx]
                                        else:
                                            top1_tool = None

                                        sample_gating_records.append({
                                            "step": step,
                                            "tools": tool_names,
                                            "alphas": alpha_vals,
                                            "predicted_top1": top1_tool,
                                            "gating_mode": "gating" if use_gating_flag else "uniform",
                                        })
                                    except Exception:
                                        pass

                                _set_all_alphas(
                                    gated_modules, alphas,
                                    device=model.device,
                                    dtype=next(model.parameters()).dtype,
                                )

                            raw = _single_step_generate(q_step)
                            print(f"original_output_step{step}:{raw}")

                            parsed = profile.parse_inference_output(
                                raw, available_tools
                            )
                            step_calls = (
                                parsed.get("parsed", [])
                                if isinstance(parsed, dict)
                                else []
                            )
                            if not step_calls:
                                break

                            for curr_call in step_calls:
                                if curr_call in history_calls:
                                    continue
                                history_calls.append(curr_call)

                            call_status = (
                                parsed.get("call_status")
                                if isinstance(parsed, dict)
                                else None
                            )
                            if call_status == "END":
                                break
                            if call_status is None:
                                status = parse_call_status(raw)
                                if status == "END":
                                    break

                        if history_calls:
                            calls_str = ", ".join(
                                _dict_to_python_call_local(
                                    list(c.keys())[0], list(c.values())[0]
                                )
                                for c in history_calls
                            )
                            output = f"[{calls_str}]"
                        else:
                            output = "[]"

                    else:
                        if gated_modules:
                            with torch.no_grad():
                                q_emb = gating_question_encoder.encode(question)
                                tool_embs = _encode_tools_for_gating(
                                    available_tools,
                                    model=model, tokenizer=tokenizer,
                                    tool_embedding=tool_embedding, tool_vocab=tool_vocab,
                                    tool_vocab_type=tool_vocab_type, unk_tool_id=unk_tool_id,
                                )

                                alphas = _compute_gating_alphas(
                                    gating_net=gating_net, model=model,
                                    q_emb=q_emb, tool_embs=tool_embs,
                                    use_gating=use_gating_flag,
                                    num_tools=len(available_tools),
                                )

                            try:
                                alpha_vals = (
                                    alphas.detach().cpu().tolist()
                                    if isinstance(alphas, torch.Tensor)
                                    else []
                                )
                                tool_names = []
                                for schema in available_tools:
                                    if isinstance(schema, dict):
                                        tool_names.append(schema.get("name", ""))
                                    else:
                                        tool_names.append(str(schema))
                                print(
                                    "[gating][infer] "
                                    f"sample={test_id}, "
                                    "tools_weights="
                                    + ", ".join(
                                        f"{name}:{val:.4f}"
                                        for name, val in zip(tool_names, alpha_vals)
                                    )
                                )
                            except Exception:
                                pass

                            if alphas is not None and len(available_tools) > 0:
                                try:
                                    alpha_vals = (
                                        alphas.detach().cpu().tolist()
                                        if isinstance(alphas, torch.Tensor)
                                        else []
                                    )
                                    tool_names = []
                                    for schema in available_tools:
                                        if isinstance(schema, dict):
                                            tool_names.append(schema.get("name", ""))
                                        else:
                                            tool_names.append(str(schema))

                                    if len(alpha_vals) > 0 and len(tool_names) > 0:
                                        top1_idx = alpha_vals.index(max(alpha_vals))
                                        top1_tool = tool_names[top1_idx]
                                    else:
                                        top1_tool = None

                                    sample_gating_records.append({
                                        "step": 0,
                                        "tools": tool_names,
                                        "alphas": alpha_vals,
                                        "predicted_top1": top1_tool,
                                        "gating_mode": "gating" if use_gating_flag else "uniform",
                                    })
                                except Exception:
                                    pass

                            _set_all_alphas(
                                gated_modules, alphas,
                                device=model.device,
                                dtype=next(model.parameters()).dtype,
                            )

                        output = _generate_with_profile(
                            question,
                            available_tools,
                            multi_step=False,
                            gen_cfg=gen_cfg,
                        )

                    print(f"original_output:{output}")
                    parsed = profile.parse_inference_output(output, available_tools)
                    parsed_calls = (
                        parsed.get("parsed") if isinstance(parsed, dict) else None
                    )
                    if parsed_calls is not None:
                        output = parsed_calls
                        print(f"mapped_output: {output}")
                    else:
                        print("[warn] profile parse returned no tool calls")
                        output = []

                    print(f"output: {output}")
                    model_output = []
                    for item in output:
                        if isinstance(item, dict) and len(item) == 1:
                            func_name = list(item.keys())[0]
                            params = item[func_name]
                            func_name_converted = func_name.replace(".", "_")
                            model_output.append({func_name_converted: params})
                except Exception as e:
                    print(f"Prediction failed: {e}, output: {output}")
                    model_output = []
                print(f"model_output: {model_output}")

                real_category = category_slug(filename)

                answer = None
                check_result = {"valid": False, "error": []}
                if dataset_obj is not None:
                    try:
                        answer = dataset_obj.get_sample_ground_truth(
                            real_category, test_id
                        )
                        print(f"standard_answer: {answer}")

                        if real_category not in bfcl_prompt_cache:
                            try:
                                bfcl_entries = load_dataset_entry(
                                    real_category,
                                    include_prereq=False,
                                    include_language_specific_hint=False,
                                )
                                bfcl_prompt_cache[real_category] = bfcl_entries
                            except Exception as _e_load:
                                print(
                                    f"[warn] failed to load BFCL prompt entries for category={real_category}: {_e_load}"
                                )
                                bfcl_prompt_cache[real_category] = []

                        func_descriptions: List[Dict[str, Any]] = data.get(
                            "function", []
                        )
                        if real_category in bfcl_prompt_cache and test_id < len(
                            bfcl_prompt_cache[real_category]
                        ):
                            bfcl_entry = bfcl_prompt_cache[real_category][test_id]
                            if isinstance(bfcl_entry, dict) and isinstance(
                                bfcl_entry.get("function"), list
                            ):
                                func_descriptions = bfcl_entry["function"]

                        evaluator = dataset_obj.ast_eval()
                        if evaluator is not None:
                            try:
                                expected_len = (
                                    len(answer) if isinstance(answer, list) else None
                                )
                                if isinstance(model_output, list) and expected_len:
                                    if len(model_output) > expected_len:
                                        model_output = model_output[:expected_len]
                                        print(
                                            f"[info] truncated model_output to {expected_len} calls "
                                            "to match ground truth length"
                                        )
                            except Exception as _e_trunc:
                                print(
                                    f"[warn] truncate model_output failed: {_e_trunc}"
                                )

                            if "parallel" in real_category:
                                check_result = evaluator.parallel_no_order(
                                    func_descriptions, model_output, answer
                                )
                            else:
                                check_result = evaluator.multiple(
                                    func_descriptions, model_output, answer
                                )
                            if target_idx is not None:
                                print(f"[ast_eval] check_result: {check_result}")
                    except Exception as e:
                        print(f"[warn] dataset evaluation failed: {e}")

                if not isinstance(check_result, dict):
                    check_result = {"valid": False, "error": []}
                else:
                    check_result = dict(check_result)
                err_list = check_result.get("error", [])
                if not isinstance(err_list, list):
                    err_list = [str(err_list)] if err_list is not None else []
                check_result["error"] = err_list

                return {
                    "test_id": test_id,
                    "question": question,
                    "tools": available_tools,
                    "output": model_output,
                    "answer": answer,
                    "gating_steps": sample_gating_records,
                    "check_result": check_result,
                }

            def _run_with_injected_adapters(resolved: List[Tuple[int, str]]) -> None:
                try:
                    _clear_gated_modules(gated_modules)
                    adapter_files = [
                        os.path.join(apath, "adapter_model.safetensors")
                        for _, apath in sorted(resolved, key=lambda x: x[0])
                    ]
                    _inject_gated_lora_experts(
                        model=model,
                        gated_modules=gated_modules,
                        adapter_files=adapter_files,
                        target_modules=gating_target_modules,
                        lora_alpha=int(getattr(args, "lora_alpha", 0)),
                        lora_rank=int(getattr(args, "lora_rank", 0)),
                    )
                    ret.append(_predict(model, test_id))
                finally:
                    _clear_gated_modules(gated_modules)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            if use_tool_finetuning_adapters:
                if tft_index is None:
                    raise RuntimeError("tool_finetuning index not loaded (internal error)")
                resolved_ft: List[Tuple[int, str]] = []
                missing_ft: List[Tuple[int, str]] = []
                for tid, schema in enumerate(available_tools):
                    tool_key = tool_key_for_schema(schema, dataset=ds_train_bfcl)
                    rec = tft_index.get(tool_key)
                    rec_is_global = False
                    if rec is not None:
                        try:
                            rec_path = os.path.abspath(rec.adapter_dir)
                            rec_is_global = (
                                os.path.commonpath([tft_tools_root, rec_path])
                                == tft_tools_root
                            )
                        except Exception:
                            rec_is_global = False
                    if rec is None or (not rec_is_global) or not adapter_exists(rec.adapter_dir):
                        missing_ft.append(
                            (
                                tid,
                                rec.adapter_dir if rec is not None else f"{tool_key} -> None",
                            )
                        )
                    else:
                        resolved_ft.append((tid, rec.adapter_dir))

                if missing_ft:
                    missing_desc = ", ".join(
                        [f"tid={tid}: {path}" for tid, path in missing_ft]
                    )
                    raise FileNotFoundError(
                        f"No tool_finetuning adapter found for sample {filename}#{test_id}. Missing: {missing_desc}"
                    )

                _run_with_injected_adapters(resolved_ft)
            else:
                tool_pretraining_root = os.path.join(
                    TOOL_PRETRAINING_ROOT_PATH,
                    "bfcl",
                    real_category_key,
                    args.model_name,
                    rank_dir,
                    lr_dir,
                    *run_segments,
                    "adapter",
                )

                resolved_pt: List[Tuple[int, str]] = []
                missing_pt: List[Tuple[int, str]] = []
                for tid in range(len(available_tools)):
                    candidate = os.path.join(
                        tool_pretraining_root, f"data_{test_id}", f"tool_{tid}"
                    )
                    chosen = _choose_existing_adapter_dir([candidate])
                    if chosen is None:
                        missing_pt.append((tid, candidate))
                    else:
                        resolved_pt.append((tid, chosen))

                if missing_pt:
                    missing_desc = ", ".join(
                        [f"tid={tid}: {path}" for tid, path in missing_pt]
                    )
                    raise FileNotFoundError(
                        f"No tool pretraining adapter found for sample {filename}#{test_id}. Missing: {missing_desc}"
                    )

                _run_with_injected_adapters(resolved_pt)
            if write_outputs and result_file is not None:
                with open(result_file, "w") as fout:
                    json.dump(ret, fout, indent=4)
        aggregate_ret.extend(ret)

    right = sum(1 for item in aggregate_ret if item["check_result"]["valid"])
    accuracy = right / len(aggregate_ret) if len(aggregate_ret) > 0 else 0.0

    print(f"Right: {right}, Total: {len(aggregate_ret)}, Accuracy: {accuracy:.4f}")

    if write_outputs and output_root_dir:
        statistics = {
            "total_samples": len(aggregate_ret),
            "correct_samples": right,
            "accuracy": accuracy,
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
        default="multiple",
        help="BFCL test category or group (e.g., multiple, parallel, non_live, live)",
    )
    parser.add_argument(
        "--only_idx",
        type=int,
        default=None,
        help="仅运行指定的样本索引（0-based）；用于单条样本调试。",
    )
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7b-instruct")
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
    args.dataset = "bfcl"
    print(args)
    run_bfcl_inference(args)

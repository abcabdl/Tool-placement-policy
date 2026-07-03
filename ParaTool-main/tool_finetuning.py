from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.tool_finetuning_dataset import ToolFinetuningEntry, ToolFinetuningQA, ToolFinetuningIndex, load_tool_finetuning_entries
from dataset.profiles import make_tool_call_profile
from dataset.tool_keys import tool_key_for_schema, tool_key_hash
from gating_network import (
    GatingNetwork,
    GatingQuestionEncoder,
    make_gating_question_encoder,
    normalize_gating_q_encoder_mode,
    is_finish_tool_schema,
    map_gating_alphas_with_finish_to_experts,
    resolve_tool_ids,
    _load_lora_experts_for_paths,
    _normalize_module_name,
)
from root_dir_path import TOOL_PRETRAINING_ROOT_PATH
from utils import (
    _create_file_logger,
    adapter_exists,
    adapter_file,
    build_tool_finetuning_root_dir,
    build_tool_pretraining_adapter_root,
    get_model,
    get_model_path,
    is_bfcl_multistep_category,
    load_json_dict,
    log_json,
    normalize_dataset_name,
    save_json_atomic,
    tool_desc,
    tool_finetuning_log_path,
    train_filename,
)


def _compute_answer_only_ce_loss(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    ignored_id = -100
    transformer = getattr(model, "model", None) or getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)
    if transformer is None or lm_head is None:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs[0]
    if hidden_states.size(1) < 2:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.reshape(-1)

    hidden_device = flat_hidden.device
    if flat_labels.device != hidden_device:
        flat_labels = flat_labels.to(hidden_device)

    # Compute CE only where labels are supervised to keep per-QA training stable.
    mask = flat_labels != ignored_id

    if not bool(mask.any().item()):
        return torch.zeros((), device=flat_hidden.device, dtype=torch.float32)

    sel_hidden = flat_hidden[mask]
    sel_labels = flat_labels[mask]

    try:
        head_device = next(lm_head.parameters()).device
    except StopIteration:
        head_device = sel_hidden.device
    if sel_hidden.device != head_device:
        sel_hidden = sel_hidden.to(head_device)
        sel_labels = sel_labels.to(head_device)

    logits = lm_head(sel_hidden)
    return F.cross_entropy(logits.float(), sel_labels)

class TrainableGatedLoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.lora_A: nn.ParameterList = nn.ParameterList()
        self.lora_B: nn.ParameterList = nn.ParameterList()
        self.scales: List[float] = []
        self.alphas: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.scales = []
        self.alphas = None

    def set_experts(
        self,
        experts: List[Dict[str, Any]],
        *,
        trainable_mask: Optional[List[bool]] = None,
    ) -> None:
        self.clear()
        if not experts:
            return
        if trainable_mask is not None and len(trainable_mask) != len(experts):
            raise ValueError(
                f"trainable_mask len={len(trainable_mask)} != experts len={len(experts)}"
            )

        device = self.base_linear.weight.device
        base_dtype = self.base_linear.weight.dtype

        A_params: List[nn.Parameter] = []
        B_params: List[nn.Parameter] = []
        scales: List[float] = []

        for i, exp in enumerate(experts):
            A = exp.get("A")
            B = exp.get("B")
            if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
                raise ValueError("expert must include tensor 'A' and 'B'")
            scale_val = float(exp.get("scale", 1.0))

            if B.shape[0] != A.shape[1] and B.shape[1] == A.shape[1]:
                B = B.t()

            trainable = True if trainable_mask is None else bool(trainable_mask[i])

            param_dtype = (
                torch.float32
                if trainable and base_dtype in (torch.float16, torch.bfloat16)
                else base_dtype
            )
            A = A.to(device=device, dtype=param_dtype)
            B = B.to(device=device, dtype=param_dtype)

            A_params.append(nn.Parameter(A.detach().clone(), requires_grad=trainable))
            B_params.append(nn.Parameter(B.detach().clone(), requires_grad=trainable))
            scales.append(scale_val)

        self.lora_A = nn.ParameterList(A_params)
        self.lora_B = nn.ParameterList(B_params)
        self.scales = scales
        self.alphas = None

    def set_alphas(self, alphas: Optional[torch.Tensor]) -> None:
        self.alphas = alphas

    def set_trainable_expert_indices(self, indices: Optional[set[int]]) -> None:
        if indices is None:
            enabled = set(range(len(self.lora_A)))
        else:
            enabled = {int(i) for i in indices}
        for i, (A, B) in enumerate(zip(self.lora_A, self.lora_B)):
            trainable = i in enabled
            A.requires_grad_(trainable)
            B.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_linear(x)
        if not self.lora_A or self.alphas is None:
            return y

        alphas = self.alphas
        if alphas.dim() != 1 or alphas.numel() != len(self.lora_A):
            raise ValueError(
                f"alphas shape {tuple(alphas.shape)} mismatch experts={len(self.lora_A)}"
            )
        if int(torch.count_nonzero(alphas).item()) == 0:
            return y

        x_device = x.device
        x_dtype = x.dtype
        compute_dtype = (
            torch.float32 if x_dtype in (torch.float16, torch.bfloat16) else x_dtype
        )

        A_stack = torch.stack([A.to(device=x_device) for A in self.lora_A], dim=0).to(
            dtype=compute_dtype
        )
        B_stack = torch.stack(
            [(B.to(device=x_device) * float(scale)) for B, scale in zip(self.lora_B, self.scales)],
            dim=0,
        ).to(dtype=compute_dtype)

        alphas = alphas.to(device=x_device, dtype=compute_dtype)
        orig_shape = x.shape
        d_in = orig_shape[-1]
        x_flat = x.view(-1, d_in).to(dtype=compute_dtype)

        XA = torch.einsum("nd,mdr->nmr", x_flat, A_stack)
        XAB = torch.einsum("nmr,mrd->nmd", XA, B_stack)
        delta_flat = (alphas.view(1, -1, 1) * XAB).sum(dim=1)

        d_out = delta_flat.shape[-1]
        delta = delta_flat.view(*orig_shape[:-1], d_out).to(device=y.device, dtype=compute_dtype)
        out = y.to(dtype=compute_dtype) + delta
        return out.to(device=y.device, dtype=y.dtype)

def _count_supervised_tokens(labels: torch.Tensor, *, ignored_id: int = -100) -> int:
    if labels.numel() == 0 or labels.size(1) < 2:
        return 0
    with torch.no_grad():
        return int((labels[:, 1:] != ignored_id).sum().item())

def _replace_with_trainable_gated_lora_linear(
    model: nn.Module,
    target_module_names: List[str],
) -> Dict[str, TrainableGatedLoraLinear]:
    name_to_module: Dict[str, TrainableGatedLoraLinear] = {}

    def _recursive(parent: nn.Module, parent_name: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_module_names):
                wrapped = TrainableGatedLoraLinear(child)
                setattr(parent, child_name, wrapped)
                name_to_module[full_name] = wrapped
            else:
                _recursive(child, full_name)

    _recursive(model, "")
    return name_to_module

def _set_all_alphas(
    gated_modules: Dict[str, TrainableGatedLoraLinear], alphas: Optional[torch.Tensor]
) -> None:
    for module in gated_modules.values():
        module.set_alphas(alphas)

def _sample_range(rng: random.Random, lo: float, hi: float) -> float:
    lo_f = float(lo)
    hi_f = float(hi)
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    if abs(hi_f - lo_f) <= 1e-12:
        return lo_f
    return rng.uniform(lo_f, hi_f)

def _label_smoothed_alphas(
    *,
    M: int,
    answer_idx: int,
    gold_mass: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gold = max(0.0, min(1.0, float(gold_mass)))
    out = torch.zeros((M,), device=device, dtype=dtype)
    if M <= 0:
        return out
    if M == 1:
        out[0] = 1.0
        return out
    base = (1.0 - gold) / float(M - 1)
    out.fill_(base)
    out[int(answer_idx)] = gold
    return out / out.sum().clamp_min(1e-8)

def _wrong_top1_alphas(
    *,
    M: int,
    answer_idx: int,
    rng: random.Random,
    gold_mass_min: float,
    gold_mass_max: float,
    wrong_top_mass_min: float,
    wrong_top_mass_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[int]]:
    if M <= 1:
        return _label_smoothed_alphas(
            M=M,
            answer_idx=answer_idx,
            gold_mass=1.0,
            device=device,
            dtype=dtype,
        ), None

    wrong_candidates = [i for i in range(M) if i != int(answer_idx)]
    wrong_idx = int(rng.choice(wrong_candidates))
    gold_mass = min(_sample_range(rng, gold_mass_min, gold_mass_max), 0.499)
    top_lo = max(float(wrong_top_mass_min), gold_mass + 1e-4)
    top_hi = min(float(wrong_top_mass_max), 1.0 - gold_mass)
    if top_hi < top_lo:
        wrong_top_mass = max(gold_mass + 1e-4, 1.0 - gold_mass)
    else:
        wrong_top_mass = _sample_range(rng, top_lo, top_hi)

    out = torch.zeros((M,), device=device, dtype=dtype)
    out[int(answer_idx)] = float(gold_mass)
    out[wrong_idx] = float(wrong_top_mass)
    rest = max(0.0, 1.0 - float(gold_mass) - float(wrong_top_mass))
    rest_indices = [i for i in range(M) if i not in (int(answer_idx), wrong_idx)]
    if rest_indices:
        rest_each = rest / float(len(rest_indices))
        for i in rest_indices:
            out[i] = rest_each
    return out / out.sum().clamp_min(1e-8), wrong_idx

def _update_alpha_summary(
    summary: Dict[str, Any],
    *,
    mode: str,
    raw_answer_alpha: float,
    augmented_answer_alpha: float,
    trainable_size: int,
) -> None:
    summary["total_steps"] = int(summary.get("total_steps", 0) or 0) + 1
    modes = summary.setdefault("modes", {})
    modes[str(mode)] = int(modes.get(str(mode), 0) or 0) + 1
    trainable_sizes = summary.setdefault("trainable_sizes", {})
    trainable_key = str(int(trainable_size))
    trainable_sizes[trainable_key] = int(trainable_sizes.get(trainable_key, 0) or 0) + 1

    total_steps = float(summary["total_steps"])
    for prefix, value in (
        ("raw_answer_alpha", raw_answer_alpha),
        ("augmented_answer_alpha", augmented_answer_alpha),
    ):
        value = float(value)
        sum_key = f"{prefix}_sum"
        summary[sum_key] = float(summary.get(sum_key, 0.0) or 0.0) + value
        summary[f"{prefix}_min"] = min(
            float(summary.get(f"{prefix}_min", value) or value),
            value,
        )
        summary[f"{prefix}_max"] = max(
            float(summary.get(f"{prefix}_max", value) or value),
            value,
        )
        summary[f"{prefix}_mean"] = float(summary[sum_key]) / total_steps

def _choose_alpha_aug_mode(
    *,
    rng: random.Random,
    stage1_prob: float,
    label_smooth_prob: float,
    wrong_top1_prob: float,
) -> str:
    weights = [
        ("stage1", max(0.0, float(stage1_prob))),
        ("label_smooth", max(0.0, float(label_smooth_prob))),
        ("wrong_top1", max(0.0, float(wrong_top1_prob))),
    ]
    total = sum(w for _, w in weights)
    if total <= 0.0:
        return "stage1"
    draw = rng.random() * total
    acc = 0.0
    for mode, weight in weights:
        acc += weight
        if draw <= acc:
            return mode
    return "stage1"

def _apply_alpha_augmentation(
    *,
    raw_alphas: torch.Tensor,
    answer_idx: int,
    rng: random.Random,
    stage1_prob: float,
    label_smooth_prob: float,
    wrong_top1_prob: float,
    label_gold_min: float,
    label_gold_max: float,
    wrong_gold_min: float,
    wrong_gold_max: float,
    wrong_top_min: float,
    wrong_top_max: float,
) -> Tuple[torch.Tensor, str, Optional[int]]:
    if raw_alphas.dim() != 1 or raw_alphas.numel() <= 0:
        return raw_alphas, "stage1", None
    M = int(raw_alphas.numel())
    if not (0 <= int(answer_idx) < M):
        return raw_alphas, "stage1", None

    mode = _choose_alpha_aug_mode(
        rng=rng,
        stage1_prob=stage1_prob,
        label_smooth_prob=label_smooth_prob,
        wrong_top1_prob=wrong_top1_prob,
    )
    if mode == "label_smooth":
        gold_mass = _sample_range(rng, label_gold_min, label_gold_max)
        return (
            _label_smoothed_alphas(
                M=M,
                answer_idx=int(answer_idx),
                gold_mass=gold_mass,
                device=raw_alphas.device,
                dtype=raw_alphas.dtype,
            ),
            mode,
            None,
        )
    if mode == "wrong_top1":
        augmented, wrong_idx = _wrong_top1_alphas(
            M=M,
            answer_idx=int(answer_idx),
            rng=rng,
            gold_mass_min=wrong_gold_min,
            gold_mass_max=wrong_gold_max,
            wrong_top_mass_min=wrong_top_min,
            wrong_top_mass_max=wrong_top_max,
            device=raw_alphas.device,
            dtype=raw_alphas.dtype,
        )
        return augmented, mode, wrong_idx
    return raw_alphas, "stage1", None

def _trainable_indices_from_alpha(
    *,
    alpha_global: torch.Tensor,
    answer_global_idx: int,
    train_all_alpha_experts: bool,
    threshold: float,
) -> set[int]:
    if not train_all_alpha_experts:
        return {int(answer_global_idx)}
    selected = {
        int(i)
        for i, a in enumerate(alpha_global.detach().cpu().tolist())
        if float(a) > float(threshold)
    }
    selected.add(int(answer_global_idx))
    return selected

def _create_init_adapter(
    save_dir: str,
    gated_modules: Dict[str, TrainableGatedLoraLinear],
    lora_rank: int,
    lora_alpha: int,
    base_model_path: str,
    target_modules: List[str],
) -> str:
    try:
        import safetensors.torch as st
    except ImportError as e:
        raise ImportError(
            "safetensors.torch is required to create LoRA adapters; "
            "please install safetensors in your training environment."
        ) from e

    os.makedirs(save_dir, exist_ok=True)

    state_dict: Dict[str, torch.Tensor] = {}

    for full_name, module in gated_modules.items():
        base_linear = module.base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        dtype = base_linear.weight.dtype

        lora_A = torch.empty(lora_rank, in_features, dtype=dtype)
        nn.init.kaiming_uniform_(lora_A, a=5**0.5)

        lora_B = torch.zeros(out_features, lora_rank, dtype=dtype)

        key_prefix = f"base_model.model.{full_name}"
        state_dict[f"{key_prefix}.lora_A.weight"] = lora_A.cpu()
        state_dict[f"{key_prefix}.lora_B.weight"] = lora_B.cpu()

    adapter_weights_path = os.path.join(save_dir, "adapter_model.safetensors")
    st.save_file(state_dict, adapter_weights_path)

    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0,
        "peft_type": "LORA",
        "r": lora_rank,
        "rank_pattern": {},
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    config_file = os.path.join(save_dir, "adapter_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return save_dir

def _ensure_finish_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(schema or {})
    base.update(
        {
            "category_name": str(base.get("category_name") or "__special__").strip(),
            "tool_name": str(base.get("tool_name") or "__finish__").strip(),
            "api_name": str(base.get("api_name") or "Finish").strip(),
            "name": str(base.get("name") or "Finish").strip() or "Finish",
            "description": str(
                base.get("description")
                or "End the tool-calling loop and output the final answer."
            ).strip(),
        }
    )
    if "parameters" not in base:
        base["parameters"] = {"type": "dict", "properties": {}, "required": []}
    return base

def _export_one_adapter(
    adapter_dir: str,
    *,
    expert_idx: int,
    gated_modules: Dict[str, TrainableGatedLoraLinear],
    base_dtype: torch.dtype,
) -> None:
    try:
        import safetensors.torch as st  # type: ignore
    except ImportError as e:
        raise ImportError(
            "safetensors.torch is required to export tool_finetuning adapters; please install safetensors."
        ) from e

    os.makedirs(adapter_dir, exist_ok=True)
    save_path = os.path.join(adapter_dir, "adapter_model.safetensors")

    state_dict: Dict[str, torch.Tensor] = {}
    for full_name, module in gated_modules.items():
        if not module.lora_A or expert_idx >= len(module.lora_A):
            continue
        A = module.lora_A[expert_idx].detach().to(device="cpu", dtype=base_dtype)
        B = module.lora_B[expert_idx].detach().to(device="cpu", dtype=base_dtype)
        state_dict[f"{full_name}.lora_A.weight"] = A.t().contiguous()
        state_dict[f"{full_name}.lora_B.weight"] = B.t().contiguous()

    st.save_file(state_dict, save_path)

def _global_tool_adapter_dir(tool_finetuning_root: str, tool_key: str) -> str:
    return os.path.join(tool_finetuning_root, "tools", tool_key_hash(tool_key, n=32))

def _copy_adapter_config_if_missing(src_dir: Optional[str], dst_dir: str) -> None:
    if not src_dir:
        return
    src = os.path.join(src_dir, "adapter_config.json")
    dst = os.path.join(dst_dir, "adapter_config.json")
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)

def _write_global_tool_metadata(
    adapter_dir: str,
    *,
    tool_key: str,
    schema: Dict[str, Any],
    dataset: str,
    num_updates: int,
    last_example_index: Optional[int],
    seed_adapter_dir: Optional[str],
) -> None:
    meta = {
        "tool_key": tool_key,
        "schema_hash": tool_key_hash(tool_key, n=32),
        "tool_name": str(schema.get("name") or "").strip() if isinstance(schema, dict) else "",
        "tool_desc": tool_desc(schema, dataset=dataset) if isinstance(schema, dict) else "",
        "num_updates": int(num_updates),
        "last_example_index": (
            int(last_example_index) if last_example_index is not None else None
        ),
        "seed_adapter_dir": seed_adapter_dir,
        "updated_at": float(time.time()),
    }
    save_json_atomic(os.path.join(adapter_dir, "metadata.json"), meta)

def train_tool_finetuning(
    *,
    dataset: str = "bfcl",
    category: str,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    tool_finetuning_lr: float,
    tool_finetuning_epochs: int,
    tool_pretraining_lr: float = 0.0006,
    tool_pretraining_epochs: int = 3,
    tool_finetuning_inner_epochs: int = 1,
    gating: bool = False,
    gating_network_ckpt_path: Optional[str] = None,
    train_all_alpha_experts: bool = False,
    trainable_alpha_threshold: float = 0.0,
    alpha_aug_stage1_prob: float = 1.0,
    alpha_aug_label_smooth_prob: float = 0.0,
    alpha_aug_wrong_top1_prob: float = 0.0,
    alpha_aug_label_gold_min: float = 0.6,
    alpha_aug_label_gold_max: float = 0.7,
    alpha_aug_wrong_gold_min: float = 0.35,
    alpha_aug_wrong_gold_max: float = 0.5,
    alpha_aug_wrong_top_min: float = 0.51,
    alpha_aug_wrong_top_max: float = 0.65,
    alpha_aug_seed: int = 42,
    run_tag: Optional[str] = None,
) -> None:
    dataset_name = normalize_dataset_name(dataset)
    profile = make_tool_call_profile(dataset=dataset_name, tool_call_ways="without")

    tool_finetuning_inner_epochs = max(1, int(tool_finetuning_inner_epochs or 1))
    alpha_aug_rng = random.Random(int(alpha_aug_seed))
    trainable_alpha_threshold = float(trainable_alpha_threshold)

    base_model, tokenizer, _generation_config = get_model(model_name, max_new_tokens=1024)
    device = base_model.device

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    base_dtype = next(base_model.parameters()).dtype
    hidden_size = int(base_model.config.hidden_size)

    gating_net: Optional[GatingNetwork] = None
    tool_embedding: Optional[nn.Embedding] = None
    tool_vocab: Dict[str, int] = {}
    tool_vocab_type: Optional[str] = None
    unk_tool_id: Optional[int] = None
    gating_network_ckpt_path_resolved: Optional[str] = None
    gating_q_encoder: Optional[GatingQuestionEncoder] = None

    if gating:
        if not gating_network_ckpt_path or not str(gating_network_ckpt_path).strip():
            raise ValueError(
                "Tool finetuning requires `--gating_network_ckpt_path` when `--gating` is enabled."
            )
        gating_network_ckpt_path_resolved = os.path.expanduser(os.path.expandvars(gating_network_ckpt_path))
        try:
            gating_network_ckpt_path_resolved = gating_network_ckpt_path_resolved.format(category=category)
        except Exception:
            pass
        if not os.path.isabs(gating_network_ckpt_path_resolved):
            raise ValueError(
                f"`--gating_network_ckpt_path` must be an absolute path, got: {gating_network_ckpt_path_resolved!r}"
            )
        if not os.path.exists(gating_network_ckpt_path_resolved):
            raise FileNotFoundError(f"GatingNetwork checkpoint not found: {gating_network_ckpt_path_resolved}")

        ckpt = torch.load(gating_network_ckpt_path_resolved, map_location=device)
        gating_dim = int(ckpt.get("hidden_size", hidden_size))
        ckpt_hidden_dim = int(ckpt.get("hidden_dim", 512))
        gating_net = GatingNetwork(dim=gating_dim, hidden_dim=ckpt_hidden_dim).to(device)
        gating_net.load_state_dict(ckpt["gating_state_dict"])
        gating_net.eval()
        for p in gating_net.parameters():
            p.requires_grad = False

        emb_state = ckpt.get("tool_embedding_state_dict")
        if isinstance(emb_state, dict) and isinstance(emb_state.get("weight"), torch.Tensor):
            weight = emb_state["weight"]
            if int(weight.shape[1]) != gating_dim:
                raise ValueError(
                    f"tool_embedding dim {int(weight.shape[1])} != gating_dim {gating_dim} from checkpoint"
                )
            tool_embedding = nn.Embedding(
                num_embeddings=int(weight.shape[0]),
                embedding_dim=int(weight.shape[1]),
                sparse=True,
            ).to(device=device, dtype=weight.dtype)
            tool_embedding.load_state_dict(emb_state)
            tool_embedding.eval()
            for p in tool_embedding.parameters():
                p.requires_grad = False

            vocab = ckpt.get("tool_vocab")
            if isinstance(vocab, dict):
                tool_vocab = {str(k): int(v) for k, v in vocab.items() if isinstance(v, int)}
            tool_vocab_type = str(ckpt.get("tool_vocab_type") or "").strip() or None
            if "unk_tool_id" in ckpt:
                try:
                    unk_tool_id = int(ckpt["unk_tool_id"])
                except Exception:
                    unk_tool_id = None

        qe_loaded = normalize_gating_q_encoder_mode(ckpt.get("q_encoder", "llm"))
        gating_q_encoder = make_gating_question_encoder(
            qe_loaded,
            device=device,
            base_model=base_model,
            tokenizer=tokenizer,
        )

    tool_pretraining_root = build_tool_pretraining_adapter_root(
        root_dir=TOOL_PRETRAINING_ROOT_PATH,
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        dataset=dataset_name,
        tool_pretraining_lr=tool_pretraining_lr,
        tool_pretraining_epochs=tool_pretraining_epochs,
        category=category,
        run_tag=run_tag,
    )
    tool_finetuning_root = build_tool_finetuning_root_dir(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        dataset=dataset_name,
        tool_pretraining_lr=tool_pretraining_lr,
        tool_pretraining_epochs=tool_pretraining_epochs,
        category=category,
        tool_finetuning_lr=tool_finetuning_lr,
        tool_finetuning_epochs=tool_finetuning_epochs,
        run_tag=run_tag,
    )
    os.makedirs(tool_finetuning_root, exist_ok=True)
    index_path = os.path.join(tool_finetuning_root, "tool_index.json")
    tool_finetuning_index = ToolFinetuningIndex.load(index_path)

    entries = load_tool_finetuning_entries(
        dataset=dataset_name,
        category=category,
    )
    raw_entries = list(entries or [])
    if not raw_entries:
        print("[tool_finetuning] no entries found, skip.")
        return

    gated_target_modules = ["down_proj", "up_proj", "gate_proj"]
    gated_modules = _replace_with_trainable_gated_lora_linear(base_model, gated_target_modules)
    print(f"[tool_finetuning] wrapped {len(gated_modules)} Linear modules with TrainableGatedLoraLinear")
    _set_all_alphas(gated_modules, None)

    def resolve_adapter_dir(
        *,
        tool_schema: Dict[str, Any],
        tool_key: str,
        entry: ToolFinetuningEntry,
        local_key_to_idx: Dict[str, int],
    ) -> Optional[str]:
        if dataset_name == "stb" and tool_key == entry.tool_key and entry.tid >= 0:
            cand = os.path.join(
                tool_pretraining_root, f"data_{entry.example_index}", f"tool_{entry.tid}"
            )
            if adapter_exists(cand):
                return cand

        local_idx = local_key_to_idx.get(tool_key)
        if local_idx is not None:
            cand = os.path.join(
                tool_pretraining_root, f"data_{entry.example_index}", f"tool_{local_idx}"
            )
            if adapter_exists(cand):
                return cand

        return None

    entries_sorted = sorted(
        raw_entries,
        key=lambda e: (int(getattr(e, "example_index", 0)), int(getattr(e, "tid", -1)), str(getattr(e, "tool_key", ""))),
    )

    grouped: Dict[int, List[ToolFinetuningEntry]] = {}
    for e in entries_sorted:
        grouped.setdefault(int(getattr(e, "example_index", 0)), []).append(e)

    merged: List[ToolFinetuningEntry] = []
    for ex_idx in sorted(grouped.keys()):
        items = grouped[ex_idx]
        base = items[0]
        merged_qas: List[ToolFinetuningQA] = []
        for it in items:
            merged_qas.extend(list(getattr(it, "qas", []) or []))
        merged.append(
            ToolFinetuningEntry(
                example_index=int(ex_idx),
                tid=-1,
                tool_schema=dict(getattr(base, "tool_schema", {}) or {}),
                tool_key=str(getattr(base, "tool_key", "") or ""),
                function_tools=list(getattr(base, "function_tools", []) or []),
                qas=merged_qas,
                source_question=getattr(base, "source_question", None),
                source_id=getattr(base, "source_id", None),
            )
        )
    entries_sorted = merged
    print(f"[tool_finetuning] loaded entries: raw={len(raw_entries)} grouped={len(entries_sorted)} (joint)")

    progress_path = os.path.join(tool_finetuning_root, "global_progress.json")
    alpha_summary_path = os.path.join(tool_finetuning_root, "alpha_aug_summary.json")
    progress = load_json_dict(progress_path)
    progress.setdefault("version", 1)
    processed = progress.setdefault("processed", {})
    tool_update_totals = progress.setdefault("tool_updates", {})
    alpha_summary = progress.setdefault("alpha_summary", {})

    num_entries = len(entries_sorted)
    print(
        f"[tool_finetuning] global joint training: outer_epochs={tool_finetuning_epochs} "
        f"inner_epochs={tool_finetuning_inner_epochs} entries={num_entries}"
    )

    for outer_ep in range(tool_finetuning_epochs):
        for entry_idx, entry in enumerate(entries_sorted, start=1):
            progress_key = f"epoch={outer_ep + 1}:example_index={entry.example_index}"
            if processed.get(progress_key):
                print(
                    f"[tool_finetuning] skip global entry epoch={outer_ep + 1}/{tool_finetuning_epochs} "
                    f"entry={entry_idx}/{num_entries} example_index={entry.example_index} "
                    f"(already processed)"
                )
                continue

            out_dir = os.path.join(
                tool_finetuning_root,
                "_logs",
                f"epoch_{outer_ep + 1}",
                f"data_{entry.example_index}",
            )
            entry_log_path = tool_finetuning_log_path(out_dir=out_dir)
            os.makedirs(os.path.dirname(entry_log_path), exist_ok=True)
            entry_logger, entry_log_handler = _create_file_logger(
                entry_log_path,
                level=logging.INFO,
            )
            try:
                entry_alpha_summary: Dict[str, Any] = {}
                log_json(
                    entry_logger,
                    {
                        "event": "global_entry_start",
                        "outer_epoch": outer_ep + 1,
                        "outer_epochs": tool_finetuning_epochs,
                        "entry_idx": entry_idx,
                        "num_entries": num_entries,
                        "example_index": entry.example_index,
                        "source_id": getattr(entry, "source_id", None),
                        "qas": len(entry.qas or []),
                        "dataset": dataset_name,
                        "category": category,
                        "tool_finetuning_inner_epochs": tool_finetuning_inner_epochs,
                        "train_all_alpha_experts": bool(train_all_alpha_experts),
                        "trainable_alpha_threshold": float(trainable_alpha_threshold),
                        "alpha_aug_probs": {
                            "stage1": float(alpha_aug_stage1_prob),
                            "label_smooth": float(alpha_aug_label_smooth_prob),
                            "wrong_top1": float(alpha_aug_wrong_top1_prob),
                        },
                        "log_path": entry_log_path,
                    },
                )

                local_key_to_idx: Dict[str, int] = {}
                for i, schema in enumerate(entry.function_tools):
                    if isinstance(schema, dict):
                        key = tool_key_for_schema(schema, dataset=dataset_name)
                        local_key_to_idx.setdefault(key, i)

                seen_keys = set()
                union_tools: List[Dict[str, Any]] = []
                union_keys: List[str] = []

                def _add_union(schema: Dict[str, Any]) -> None:
                    key = tool_key_for_schema(schema, dataset=dataset_name)
                    if key in seen_keys:
                        return
                    seen_keys.add(key)
                    union_tools.append(schema)
                    union_keys.append(key)

                for schema in entry.function_tools:
                    if isinstance(schema, dict):
                        _add_union(schema)
                for qa in entry.qas:
                    for schema in qa.tools:
                        if isinstance(schema, dict):
                            _add_union(schema)

                if not union_tools:
                    log_json(
                        entry_logger,
                        {
                            "event": "global_entry_skip_no_tools",
                            "example_index": entry.example_index,
                        },
                    )
                    processed[progress_key] = {
                        "completed_at": time.time(),
                        "skipped": "no_tools",
                    }
                    save_json_atomic(progress_path, progress)
                    continue

                union_adapter_dirs: List[str] = []
                kept_tools: List[Dict[str, Any]] = []
                kept_keys: List[str] = []
                adapter_seed_dirs: Dict[str, Optional[str]] = {}
                adapter_sources: Dict[str, str] = {}
                base_model_path = get_model_path(model_name)

                for schema, key in zip(union_tools, union_keys):
                    global_dir = _global_tool_adapter_dir(tool_finetuning_root, key)
                    rec = tool_finetuning_index.get(key)
                    adapter_dir: Optional[str] = None
                    source = "missing"
                    seed_dir: Optional[str] = None
                    rec_is_global = False
                    if rec is not None:
                        try:
                            tools_root = os.path.abspath(os.path.join(tool_finetuning_root, "tools"))
                            rec_path = os.path.abspath(rec.adapter_dir)
                            rec_is_global = os.path.commonpath([tools_root, rec_path]) == tools_root
                        except Exception:
                            rec_is_global = False
                    if rec is not None and rec_is_global and adapter_exists(rec.adapter_dir):
                        adapter_dir = rec.adapter_dir
                        source = "tool_finetuning_global"
                        seed_dir = str((rec.meta or {}).get("seed_adapter_dir") or "")
                        if key not in tool_update_totals:
                            try:
                                tool_update_totals[key] = int(
                                    (rec.meta or {}).get("num_updates", 0) or 0
                                )
                            except Exception:
                                tool_update_totals[key] = 0
                    else:
                        adapter_dir = resolve_adapter_dir(
                            tool_schema=schema,
                            tool_key=key,
                            entry=entry,
                            local_key_to_idx=local_key_to_idx,
                        )
                        if adapter_dir is not None and adapter_exists(adapter_dir):
                            source = "tool_pretraining"
                            seed_dir = adapter_dir
                        else:
                            adapter_dir = global_dir
                            if not adapter_exists(adapter_dir):
                                _create_init_adapter(
                                    save_dir=adapter_dir,
                                    gated_modules=gated_modules,
                                    lora_rank=lora_rank,
                                    lora_alpha=lora_alpha,
                                    base_model_path=base_model_path,
                                    target_modules=gated_target_modules,
                                )
                            source = "init"
                            seed_dir = None

                    kept_tools.append(schema)
                    kept_keys.append(key)
                    union_adapter_dirs.append(adapter_dir)
                    adapter_seed_dirs[key] = seed_dir
                    adapter_sources[key] = source

                union_key_to_idx = {k: i for i, k in enumerate(kept_keys)}
                adapter_files = [adapter_file(d) for d in union_adapter_dirs]
                log_json(
                    entry_logger,
                    {
                        "event": "global_entry_candidate_pool",
                        "outer_epoch": outer_ep + 1,
                        "example_index": entry.example_index,
                        "pool_size": len(kept_keys),
                        "pool": [
                            {
                                "i": i,
                                "tool_key": key,
                                "tool_desc": tool_desc(schema, dataset=dataset_name),
                                "adapter_dir": union_adapter_dirs[i],
                                "source": adapter_sources.get(key),
                            }
                            for i, (key, schema) in enumerate(zip(kept_keys, kept_tools))
                        ],
                    },
                )

                experts_per_module = _load_lora_experts_for_paths(
                    adapter_files,
                    target_modules=gated_target_modules,
                    lora_alpha=lora_alpha,
                    r=lora_rank,
                    device=device,
                    dtype=base_dtype,
                )
                for full_name, module in gated_modules.items():
                    matched_experts: List[Dict[str, Any]] = []
                    norm_full = _normalize_module_name(full_name)
                    for mod_name, exps in experts_per_module.items():
                        short = _normalize_module_name(mod_name)
                        if norm_full == short:
                            matched_experts.extend(exps)
                    if len(matched_experts) != len(kept_keys):
                        raise ValueError(
                            f"expert count mismatch for module={full_name}: "
                            f"{len(matched_experts)} != {len(kept_keys)}"
                        )
                    module.set_experts(matched_experts, trainable_mask=[True] * len(kept_keys))
                    module.set_alphas(None)

                all_params: List[nn.Parameter] = []
                for module in gated_modules.values():
                    for p in list(module.lora_A.parameters()) + list(module.lora_B.parameters()):
                        all_params.append(p)
                if not all_params:
                    raise RuntimeError(
                        f"No trainable LoRA params for global entry example_index={entry.example_index}"
                    )
                optimizer = torch.optim.AdamW(all_params, lr=tool_finetuning_lr)
                entry_update_counts: Dict[str, int] = {k: 0 for k in kept_keys}

                msg = (
                    f"[tool_finetuning] global epoch {outer_ep + 1}/{tool_finetuning_epochs} "
                    f"entry {entry_idx}/{num_entries} example_index={entry.example_index} "
                    f"qas={len(entry.qas)} union_tools={len(kept_keys)}"
                )
                print(msg)
                entry_logger.info(msg)

                for inner_ep in range(tool_finetuning_inner_epochs):
                    for step_in_ep, qa in enumerate(entry.qas, start=1):
                        answer_key = str(getattr(qa, "answer_tool_key", "") or "").strip()
                        if not answer_key:
                            if isinstance(qa.answer_idx, int) and 0 <= qa.answer_idx < len(qa.tools):
                                answer_schema = qa.tools[qa.answer_idx]
                                if isinstance(answer_schema, dict):
                                    answer_key = tool_key_for_schema(
                                        answer_schema,
                                        dataset=dataset_name,
                                    )
                        answer_global_idx = union_key_to_idx.get(answer_key)
                        if answer_global_idx is None:
                            log_json(
                                entry_logger,
                                {
                                    "event": "qa_skip_answer_not_in_pool",
                                    "outer_epoch": outer_ep + 1,
                                    "inner_epoch": inner_ep + 1,
                                    "step": step_in_ep,
                                    "example_index": entry.example_index,
                                    "answer_tool_key": answer_key,
                                },
                            )
                            continue

                        tools_experts: List[Dict[str, Any]] = []
                        union_indices: List[int] = []
                        finish_schema: Optional[Dict[str, Any]] = None
                        answer_idx_expert: Optional[int] = None

                        for schema in qa.tools:
                            if not isinstance(schema, dict):
                                continue
                            if finish_schema is None and is_finish_tool_schema(schema):
                                finish_schema = schema
                            k = tool_key_for_schema(schema, dataset=dataset_name)
                            uidx = union_key_to_idx.get(k)
                            if uidx is None:
                                continue
                            if answer_idx_expert is None and k == answer_key:
                                answer_idx_expert = len(tools_experts)
                            tools_experts.append(schema)
                            union_indices.append(int(uidx))

                        tools_gating: List[Dict[str, Any]] = list(tools_experts)
                        if finish_schema is not None:
                            tools_gating.append(_ensure_finish_schema(finish_schema))
                        if answer_idx_expert is None or not tools_gating:
                            continue

                        M = len(tools_gating)
                        if not gating:
                            alphas_gating = torch.full(
                                (M,), 1.0 / float(M), device=device, dtype=base_dtype
                            )
                        else:
                            _set_all_alphas(gated_modules, None)
                            assert gating_net is not None
                            assert tool_embedding is not None and tool_vocab and isinstance(unk_tool_id, int), \
                                "gating=True requires a checkpoint with tool_embedding and tool_vocab"
                            assert gating_q_encoder is not None
                            dtype = gating_net.fc1.weight.dtype
                            q_emb = gating_q_encoder.encode(qa.question).to(dtype=dtype)
                            tool_ids = resolve_tool_ids(
                                tools_gating,
                                tool_vocab=tool_vocab,
                                tool_vocab_type=tool_vocab_type,
                                unk_tool_id=unk_tool_id,
                            )
                            tool_ids_tensor = torch.tensor(
                                tool_ids, dtype=torch.long, device=device
                            )
                            tool_embs = tool_embedding(tool_ids_tensor).unsqueeze(0).to(dtype=dtype)
                            tool_mask = torch.ones(1, M, device=device, dtype=torch.bool)
                            alphas_gating, _ = gating_net(
                                q_emb, tool_embs, tool_mask, return_scores=False
                            )
                            alphas_gating = alphas_gating[0].to(dtype=base_dtype)

                        raw_alphas_gating = alphas_gating.detach().clone()
                        alphas_gating, alpha_aug_mode, alpha_aug_wrong_idx = _apply_alpha_augmentation(
                            raw_alphas=raw_alphas_gating,
                            answer_idx=int(answer_idx_expert),
                            rng=alpha_aug_rng,
                            stage1_prob=alpha_aug_stage1_prob,
                            label_smooth_prob=alpha_aug_label_smooth_prob,
                            wrong_top1_prob=alpha_aug_wrong_top1_prob,
                            label_gold_min=alpha_aug_label_gold_min,
                            label_gold_max=alpha_aug_label_gold_max,
                            wrong_gold_min=alpha_aug_wrong_gold_min,
                            wrong_gold_max=alpha_aug_wrong_gold_max,
                            wrong_top_min=alpha_aug_wrong_top_min,
                            wrong_top_max=alpha_aug_wrong_top_max,
                        )

                        expert_indices = list(range(len(tools_experts)))
                        applied_local, finish_selected, finish_idx = map_gating_alphas_with_finish_to_experts(
                            alphas_gating,
                            tools_gating=tools_gating,
                            expert_indices=expert_indices,
                        )
                        if finish_selected:
                            continue

                        alpha_global = torch.zeros(len(kept_keys), device=device, dtype=base_dtype)
                        for a, uidx in zip(applied_local, union_indices):
                            alpha_global[int(uidx)] = alpha_global[int(uidx)] + a
                        _set_all_alphas(gated_modules, alpha_global)
                        trainable_global_indices = _trainable_indices_from_alpha(
                            alpha_global=alpha_global,
                            answer_global_idx=int(answer_global_idx),
                            train_all_alpha_experts=bool(train_all_alpha_experts),
                            threshold=float(trainable_alpha_threshold),
                        )
                        for module in gated_modules.values():
                            module.set_trainable_expert_indices(trainable_global_indices)
                        raw_answer_alpha = (
                            float(raw_alphas_gating[int(answer_idx_expert)].detach().cpu().item())
                            if 0 <= int(answer_idx_expert) < int(raw_alphas_gating.numel())
                            else 0.0
                        )
                        augmented_answer_alpha = (
                            float(alphas_gating[int(answer_idx_expert)].detach().cpu().item())
                            if 0 <= int(answer_idx_expert) < int(alphas_gating.numel())
                            else 0.0
                        )

                        enc = profile.encode_train_sample(
                            tokenizer,
                            qa.question,
                            tools_gating,
                            qa.answer_text,
                            multi_step=is_bfcl_multistep_category(dataset_name, category),
                        )
                        input_ids_tensor = torch.tensor(
                            enc.input_ids, dtype=torch.long, device=device
                        ).unsqueeze(0)
                        attention_mask = torch.ones_like(input_ids_tensor, device=device)
                        labels = input_ids_tensor.clone()
                        labels[:, : enc.prompt_len] = -100
                        num_supervised = _count_supervised_tokens(labels)
                        if num_supervised <= 0:
                            continue

                        optimizer.zero_grad(set_to_none=True)
                        loss = _compute_answer_only_ce_loss(
                            base_model,
                            input_ids=input_ids_tensor,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss.backward()

                        optimizer.step()

                        for trainable_idx in sorted(trainable_global_indices):
                            if not (0 <= int(trainable_idx) < len(kept_keys)):
                                continue
                            trainable_key = kept_keys[int(trainable_idx)]
                            entry_update_counts[trainable_key] = entry_update_counts.get(trainable_key, 0) + 1
                            tool_update_totals[trainable_key] = (
                                int(tool_update_totals.get(trainable_key, 0) or 0) + 1
                            )
                        _update_alpha_summary(
                            alpha_summary,
                            mode=alpha_aug_mode,
                            raw_answer_alpha=raw_answer_alpha,
                            augmented_answer_alpha=augmented_answer_alpha,
                            trainable_size=len(trainable_global_indices),
                        )
                        _update_alpha_summary(
                            entry_alpha_summary,
                            mode=alpha_aug_mode,
                            raw_answer_alpha=raw_answer_alpha,
                            augmented_answer_alpha=augmented_answer_alpha,
                            trainable_size=len(trainable_global_indices),
                        )

                        log_json(
                            entry_logger,
                            {
                                "event": "qa_loss",
                                "outer_epoch": outer_ep + 1,
                                "inner_epoch": inner_ep + 1,
                                "step": step_in_ep,
                                "example_index": entry.example_index,
                                "answer_tool_key": answer_key,
                                "loss": float(loss.item()),
                                "loss_mode": "answer_only_ce",
                                "num_supervised_tokens": int(num_supervised),
                                "alpha_aug_mode": alpha_aug_mode,
                                "alpha_aug_wrong_idx": (
                                    int(alpha_aug_wrong_idx)
                                    if alpha_aug_wrong_idx is not None
                                    else None
                                ),
                                "raw_alphas": [
                                    float(x) for x in raw_alphas_gating.detach().cpu().tolist()
                                ],
                                "alphas": [
                                    float(x) for x in alphas_gating.detach().cpu().tolist()
                                ],
                                "applied_alphas": [
                                    float(x) for x in applied_local.detach().cpu().tolist()
                                ],
                                "train_all_alpha_experts": bool(train_all_alpha_experts),
                                "trainable_alpha_threshold": float(trainable_alpha_threshold),
                                "trainable_expert_indices": [
                                    int(i) for i in sorted(trainable_global_indices)
                                ],
                                "trainable_tool_descs": [
                                    tool_desc(kept_tools[int(i)], dataset=dataset_name)
                                    for i in sorted(trainable_global_indices)
                                    if 0 <= int(i) < len(kept_tools)
                                ],
                                "finish_idx": int(finish_idx) if finish_idx is not None else None,
                            },
                        )
                        if step_in_ep % 10 == 0:
                            msg = (
                                f"[tool_finetuning] global epoch {outer_ep + 1}/{tool_finetuning_epochs} "
                                f"entry {entry_idx}/{num_entries} inner {inner_ep + 1}/{tool_finetuning_inner_epochs} "
                                f"step {step_in_ep}/{len(entry.qas)} loss={loss.item():.4f}"
                            )
                            print(msg)
                            entry_logger.info(msg)

                for module in gated_modules.values():
                    module.set_trainable_expert_indices(None)
                _set_all_alphas(gated_modules, None)

                for uidx, (key, schema) in enumerate(zip(kept_keys, kept_tools)):
                    global_dir = _global_tool_adapter_dir(tool_finetuning_root, key)
                    _export_one_adapter(
                        global_dir,
                        expert_idx=int(uidx),
                        gated_modules=gated_modules,
                        base_dtype=base_dtype,
                    )
                    _copy_adapter_config_if_missing(adapter_seed_dirs.get(key), global_dir)

                    prev_record = tool_finetuning_index.get(key)
                    prev_meta = dict(prev_record.meta) if prev_record is not None else {}
                    seed_adapter_dir = (
                        prev_meta.get("seed_adapter_dir") or adapter_seed_dirs.get(key)
                    )
                    num_updates = max(
                        int(tool_update_totals.get(key, 0) or 0),
                        int(prev_meta.get("num_updates", 0) or 0),
                    )
                    meta = {
                        "dataset": dataset_name,
                        "category": category,
                        "data_format": "jsonl",
                        "data_schema": "function_training_data_v1",
                        "run_tag": run_tag,
                        "model_name": model_name,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                        "tool_pretraining_lr": tool_pretraining_lr,
                        "tool_pretraining_epochs": tool_pretraining_epochs,
                        "tool_finetuning_lr": tool_finetuning_lr,
                        "tool_finetuning_epochs": tool_finetuning_epochs,
                        "tool_finetuning_inner_epochs": tool_finetuning_inner_epochs,
                        "train_all_alpha_experts": bool(train_all_alpha_experts),
                        "trainable_alpha_threshold": float(trainable_alpha_threshold),
                        "alpha_aug_stage1_prob": float(alpha_aug_stage1_prob),
                        "alpha_aug_label_smooth_prob": float(alpha_aug_label_smooth_prob),
                        "alpha_aug_wrong_top1_prob": float(alpha_aug_wrong_top1_prob),
                        "tool_desc": tool_desc(schema, dataset=dataset_name),
                        "num_updates": num_updates,
                        "last_example_index": int(entry.example_index)
                        if entry_update_counts.get(key, 0) > 0
                        else prev_meta.get("last_example_index"),
                        "seed_adapter_dir": seed_adapter_dir,
                    }
                    tool_finetuning_index.upsert(
                        key,
                        adapter_dir=global_dir,
                        status="done",
                        meta=meta,
                    )
                    _write_global_tool_metadata(
                        global_dir,
                        tool_key=key,
                        schema=schema,
                        dataset=dataset_name,
                        num_updates=num_updates,
                        last_example_index=meta.get("last_example_index"),
                        seed_adapter_dir=seed_adapter_dir,
                    )

                tool_finetuning_index.save()
                processed[progress_key] = {
                    "completed_at": time.time(),
                    "outer_epoch": outer_ep + 1,
                    "example_index": int(entry.example_index),
                    "entry_update_counts": entry_update_counts,
                    "alpha_summary": entry_alpha_summary,
                }
                save_json_atomic(progress_path, progress)
                save_json_atomic(
                    alpha_summary_path,
                    {
                        "tool_finetuning_root": tool_finetuning_root,
                        "progress_path": progress_path,
                        "summary": alpha_summary,
                    },
                )
                entry_logger.info(
                    json.dumps(
                        {
                            "event": "global_entry_export_done",
                            "example_index": entry.example_index,
                            "entry_update_counts": entry_update_counts,
                            "alpha_summary": entry_alpha_summary,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                entry_logger.exception(
                    f"Global tool_finetuning entry failed: outer_epoch={outer_ep + 1} "
                    f"entry_idx={entry_idx} example_index={entry.example_index}"
                )
                raise
            finally:
                _set_all_alphas(gated_modules, None)
                entry_logger.removeHandler(entry_log_handler)
                try:
                    entry_log_handler.close()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"[tool_finetuning] global joint training complete. index={index_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool finetuning: train LoRA experts with soft routing (export to TOOL_FINETUNING_ROOT_PATH)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (keys match CLI args). CLI args override YAML.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bfcl",
        choices=["bfcl", "stb", "stabletoolbench", "toolbench"],
    )
    parser.add_argument("--category", type=str, default="multiple")
    parser.add_argument("--model_name", type=str, default="llama3.1-8b-instruct")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--tool_finetuning_lr", type=float, default=1e-5)
    parser.add_argument("--tool_finetuning_epochs", type=int, default=2)
    parser.add_argument(
        "--tool_pretraining_lr",
        type=float,
        default=0.0006,
        help="Tool pretraining(single-tool) learning rate used to locate base adapters.",
    )
    parser.add_argument(
        "--tool_pretraining_epochs",
        type=int,
        default=3,
        help="Tool pretraining(single-tool) epochs used to locate base adapters.",
    )
    parser.add_argument(
        "--tool_finetuning_inner_epochs",
        type=int,
        default=1,
        help="Number of QA passes inside each question for the global cumulative joint mode.",
    )

    parser.add_argument(
        "--gating",
        action="store_true",
        help=(
            "Use GatingNetwork checkpoint for expert mixing (requires --gating_network_ckpt_path). "
            "If omitted, use uniform weights."
        ),
    )
    parser.add_argument(
        "--train_all_alpha_experts",
        action="store_true",
        help="If set, every loaded candidate expert with alpha > threshold is trainable for each QA.",
    )
    parser.add_argument(
        "--trainable_alpha_threshold",
        type=float,
        default=0.0,
        help="Minimum alpha required for a candidate expert to be trainable when train_all_alpha_experts is set.",
    )
    parser.add_argument("--alpha_aug_stage1_prob", type=float, default=1.0)
    parser.add_argument("--alpha_aug_label_smooth_prob", type=float, default=0.0)
    parser.add_argument("--alpha_aug_wrong_top1_prob", type=float, default=0.0)
    parser.add_argument("--alpha_aug_label_gold_min", type=float, default=0.6)
    parser.add_argument("--alpha_aug_label_gold_max", type=float, default=0.7)
    parser.add_argument("--alpha_aug_wrong_gold_min", type=float, default=0.35)
    parser.add_argument("--alpha_aug_wrong_gold_max", type=float, default=0.5)
    parser.add_argument("--alpha_aug_wrong_top_min", type=float, default=0.51)
    parser.add_argument("--alpha_aug_wrong_top_max", type=float, default=0.65)
    parser.add_argument("--alpha_aug_seed", type=int, default=42)
    parser.add_argument(
        "--gating_network_ckpt_path",
        type=str,
        default=None,
        help=(
            "Absolute GatingNetwork checkpoint path to use when --gating is set (no auto-discovery). "
            "Supports `{category}` placeholder."
        ),
    )

    pre_args, _ = parser.parse_known_args()
    if getattr(pre_args, "config", None):
        from utils import apply_env_from_config, load_yaml_file, set_parser_defaults_from_config

        cfg = load_yaml_file(pre_args.config)
        apply_env_from_config(cfg, config_path=pre_args.config)
        set_parser_defaults_from_config(parser, cfg)

    args = parser.parse_args()
    train_tool_finetuning(
        dataset=args.dataset,
        category=args.category,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        tool_finetuning_lr=args.tool_finetuning_lr,
        tool_finetuning_epochs=args.tool_finetuning_epochs,
        tool_pretraining_lr=args.tool_pretraining_lr,
        tool_pretraining_epochs=args.tool_pretraining_epochs,
        tool_finetuning_inner_epochs=int(getattr(args, "tool_finetuning_inner_epochs", 1) or 1),
        gating=bool(getattr(args, "gating", False)),
        gating_network_ckpt_path=getattr(args, "gating_network_ckpt_path", None),
        train_all_alpha_experts=bool(getattr(args, "train_all_alpha_experts", False)),
        trainable_alpha_threshold=float(getattr(args, "trainable_alpha_threshold", 0.0) or 0.0),
        alpha_aug_stage1_prob=float(getattr(args, "alpha_aug_stage1_prob", 1.0) or 0.0),
        alpha_aug_label_smooth_prob=float(
            getattr(args, "alpha_aug_label_smooth_prob", 0.0) or 0.0
        ),
        alpha_aug_wrong_top1_prob=float(
            getattr(args, "alpha_aug_wrong_top1_prob", 0.0) or 0.0
        ),
        alpha_aug_label_gold_min=float(getattr(args, "alpha_aug_label_gold_min", 0.6)),
        alpha_aug_label_gold_max=float(getattr(args, "alpha_aug_label_gold_max", 0.7)),
        alpha_aug_wrong_gold_min=float(getattr(args, "alpha_aug_wrong_gold_min", 0.35)),
        alpha_aug_wrong_gold_max=float(getattr(args, "alpha_aug_wrong_gold_max", 0.5)),
        alpha_aug_wrong_top_min=float(getattr(args, "alpha_aug_wrong_top_min", 0.51)),
        alpha_aug_wrong_top_max=float(getattr(args, "alpha_aug_wrong_top_max", 0.65)),
        alpha_aug_seed=int(getattr(args, "alpha_aug_seed", 42) or 42),
        run_tag=getattr(args, "run_tag", None),
    )

__all__ = ["train_tool_finetuning", "TrainableGatedLoraLinear"]


if __name__ == "__main__":
    main()

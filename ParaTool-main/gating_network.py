import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.tool_keys import canonical_schema, stb_triplet
from root_dir_path import ROOT_DIR
from utils import _normalize_question_to_text


_LORA_PROFILE_ENABLED = False
_LORA_PROFILE_TOTAL = 0.0
_LORA_PROFILE_CALLS = 0


def enable_lora_profile(enabled: bool = True) -> None:
    global _LORA_PROFILE_ENABLED
    _LORA_PROFILE_ENABLED = enabled

def reset_lora_profile() -> None:
    global _LORA_PROFILE_TOTAL, _LORA_PROFILE_CALLS
    _LORA_PROFILE_TOTAL = 0.0
    _LORA_PROFILE_CALLS = 0

def get_lora_profile_stats() -> Dict[str, float]:
    return {
        "lora_matmul_time": float(_LORA_PROFILE_TOTAL),
        "lora_calls": float(_LORA_PROFILE_CALLS),
    }

def _normalize_module_name(name: str) -> str:
    # Adapter checkpoints may use different PEFT prefixes than named_modules().
    if name.startswith("base_model."):
        name = name[len("base_model.") :]
    if name.startswith("model.model."):
        name = name[len("model.") :]
    return name

def is_finish_tool_schema(schema: Any, *, finish_name: str = "Finish") -> bool:
    target = str(finish_name or "Finish").strip() or "Finish"

    if isinstance(schema, dict):
        for key in ("name", "api_name", "tool_name"):
            val = str(schema.get(key) or "").strip()
            if val and val == target:
                return True
        return False

    return str(schema or "").strip() == target

def map_gating_alphas_with_finish_to_experts(
    alphas_gating: torch.Tensor,
    *,
    tools_gating: List[Any],
    expert_indices: List[int],
    finish_name: str = "Finish",
    eps: float = 1e-8,
) -> "Tuple[torch.Tensor, bool, Optional[int]]":
    # Finish participates in routing but has no LoRA expert; selecting it disables deltas.
    if not isinstance(alphas_gating, torch.Tensor):
        raise TypeError("alphas_gating must be a torch.Tensor")
    if alphas_gating.dim() != 1:
        raise ValueError(f"alphas_gating must be 1D, got shape={tuple(alphas_gating.shape)}")

    finish_idx: Optional[int] = None
    for i, schema in enumerate(tools_gating or []):
        if is_finish_tool_schema(schema, finish_name=finish_name):
            finish_idx = int(i)
            break

    finish_selected = False
    if finish_idx is not None and 0 <= finish_idx < int(alphas_gating.numel()):
        top_idx = int(alphas_gating.argmax(dim=0).item())
        finish_selected = top_idx == finish_idx

    M = len(expert_indices)
    if M <= 0:
        out = alphas_gating.new_zeros((0,))
        return out, finish_selected, finish_idx

    if finish_selected:
        out = alphas_gating.new_zeros((M,))
        return out, True, finish_idx

    idx_tensor = torch.tensor(expert_indices, dtype=torch.long, device=alphas_gating.device)
    out = alphas_gating.index_select(0, idx_tensor)

    s = out.sum().clamp_min(eps)
    out = out / s
    return out, False, finish_idx

class GatingNetwork(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim

        # Score each candidate with question, tool, product, and distance features.
        in_dim = dim * 4
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        q_emb: torch.Tensor,
        tool_embs: torch.Tensor,
        tool_mask: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, M, d = tool_embs.shape
        assert q_emb.shape == (B, d), "q_emb shape must be (B, d)"
        assert tool_mask.shape[:2] == (B, M), "tool_mask shape must be (B, M)"

        q_rep = q_emb.unsqueeze(1).expand(-1, M, -1)

        prod = q_rep * tool_embs
        diff = torch.abs(q_rep - tool_embs)
        feat = torch.cat([q_rep, tool_embs, prod, diff], dim=-1)

        feat_flat = feat.view(B * M, -1)
        h = F.gelu(self.fc1(feat_flat))
        if self.dropout is not None:
            h = self.dropout(h)
        h = F.gelu(self.fc2(h))
        if self.dropout is not None:
            h = self.dropout(h)
        scores_flat = self.fc_out(h)
        scores = scores_flat.view(B, M)

        if tool_mask.dtype == torch.bool:
            mask = tool_mask
        else:
            mask = tool_mask > 0

        scores_masked = scores.masked_fill(~mask, float("-1e9"))
        weights = F.softmax(scores_masked, dim=1)

        if not return_scores:
            return weights, None

        return weights, scores

def _canonical_schema(schema: Dict) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False)

def build_schema_to_lora_map(
    category: str,
    lora_root_dir: str,
) -> Dict[str, List[str]]:
    if not os.path.isdir(lora_root_dir):
        raise FileNotFoundError(f"LoRA root dir not found: {lora_root_dir}")

    schema_to_paths: Dict[str, List[str]] = {}

    tool_index_path = os.path.join(
        ROOT_DIR,
        "berkeley-function-call-leaderboard",
        "bfcl_eval",
        "data",
        "tool_index",
        f"BFCL_v4_{category}_tool_index.json",
    )

    if not os.path.exists(tool_index_path):
        raise FileNotFoundError(
            f"Reverse index not found: {tool_index_path}\n"
            f"Please ensure the BFCL_v4_{category}_tool_index.json file exists."
        )

    with open(tool_index_path, "r", encoding="utf-8") as f:
        tool_index = json.load(f)
    tools_info = tool_index.get("tools") or {}

    for _name, info in tools_info.items():
        schema = info.get("schema")
        if not isinstance(schema, dict):
            continue
        schema_key = _canonical_schema(schema)

        occurrences = info.get("occurrences") or []
        if not isinstance(occurrences, list):
            continue

        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            ex_idx = occ.get("example_index")
            tool_idx = occ.get("tool_idx")
            if not isinstance(ex_idx, int) or not isinstance(tool_idx, int):
                continue

            base_dir = os.path.join(lora_root_dir, f"data_{ex_idx}")
            updated_dir = os.path.join(base_dir, f"tool_{tool_idx}_updated")
            orig_dir = os.path.join(base_dir, f"tool_{tool_idx}")

            candidate_paths = [
                os.path.join(updated_dir, "adapter_model.safetensors"),
                os.path.join(orig_dir, "adapter_model.safetensors"),
            ]
            adapter_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    adapter_path = p
                    break

            if adapter_path is None:
                continue

            schema_to_paths.setdefault(schema_key, []).append(adapter_path)

    return schema_to_paths

class GatedLoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear

        self.experts: List[Dict[str, torch.Tensor]] = []

        self.alphas: Optional[torch.Tensor] = None

        self._A_stack: Optional[torch.Tensor] = None
        self._B_stack: Optional[torch.Tensor] = None
        self._stack_device: Optional[torch.device] = None
        self._stack_dtype: Optional[torch.dtype] = None

    def set_experts(self, experts: List[Dict[str, torch.Tensor]]) -> None:
        self.experts = experts or []

        self._A_stack = None
        self._B_stack = None
        self._stack_device = None
        self._stack_dtype = None

    def set_alphas(self, alphas: Optional[torch.Tensor]) -> None:
        self.alphas = alphas

    def _build_stacked_lora(
        self,
        x_device: torch.device,
        x_dtype: torch.dtype,
    ) -> None:
        if not self.experts:
            self._A_stack = None
            self._B_stack = None
            self._stack_device = None
            self._stack_dtype = None
            return

        A_list: List[torch.Tensor] = []
        B_list: List[torch.Tensor] = []

        for exp in self.experts:
            A = exp["A"]
            B = exp["B"]
            scale = float(exp.get("scale", 1.0))

            if A.device != x_device or A.dtype != x_dtype:
                A = A.to(device=x_device, dtype=x_dtype)
                exp["A"] = A
            if B.device != x_device or B.dtype != x_dtype:
                B = B.to(device=x_device, dtype=x_dtype)
                exp["B"] = B

            if B.shape[0] != A.shape[1] and B.shape[1] == A.shape[1]:
                B = B.t()
                exp["B"] = B

            B_scaled = B * scale

            A_list.append(A)
            B_list.append(B_scaled)

        self._A_stack = torch.stack(A_list, dim=0)
        self._B_stack = torch.stack(B_list, dim=0)
        self._stack_device = x_device
        self._stack_dtype = x_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_linear(x)
        if not self.experts or self.alphas is None:
            return y

        x_device = x.device
        x_dtype = x.dtype

        alphas = self.alphas
        assert alphas is not None
        assert alphas.dim() == 1 and alphas.numel() == len(
            self.experts
        ), "alphas dim mismatch with experts"
        if int(torch.count_nonzero(alphas).item()) == 0:
            return y

        if (
            self._A_stack is None
            or self._B_stack is None
            or self._stack_device != x_device
            or self._stack_dtype != x_dtype
        ):
            self._build_stacked_lora(x_device, x_dtype)

        if self._A_stack is None or self._B_stack is None:
            return y

        A_stack = self._A_stack
        B_stack = self._B_stack

        alphas = alphas.to(device=A_stack.device, dtype=A_stack.dtype)
        self.alphas = alphas

        if _LORA_PROFILE_ENABLED and x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter() if _LORA_PROFILE_ENABLED else 0.0

        orig_shape = x.shape
        d_in = orig_shape[-1]
        x_flat = x.view(-1, d_in)

        # Stack experts so all LoRA deltas are computed by two batched contractions.
        XA = torch.einsum("nd,mdr->nmr", x_flat, A_stack)

        XAB = torch.einsum("nmr,mrd->nmd", XA, B_stack)

        alpha = alphas.view(1, -1, 1)
        delta_flat = (alpha * XAB).sum(dim=1)

        d_out = delta_flat.shape[-1]
        delta = delta_flat.view(*orig_shape[:-1], d_out).to(device=y.device, dtype=y.dtype)

        if _LORA_PROFILE_ENABLED:
            if x.is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            global _LORA_PROFILE_TOTAL, _LORA_PROFILE_CALLS
            _LORA_PROFILE_TOTAL += t1 - t0
            _LORA_PROFILE_CALLS += 1

        return y + delta

def _replace_with_gated_lora_linear(
    model: nn.Module,
    target_module_names: List[str],
) -> Dict[str, GatedLoraLinear]:
    name_to_module: Dict[str, GatedLoraLinear] = {}

    def _recursive_replace(parent: nn.Module, parent_name: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and any(
                t in full_name for t in target_module_names
            ):
                wrapped = GatedLoraLinear(child)
                setattr(parent, child_name, wrapped)
                name_to_module[full_name] = wrapped
            else:
                _recursive_replace(child, full_name)

    _recursive_replace(model, "")
    return name_to_module

def _load_lora_experts_for_paths(
    adapter_paths: List[str],
    target_modules: List[str],
    lora_alpha: int,
    r: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    try:
        import safetensors.torch as st  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "safetensors.torch is required to load LoRA experts; "
            "please install safetensors in your training environment."
        ) from e

    scale_val = float(lora_alpha) / float(r)
    experts_per_module: Dict[str, List[Dict[str, torch.Tensor]]] = {}

    for path in adapter_paths:
        if not os.path.exists(path):
            continue
        state = st.load_file(path)

        for key, tensor in state.items():
            if not key.endswith("lora_A.weight"):
                continue

            module_full_name = key[: -len(".lora_A.weight")]

            if not any(t in module_full_name for t in target_modules):
                continue
            key_B = module_full_name + ".lora_B.weight"
            if key_B not in state:
                continue

            A = state[key].to(device=device, dtype=dtype)
            B = state[key_B].to(device=device, dtype=dtype)
            exp = {
                "A": A.t(),
                "B": B.t(),

                "scale": scale_val,
            }
            experts_per_module.setdefault(module_full_name, []).append(exp)

    return experts_per_module

__all__ = [
    "GatingNetwork",
    "build_schema_to_lora_map",
    "is_finish_tool_schema",
    "map_gating_alphas_with_finish_to_experts",
    "GatedLoraLinear",
    "_replace_with_gated_lora_linear",
    "_load_lora_experts_for_paths",
    "_normalize_module_name",
    "_canonical_schema",
    "enable_lora_profile",
    "reset_lora_profile",
    "get_lora_profile_stats",
    "encode_question_embedding",
    "GatingQuestionEncoder",
    "make_gating_question_encoder",
    "normalize_gating_q_encoder_mode",
    "BGE_QENC_MODEL_PATH",
    "tool_vocab_key",
    "resolve_tool_ids",
]


def tool_vocab_key(schema: Any, *, tool_vocab_type: Optional[str]) -> str:
    if not isinstance(schema, dict):
        return str(schema)
    t = str(tool_vocab_type or "").strip().lower()
    if t == "stb_triplet":
        cat, tool, api = stb_triplet(schema)
        key_obj = {"category_name": cat, "tool_name": tool, "api_name": api}
        if any(key_obj.values()):
            return json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
    return canonical_schema(schema)

def resolve_tool_ids(
    tools: List[Dict[str, Any]],
    *,
    tool_vocab: Dict[str, int],
    tool_vocab_type: Optional[str],
    unk_tool_id: int,
) -> List[int]:
    tool_ids: List[int] = []
    for schema in tools:
        if not isinstance(schema, dict):
            tool_ids.append(unk_tool_id)
            continue
        key = tool_vocab_key(schema, tool_vocab_type=tool_vocab_type)
        tool_ids.append(tool_vocab.get(key, unk_tool_id))
    return tool_ids


BGE_QENC_MODEL_PATH = os.environ.get(
    "BGE_QENC_MODEL_PATH", "BAAI/bge-small-en-v1.5"
)

# (resolved_path, device_str) -> (model, tokenizer)
_bge_encoder_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def normalize_gating_q_encoder_mode(value: Any) -> str:
    m = str(value or "llm").strip().lower()
    if m == "bge":
        return "bge"
    return "llm"


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


class GatingQuestionEncoder:
    """Unified question embedding for gating only: ``encode(text) -> (1, D)``."""

    def __init__(
        self,
        mode: str,
        *,
        device: torch.device,
        base_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        bge_model_path: Optional[str] = None,
    ) -> None:
        self.mode = normalize_gating_q_encoder_mode(mode)
        self.device = device
        self._base_model = base_model
        self._tokenizer = tokenizer
        self._bge_path = (bge_model_path or BGE_QENC_MODEL_PATH).strip() or BGE_QENC_MODEL_PATH
        self._bge_model: Optional[nn.Module] = None
        self._bge_tokenizer: Optional[Any] = None
        self._embed_dim: Optional[int] = None

        if self.mode == "llm":
            if base_model is None or tokenizer is None:
                raise ValueError("llm gating question encoder requires base_model and tokenizer")
            cfg = getattr(base_model, "config", None)
            self._embed_dim = int(getattr(cfg, "hidden_size", 0))
        else:
            self._ensure_bge_loaded()

    def _ensure_bge_loaded(self) -> None:
        if self._bge_model is not None:
            return
        path = os.path.expanduser(os.path.expandvars(self._bge_path))
        key = (path, str(self.device))
        cached = _bge_encoder_cache.get(key)
        if cached is not None:
            self._bge_model, self._bge_tokenizer = cached
            cfg = getattr(self._bge_model, "config", None)
            self._embed_dim = int(getattr(cfg, "hidden_size", 0))
            return

        from transformers import AutoModel, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(path, trust_remote_code=True)
        mdl.eval()
        mdl.to(self.device)
        for p in mdl.parameters():
            p.requires_grad = False
        _bge_encoder_cache[key] = (mdl, tok)
        self._bge_model = mdl
        self._bge_tokenizer = tok
        cfg = getattr(mdl, "config", None)
        self._embed_dim = int(getattr(cfg, "hidden_size", 0))

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._ensure_bge_loaded()
        assert self._embed_dim is not None
        return int(self._embed_dim)

    def encode(self, q_text: str) -> torch.Tensor:
        if self.mode == "llm":
            assert self._base_model is not None and self._tokenizer is not None
            return encode_question_embedding(
                self._base_model, self._tokenizer, self.device, q_text
            )
        self._ensure_bge_loaded()
        assert self._bge_model is not None and self._bge_tokenizer is not None
        text = _normalize_question_to_text(q_text)
        inputs = self._bge_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._bge_model(**inputs)
            emb = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        return emb


def make_gating_question_encoder(
    mode: Union[str, Any],
    *,
    device: torch.device,
    base_model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    bge_model_path: Optional[str] = None,
) -> GatingQuestionEncoder:
    return GatingQuestionEncoder(
        str(mode),
        device=device,
        base_model=base_model,
        tokenizer=tokenizer,
        bge_model_path=bge_model_path,
    )


def encode_question_embedding(
    base_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    q_text: str,
) -> torch.Tensor:
    text = _normalize_question_to_text(q_text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = base_model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
    return hidden.mean(dim=1)

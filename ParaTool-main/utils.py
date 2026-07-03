from __future__ import annotations

import hashlib
import os
import json
import logging
import re
import shutil
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from root_dir_path import DATA_ROOT_DIR, ROOT_DIR, TOOL_FINETUNING_ROOT_PATH

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

def load_yaml_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise ModuleNotFoundError(
            "Missing dependency PyYAML. Install it via `pip install PyYAML` "
            "or add it to your environment before using `--config`."
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return obj

def resolve_path(path: str, *, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))

def _extract_openai_block(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    openai = cfg.get("openai")
    if isinstance(openai, dict):
        return dict(openai)
    return dict(cfg)

def apply_openai_env_from_mapping(
    openai_cfg: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    api_key = (
        openai_cfg.get("api_key")
        or openai_cfg.get("OPENAI_API_KEY")
        or openai_cfg.get("key")
    )
    api_base = (
        openai_cfg.get("api_base")
        or openai_cfg.get("base_url")
        or openai_cfg.get("OPENAI_API_BASE")
        or openai_cfg.get("OPENAI_BASE_URL")
    )

    if api_key and (overwrite or not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"))):
        os.environ["OPENAI_API_KEY"] = str(api_key)
        os.environ.setdefault("OPENAI_KEY", str(api_key))

    if api_base and (
        overwrite
        or not (os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    ):
        os.environ["OPENAI_API_BASE"] = str(api_base)
        os.environ.setdefault("OPENAI_BASE_URL", str(api_base))

    if "model" in openai_cfg and openai_cfg.get("model") and (overwrite or not os.environ.get("OPENAI_MODEL")):
        os.environ["OPENAI_MODEL"] = str(openai_cfg.get("model"))
        os.environ.setdefault("EVAL_MODEL", str(openai_cfg.get("model")))

    if "temperature" in openai_cfg and openai_cfg.get("temperature") is not None and (overwrite or not os.environ.get("OPENAI_TEMPERATURE")):
        os.environ["OPENAI_TEMPERATURE"] = str(openai_cfg.get("temperature"))

def apply_env_from_config(
    cfg: Mapping[str, Any],
    *,
    config_path: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    base_dir = os.getcwd()
    if config_path:
        base_dir = os.path.dirname(os.path.abspath(config_path))

    env_cfg = cfg.get("env")
    if isinstance(env_cfg, dict):
        for k, v in env_cfg.items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            if overwrite or k not in os.environ:
                os.environ[str(k)] = str(v)

    openai_config_path = cfg.get("openai_config") or cfg.get("secrets_file")
    if isinstance(openai_config_path, str) and openai_config_path.strip():
        resolved = resolve_path(openai_config_path.strip(), base_dir=base_dir)
        if os.path.exists(resolved):
            openai_cfg = load_yaml_file(resolved)
            apply_openai_env_from_mapping(_extract_openai_block(openai_cfg), overwrite=overwrite)

    openai_inline = cfg.get("openai")
    if isinstance(openai_inline, dict):
        apply_openai_env_from_mapping(openai_inline, overwrite=overwrite)

    eval_model = cfg.get("eval_model")
    if isinstance(eval_model, str) and eval_model.strip():
        if overwrite or "EVAL_MODEL" not in os.environ:
            os.environ["EVAL_MODEL"] = eval_model.strip()

def _is_blank_config_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False

def set_parser_defaults_from_config(parser: Any, cfg: Mapping[str, Any]) -> None:
    arg_names = set()
    for action in getattr(parser, "_actions", []):
        dest = getattr(action, "dest", None)
        if dest and dest != "help":
            arg_names.add(dest)

    defaults: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in arg_names:
            if _is_blank_config_value(v):
                continue
            defaults[k] = v
    if defaults:
        parser.set_defaults(**defaults)

_MODEL_SHORTCUTS = {
    "llama3.1-8b-instruct": ("LLAMA31_8B_INSTRUCT_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5-7b-instruct": ("QWEN25_7B_INSTRUCT_PATH", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen2.5-14b-instruct": ("QWEN25_14B_INSTRUCT_PATH", "Qwen/Qwen2.5-14B-Instruct"),
}


def get_model_path(model_name: str) -> str:
    shortcut = _MODEL_SHORTCUTS.get(model_name.lower())
    if shortcut is None:
        return os.path.expanduser(os.path.expandvars(model_name))

    env_key, hf_model_id = shortcut
    override = os.environ.get(env_key)
    if override:
        return os.path.expanduser(os.path.expandvars(override))
    return hf_model_id

def get_model(
    model_name: str,
    max_new_tokens: int = 20,
    device_id: Optional[int] = None,
):
    model_path = get_model_path(model_name)

    if device_id is not None:
        device_map = {"": f"cuda:{device_id}"}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_gen = GenerationConfig.from_model_config(model.config)
    base_gen.num_beams = 1
    base_gen.do_sample = True
    base_gen.temperature = 0.7
    base_gen.max_new_tokens = max_new_tokens
    base_gen.return_dict_in_generate = True
    base_gen.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, base_gen

def load_data(data_name, category, model_name=None):
    alias = category.lower().replace("-", "_").replace(" ", "_")
    group_map = {
        "non_live": [
            "parallel",
            "multiple",
            "parallel_multiple",
            "irrelevance",
        ],
        "live": [
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "live_irrelevance",
            "live_relevance",
        ],
    }

    from dataset import BFCLDataset

    bfcl_ds = BFCLDataset()

    def _load_single_bfcl(concrete_cat: str):
        dataset = bfcl_ds.load_data(concrete_cat)
        return [(f"{concrete_cat}.jsonl", dataset)]

    def _load_single_aug(concrete_cat: str):
        fname = f"{concrete_cat}.json"
        base_dir = os.path.join(DATA_ROOT_DIR, data_name)
        aug_path = os.path.join(base_dir, "python", model_name, fname)
        if not os.path.exists(aug_path):
            raise FileNotFoundError(
                f"Augment file not found: {aug_path}. "
                "Please generate it via the corresponding scripts under data_generation/."
            )
        with open(aug_path, "r", encoding="utf-8") as f:
            return [(fname, json.load(f))]

    loader = _load_single_bfcl if data_name.lower() == "bfcl" else _load_single_aug

    if alias in group_map:
        results = []
        for concrete_cat in group_map[alias]:
            results.extend(loader(concrete_cat))
        return results

    return loader(category)

def _normalize_question_to_text(question):
    if isinstance(question, str):
        return question

    try:
        if isinstance(question, list) and len(question) > 0:
            if all(isinstance(x, list) for x in question):
                flat = []
                for sub in question:
                    flat.extend(sub)
                question = flat

            parts = []
            for msg in question:
                if isinstance(msg, dict) and "content" in msg:
                    parts.append(str(msg["content"]))
                else:
                    parts.append(str(msg))
            return "\n".join(parts)
    except Exception:
        pass

    return str(question)

def _ast_get_full_name(node):
    import ast

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _ast_get_full_name(node.value)
        if left is None:
            return None
        return f"{left}.{node.attr}"
    return None

def _ast_literal_eval_safe(node):
    import ast

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.NameConstant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_literal_eval_safe(node.operand)
        if isinstance(val, (int, float)):
            return -val
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_literal_eval_safe(elt) for elt in node.elts]
    if isinstance(node, ast.Dict):
        keys = [_ast_literal_eval_safe(k) for k in node.keys]
        vals = [_ast_literal_eval_safe(v) for v in node.values]
        return {k: v for k, v in zip(keys, vals)}

    try:
        import ast as _ast

        if hasattr(_ast, "unparse"):
            return _ast.unparse(node)
    except Exception:
        pass
    return None

def parse_python_tool_calls(text):
    import ast

    if not isinstance(text, str):
        raise ValueError("Input must be string")

    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")

        s = s.split("\n", 1)[-1]

    def _wrap_java_class_literals(expr: str) -> str:
        import re as _re

        parts = _re.split(r'(".*?"|\'.*?\')', expr)
        for i, part in enumerate(parts):
            if i % 2 == 0 and part:
                parts[i] = _re.sub(
                    r"\b([\w$.]+\.class)\b", r'"\\1"', part
                )
        return "".join(parts)

    def _fix_keyword_arg_names(expr: str) -> str:
        try:
            import io as _io
            import tokenize as _tokenize
            import keyword as _keyword

            tokens = list(
                _tokenize.generate_tokens(_io.StringIO(expr).readline)
            )
            out_tokens = []
            n = len(tokens)
            i = 0
            while i < n:
                tok = tokens[i]

                if tok.type == _tokenize.NAME and _keyword.iskeyword(tok.string):
                    j = i + 1
                    while j < n and tokens[j].type in (
                        _tokenize.NL,
                        _tokenize.NEWLINE,
                        _tokenize.INDENT,
                        _tokenize.DEDENT,
                    ):
                        j += 1
                    if (
                        j < n
                        and tokens[j].type == _tokenize.OP
                        and tokens[j].string == "="
                    ):
                        tok = _tokenize.TokenInfo(
                            tok.type,
                            tok.string + "_",
                            tok.start,
                            tok.end,
                            tok.line,
                        )
                out_tokens.append(tok)
                i += 1
            return _tokenize.untokenize(out_tokens)
        except Exception:
            return expr

    def _try_parse(expr: str):
        try:
            node = ast.parse(expr, mode="eval").body
        except Exception:
            fixed = _fix_keyword_arg_names(expr)
            if fixed != expr:
                try:
                    node = ast.parse(fixed, mode="eval").body
                except Exception:
                    return None
            else:
                return None

        if isinstance(node, ast.List):
            seq = node.elts
        elif isinstance(node, ast.Tuple):
            seq = node.elts
        elif isinstance(node, ast.Call):
            seq = [node]
        else:
            return None

        calls = []
        for call in seq:
            if isinstance(call, ast.Call):
                calls.append(call)
        return calls or None

    s = _wrap_java_class_literals(s)

    seq = _try_parse(s)

    if seq is None and "[" in s and "]" in s:
        start = s.find("[")
        end = s.rfind("]")
        if start < end:
            sliced = s[start : end + 1]
            seq = _try_parse(sliced)

    if seq is None:
        wrapped = f"[{s}]"
        seq = _try_parse(wrapped)
        if seq is None and "[" in s and "]" in s:
            start = s.find("[")
            end = s.rfind("]")
            if start < end:
                wrapped_sliced = f"[{s[start : end + 1]}]"
                seq = _try_parse(wrapped_sliced)

    if seq is None:
        raise ValueError("Failed to parse python calls")

    calls = []
    for call in seq:
        if not isinstance(call, ast.Call):
            continue
        fn = _ast_get_full_name(call.func)
        if not fn:
            continue
        params = {}

        if getattr(call, "args", None):
            pos_vals = []
            for arg_node in call.args:
                try:
                    pos_vals.append(_ast_literal_eval_safe(arg_node))
                except Exception:
                    try:
                        import ast as _ast

                        if hasattr(_ast, "unparse"):
                            pos_vals.append(_ast.unparse(arg_node))
                        else:
                            pos_vals.append(None)
                    except Exception:
                        pos_vals.append(None)
            if pos_vals:
                params["__positional_args__"] = pos_vals
        for kw in call.keywords:
            if kw.arg is None:
                continue
            params[kw.arg] = _ast_literal_eval_safe(kw.value)
        calls.append({fn: params})

    if not calls:
        raise ValueError("No valid function calls extracted")
    return calls

def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)

def _safe_filename_component(value: Any, *, max_len: int = 80) -> str:
    s = str(value) if value is not None else ""
    s = s.strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        return "unknown"
    return s[:max_len]

def gating_checkpoint_filename(
    *,
    model_name: str,
    category: str,
    entropy_reg_weight: float = 0.2,
    q_encoder: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> str:
    reg_str = f"{float(entropy_reg_weight):.6g}"
    stem = f"gating_{model_name}_cat-{category}_entreg-{reg_str}"
    if str(q_encoder or "llm").strip().lower() == "bge":
        stem += "_qenc-bge"
    tag = str(run_tag or "").strip()
    if tag:
        stem += f"_run-{_safe_filename_component(tag)}"
    return f"{stem}.pt"

def _extract_tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return "unknown"
    for key in ("api_name", "name", "tool_name"):
        val = tool.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown"

def _extract_tool_names(tools: Any) -> List[str]:
    if not isinstance(tools, list):
        return []
    names: List[str] = []
    seen = set()
    for t in tools:
        name = _extract_tool_name(t)
        if name == "unknown" or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names

def _create_file_logger(
    log_path: str, *, level: int = logging.INFO
) -> Tuple[logging.Logger, logging.Handler]:
    digest = hashlib.md5(log_path.encode("utf-8")).hexdigest()[:16]
    logger = logging.getLogger(f"tool_pretraining.train.{digest}")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler

def compute_qa_hash(qa_plus: list) -> str:
    return hashlib.md5(json.dumps(qa_plus, sort_keys=True).encode()).hexdigest()[:16]

def get_tool_pretraining_dedup_key(
    tool_schema: dict, qa_plus: list
) -> Tuple[str, str, str, str]:
    return (
        tool_schema.get("category_name", ""),
        tool_schema.get("tool_name", ""),
        tool_schema.get("api_name", ""),
        compute_qa_hash(qa_plus),
    )

STB_DATASET_NAMES = frozenset({"stb", "stabletoolbench", "toolbench"})


def normalize_dataset_name(dataset: str) -> str:
    ds = str(dataset or "").strip().lower()
    if ds in STB_DATASET_NAMES:
        return "stb"
    return ds or "bfcl"

def is_bfcl_multistep_category(dataset: str, category: str) -> bool:
    return normalize_dataset_name(dataset) == "bfcl" and str(category or "").strip() in {
        "parallel_multiple",
        "live_parallel_multiple",
    }

def train_filename(category: str) -> str:
    cat = category_slug(category)
    return f"{cat}.jsonl"

def category_slug(category: str) -> str:
    cat = str(category or "").strip()
    if not cat:
        raise ValueError("Category must be a non-empty string.")
    if cat.endswith(".json"):
        raise ValueError(
            f"JSON array category names are no longer supported: {cat!r}. "
            "Use a bare category name or .jsonl."
        )
    if cat.endswith(".jsonl"):
        cat = cat[: -len(".jsonl")]
    if not cat:
        raise ValueError("Category must not resolve to an empty name.")
    return cat

def _run_tag_path_segment(run_tag: Optional[str]) -> List[str]:
    tag = str(run_tag or "").strip()
    if not tag:
        return []
    return [f"run={_safe_filename_component(tag)}"]

def build_tool_pretraining_adapter_root(
    *,
    root_dir: str,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    dataset: str,
    tool_pretraining_lr: float,
    tool_pretraining_epochs: int,
    category: str,
    run_tag: Optional[str] = None,
) -> str:
    ds = str(dataset or "").strip().lower()
    cat = category_slug(category)

    return os.path.join(
        root_dir,
        ds,
        cat,
        model_name,
        f"rank={lora_rank}_alpha={lora_alpha}",
        f"lr={tool_pretraining_lr}_epoch={tool_pretraining_epochs}",
        *_run_tag_path_segment(run_tag),
        "adapter",
    )

def build_tool_finetuning_root_dir(
    *,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    dataset: str,
    tool_pretraining_lr: float,
    tool_pretraining_epochs: int,
    category: str,
    tool_finetuning_lr: float,
    tool_finetuning_epochs: int,
    run_tag: Optional[str] = None,
) -> str:
    filename = train_filename(category)
    return os.path.join(
        TOOL_FINETUNING_ROOT_PATH,
        model_name,
        f"rank={lora_rank}_alpha={lora_alpha}",
        dataset,
        f"lr={tool_pretraining_lr}_epoch={tool_pretraining_epochs}",
        f"tool_finetuning_lr={tool_finetuning_lr}_epoch={tool_finetuning_epochs}",
        *_run_tag_path_segment(run_tag),
        filename,
    )

def adapter_file(adapter_dir: str) -> str:
    return os.path.join(adapter_dir, "adapter_model.safetensors")

def adapter_exists(adapter_dir: str) -> bool:
    return os.path.exists(adapter_file(adapter_dir))

def copy_adapter_dir(src_dir: str, dst_dir: str) -> None:
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(src_dir, dst_dir)

def tool_finetuning_log_path(*, out_dir: str) -> str:
    log_root = os.path.join(ROOT_DIR, "logs", "tool_finetuning")
    rel_out = None
    try:
        rel_out = os.path.relpath(out_dir, start=TOOL_FINETUNING_ROOT_PATH)
        if rel_out.startswith(".."):
            rel_out = None
    except Exception:
        rel_out = None

    if rel_out:
        log_dir = os.path.join(log_root, rel_out)
    else:
        safe_leaf = os.path.basename(os.path.normpath(out_dir)) or "unknown"
        log_dir = os.path.join(log_root, "adapters", safe_leaf)
    return os.path.join(log_dir, "train.log")

def tool_desc(schema: Dict[str, Any], *, dataset: str) -> str:
    if not isinstance(schema, dict):
        return str(schema)
    ds = str(dataset or "").strip().lower()
    if ds in STB_DATASET_NAMES:
        from dataset.tool_keys import stb_triplet
        cat, tool, api = stb_triplet(schema)
        if cat or tool or api:
            return f"{cat}/{tool}/{api}"
    name = str(schema.get("name") or "").strip()
    return name or json.dumps(schema, ensure_ascii=False)

def log_json(logger: logging.Logger, payload: Dict[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info(str(payload))

def save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def load_json_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

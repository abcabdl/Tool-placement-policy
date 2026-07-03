import torch
import os
import json
import gc
import shutil
import logging
import math
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
from transformers import DefaultDataCollator
import argparse
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import time
from utils import (
    _create_file_logger,
    _extract_tool_name,
    _extract_tool_names,
    _json_dumps_safe,
    _normalize_question_to_text,
    build_tool_pretraining_adapter_root,
    get_model,
    get_tool_pretraining_dedup_key,
)
from dataset.data_io import iter_training_blocks
from dataset.tool_pretraining_dataset import (
    build_tool_pretraining_prompt_data,
    load_tool_pretraining_data,
)
from tqdm import tqdm
from root_dir_path import TOOL_PRETRAINING_ROOT_PATH, ROOT_DIR


class ToolPretrainingDataset(Dataset):
    ignored_id = -100
    min_answer_tokens = 5

    def __init__(
        self,
        prompt_data,
        tokenizer,
        max_length=3000,
    ):
        self.max_length = max_length
        self.dataset = []
        self.seq_lens: List[int] = []

        skipped_short_answer = 0
        skipped_truncated_answer = 0

        for idx, (input_ids, prompt_len) in enumerate(prompt_data):
            labels = input_ids.copy()

            for i in range(min(prompt_len, len(labels))):
                labels[i] = self.ignored_id

            answer_token_count = len(input_ids) - prompt_len

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

                if prompt_len < max_length:
                    answer_token_count = max_length - prompt_len
                else:
                    answer_token_count = 0

            if answer_token_count < self.min_answer_tokens:
                if len(input_ids) >= max_length:
                    skipped_truncated_answer += 1
                else:
                    skipped_short_answer += 1
                continue

            attention_mask = [1] * len(input_ids)

            self.dataset.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                    "answer_token_count": answer_token_count,
                }
            )
            self.seq_lens.append(len(input_ids))

        self.total_len = len(self.dataset)

        if skipped_short_answer > 0 or skipped_truncated_answer > 0:
            print(
                f"[ToolPretrainingDataset] 过滤了 {skipped_short_answer + skipped_truncated_answer} 个样本："
                f" {skipped_short_answer} 个 answer 过短, "
                f"{skipped_truncated_answer} 个被 max_length 截断"
            )

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]

class ToolPretrainingCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device, *, pad_to_multiple_of: Optional[int] = 8):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.pad_to_multiple_of = (
            int(pad_to_multiple_of)
            if pad_to_multiple_of is not None and int(pad_to_multiple_of) > 0
            else None
        )
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]
        answer_token_count = [
            example.get("answer_token_count", 0) for example in examples
        ]

        max_len = max((len(ids) for ids in input_ids), default=0)
        if self.pad_to_multiple_of and max_len:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        padded_input_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        padded_attention_mask: List[List[int]] = []
        for ids, labs, mask in zip(input_ids, labels, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labs + [ToolPretrainingDataset.ignored_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids).to(self.device),
            "labels": torch.tensor(padded_labels).to(self.device),
            "attention_mask": torch.tensor(padded_attention_mask).to(self.device),
            "answer_token_count": torch.tensor(answer_token_count).to(self.device),
        }

def train_tool_adapter(
    question,
    augments,
    args,
    model,
    tokenizer,
    init_adapter_path,
    save_path,
    tools,
    *,
    log_path: Optional[str] = None,
    log_meta: Optional[Dict[str, Any]] = None,
):
    logger: Optional[logging.Logger] = None
    log_handler: Optional[logging.Handler] = None
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        level_str = str(getattr(args, "log_level", "INFO") or "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logger, log_handler = _create_file_logger(log_path, level=level)

    def _log_info(msg: str, *vals: Any) -> None:
        if logger is None:
            return
        try:
            logger.info(msg, *vals)
        except Exception:
            pass

    def _log_exception(msg: str, *vals: Any) -> None:
        if logger is None:
            return
        try:
            logger.exception(msg, *vals)
        except Exception:
            pass

    question_text = _normalize_question_to_text(question)
    target_tool_schema = {}
    try:
        if isinstance(augments, list) and augments and isinstance(augments[0], dict):
            target_tool_schema = augments[0].get("tool_schema") or {}
    except Exception:
        target_tool_schema = {}

    _log_info("[train_start] save_path=%s", save_path)
    _log_info("[train_start] init_adapter_path=%s", init_adapter_path)
    _log_info("[train_start] question=%s", question_text)
    _log_info("[train_start] target_tool=%s", _extract_tool_name(target_tool_schema))
    _log_info(
        "[train_start] target_tool_schema=%s", _json_dumps_safe(target_tool_schema)
    )
    _log_info(
        "[train_start] available_tools(count=%s) names=%s",
        len(tools) if isinstance(tools, list) else 0,
        _extract_tool_names(tools),
    )
    _log_info("[train_start] available_tools_json=%s", _json_dumps_safe(tools))
    if log_meta:
        _log_info("[train_start] meta=%s", _json_dumps_safe(log_meta))
    try:
        _log_info("[train_start] args=%s", _json_dumps_safe(vars(args)))
    except Exception:
        _log_info("[train_start] args=%s", str(args))

    try:
        prompt_data = build_tool_pretraining_prompt_data(
            augments,
            tokenizer,
            tools,
            dataset=args.dataset,
            category=args.category,
        )

        if not prompt_data:
            msg = (
                "[skip] no usable training_data for current function; "
                f"question={question_text!r}, "
                f"save_path={save_path}"
            )
            print(msg)
            _log_info(msg)
            return model

        train_data = ToolPretrainingDataset(
            prompt_data,
            tokenizer,
            max_length=int(getattr(args, "max_seq_length", 8000) or 8000),
        )
        _log_info(
            "[train_data] raw_prompts=%d filtered_samples=%d",
            len(prompt_data),
            len(train_data),
        )
        try:
            if getattr(train_data, "seq_lens", None):
                lens = sorted(train_data.seq_lens)
                p50 = lens[len(lens) // 2]
                p90 = lens[int(len(lens) * 0.9)]
                p99 = lens[int(len(lens) * 0.99)]
                _log_info(
                    "[train_data] seq_len p50=%d p90=%d p99=%d max=%d",
                    int(p50),
                    int(p90),
                    int(p99),
                    int(lens[-1]),
                )
        except Exception:
            pass
        if len(train_data) == 0:
            msg = (
                "[skip] no usable training samples after filtering (answer-only CE); "
                f"question={question_text!r}, "
                f"save_path={save_path}"
            )
            print(msg)
            _log_info(msg)
            return model

        pad_to_multiple_of = getattr(args, "pad_to_multiple_of", 8)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.per_device_train_batch_size,
            collate_fn=ToolPretrainingCollator(
                tokenizer, model.device, pad_to_multiple_of=pad_to_multiple_of
            ),
            shuffle=True,
        )

        if isinstance(model, PeftModel):
            model = model.unload()
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
        model.is_parallelizable = True
        model.model_parallel = True
        model.train()
        try:
            model.config.use_cache = False
        except Exception:
            pass

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        try:
            optimizer = torch.optim.AdamW(
                model_parameters,
                lr=args.learning_rate,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

        def _compute_answer_only_ce_loss(batch_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            input_ids = batch_dict.get("input_ids")
            labels = batch_dict.get("labels")
            attention_mask = batch_dict.get("attention_mask")
            if input_ids is None or labels is None:
                out = model(**batch_dict)
                return out.loss

            # Avoid materializing full vocab logits for prefix tokens that are masked out.
            causal_lm = model
            try:
                base = getattr(model, "base_model", None)
                if base is not None and hasattr(base, "model"):
                    causal_lm = base.model
                else:
                    causal_lm = model.get_base_model()  # type: ignore[attr-defined]
            except Exception:
                causal_lm = model

            transformer = getattr(causal_lm, "model", None) or getattr(
                causal_lm, "transformer", None
            )
            lm_head = getattr(causal_lm, "lm_head", None)
            if transformer is None or lm_head is None:
                out = model(**batch_dict)
                return out.loss

            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs[0]
            if hidden_states.size(1) < 2:
                out = model(**batch_dict)
                return out.loss

            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:]
            flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
            flat_labels = shift_labels.reshape(-1)

            hidden_device = flat_hidden.device
            if flat_labels.device != hidden_device:
                flat_labels = flat_labels.to(hidden_device)

            mask = flat_labels != ToolPretrainingDataset.ignored_id
            if not bool(mask.any().item()):
                out = model(**batch_dict)
                return out.loss

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

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                answer_token_counts = batch.pop("answer_token_count", None)
                optimizer.zero_grad(set_to_none=True)
                loss = _compute_answer_only_ce_loss(batch)
                loss.backward()

                avg_answer_tokens = 0.0
                if answer_token_counts is not None:
                    avg_answer_tokens = answer_token_counts.float().mean().item()

                file_every = int(getattr(args, "file_log_every_n_steps", 1) or 0)
                need_file_log = file_every > 0 and (
                    step == 0 or (step + 1) % file_every == 0
                )

                console_every = int(getattr(args, "console_log_every_n_steps", 10) or 0)
                need_console_log = console_every > 0 and (
                    step == 0 or (step + 1) % console_every == 0
                )

                total_norm = 0.0
                if need_file_log or need_console_log:
                    # Gradients can live on different GPUs under model parallelism.
                    total_norm_sq_by_device: Dict[torch.device, torch.Tensor] = {}
                    for p in trainable_params:
                        if p.grad is None:
                            continue
                        grad = p.grad.detach()
                        dev = grad.device
                        acc = total_norm_sq_by_device.get(dev)
                        if acc is None:
                            acc = torch.zeros((), device=dev, dtype=torch.float32)
                            total_norm_sq_by_device[dev] = acc
                        acc += grad.float().pow(2).sum()

                    total_norm_sq_cpu = 0.0
                    for acc in total_norm_sq_by_device.values():
                        total_norm_sq_cpu += float(acc.item())
                    total_norm = math.sqrt(total_norm_sq_cpu)

                if need_file_log:
                    _log_info(
                        "epoch=%d step=%d loss=%.6f grad_norm=%.6f avg_answer_tokens=%.2f",
                        epoch + 1,
                        step + 1,
                        float(loss.item()),
                        float(total_norm),
                        float(avg_answer_tokens),
                    )

                if need_console_log:
                    print(
                        f"Epoch {epoch+1}, Step {step+1}, "
                        f"loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}, "
                        f"Avg Answer Tokens: {avg_answer_tokens:.1f}"
                    )

                optimizer.step()

        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)

        model = model.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _log_info("[train_end] saved_adapter=%s", save_path)
        return model
    except Exception as e:
        _log_exception("[train_error] %s", e)
        raise
    finally:
        if log_handler is not None and logger is not None:
            logger.removeHandler(log_handler)
            log_handler.close()

def main(args):
    data_list = load_tool_pretraining_data(
        args.dataset,
        args.category,
        getattr(args, "tool_pretraining_data_path", None),
    )

    model, tokenizer, _ = get_model(args.model_name)
    init_adapter_path = os.path.join(
        TOOL_PRETRAINING_ROOT_PATH,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        "base_weight",
    )

    desired_targets = sorted(["down_proj", "gate_proj", "up_proj"])
    need_create = True
    adapter_ckpt = os.path.join(init_adapter_path, "adapter_model.safetensors")
    cfg_path = os.path.join(init_adapter_path, "adapter_config.json")
    if os.path.exists(adapter_ckpt) and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cur_targets = sorted(cfg.get("target_modules", []) or [])
            if cur_targets == desired_targets:
                need_create = False
            else:
                print(
                    f"[warn] base_weight target_modules {cur_targets} != desired {desired_targets}; rebuilding base weight"
                )
        except Exception as e:
            print(
                f"[warn] failed to read existing base adapter config: {e}; recreating base weight..."
            )
    if need_create:
        print("Create LoRA base weight (FFN targets)...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=desired_targets,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(1)
        assert os.path.exists(adapter_ckpt)

        model = model.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    for filename, full_dataset in data_list:
        output_dir = build_tool_pretraining_adapter_root(
            root_dir=TOOL_PRETRAINING_ROOT_PATH,
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            dataset=args.dataset,
            tool_pretraining_lr=args.learning_rate,
            tool_pretraining_epochs=args.num_train_epochs,
            category=args.category,
            run_tag=getattr(args, "run_tag", None),
        )
        os.makedirs(output_dir, exist_ok=True)
        skipped = 0

        trained_keys: Dict[Tuple[str, str, str, str], str] = {}

        for scan_block in iter_training_blocks(full_dataset):
            if not scan_block.training_data:
                continue
            scan_path = os.path.join(
                output_dir, f"data_{scan_block.example_index}", f"tool_{scan_block.tid}"
            )
            adapter_weights_path = os.path.join(
                scan_path, "adapter_model.safetensors"
            )
            if os.path.exists(adapter_weights_path):
                tool_schema = scan_block.tool_schema
                qa_plus = scan_block.training_data
                key = get_tool_pretraining_dedup_key(tool_schema, qa_plus)
                trained_keys[key] = scan_path

        print(
            f"[resume] 已找到 {len(trained_keys)} 个已训练的 adapter，可用于断点续跑和去重复制"
        )

        for block in tqdm(list(iter_training_blocks(full_dataset)), desc="训练数据"):
            idx = block.example_index
            tid = block.tid
            save_path = os.path.join(output_dir, f"data_{idx}", f"tool_{tid}")
            adapter_weights_path = os.path.join(
                save_path, "adapter_model.safetensors"
            )

            tool_schema = block.tool_schema
            qa_plus = block.training_data
            if not isinstance(qa_plus, list) or len(qa_plus) == 0:
                skipped += 1
                print(
                    f"[skip] empty training_data for {filename}:data_{idx}/tool_{tid}; "
                    f"save_path={save_path}"
                )
                continue
            dedup_key = get_tool_pretraining_dedup_key(tool_schema, qa_plus)

            if os.path.exists(save_path) and os.path.exists(adapter_weights_path):
                print(f"[skip] 已存在 {save_path}")
                continue

            if dedup_key in trained_keys:
                src_path = trained_keys[dedup_key]
                src_adapter_weights_path = os.path.join(
                    src_path, "adapter_model.safetensors"
                )

                if not os.path.exists(src_adapter_weights_path):
                    print(
                        f"[warn] stale dedup key points to missing adapter: key={dedup_key} src={src_path}. "
                        "Will retrain/skip this sample instead of copying."
                    )
                    try:
                        del trained_keys[dedup_key]
                    except Exception:
                        pass
                else:
                    triplet_info = (
                        f"({dedup_key[0]}, {dedup_key[1]}, {dedup_key[2]})"
                    )
                    print(
                        f"[copy] {triplet_info} 已训练于 {src_path}，复制到 {save_path}"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    try:
                        shutil.copytree(src_path, save_path)
                        continue
                    except FileNotFoundError as e:
                        print(
                            f"[warn] copytree failed, will retrain: key={dedup_key} "
                            f"src={src_path} dst={save_path} err={e}"
                        )
                        try:
                            del trained_keys[dedup_key]
                        except Exception:
                            pass

            full_tools = block.function_tools
            log_root = os.path.join(ROOT_DIR, "logs")
            rel_save_path = os.path.relpath(save_path, start=TOOL_PRETRAINING_ROOT_PATH)
            if rel_save_path.startswith(".."):
                adapter_log_dir = os.path.join(log_root, "tool_pretraining", "_unmapped")
            else:
                adapter_log_dir = os.path.join(log_root, "tool_pretraining", rel_save_path)
            adapter_log_path = os.path.join(adapter_log_dir, "train.log")

            api_name = _extract_tool_name(tool_schema)
            tqdm.write(
                f"[train] data_{idx}/tool_{tid} api={api_name} log={adapter_log_path}"
            )
            base_aug = {"tool_schema": tool_schema, "training_data": qa_plus}

            model = train_tool_adapter(
                None,
                [base_aug],
                args,
                model,
                tokenizer,
                init_adapter_path,
                save_path,
                tools=full_tools,
                log_path=adapter_log_path,
                log_meta={
                    "filename": filename,
                    "data_idx": idx,
                    "tool_idx": tid,
                    "dedup_key": dedup_key,
                    "api_name": api_name,
                    "save_path": save_path,
                    "source_id": block.source_id,
                },
            )

            if os.path.exists(adapter_weights_path):
                trained_keys[dedup_key] = save_path
            else:
                print(
                    f"[warn] adapter file missing after train; "
                    f"will not cache dedup_key for save_path={save_path}"
                )

        if skipped:
            print(f"Skipped {skipped} functions without training_data in {filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (keys match CLI args). CLI args override YAML.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="模型名称",
        default="Qwen2.5-7b-instruct",
    )
    parser.add_argument("--dataset", type=str, help="数据集名称", default="stb")
    parser.add_argument(
        "--category",
        type=str,
        help="数据集类别",
        default="G1_category",
    )
    parser.add_argument(
        "--tool_pretraining_data_path",
        type=str,
        default=None,
        help=(
            "Optional tool pretraining data JSONL override. Keeps --category semantics for prompts, "
            "official debug, and output paths while loading data from this file."
        ),
    )
    parser.add_argument("--run_tag", type=str, default=None)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="每设备批量大小",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8000,
        help="最大序列长度（仅用于截断；实际 padding 为 batch 内动态长度）。",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=8,
        help="动态 padding 时向上补齐到该倍数；设为 0/负数可关闭。",
    )
    parser.add_argument(
        "--file_log_every_n_steps",
        type=int,
        default=1,
        help="写入 train.log 的步频；设为 0 可关闭文件 step 日志。",
    )
    parser.add_argument(
        "--console_log_every_n_steps",
        type=int,
        default=10,
        help="控制台打印 loss 的步频；设为 0 可关闭控制台 step 日志。",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="日志级别（写入 train.log 时使用）：DEBUG/INFO/WARNING/ERROR",
    )

    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha参数")
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
    print("参数配置：", args)
    main(args)

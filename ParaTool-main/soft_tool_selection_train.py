import os
import json
import random
import re
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from gating_network import (
    GatingNetwork,
    make_gating_question_encoder,
    normalize_gating_q_encoder_mode,
    resolve_tool_ids,
    tool_vocab_key,
)
from root_dir_path import ROOT_DIR, SOFT_TOOL_SELECTION_ROOT_PATH
from dataset.soft_tool_selection_dataset import load_soft_tool_selection_samples
from utils import gating_checkpoint_filename, get_model


def _safe_log_component(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return "unknown"
    return re.sub(r"[^a-zA-Z0-9._=-]+", "_", s)

def _qenc_path_segment(q_encoder: str) -> str:
    return "qenc=bge" if normalize_gating_q_encoder_mode(q_encoder) == "bge" else ""


def _soft_tool_selection_train_log_path(
    *,
    dataset: str,
    model_name: str,
    category: str,
    learning_rate: float,
    num_epochs: int,
    hidden_dim: int,
    entropy_reg_weight: float,
    q_encoder: str = "llm",
    run_tag: Optional[str] = None,
) -> str:
    run_segments = [f"run={_safe_log_component(run_tag)}"] if str(run_tag or "").strip() else []
    log_dir = os.path.join(
        ROOT_DIR,
        "logs",
        "soft_tool_selection",
        _safe_log_component(model_name),
        _safe_log_component(dataset),
        f"cat={_safe_log_component(category)}",
        (
            f"lr={float(learning_rate)}_epoch={int(num_epochs)}_hid={int(hidden_dim)}"
            f"_reg={float(entropy_reg_weight)}"
        ),
        *run_segments,
    )
    seg = _qenc_path_segment(q_encoder)
    if seg:
        log_dir = os.path.join(log_dir, seg)
    return os.path.join(log_dir, "train.log")

def _soft_tool_selection_run_output_dir(
    *,
    dataset: str,
    model_name: str,
    category: str,
    learning_rate: float,
    num_epochs: int,
    hidden_dim: int,
    entropy_reg_weight: float,
    q_encoder: str = "llm",
    run_tag: Optional[str] = None,
) -> str:
    run_segments = [f"run={_safe_log_component(run_tag)}"] if str(run_tag or "").strip() else []
    out_dir = os.path.join(
        SOFT_TOOL_SELECTION_ROOT_PATH,
        _safe_log_component(dataset),
        "runs",
        _safe_log_component(model_name),
        f"cat={_safe_log_component(category)}",
        (
            f"lr={float(learning_rate)}_epoch={int(num_epochs)}_hid={int(hidden_dim)}"
            f"_reg={float(entropy_reg_weight)}"
        ),
        *run_segments,
    )
    seg = _qenc_path_segment(q_encoder)
    if seg:
        out_dir = os.path.join(out_dir, seg)
    return out_dir

def evaluate_samples(
    samples: List,
    gating_net: GatingNetwork,
    tool_embedding: nn.Embedding,
    encode_question,
    tool_vocab: Dict[str, int],
    tool_vocab_type: Optional[str],
    unk_tool_id: int,
    entropy_reg_weight: float,
    device: torch.device,
    eps: float = 1e-8,
) -> Dict[str, float]:
    gating_net.eval()

    total_correct = 0
    total_labeled = 0
    total_loss = 0.0
    total_ce_loss = 0.0
    total_entropy = 0.0

    with torch.no_grad():
        for sample in samples:
            q_text = sample.question
            tools = sample.tools
            answer_idx = sample.answer_idx

            if not isinstance(answer_idx, int):
                continue

            tool_ids = resolve_tool_ids(
                tools,
                tool_vocab=tool_vocab,
                tool_vocab_type=tool_vocab_type,
                unk_tool_id=unk_tool_id,
            )
            if not tool_ids:
                continue

            M = len(tool_ids)
            if not (0 <= int(answer_idx) < M):
                continue

            q_emb = encode_question(q_text)
            q_emb = q_emb.to(dtype=gating_net.fc1.weight.dtype)

            tool_ids_tensor = torch.tensor(tool_ids, dtype=torch.long, device=device)
            tool_embs = tool_embedding(tool_ids_tensor).unsqueeze(0)
            tool_mask = torch.ones(1, M, device=device, dtype=torch.bool)

            alphas, scores = gating_net(q_emb, tool_embs, tool_mask, return_scores=True)
            scores = scores if scores is not None else alphas

            target = torch.tensor([int(answer_idx)], device=device, dtype=torch.long)
            ce_loss = F.cross_entropy(scores, target)

            alphas_vec = alphas[0]
            entropy = -(alphas_vec * (alphas_vec + eps).log()).sum()
            entropy_loss = -entropy
            loss = ce_loss + entropy_reg_weight * entropy_loss

            pred_idx = int(scores[0].argmax(dim=-1).item())
            total_labeled += 1
            if pred_idx == int(answer_idx):
                total_correct += 1

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_entropy += entropy.item()

    gating_net.train()

    if total_labeled == 0:
        return {
            'accuracy': 0.0,
            'avg_loss': 0.0,
            'avg_ce_loss': 0.0,
            'avg_entropy': 0.0,
            'num_samples': 0,
        }

    return {
        'accuracy': total_correct / total_labeled,
        'avg_loss': total_loss / total_labeled,
        'avg_ce_loss': total_ce_loss / total_labeled,
        'avg_entropy': total_entropy / total_labeled,
        'num_samples': total_labeled,
    }

def save_epoch_checkpoint(
    epoch: int,
    num_epochs: int,
    gating_net: GatingNetwork,
    tool_embedding: nn.Embedding,
    tool_vocab: Dict[str, int],
    run_output_dir: str,
    base_filename: str,
    checkpoint_metadata: Dict,
    train_metrics: Dict[str, float],
    is_best: bool = False,
    also_save_paths: Optional[List[str]] = None,
) -> str:
    def _atomic_torch_save(obj: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    base_name, ext = os.path.splitext(base_filename)
    epoch_filename = f"{base_name}_epoch{epoch + 1}{ext}"
    epoch_path = os.path.join(run_output_dir, epoch_filename)

    ckpt_obj = {
        **checkpoint_metadata,
        "gating_state_dict": gating_net.state_dict(),
        "tool_embedding_state_dict": tool_embedding.state_dict(),
        "tool_vocab": tool_vocab,
        "current_epoch": epoch + 1,
        "total_epochs": num_epochs,
        "train_accuracy": train_metrics['accuracy'],
        "train_loss": train_metrics['avg_loss'],
    }

    os.makedirs(run_output_dir, exist_ok=True)
    _atomic_torch_save(ckpt_obj, epoch_path)

    if is_best:
        best_filename = f"{base_name}_best{ext}"
        best_path = os.path.join(run_output_dir, best_filename)
        _atomic_torch_save(ckpt_obj, best_path)

    for extra_path in (also_save_paths or []):
        if not extra_path:
            continue
        try:
            _atomic_torch_save(ckpt_obj, extra_path)
        except Exception as e:
            print(f"[gating][tool_ce] warning: failed to save extra checkpoint to {extra_path}: {e}")

    return epoch_path

def train_soft_tool_selection(
    model_name: str = "llama3.1-8b-instruct",
    category: str = "multiple",
    dataset: str = "bfcl",
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    entropy_reg_weight: float = 0.2,
    hidden_dim: int = 512,
    q_encoder: str = "llm",
    bge_model_path: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> None:
    dataset_name = str(dataset or "bfcl").lower()
    if dataset_name in ("stabletoolbench", "toolbench"):
        dataset_name = "stb"

    entropy_reg_weight = float(entropy_reg_weight)

    save_dir = os.path.join(SOFT_TOOL_SELECTION_ROOT_PATH, dataset_name)

    save_name = gating_checkpoint_filename(
        model_name=model_name,
        category=category,
        entropy_reg_weight=entropy_reg_weight,
        q_encoder=q_encoder,
        run_tag=run_tag,
    )
    save_path = os.path.join(save_dir, save_name)

    qe_mode = normalize_gating_q_encoder_mode(q_encoder)
    if qe_mode == "bge":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        question_encoder = make_gating_question_encoder(
            "bge",
            device=device,
            bge_model_path=bge_model_path,
        )
        hidden_size = question_encoder.embed_dim
        base_model = None
        tokenizer = None
    else:
        base_model, tokenizer, _ = get_model(model_name, max_new_tokens=1024)
        device = base_model.device
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False
        hidden_size = int(base_model.config.hidden_size)
        question_encoder = make_gating_question_encoder(
            "llm",
            device=device,
            base_model=base_model,
            tokenizer=tokenizer,
        )

    def encode_question(q_text: str) -> torch.Tensor:
        return question_encoder.encode(q_text)

    samples_list = load_soft_tool_selection_samples(
        dataset=dataset_name,
        category=category,
    )
    print(f"[gating][tool_ce] total training samples: {len(samples_list)}")
    if not samples_list:
        print("[gating][tool_ce] no samples available, skip soft_tool_selection.")
        return

    train_samples = samples_list
    print(f"[gating][tool_ce] using all {len(train_samples)} samples for training")

    tool_vocab: Dict[str, int] = {}
    tool_vocab_type: str
    if dataset_name == "stb":
        tool_vocab_type = "stb_triplet"
    else:
        tool_vocab_type = "canonical_schema"

    def _tool_key(schema: Dict) -> str:
        if not isinstance(schema, dict):
            return ""
        return tool_vocab_key(schema, tool_vocab_type=tool_vocab_type)

    save_dir = os.path.join(SOFT_TOOL_SELECTION_ROOT_PATH, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    tool_vocab_path = os.path.join(save_dir, "tool_vocab.json")
    if os.path.exists(tool_vocab_path):
        try:
            with open(tool_vocab_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(k, str) and isinstance(v, int):
                        tool_vocab[k] = v
        except Exception as e:
            print(
                f"[gating][tool_ce] warning: failed to load tool_vocab from {tool_vocab_path}: {e}"
            )

    next_id = max(tool_vocab.values(), default=-1) + 1
    for sample in samples_list:
        for schema in sample.tools:
            if not isinstance(schema, dict):
                continue
            key = _tool_key(schema)
            if not key:
                continue
            if key in tool_vocab:
                continue
            tool_vocab[key] = next_id
            next_id += 1

    unk_tool_id = next_id
    num_tools = unk_tool_id + 1

    try:
        with open(tool_vocab_path, "w", encoding="utf-8") as f:
            json.dump(tool_vocab, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(
            f"[gating][tool_ce] warning: failed to save tool_vocab to {tool_vocab_path}: {e}"
        )

    gating_net = GatingNetwork(dim=hidden_size, hidden_dim=hidden_dim).to(device)
    tool_embedding = nn.Embedding(
        num_embeddings=num_tools,
        embedding_dim=hidden_size,
        sparse=True,
    ).to(device=device, dtype=gating_net.fc1.weight.dtype)
    nn.init.normal_(tool_embedding.weight, mean=0.0, std=0.02)

    gating_opt = torch.optim.AdamW(gating_net.parameters(), lr=learning_rate)
    tool_opt = torch.optim.SparseAdam(tool_embedding.parameters(), lr=learning_rate)

    sample_count = 0
    total_top1_correct = 0
    total_top1_labeled = 0
    eps = 1e-8
    log_interval = 20
    best_epoch = -1

    run_output_dir = _soft_tool_selection_run_output_dir(
        dataset=dataset_name,
        model_name=model_name,
        category=category,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        entropy_reg_weight=entropy_reg_weight,
        q_encoder=q_encoder,
        run_tag=run_tag,
    )
    run_save_path = os.path.join(run_output_dir, save_name)

    log_path = _soft_tool_selection_train_log_path(
        dataset=dataset_name,
        model_name=model_name,
        category=category,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        entropy_reg_weight=entropy_reg_weight,
        q_encoder=q_encoder,
        run_tag=run_tag,
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[gating][tool_ce] logging details to: {log_path}")
    print(f"[gating][tool_ce] run output dir: {run_output_dir}")

    train_metrics: Dict[str, float] = {
        "accuracy": 0.0,
        "avg_loss": 0.0,
        "avg_ce_loss": 0.0,
        "avg_entropy": 0.0,
        "num_samples": 0,
    }
    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(num_epochs):
            print(f"[gating][tool_ce] epoch {epoch + 1}/{num_epochs}")
            random.shuffle(train_samples)

            epoch_top1_correct = 0
            epoch_top1_labeled = 0
            for step_in_epoch, sample in enumerate(train_samples, start=1):
                sample_count += 1

                q_text = sample.question
                tools = sample.tools
                answer_idx = sample.answer_idx
                if not isinstance(answer_idx, int):
                    continue

                tool_ids = resolve_tool_ids(
                    tools,
                    tool_vocab=tool_vocab,
                    tool_vocab_type=tool_vocab_type,
                    unk_tool_id=unk_tool_id,
                )
                if not tool_ids:
                    continue
                M = len(tool_ids)
                if not (0 <= int(answer_idx) < M):
                    continue

                q_emb = encode_question(q_text)
                q_emb = q_emb.to(dtype=gating_net.fc1.weight.dtype)

                tool_ids_tensor = torch.tensor(tool_ids, dtype=torch.long, device=device)
                tool_embs = tool_embedding(tool_ids_tensor).unsqueeze(0)
                tool_mask = torch.ones(1, M, device=device, dtype=torch.bool)
                alphas, scores = gating_net(q_emb, tool_embs, tool_mask, return_scores=True)
                scores = scores if scores is not None else alphas

                target = torch.tensor([int(answer_idx)], device=device, dtype=torch.long)
                ce_loss = F.cross_entropy(scores, target)

                alphas_vec = alphas[0]
                entropy = -(alphas_vec * (alphas_vec + eps).log()).sum()
                entropy_loss = -entropy
                loss = ce_loss + entropy_reg_weight * entropy_loss

                pred_idx = int(scores[0].argmax(dim=-1).item())
                epoch_top1_labeled += 1
                total_top1_labeled += 1
                if pred_idx == int(answer_idx):
                    epoch_top1_correct += 1
                    total_top1_correct += 1

                gating_opt.zero_grad()
                tool_opt.zero_grad()
                loss.backward()
                gating_opt.step()
                tool_opt.step()

                if (
                    step_in_epoch == 1
                    or step_in_epoch % log_interval == 0
                    or step_in_epoch == len(train_samples)
                ):
                    print(
                        f"[gating][tool_ce] epoch {epoch + 1}/{num_epochs}, "
                        f"step {step_in_epoch}/{len(train_samples)}, "
                        f"loss={loss.item():.4f}, ce={ce_loss.item():.4f}, "
                        f"entropy={entropy.item():.4f}, ent_w={entropy_reg_weight:.6g}"
                    )

                    tool_names: List[str] = []
                    for schema in tools:
                        if isinstance(schema, dict):
                            tool_names.append(str(schema.get("name", "")))
                        else:
                            tool_names.append(str(schema))

                    prob_vals = [float(x) for x in alphas_vec.detach().cpu().tolist()]
                    label_vals = [1 if i == int(answer_idx) else 0 for i in range(M)]

                    pred_name = tool_names[pred_idx] if 0 <= pred_idx < M else ""
                    gold_name = (
                        tool_names[int(answer_idx)] if 0 <= int(answer_idx) < M else ""
                    )
                    pred_prob = prob_vals[pred_idx] if 0 <= pred_idx < M else 0.0

                    probs_bracket = (
                        "[" + ", ".join(f"{n}:{p:.4f}" for n, p in zip(tool_names, prob_vals)) + "]"
                    )
                    label_bracket = "[" + ", ".join(str(v) for v in label_vals) + "]"

                    log_f.write(
                        f"[gating][tool_ce] epoch {epoch + 1}/{num_epochs}, "
                        f"step {step_in_epoch}/{len(train_samples)}, "
                        f"example_index={sample.example_index}, "
                        f"loss={loss.item():.4f}, ce={ce_loss.item():.4f}, "
                        f"entropy={entropy.item():.4f}, ent_w={entropy_reg_weight:.6g}, "
                        f"pred={pred_idx}, gold={int(answer_idx)}\n"
                        f"[gating][tool_ce][pred] [{pred_name}()] p={pred_prob:.4f}\n"
                        f"[gating][tool_ce][gold] [{gold_name}()]\n"
                        f"[gating][tool_ce][probs] {probs_bracket}\n"
                        f"[gating][tool_ce][label] {label_bracket}\n\n"
                    )
                    log_f.flush()

            if epoch_top1_labeled > 0:
                acc = epoch_top1_correct / float(epoch_top1_labeled)
                epoch_stats_line = (
                    f"[gating][tool_ce][epoch_stats] epoch {epoch + 1}/{num_epochs} "
                    f"labeled_samples={epoch_top1_labeled}, "
                    f"top1_correct={epoch_top1_correct}, "
                    f"top1_acc={acc:.4f}"
                )
                print(epoch_stats_line)
                log_f.write(epoch_stats_line + "\n\n")
                log_f.flush()
            else:
                epoch_stats_line = (
                    f"[gating][tool_ce][epoch_stats] epoch {epoch + 1}/{num_epochs} "
                    "labeled_samples=0 (no top1 stats)."
                )
                print(epoch_stats_line)
                log_f.write(epoch_stats_line + "\n\n")
                log_f.flush()

            print(f"[gating][tool_ce] evaluating epoch {epoch + 1}/{num_epochs} on training samples...")

            train_metrics = evaluate_samples(
                samples=train_samples,
                gating_net=gating_net,
                tool_embedding=tool_embedding,
                encode_question=encode_question,
                tool_vocab=tool_vocab,
                tool_vocab_type=tool_vocab_type,
                unk_tool_id=unk_tool_id,
                entropy_reg_weight=entropy_reg_weight,
                device=device,
                eps=eps,
            )

            epoch_eval_line = (
                f"[gating][tool_ce][epoch_eval] epoch {epoch + 1}/{num_epochs}\n"
                f"  Train: acc={train_metrics['accuracy']:.4f}, "
                f"loss={train_metrics['avg_loss']:.4f}, "
                f"ce={train_metrics['avg_ce_loss']:.4f}, "
                f"entropy={train_metrics['avg_entropy']:.4f}, "
                f"samples={train_metrics['num_samples']}"
            )
            print(epoch_eval_line)
            log_f.write(epoch_eval_line + "\n\n")
            log_f.flush()

            is_best = epoch + 1 == num_epochs
            if is_best:
                best_epoch = epoch + 1
                best_msg = f"[gating][tool_ce] saving final epoch {best_epoch} as best"
                print(best_msg)
                log_f.write(best_msg + "\n\n")
                log_f.flush()

            extra_save_paths: List[str] = [run_save_path]

            if is_best:
                extra_save_paths.append(save_path)
            ckpt_meta = {
                "format_version": 3,
                "objective": "tool_ce",
                "dataset": dataset_name,
                "unk_tool_id": int(unk_tool_id),
                "tool_vocab_type": tool_vocab_type,
                "hidden_size": hidden_size,
                "hidden_dim": hidden_dim,
                "learning_rate": float(learning_rate),
                "num_epochs": int(num_epochs),
                "model_name": model_name,
                "category": category,
                "entropy_reg_weight": float(entropy_reg_weight),
                "data_format": "jsonl",
                "data_schema": "function_training_data_v1",
                "run_tag": run_tag,
            }
            if qe_mode == "bge":
                ckpt_meta["q_encoder"] = "bge"
            epoch_ckpt_path = save_epoch_checkpoint(
                epoch=epoch,
                num_epochs=num_epochs,
                gating_net=gating_net,
                tool_embedding=tool_embedding,
                tool_vocab=tool_vocab,
                run_output_dir=run_output_dir,
                base_filename=save_name,
                checkpoint_metadata=ckpt_meta,
                train_metrics=train_metrics,
                is_best=is_best,
                also_save_paths=extra_save_paths,
            )
            ckpt_msg = f"[gating][tool_ce] epoch {epoch + 1} checkpoint saved to: {epoch_ckpt_path}"
            print(ckpt_msg)
            log_f.write(ckpt_msg + "\n\n")
            log_f.flush()

    with open(log_path, "a", encoding="utf-8") as log_f:
        final_summary = (
            f"[gating][tool_ce][final_summary]\n"
            f"  Total training samples (all epochs): {sample_count}\n"
            f"  Best model: epoch {best_epoch}\n"
            f"  Final epoch: {num_epochs}\n"
        )
        log_f.write(final_summary + "\n")
        log_f.flush()

    print(f"[gating][tool_ce] best model saved from epoch {best_epoch}")

    ckpt_obj = {
        "format_version": 3,
        "objective": "tool_ce",
        "dataset": dataset_name,
        "gating_state_dict": gating_net.state_dict(),
        "tool_embedding_state_dict": tool_embedding.state_dict(),
        "tool_vocab": tool_vocab,
        "tool_vocab_type": tool_vocab_type,
        "unk_tool_id": int(unk_tool_id),
        "hidden_size": hidden_size,
        "hidden_dim": hidden_dim,
        "learning_rate": float(learning_rate),
        "num_epochs": int(num_epochs),
        "current_epoch": int(num_epochs),
        "total_epochs": int(num_epochs),
        "train_accuracy": float(train_metrics.get("accuracy", 0.0)),
        "train_loss": float(train_metrics.get("avg_loss", 0.0)),
        "model_name": model_name,
        "category": category,
        "entropy_reg_weight": float(entropy_reg_weight),
        "data_format": "jsonl",
        "data_schema": "function_training_data_v1",
        "run_tag": run_tag,
    }
    if qe_mode == "bge":
        ckpt_obj["q_encoder"] = "bge"

    os.makedirs(run_output_dir, exist_ok=True)
    tmp_path = f"{run_save_path}.tmp"
    torch.save(ckpt_obj, tmp_path)
    os.replace(tmp_path, run_save_path)

    print(f"[gating][tool_ce] training complete, total_samples={sample_count}")
    if total_top1_labeled > 0:
        total_acc = total_top1_correct / float(total_top1_labeled)
        print(
            f"[gating][tool_ce][train_stats] total_labeled_samples={total_top1_labeled}, "
            f"top1_correct={total_top1_correct}, "
            f"top1_acc={total_acc:.4f}"
        )
    else:
        print("[gating][tool_ce][train_stats] no labeled samples for top1 stats.")
    print(f"[gating][tool_ce] gating network saved to: {run_save_path}")
    print(f"[gating][tool_ce] canonical checkpoint path: {save_path}")

__all__ = ["train_soft_tool_selection"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Soft tool selection: train GatingNetwork (tool_ce).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (keys match CLI args). CLI args override YAML.",
    )
    parser.add_argument("--model_name", type=str, default="llama3.1-8b-instruct")
    parser.add_argument("--category", type=str, default="multiple")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default="bfcl",
        choices=["bfcl", "stb", "stabletoolbench", "toolbench"],
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--entropy_reg_weight",
        type=float,
        default=0.2,
        help=(
            "Entropy regularization weight (positive encourages higher-entropy / less-peaky "
            "gating). Loss = CE + entropy_reg_weight * (-entropy)."
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--q_encoder",
        type=str,
        default="llm",
        choices=["llm", "bge"],
        help="Question encoder for gating: llm=base model hidden mean-pool; bge=local BGE-small.",
    )
    parser.add_argument(
        "--bge_model_path",
        type=str,
        default=None,
        help="Override BGE model directory (default: env BGE_QENC_MODEL_PATH or built-in path; not saved in ckpt).",
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
    print(args)
    train_soft_tool_selection(
        model_name=args.model_name,
        category=args.category,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        entropy_reg_weight=args.entropy_reg_weight,
        hidden_dim=args.hidden_dim,
        q_encoder=args.q_encoder,
        bge_model_path=args.bge_model_path,
        run_tag=args.run_tag,
    )

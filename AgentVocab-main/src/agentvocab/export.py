from __future__ import annotations

import gc
import re
import shutil
from pathlib import Path

from tqdm import tqdm

from .io import get_logger


TOKENIZER_FILE_NAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
]

EXTRA_FILE_NAMES = [
    "configuration.json",
    "LICENSE",
    "README.md",
]

LOGGER = get_logger(__name__)


def copy_existing_files(src_dir: str | Path, dst_dir: str | Path, names: list[str]) -> None:
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_path / name
        if src.exists():
            shutil.copy2(src, dst_path / name)


def find_lora_checkpoints(series_dir: str | Path) -> list[tuple[Path, int]]:
    checkpoints: list[tuple[Path, int]] = []
    pattern = re.compile(r"^checkpoint-(\d+)$")
    for train_dir in Path(series_dir).glob("train-*"):
        if not train_dir.is_dir():
            continue
        for checkpoint_dir in train_dir.rglob("checkpoint-*"):
            match = pattern.match(checkpoint_dir.name)
            if not match:
                continue
            if (checkpoint_dir / "adapter_model.safetensors").exists() or (checkpoint_dir / "adapter_model.bin").exists():
                checkpoints.append((checkpoint_dir, int(match.group(1))))
    checkpoints.sort(key=lambda item: item[1])
    return checkpoints


def merge_lora_checkpoints(
    series_dir: str | Path,
    base_model_path: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    *,
    device_map: str = "cpu",
) -> list[Path]:
    """Merge LoRA checkpoints into full model directories for evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    merged_paths: list[Path] = []
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = find_lora_checkpoints(series_dir)
    LOGGER.info("Found %d LoRA checkpoints under %s", len(checkpoints), series_dir)
    for checkpoint_dir, step in tqdm(checkpoints, desc="merge LoRA checkpoints", unit="ckpt"):
        target_dir = output_root / f"{output_prefix}{step}"
        if (target_dir / "config.json").exists():
            LOGGER.info("Skipping existing merged checkpoint %s", target_dir)
            merged_paths.append(target_dir)
            continue

        base_model = None
        peft_model = None
        merged_model = None
        try:
            LOGGER.info("Merging %s -> %s", checkpoint_dir, target_dir)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(target_dir, safe_serialization=True)
            copy_existing_files(base_model_path, target_dir, TOKENIZER_FILE_NAMES)
            copy_existing_files(base_model_path, target_dir, EXTRA_FILE_NAMES)
            merged_paths.append(target_dir)
            LOGGER.info("Saved merged checkpoint to %s", target_dir)
        finally:
            del merged_model, peft_model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return merged_paths

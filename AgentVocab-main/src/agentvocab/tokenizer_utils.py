from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .io import get_logger
from .io import load_token_list


LOGGER = get_logger(__name__)


def expand_vocab_with_mean_initialization(
    base_model_path: str,
    new_tokens_file: str | Path,
    output_dir: str | Path,
    *,
    device_map: str = "cpu",
    torch_dtype: str = "bfloat16",
) -> int:
    """Add new tokens and initialize their embeddings from base-token means."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    LOGGER.info("Loading tokenizer/model from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    new_tokens = load_token_list(new_tokens_file)
    old_embeddings = model.get_input_embeddings().weight.data.clone()
    old_lm_head = model.get_output_embeddings().weight.data.clone()

    mean_embeddings = []
    mean_lm_heads = []
    LOGGER.info("Preparing mean initialization for %d new tokens", len(new_tokens))
    for token in tqdm(new_tokens, desc="compute mean initialization"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            mean_embeddings.append(old_embeddings[token_ids].to(torch.float32).mean(dim=0).to(dtype))
            mean_lm_heads.append(old_lm_head[token_ids].to(torch.float32).mean(dim=0).to(dtype))
        else:
            hidden_size = old_embeddings.size(1)
            mean_embeddings.append(torch.zeros(hidden_size, dtype=dtype, device=old_embeddings.device))
            mean_lm_heads.append(torch.zeros(hidden_size, dtype=dtype, device=old_lm_head.device))

    added = tokenizer.add_tokens(new_tokens)
    LOGGER.info("Tokenizer accepted %d new tokens; resizing model embeddings", added)
    model.resize_token_embeddings(len(tokenizer))

    new_embeddings = model.get_input_embeddings().weight.data
    new_lm_head = model.get_output_embeddings().weight.data
    for index, token in enumerate(tqdm(new_tokens, desc="inject new token weights")):
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_embeddings[token_id] = mean_embeddings[index]
        new_lm_head[token_id] = mean_lm_heads[index]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving expanded tokenizer/model to %s", output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    LOGGER.info("Finished vocabulary expansion")
    return added

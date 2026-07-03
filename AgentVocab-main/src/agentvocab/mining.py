from __future__ import annotations

import math
import os
import re
import random
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import dump_json
from .io import count_text_lines, get_logger, iter_text_field, load_json, progress


LOGGER = get_logger(__name__)


DEFAULT_BANNED_WORDS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "shall", "can", "could", "would", "should",
    "to", "for", "of", "in", "on", "at", "by", "with", "about", "as", "from",
    "and", "but", "or", "so", "if", "then", "let", "my", "your",
    "a", "an", "the", "this", "that", "these", "those",
}


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.fail: TrieNode | None = None
        self.is_leaf = False
        self.word: str | None = None
        self.depth = 0


class TokenACAutomaton:
    """Aho-Corasick automaton over token-id sequences."""

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: list[int], word: str) -> None:
        node = self.root
        for depth, token_id in enumerate(token_ids, 1):
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
            node.depth = depth
        node.is_leaf = True
        node.word = word

    def build_fail_pointers(self) -> None:
        queue: deque[TrieNode] = deque()
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for token_id, child in current.children.items():
                fail_node = current.fail
                while fail_node is not None and token_id not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = self.root if fail_node is None else fail_node.children[token_id]
                queue.append(child)


def token_savings(length: int, frequency: int) -> float:
    """Approximate decoding-token savings from replacing a span with one token."""
    if length <= 1 or frequency <= 0:
        return 0.0
    return ((length - 1) * frequency) / math.sqrt(length)


def mine_structural_tokens(
    input_file: str | Path,
    tokenizer_path: str,
    output_file: str | Path,
    *,
    max_new_tokens: int = 10_000,
    min_frequency: int = 10,
) -> list[dict[str, Any]]:
    """Mine high-frequency structural spans ranked by token savings."""
    from transformers import AutoTokenizer

    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    span_counter: Counter[str] = Counter()
    with Path(input_file).open("r", encoding="utf-8") as f:
        for line in progress(f, desc="read structural spans", total=count_text_lines(input_file), unit="line"):
            span = line.rstrip("\n\r")
            if len(span.strip()) >= 2:
                span_counter[span] += 1
    LOGGER.info("Counted %d unique structural spans", len(span_counter))

    candidates: dict[str, dict[str, Any]] = {}
    for span, frequency in span_counter.items():
        if frequency < min_frequency:
            continue
        token_ids = tokenizer.encode(span, add_special_tokens=False)
        if len(token_ids) <= 1:
            continue
        candidates[span] = {
            "token": span,
            "original_freq": frequency,
            "current_freq": frequency,
            "original_len": len(token_ids),
            "current_len": len(token_ids),
            "tokenizer_view": " | ".join(tokenizer.decode([token_id]) for token_id in token_ids),
            "marginal_savings": token_savings(len(token_ids), frequency),
        }
    LOGGER.info("Kept %d structural candidates after frequency/token-length filtering", len(candidates))

    selected: list[dict[str, Any]] = []
    available = set(candidates)
    with tqdm(total=min(max_new_tokens, len(available)), desc="select structural tokens") as pbar:
        while available and len(selected) < max_new_tokens:
            best = max(available, key=lambda span: (candidates[span]["marginal_savings"], len(span)))
            best_stat = candidates[best]
            if best_stat["marginal_savings"] <= 0:
                break

            selected.append(dict(best_stat))
            available.remove(best)
            tokenizer.add_tokens([best])

            to_remove: list[str] = []
            for span in list(available):
                stat = candidates[span]
                if best not in span:
                    continue
                new_ids = tokenizer.encode(span, add_special_tokens=False)
                if len(new_ids) < stat["current_len"]:
                    stat["current_len"] = len(new_ids)
                    stat["tokenizer_view"] = " | ".join(tokenizer.decode([token_id]) for token_id in new_ids)
                    stat["marginal_savings"] = token_savings(stat["current_len"], stat["current_freq"])
                    if stat["marginal_savings"] <= 0:
                        to_remove.append(span)
            for span in to_remove:
                available.remove(span)
            pbar.update(1)

    selected.sort(key=lambda item: item["marginal_savings"], reverse=True)
    dump_json(output_file, selected)
    LOGGER.info("Wrote %d structural tokens to %s", len(selected), output_file)
    return selected


def is_noisy_content_candidate(text: str, original_ids: list[int]) -> bool:
    text_stripped = text.strip()
    if len(text_stripped) < 3:
        return True
    if len(text_stripped) >= 2 and text_stripped[0] == "n" and text_stripped[1].isupper():
        return True
    original_case_words = text_stripped.split()
    if len(original_case_words) >= 2 and len(original_case_words[-1]) == 1:
        return True
    if len(set(original_ids)) == 1 and len(original_ids) > 1:
        return True
    text_no_space = text_stripped.replace(" ", "")
    if len(set(text_no_space)) <= 1 and len(text_no_space) > 1:
        return True
    if re.search(r"(.{2,})\1{3,}", text_no_space):
        return True
    if len(text_no_space) > 15 and sum(1 for char in text_no_space if char.isupper()) / len(text_no_space) > 0.4:
        return True
    if len(text_no_space) > 10 and not any(vowel in text_no_space.lower() for vowel in "aeiouy"):
        return True

    stripped_words = text_stripped.split()
    if len(stripped_words) >= 2 and (len(stripped_words[0]) < 2 or len(stripped_words[-1]) < 2):
        return True

    words_lower = set(re.findall(r"[a-z]+", text.lower()))
    if words_lower.intersection(DEFAULT_BANNED_WORDS):
        return True

    if re.fullmatch(r"[\-_]?[A-Za-z0-9_]+", text_stripped):
        return False

    if " " in text:
        if len(text_stripped.split()) > 3:
            return True
        if not re.fullmatch(r"[A-Za-z0-9_ ]+", text_stripped):
            return True

    first_char = text[0]
    if first_char == " " or first_char.isupper() or first_char in {"-", "_"}:
        return False
    if " " in text:
        first_part = text.split(" ")[0]
        if len(first_part) <= 2 or first_part in {"ing", "ed", "ly", "tion", "ment", "able"}:
            return True
    elif len(text) < 5:
        return True

    return False


def mine_content_candidates_tlbpe(
    input_file: str | Path,
    tokenizer_path: str,
    output_file: str | Path,
    *,
    max_new_tokens: int = 10_000,
    min_frequency: int = 10,
    max_subwords: int = 4,
    offset: int = 100_000,
    work_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Mine content-token candidates with token-level BPE and dynamic savings.

    This mirrors the first stage of the TV3 content-token pipeline: token-level
    BPE proposes spans, noisy spans are filtered, real frequencies are counted,
    and a dynamic tokenizer simulation greedily keeps high-savings candidates.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from transformers import AutoTokenizer

    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    work_path = Path(work_dir or Path(output_file).parent)
    work_path.mkdir(parents=True, exist_ok=True)
    fake_corpus = work_path / "tlbpe_fake_corpus.txt"

    unique_base_tokens: set[int] = set()
    fake_sentence_counter: Counter[str] = Counter()
    LOGGER.info("Encoding content corpus %s for TL-BPE", input_file)
    with Path(input_file).open("r", encoding="utf-8") as f_in, fake_corpus.open("w", encoding="utf-8") as f_out:
        for line in progress(f_in, desc="encode content spans", total=count_text_lines(input_file), unit="line"):
            text = line.rstrip("\r\n\t ")
            if not text:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < 2:
                continue
            unique_base_tokens.update(token_ids)
            fake_line = "".join(chr(token_id + offset) for token_id in token_ids)
            fake_sentence_counter[fake_line] += 1
            f_out.write(fake_line + "\n")
    LOGGER.info("Prepared %d unique fake sentences with %d unique base tokens", len(fake_sentence_counter), len(unique_base_tokens))

    fast_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=len(unique_base_tokens) + max_new_tokens * 2,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"],
    )
    LOGGER.info("Training TL-BPE tokenizer on %s", fake_corpus)
    fast_tokenizer.train(files=[str(fake_corpus)], trainer=trainer)

    clean_candidates: set[str] = set()
    candidate_info: dict[str, dict[str, Any]] = {}
    tlbpe_vocab = fast_tokenizer.get_vocab()
    for fake_word, _ in progress(tlbpe_vocab.items(), desc="filter TL-BPE candidates", total=len(tlbpe_vocab), unit="token"):
        if fake_word == "[UNK]" or len(fake_word) < 2:
            continue
        original_ids = [ord(char) - offset for char in fake_word]
        if any(token_id < 0 or token_id > 1_000_000 for token_id in original_ids):
            continue
        text = tokenizer.decode(original_ids)
        if is_noisy_content_candidate(text, original_ids):
            continue

        clean_candidates.add(fake_word)
        candidate_info[fake_word] = {"token": text, "original_ids": original_ids}
    LOGGER.info("Kept %d clean TL-BPE candidates after noise filtering", len(clean_candidates))

    candidate_freqs: dict[str, int] = {}
    for fake_word in tqdm(clean_candidates, desc="count TL-BPE candidate frequency"):
        frequency = 0
        for sentence, count in fake_sentence_counter.items():
            occurrences = sentence.count(fake_word)
            if occurrences:
                frequency += occurrences * count
        if frequency >= min_frequency:
            candidate_freqs[fake_word] = frequency
    LOGGER.info("Kept %d content candidates after minimum-frequency filtering", len(candidate_freqs))

    candidate_stats: dict[str, dict[str, Any]] = {}
    for fake_word, frequency in candidate_freqs.items():
        info = candidate_info[fake_word]
        original_ids = info["original_ids"]
        original_len = len(original_ids)
        candidate_stats[fake_word] = {
            "token": info["token"],
            "fake_word": fake_word,
            "original_ids": original_ids,
            "original_freq": frequency,
            "current_freq": frequency,
            "original_len": original_len,
            "current_len": original_len,
            "tokenizer_view": " | ".join(tokenizer.decode([token_id]) for token_id in original_ids),
            "marginal_savings": token_savings(original_len, frequency),
        }

    selected: list[dict[str, Any]] = []
    available = set(candidate_stats)
    dynamic_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    with tqdm(total=min(max_new_tokens, len(available)), desc="select TL-BPE content candidates") as pbar:
        while available and len(selected) < max_new_tokens:
            valid = [
                fake_word
                for fake_word in available
                if candidate_stats[fake_word]["current_len"] <= max_subwords
                and candidate_stats[fake_word]["marginal_savings"] > 0
            ]
            if not valid:
                break

            best_key = max(valid, key=lambda key: (candidate_stats[key]["marginal_savings"], candidate_stats[key]["original_len"]))
            best = candidate_stats[best_key]
            selected.append(
                {
                    "token": best["token"],
                    "original_ids": best["original_ids"],
                    "original_freq": best["original_freq"],
                    "current_freq": best["current_freq"],
                    "original_len": best["original_len"],
                    "current_len": best["current_len"],
                    "tokenizer_view": best["tokenizer_view"],
                    "marginal_savings": best["marginal_savings"],
                }
            )

            available.remove(best_key)
            dynamic_tokenizer.add_tokens([best["token"]])

            to_remove: list[str] = []
            for candidate_key in list(available):
                stat = candidate_stats[candidate_key]
                updated = False

                if candidate_key in best_key:
                    occurrences = best_key.count(candidate_key)
                    stat["current_freq"] -= occurrences * best["current_freq"]
                    updated = True
                elif best_key in candidate_key:
                    new_ids = dynamic_tokenizer.encode(stat["token"], add_special_tokens=False)
                    if len(new_ids) < stat["current_len"]:
                        stat["current_len"] = len(new_ids)
                        stat["tokenizer_view"] = " | ".join(dynamic_tokenizer.decode([token_id]) for token_id in new_ids)
                        updated = True

                if updated:
                    if stat["current_freq"] < min_frequency:
                        to_remove.append(candidate_key)
                    else:
                        stat["marginal_savings"] = token_savings(stat["current_len"], stat["current_freq"])
                        if stat["marginal_savings"] <= 0 and stat["current_len"] <= 1:
                            to_remove.append(candidate_key)

            for candidate_key in to_remove:
                available.remove(candidate_key)
            pbar.update(1)

    if fake_corpus.exists():
        fake_corpus.unlink()

    dump_json(output_file, selected)
    LOGGER.info("Wrote %d TL-BPE content candidates to %s", len(selected), output_file)
    return selected


def score_content_candidates_by_gradient(
    candidates_file: str | Path,
    corpus_jsonl: str | Path,
    model_path: str,
    output_file: str | Path,
    *,
    text_field: str = "actual_input",
    sample_size: int = 256,
    max_seq_len: int = 4096,
    device: str = "cuda",
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Re-rank TL-BPE content candidates with VEGAD-style gradient scores.

    This follows the original AgentVocab TV3 pipeline more closely:
    candidate token-id spans are matched with an AC automaton and scored by
    accumulated input-embedding and output-logit gradients.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    LOGGER.info("Loading tokenizer/model from %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    candidates = [item for item in load_json(candidates_file) if isinstance(item, dict) and item.get("token")]
    LOGGER.info("Loaded %d content candidates from %s", len(candidates), candidates_file)
    ac = TokenACAutomaton()
    for item in candidates:
        item["token_ids"] = item.get("original_ids") or tokenizer.encode(item["token"], add_special_tokens=False)
        item["vegad_gradient_score"] = 0.0
        item["gradient_hits"] = 0
        if item["token_ids"]:
            ac.insert(item["token_ids"], item["token"])
    ac.build_fail_pointers()
    item_by_token = {item["token"]: item for item in candidates}

    texts = list(progress(iter_text_field(corpus_jsonl, text_field), desc="load gradient corpus", total=count_text_lines(corpus_jsonl), unit="row"))
    if len(texts) > sample_size:
        rng = random.Random(random_seed)
        texts = rng.sample(texts, sample_size)
    LOGGER.info("Scoring content candidates on %d sampled texts", len(texts))
    embedding_layer = model.get_input_embeddings()
    special_token_ids = set(tokenizer.all_special_ids)

    for text in tqdm(texts, desc="gradient score content candidates"):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
        if input_ids.numel() < 2:
            continue

        embeds = embedding_layer(input_ids).detach().clone()
        embeds.requires_grad_(True)
        outputs = model(inputs_embeds=embeds)
        logits = outputs.logits
        logits.retain_grad()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()

        g_embed = embeds.grad[0].float()
        g_lmhead = logits.grad[0].float()
        input_row = input_ids[0]
        special_mask = torch.isin(input_row, torch.tensor(list(special_token_ids), device=device))
        g_lmhead[1:][special_mask[1:]] = 0.0

        zero_embed = torch.zeros((1, g_embed.size(1)), device=device)
        zero_lmhead = torch.zeros((1, g_lmhead.size(1)), device=device)
        cum_embed = torch.cat([zero_embed, torch.cumsum(g_embed, dim=0)], dim=0)
        cum_lmhead = torch.cat([zero_lmhead, torch.cumsum(g_lmhead, dim=0)], dim=0)

        token_ids = input_row.detach().cpu().tolist()
        node = ac.root
        for index, token_id in enumerate(token_ids):
            if token_id in special_token_ids:
                node = ac.root
                continue

            while node is not ac.root and token_id not in node.children:
                node = node.fail if node.fail is not None else ac.root
            if token_id in node.children:
                node = node.children[token_id]
            else:
                node = ac.root

            temp_node = node
            while temp_node is not ac.root:
                if temp_node.is_leaf and temp_node.word is not None:
                    word = temp_node.word
                    depth = temp_node.depth
                    start = index - depth + 1
                    end = index

                    sum_embed = cum_embed[end + 1] - cum_embed[start]
                    norm_embed = torch.norm(sum_embed, p=2).item()

                    if start > 0:
                        sum_lmhead = cum_lmhead[end] - cum_lmhead[start - 1]
                    else:
                        sum_lmhead = cum_lmhead[end]
                    norm_lmhead = torch.norm(sum_lmhead, p=1).item()

                    matched_item = item_by_token[word]
                    matched_item["vegad_gradient_score"] += norm_embed + norm_lmhead
                    matched_item["gradient_hits"] += 1
                temp_node = temp_node.fail if temp_node.fail is not None else ac.root

    for item in candidates:
        item["vegad_gradient_score"] = round(float(item["vegad_gradient_score"]), 4)
        item["bpe_marginal_savings"] = round(float(item.get("marginal_savings", 0.0)), 4)
        item["score"] = item["vegad_gradient_score"]
        if "original_freq" in item and "current_freq" in item:
            item["freq_before_after"] = f"{item['original_freq']} -> {item['current_freq']}"
        if "original_len" in item and "current_len" in item:
            item["len_before_after"] = f"{item['original_len']} -> {item['current_len']}"

    candidates.sort(key=lambda item: item["vegad_gradient_score"], reverse=True)
    dump_json(output_file, candidates)
    LOGGER.info("Wrote %d gradient-ranked content candidates to %s", len(candidates), output_file)
    return candidates

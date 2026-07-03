from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from dataset.data_io import (
    candidate_tools_from_qa,
    data_filename,
    load_jsonl_records,
    resolve_training_jsonl,
)
from dataset.profiles import (
    BfclPythonProfile,
    StbReactProfile,
    build_stb_react_answer_from_qa,
)


def build_tool_pretraining_prompt_data(
    augments,
    tokenizer,
    tools,
    *,
    dataset: str,
    category: str,
):
    prompt_data: List[Tuple[List[int], int]] = []

    dataset_name = (dataset or "").lower()
    is_bfcl = dataset_name == "bfcl"
    multi_turn_categories = {"parallel_multiple", "live_parallel_multiple"}
    is_multi_step_category = str(category or "") in multi_turn_categories

    bfcl_with_profile = BfclPythonProfile("with")
    bfcl_without_profile = BfclPythonProfile("without")

    stb_with_profile = StbReactProfile("with")
    stb_without_profile = StbReactProfile("without")

    for aug in augments:
        qas = aug.get("training_data", [])
        if not isinstance(qas, list) or not qas:
            continue

        for qa in qas:
            qa_question = qa.get("question", "")
            tools_for_this_qa = candidate_tools_from_qa(qa)
            if not tools_for_this_qa:
                continue
            if is_bfcl:
                ans = qa.get("answer")
                if not isinstance(ans, str) or not ans.strip():
                    continue
                if tools_for_this_qa:
                    enc_with = bfcl_with_profile.encode_train_sample(
                        tokenizer,
                        qa_question,
                        tools_for_this_qa,
                        ans,
                        multi_step=is_multi_step_category,
                    )
                    prompt_data.append((enc_with.input_ids, enc_with.prompt_len))

                enc_without = bfcl_without_profile.encode_train_sample(
                    tokenizer,
                    qa_question,
                    tools_for_this_qa,
                    ans,
                    multi_step=is_multi_step_category,
                )
                prompt_data.append((enc_without.input_ids, enc_without.prompt_len))
            else:
                react_answer = build_stb_react_answer_from_qa(qa)
                if not react_answer:
                    continue

                enc_with = stb_with_profile.encode_train_sample(
                    tokenizer,
                    qa_question,
                    tools_for_this_qa,
                    react_answer,
                    multi_step=is_multi_step_category,
                )
                prompt_data.append((enc_with.input_ids, enc_with.prompt_len))

                enc_without = stb_without_profile.encode_train_sample(
                    tokenizer,
                    qa_question,
                    tools_for_this_qa,
                    react_answer,
                    multi_step=is_multi_step_category,
                )
                prompt_data.append((enc_without.input_ids, enc_without.prompt_len))

    return prompt_data

def load_tool_pretraining_data(
    dataset: str,
    category: str,
    tool_pretraining_data_path: Optional[str] = None,
):
    if tool_pretraining_data_path:
        path = os.path.abspath(os.path.expanduser(tool_pretraining_data_path))
    else:
        path = resolve_training_jsonl(dataset, category)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tool pretraining JSONL data file not found: {path}."
        )
    data = load_jsonl_records(path)
    return [(os.path.basename(path) if tool_pretraining_data_path else data_filename(category), data)]

__all__ = ["build_tool_pretraining_prompt_data", "load_tool_pretraining_data"]

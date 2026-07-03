from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol

import sitecustomize  # noqa: F401  -- forces third_party / BFCL paths to load

from bfcl_eval.constants.enums import Language  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import (  # noqa: E402
    multiple_function_checker,
    parallel_function_checker_no_order,
    simple_function_checker,
)
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry  # noqa: E402


class Dataset(Protocol):
    name: str

    def load_data(self, category: str) -> List[Dict[str, Any]]: ...

    def get_ground_truth(self, category: str) -> List[Any]: ...

    def get_sample(self, category: str, index: int) -> Dict[str, Any]: ...

    def get_sample_ground_truth(self, category: str, index: int) -> Any: ...

    def ast_eval(self): ...

class BFCLDataset:
    name = "bfcl"

    def __init__(
        self,
        default_language: Language = Language.PYTHON,
        model_name: str = "glm-4.5-FC",
    ):
        self.default_language = default_language
        self.model_name = model_name

    def load_data(self, category: str) -> List[Dict[str, Any]]:
        return load_dataset_entry(
            category,
            include_prereq=False,
            include_language_specific_hint=False,
        )

    def get_ground_truth(self, category: str) -> List[Any]:
        return load_ground_truth_entry(category)

    def get_sample(self, category: str, index: int) -> Dict[str, Any]:
        return self.load_data(category)[index]

    def get_sample_ground_truth(self, category: str, index: int) -> Any:
        return self.get_ground_truth(category)[index]["ground_truth"]

    def ast_eval(self):
        default_lang = self.default_language
        model_name = self.model_name

        class _ASTEvalFacade:
            def simple(
                self,
                func_description: Dict[str, Any],
                model_output: Dict[str, Any],
                possible_answer: Dict[str, Any],
                language: Optional[Language] = None,
            ) -> Dict[str, Any]:
                return simple_function_checker(
                    func_description,
                    model_output,
                    possible_answer,
                    language or default_lang,
                    model_name,
                )

            def multiple(
                self,
                func_descriptions: List[Dict[str, Any]],
                model_output: List[Dict[str, Any]],
                possible_answers: List[Dict[str, Any]],
                language: Optional[Language] = None,
            ) -> Dict[str, Any]:
                return multiple_function_checker(
                    func_descriptions,
                    model_output,
                    possible_answers,
                    language or default_lang,
                    model_name,
                )

            def parallel_no_order(
                self,
                func_descriptions: List[Dict[str, Any]],
                model_output: List[Dict[str, Any]],
                possible_answers: List[Dict[str, Any]],
                language: Optional[Language] = None,
            ) -> Dict[str, Any]:
                return parallel_function_checker_no_order(
                    func_descriptions,
                    model_output,
                    possible_answers,
                    language or default_lang,
                    model_name,
                )

        return _ASTEvalFacade()

class StableToolBenchDataset:
    name = "stabletoolbench"

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "test_instruction",
    ):
        if root_dir is None:
            from root_dir_path import ROOT_DIR

            root_dir = os.path.join(ROOT_DIR, "StableToolBench")

        self.root_dir = root_dir
        self.split = split
        self.solvable_queries_dir = os.path.join(root_dir, "solvable_queries", split)

        if not os.path.exists(self.solvable_queries_dir):
            raise FileNotFoundError(
                f"StableToolBench solvable_queries directory not found: {self.solvable_queries_dir}"
            )

    def load_data(self, category: str) -> List[Dict[str, Any]]:
        if not category.endswith(".json"):
            category_file = f"{category}.json"
        else:
            category_file = category

        data_path = os.path.join(self.solvable_queries_dir, category_file)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"StableToolBench category file not found: {data_path}"
            )

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a list in {data_path}, got {type(data).__name__}"
            )

        return data

    def get_ground_truth(self, category: str) -> List[Dict[str, Any]]:
        return self.load_data(category)

    def get_sample(self, category: str, index: int) -> Dict[str, Any]:
        data = self.load_data(category)
        if index < 0 or index >= len(data):
            raise IndexError(
                f"Sample index {index} out of range for category {category} (total: {len(data)})"
            )
        return data[index]

    def get_sample_ground_truth(self, category: str, index: int) -> Any:
        sample = self.get_sample(category, index)
        return sample.get("relevant APIs", [])

    def ast_eval(self):
        return None

def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    name = (dataset_name or "").lower()
    if name == "bfcl":
        return BFCLDataset(**kwargs)
    if name in ("stabletoolbench", "toolbench", "stb"):
        return StableToolBenchDataset(**kwargs)
    raise ValueError(f"Unknown dataset: {dataset_name}")

__all__ = [
    "Dataset",
    "get_dataset",
    "BFCLDataset",
    "StableToolBenchDataset",
    "Language",
]

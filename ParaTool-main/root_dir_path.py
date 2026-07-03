import os

import sitecustomize  # noqa: F401  -- forces third_party / BFCL paths to load

_DEFAULT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.environ.get("ROOT_DIR", _DEFAULT_ROOT_DIR)


_DEFAULT_PARAM_ROOT_PATH = os.path.join(ROOT_DIR, "output", "paratool")
PARAM_ROOT_PATH = os.environ.get("PARAM_ROOT_PATH", _DEFAULT_PARAM_ROOT_PATH)

TOOL_PRETRAINING_ROOT_PATH = os.environ.get(
    "TOOL_PRETRAINING_ROOT_PATH", os.path.join(PARAM_ROOT_PATH, "tool_pretraining")
)
SOFT_TOOL_SELECTION_ROOT_PATH = os.environ.get(
    "SOFT_TOOL_SELECTION_ROOT_PATH", os.path.join(PARAM_ROOT_PATH, "soft_tool_selection")
)
TOOL_FINETUNING_ROOT_PATH = os.environ.get(
    "TOOL_FINETUNING_ROOT_PATH", os.path.join(PARAM_ROOT_PATH, "tool_finetuning")
)


DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")

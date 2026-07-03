from __future__ import annotations


def get_dataset(*args, **kwargs):
    from .dataset import get_dataset as _get_dataset

    return _get_dataset(*args, **kwargs)


def __getattr__(name: str):
    if name in {"BFCLDataset", "Dataset", "Language", "StableToolBenchDataset"}:
        from . import dataset as _dataset

        return getattr(_dataset, name)
    raise AttributeError(name)

__all__ = [
    "Dataset",
    "get_dataset",
    "BFCLDataset",
    "StableToolBenchDataset",
    "Language",
]

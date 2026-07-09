"""Bootstrap MultimodalDataset from cached bytecode."""
from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path

import data.meld_context  # noqa: F401

_CACHE_DIR = Path(__file__).resolve().parent / "__pycache__"
_TAG = sys.implementation.cache_tag
_PYC_CANDIDATES = [
    _CACHE_DIR / f"multimodal_dataset_impl.{_TAG}.pyc",
    _CACHE_DIR / "multimodal_dataset_impl.cpython-310.pyc",
    _CACHE_DIR / f"dataset.{_TAG}.pyc",
    _CACHE_DIR / "dataset.cpython-38.pyc",
]
if sys.version_info < (3, 10):
    raise ImportError(
        "MultimodalDataset requires Python 3.10+ (conda env myenv310). "
        f"Current interpreter: {sys.version.split()[0]}"
    )
_pyc_path = next((p for p in _PYC_CANDIDATES if p.is_file() and p.stat().st_size > 4096), None)
if _pyc_path is None:
    raise ImportError(f"Missing dataset bytecode in {_CACHE_DIR}")

with _pyc_path.open("rb") as f:
    f.read(16)
    _code = marshal.load(f)

_impl = types.ModuleType("data._dataset_impl")
_impl.__file__ = str(_pyc_path)
sys.modules["data._dataset_impl"] = _impl
exec(_code, _impl.__dict__)  # noqa: S102

MultimodalDataset = _impl.MultimodalDataset

for _name, _value in _impl.__dict__.items():
    if not _name.startswith("_"):
        globals()[_name] = _value

__all__ = ["MultimodalDataset"]

"""Load model implementation modules from bytecode cache."""
from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path
from types import ModuleType


def load_model_pyc(stub_module: str, pyc_basename: str) -> ModuleType:
    cache_dir = Path(__file__).resolve().parent / "__pycache__"
    candidates = [
        cache_dir / f"{pyc_basename}.cpython-38.pyc",
        cache_dir / f"{pyc_basename}.cpython-310.pyc",
        cache_dir / f"{pyc_basename}.{sys.implementation.cache_tag}.pyc",
    ]
    pyc_path = next((p for p in candidates if p.is_file() and p.stat().st_size > 1024), None)
    if pyc_path is None:
        raise ImportError(f"Missing model bytecode for {pyc_basename} in {cache_dir}")

    impl_name = f"models._pyc_{pyc_basename}"
    if impl_name in sys.modules:
        return sys.modules[impl_name]

    with pyc_path.open("rb") as f:
        f.read(16)
        code = marshal.load(f)

    mod = types.ModuleType(impl_name)
    mod.__file__ = str(pyc_path)
    sys.modules[impl_name] = mod
    exec(code, mod.__dict__)  # noqa: S102
    return mod

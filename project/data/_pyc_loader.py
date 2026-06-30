"""Load compiled implementation from __pycache__ when .py source is unavailable."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_pyc_module(module_name: str, pyc_basename: str) -> ModuleType:
    cache_dir = Path(__file__).resolve().parent / "__pycache__"
    tag = sys.implementation.cache_tag
    pyc_path = cache_dir / f"{pyc_basename}.{tag}.pyc"
    if not pyc_path.is_file():
        raise ImportError(f"Missing bytecode module: {pyc_path}")
    if module_name in sys.modules:
        return sys.modules[module_name]
    loader = importlib.machinery.SourcelessFileLoader(module_name, str(pyc_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f"Cannot create spec for {pyc_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    loader.exec_module(mod)
    return mod

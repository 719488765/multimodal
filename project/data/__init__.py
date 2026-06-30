"""
数据加载模块
"""

__all__ = ["MultimodalDataset", "multimodal_collate_fn"]


def __getattr__(name: str):
    if name == "MultimodalDataset":
        from .dataset import MultimodalDataset

        return MultimodalDataset
    if name == "multimodal_collate_fn":
        from .collate import multimodal_collate_fn

        return multimodal_collate_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

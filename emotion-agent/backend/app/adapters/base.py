from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class EmotionModelAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

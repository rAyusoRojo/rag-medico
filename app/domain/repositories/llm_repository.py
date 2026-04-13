"""Puerto de dominio para el modelo de lenguaje (una sola operación: generar con contexto)."""
from abc import ABC, abstractmethod
from typing import Any


class LLMRepository(ABC):

    @abstractmethod
    def generate(self, question: str, context: str, *, timing: Any = None) -> str:
        raise NotImplementedError

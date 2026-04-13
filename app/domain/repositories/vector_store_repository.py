"""Puerto de dominio: abstrae Chroma (o un almacén en memoria en tests)."""
from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.retrieved_chunk import RetrievedChunk


class VectorStoreRepository(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        sources: list[str] | None = None,
        *,
        timing: Any = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def list_sources(self) -> list[str]:
        raise NotImplementedError

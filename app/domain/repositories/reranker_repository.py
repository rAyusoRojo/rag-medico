"""
Puerto: reordena fragmentos recuperados respecto a la pregunta.

Implementaciones: `LocalCrossEncoderReranker` (modelo HF) y `NoOpReranker` (sin reordenar, solo recorte).
"""
from abc import ABC, abstractmethod

from app.domain.entities.retrieved_chunk import RetrievedChunk


class RerankerRepository(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

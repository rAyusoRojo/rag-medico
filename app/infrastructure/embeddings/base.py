"""Contrato común para calcular embeddings de consulta y de lotes en ingesta."""
from typing import Protocol


class EmbeddingProvider(Protocol):

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

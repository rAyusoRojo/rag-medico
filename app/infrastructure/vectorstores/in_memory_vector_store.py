"""Stub en memoria para pruebas unitarias o demos sin Chroma (respuesta fija)."""
from typing import Any

from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.domain.repositories.vector_store_repository import VectorStoreRepository


class InMemoryVectorStore(VectorStoreRepository):
    def search(
        self,
        query: str,
        top_k: int,
        sources: list[str] | None = None,
        *,
        timing: Any = None,
    ) -> list[RetrievedChunk]:
        _ = (query, top_k, sources, timing)
        return [
            RetrievedChunk(
                content="El higado es un organo vital del metabolismo y detoxificacion.",
                source="anatomia_basica.md",
            )
        ]

    def list_sources(self) -> list[str]:
        return ["anatomia_basica.md"]

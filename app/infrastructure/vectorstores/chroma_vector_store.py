"""
Implementación del repositorio vectorial con Chroma: elige embedder según OPENAI_API_KEY
y usa el mismo sufijo de nombre de colección que `ingest.py` para no mezclar vectores incompatibles.
"""
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.db.vector_store import ChromaVectorStoreDB
from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.domain.repositories.vector_store_repository import VectorStoreRepository
from app.infrastructure.embeddings.base import EmbeddingProvider
from app.infrastructure.embeddings.local_embedder import LocalHashEmbedder
from app.infrastructure.embeddings.openai_embedder import OpenAIEmbedder


@lru_cache
def get_chroma_vector_store_repository() -> "ChromaVectorStoreRepository":
    # Una instancia por proceso: mismo embedder y colección en todas las peticiones.
    return ChromaVectorStoreRepository()


class ChromaVectorStoreRepository(VectorStoreRepository):
    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedder = self._build_embedder()
        collection_name = self._build_collection_name()
        self._db = ChromaVectorStoreDB(embedder=self._embedder, collection_name=collection_name)
        # Si no hay datos y existen .md/.txt, indexado mínimo para no devolver siempre vacío.
        self._db.ensure_index()

    def search(
        self,
        query: str,
        top_k: int,
        sources: list[str] | None = None,
        *,
        timing: Any = None,
    ) -> list[RetrievedChunk]:
        return self._db.search(query=query, top_k=top_k, sources=sources, timing=timing)

    def list_sources(self) -> list[str]:
        return self._db.list_sources()

    def _build_embedder(self) -> EmbeddingProvider:
        if self._settings.openai_api_key.strip():
            return OpenAIEmbedder()
        return LocalHashEmbedder()

    def _build_collection_name(self) -> str:
        # Duplicado deliberado con ingest: cambiar aquí implica cambiar la ingesta.
        base_name = self._settings.chroma_collection.strip() or "rag_medico_docs"
        if self._settings.openai_api_key.strip():
            model = self._settings.embedding_model.strip() or "openai"
            model_slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
            return f"{base_name}__openai__{model_slug}"
        return f"{base_name}__localhash_256"

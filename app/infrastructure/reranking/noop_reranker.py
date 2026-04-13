"""
Implementación de `RerankerRepository` sin modelo: mantiene el orden de similitud vectorial (Chroma)
y devuelve los primeros `top_k` fragmentos.

Se inyecta desde `get_rag_service()` cuando `RERANKING_ENABLED=false` en `.env`, evitando cargar
el cross-encoder y el coste de inferencia en el rerank.
"""
from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.domain.repositories.reranker_repository import RerankerRepository


class NoOpReranker(RerankerRepository):
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        _ = query
        if not chunks:
            return []
        k = max(1, top_k)
        return chunks[:k]

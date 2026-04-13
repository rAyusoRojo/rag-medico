"""
Reranking local con CrossEncoder (sentence-transformers): puntuación query–pasaje y recorte top_k.
El modelo por defecto (`nflechas/spanish-BERT-sts`) está entrenado en similitud textual en español (STS),
no en passage ranking MS MARCO; sirve para reordenar por alineación pregunta–fragmento en corpus en español.
Para rerank multilingüe, configura `CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` en `.env`.
El modelo se carga una vez por proceso vía factory cacheada.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import DEFAULT_CROSS_ENCODER_MODEL, get_settings
from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.domain.repositories.reranker_repository import RerankerRepository


class LocalCrossEncoderReranker(RerankerRepository):
    def __init__(self, model_name: str) -> None:
        # max_length acorde a trozos largos en español (subpalabras); 512 es el estándar de estos modelos.
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        top_k = max(1, top_k)
        pairs = [(query, c.content) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]


@lru_cache
def get_local_cross_encoder_reranker() -> LocalCrossEncoderReranker:
    settings = get_settings()
    model = settings.cross_encoder_model.strip() or DEFAULT_CROSS_ENCODER_MODEL
    return LocalCrossEncoderReranker(model_name=model)

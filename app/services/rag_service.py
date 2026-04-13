"""
Fachada de aplicación para el RAG: ensambla el caso de uso de preguntas con Chroma + OpenAI + reranker.
`get_rag_service` está cacheado para reutilizar la misma instancia (y conexiones) en cada petición HTTP.

Si `RERANKING_ENABLED` es falso, se usa `NoOpReranker` (sin cargar Hugging Face); si es verdadero,
`get_local_cross_encoder_reranker()` (modelo según `CROSS_ENCODER_MODEL`).
"""
from functools import lru_cache

from app.application.use_cases.ask_question import AskQuestionUseCase
from app.core.config import get_settings
from app.infrastructure.llm.openai_llm_repository import OpenAILLMRepository
from app.infrastructure.reranking.local_cross_encoder_reranker import get_local_cross_encoder_reranker
from app.infrastructure.reranking.noop_reranker import NoOpReranker
from app.infrastructure.vectorstores.chroma_vector_store import get_chroma_vector_store_repository
from app.models.qa import AskResponse


class RagService:
    """Delgada capa sobre `AskQuestionUseCase` para inyectar dependencias desde FastAPI."""

    def __init__(self, use_case: AskQuestionUseCase) -> None:
        self._use_case = use_case

    def ask(self, question: str, sources: list[str] | None = None) -> AskResponse:
        return self._use_case.execute(question=question, sources=sources)


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    # Rerank real solo si está habilitado: evita descargar/cargar el cross-encoder en procesos que no lo usan.
    reranker = (
        get_local_cross_encoder_reranker()
        if settings.reranking_enabled
        else NoOpReranker()
    )
    use_case = AskQuestionUseCase(
        vector_store=get_chroma_vector_store_repository(),
        llm=OpenAILLMRepository(),
        reranker=reranker,
    )
    return RagService(use_case=use_case)

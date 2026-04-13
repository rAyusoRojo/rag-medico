"""
Caso de uso con Chroma real y OpenAI (requiere índice con datos y API key si el LLM la usa).

Usa `NoOpReranker` para no cargar el cross-encoder en tests; la ordenación sigue la similitud vectorial.
Comprueba que la leyenda de citas refleja el orden de fragmentos recuperados.
"""
from app.application.use_cases.ask_question import AskQuestionUseCase
from app.infrastructure.llm.openai_llm_repository import OpenAILLMRepository
from app.infrastructure.reranking.noop_reranker import NoOpReranker
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStoreRepository


def test_ask_question_returns_sources() -> None:
    use_case = AskQuestionUseCase(
        vector_store=ChromaVectorStoreRepository(),
        llm=OpenAILLMRepository(),
        reranker=NoOpReranker(),
    )
    result = use_case.execute("Que es el higado?")
    assert result.sources
    assert result.citations
    assert result.citations[0].ref == 1
    assert result.citations[0].document

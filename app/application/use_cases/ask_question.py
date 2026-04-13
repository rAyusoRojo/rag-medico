"""
Caso de uso principal: búsqueda vectorial → rerank (según `RERANKING_ENABLED`) → contexto numerado → LLM.

El recall en Chroma usa `VECTOR_RECALL_K` frente a `VECTOR_TOP_K` solo cuando el rerank está activo;
si no, se pide directamente `VECTOR_TOP_K` candidatos por embedding.
Los índices [1], [2], … en el contexto coinciden con `citations` en la respuesta (trazabilidad).
"""
import logging
from contextlib import nullcontext

from app.core.config import get_settings
from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.domain.repositories.llm_repository import LLMRepository
from app.domain.repositories.reranker_repository import RerankerRepository
from app.domain.repositories.vector_store_repository import VectorStoreRepository
from app.models.qa import AskResponse, ChunkCitation, SourceReference
from app.infrastructure.timing.run_timing_log import RunTimingLog

logger = logging.getLogger(__name__)


class AskQuestionUseCase:
    def __init__(
        self,
        vector_store: VectorStoreRepository,
        llm: LLMRepository,
        reranker: RerankerRepository,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._reranker = reranker
        self._settings = get_settings()

    def execute(self, question: str, sources: list[str] | None = None) -> AskResponse:
        timing: RunTimingLog | None = (
            RunTimingLog("query") if self._settings.timing_logs_enabled else None
        )
        if timing:
            timing.meta(
                reranking_enabled=self._settings.reranking_enabled,
                vector_recall_k=self._settings.vector_recall_k,
                vector_top_k=self._settings.vector_top_k,
                question_chars=len(question),
            )
            if sources:
                timing.meta(sources_filter_count=len(sources))
            logger.info("Log de tiempos (consulta): %s", timing.path)

        chunks_retrieved = 0
        answer_chars = 0
        try:
            # Con rerank: recall amplio y cross-encoder. Sin rerank: solo top_k por embedding (como antes).
            recall_k = (
                max(self._settings.vector_recall_k, self._settings.vector_top_k)
                if self._settings.reranking_enabled
                else self._settings.vector_top_k
            )
            chunks = self._vector_store.search(
                question,
                top_k=recall_k,
                sources=sources,
                timing=timing,
            )
            rerank_cm = (
                timing.step("cross_encoder_rerank")
                if (timing and self._settings.reranking_enabled)
                else nullcontext()
            )
            with rerank_cm:
                chunks = self._reranker.rerank(
                    question,
                    chunks,
                    top_k=self._settings.vector_top_k,
                )
            chunks_retrieved = len(chunks)

            if not chunks:
                answer = (
                    "No se encontro contexto relevante para responder con fiabilidad. "
                    "Reformula la pregunta o amplia la base documental."
                )
                return AskResponse(answer=answer, sources=[], citations=[])

            ctx_cm = timing.step("build_context") if timing else nullcontext()
            with ctx_cm:
                context = self._build_context(chunks)

            raw_answer = self._llm.generate(question=question, context=context, timing=timing)
            answer = raw_answer.strip()
            answer_chars = len(answer)
            citations = self._citation_legend(chunks)
            return AskResponse(
                answer=answer,
                sources=self._collect_sources(chunks),
                citations=citations,
            )
        finally:
            if timing:
                timing.finish(chunks_retrieved=chunks_retrieved, answer_chars=answer_chars)

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        # Formato acordado con el prompt del LLM: cada bloque lleva ref=[n] para citas en la respuesta.
        blocks: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            page_line = (
                f"Pagina: {chunk.page}\n"
                if chunk.page is not None
                else "Pagina: (no aplica / documento de texto)\n"
            )
            blocks.append(
                f"<<< FRAGMENTO ref=[{idx}] >>>\n"
                f"Documento: {chunk.source}\n"
                f"{page_line}"
                f"---\n"
                f"{chunk.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _citation_legend(chunks: list[RetrievedChunk]) -> list[ChunkCitation]:
        return [
            ChunkCitation(ref=i, document=c.source, page=c.page)
            for i, c in enumerate(chunks, start=1)
        ]

    @staticmethod
    def _collect_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
        # Una entrada por par (documento, página) sin duplicar si varios chunks vienen de la misma página.
        seen: set[tuple[str, int | None]] = set()
        ordered: list[SourceReference] = []
        for chunk in chunks:
            key = (chunk.source, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(SourceReference(document=chunk.source, page=chunk.page))
        return ordered

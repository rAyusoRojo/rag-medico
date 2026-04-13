"""
Endpoint de consulta RAG: recibe la pregunta (y opcionalmente fuentes) y devuelve respuesta + citas.
Los errores internos se traducen a HTTP 500 con un mensaje breve para no exponer trazas al cliente.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.qa import AskResponse
from app.services.rag_service import RagService, get_rag_service

router = APIRouter(tags=["rag"])
logger = logging.getLogger(__name__)


@router.get("/ask", response_model=AskResponse)
def ask(
    question: str = Query(min_length=3, max_length=500),
    sources: list[str] | None = Query(
        default=None,
        description="Filtrar por rutas exactas de source (repetir param o varios valores)",
    ),
    rag_service: RagService = Depends(get_rag_service),
) -> AskResponse:
    try:
        return rag_service.ask(question=question, sources=sources)
    except Exception as exc:
        logger.exception("Error no controlado en /ask")
        error_type = type(exc).__name__
        error_message = str(exc).strip().replace("\n", " ")
        concise_message = (error_message[:120] + "...") if len(error_message) > 120 else error_message
        detail = f"Error procesando la consulta ({error_type})"
        if concise_message:
            detail = f"{detail}: {concise_message}"
        raise HTTPException(status_code=500, detail=detail) from exc


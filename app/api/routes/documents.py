"""
Lista los valores `source` presentes en Chroma (rutas relativas a la carpeta de documentos).
La UI usa esto para checkboxes de filtrado en /ask.
"""
from fastapi import APIRouter, Depends

from app.infrastructure.vectorstores.chroma_vector_store import (
    ChromaVectorStoreRepository,
    get_chroma_vector_store_repository,
)
from app.models.documents import DocumentListResponse

router = APIRouter(tags=["rag"])


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    vector_store: ChromaVectorStoreRepository = Depends(get_chroma_vector_store_repository),
) -> DocumentListResponse:
    return DocumentListResponse(sources=vector_store.list_sources())

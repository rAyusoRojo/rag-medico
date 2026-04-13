"""Respuesta de GET /documents: lista de `source` conocidos en el índice."""
from pydantic import BaseModel, Field


class DocumentListResponse(BaseModel):
    sources: list[str] = Field(description="Rutas relativas de documentos indexados (campo source)")

"""
Esquemas de respuesta HTTP para /ask: respuesta, fuentes únicas y leyenda alineada con ref=[n] del contexto.
"""
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    document: str = Field(description="Ruta relativa del documento (campo source en el indice)")
    page: int | None = Field(default=None, description="Numero de pagina si el fragmento proviene de un PDF")


class ChunkCitation(BaseModel):
    ref: int = Field(ge=1, description="Indice [n] del fragmento en el contexto enviado al modelo")
    document: str = Field(description="Ruta relativa del documento de ese fragmento")
    page: int | None = Field(default=None, description="Pagina del PDF para ese fragmento, si aplica")


class AskResponse(BaseModel):
    answer: str = Field(description="Respuesta generada por el sistema RAG")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Fuentes unicas (documento y pagina) entre los fragmentos recuperados",
    )
    citations: list[ChunkCitation] = Field(
        default_factory=list,
        description="Leyenda ordenada: cada elemento coincide con [n] en el contexto y en la respuesta",
    )

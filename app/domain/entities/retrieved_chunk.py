"""Fragmento recuperado del vector store antes de pasarlo al LLM (texto + procedencia)."""
from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    source: str
    page: int | None = None
    chunk_id: str | None = None

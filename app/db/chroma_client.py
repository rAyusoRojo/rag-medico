"""
Cliente Chroma persistente: los vectores viven en disco bajo CHROMA_PERSIST_DIRECTORY.
La similaridad usa coseno, alineada con cómo se suelen normalizar embeddings OpenAI.
"""
import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings


def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_directory)


def get_chroma_collection(collection_name: str | None = None) -> Collection:
    """
    Colección con distancia HNSW coseno (coherente con embeddings normalizados).

    `collection_name` permite anular el nombre lógico (p. ej. sufijo por modelo de embedding
    en `ChromaVectorStoreRepository` para no mezclar vectores incompatibles con la ingesta).
    """
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name or settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

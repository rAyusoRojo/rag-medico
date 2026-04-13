"""
Acceso de bajo nivel a Chroma para el repositorio vectorial: consulta por embedding y filtro `source`.

Incluye `ensure_index`: relleno mínimo desde .md/.txt si la colección está vacía (camino distinto
a la ingesta principal en `ingest.py`, que indexa PDF/docx con otro esquema de ids).
"""
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection

from app.core.config import get_settings
from app.db.chroma_client import get_chroma_collection
from app.domain.entities.retrieved_chunk import RetrievedChunk
from app.infrastructure.embeddings.base import EmbeddingProvider


def _metadata_page(metadata: dict | None) -> int | None:
    if not metadata or not isinstance(metadata, dict):
        return None
    raw = metadata.get("page")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


class ChromaVectorStoreDB:
    def __init__(self, embedder: EmbeddingProvider, collection_name: str | None = None) -> None:
        self._settings = get_settings()
        self._embedder = embedder
        self._collection: Collection = get_chroma_collection(collection_name=collection_name)

    def search(
        self,
        query: str,
        top_k: int,
        sources: list[str] | None = None,
        *,
        timing: Any = None,
    ) -> list[RetrievedChunk]:
        # Chroma devuelve los `n_results` más cercanos en espacio de embeddings (coseno).
        top_k = max(1, top_k)
        if timing is not None:
            with timing.step("embedding_query"):
                query_embedding = self._embedder.embed_query(query)
        else:
            query_embedding = self._embedder.embed_query(query)
        where = self._sources_where_clause(sources)
        if timing is not None:
            with timing.step("chroma_query"):
                result = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas"],
                    where=where,
                )
        else:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas"],
                where=where,
            )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids_list = result.get("ids", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for idx, document in enumerate(documents):
            if not document or not str(document).strip():
                continue
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            meta_dict = metadata if isinstance(metadata, dict) else {}
            source = str(meta_dict.get("source", "fuente_desconocida"))
            page = _metadata_page(meta_dict)
            chunk_id = ids_list[idx] if idx < len(ids_list) else None
            chunks.append(RetrievedChunk(
                content=str(document).strip(),
                source=source,
                page=page,
                chunk_id=chunk_id,
            ))
        return chunks

    def list_sources(self) -> list[str]:
        # Recorre metadatos de todos los puntos (puede crecer con el corpus).
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        found: set[str] = set()
        for meta in metadatas:
            if not meta or not isinstance(meta, dict):
                continue
            src = meta.get("source")
            if src:
                found.add(str(src))
        return sorted(found)

    @staticmethod
    def _sources_where_clause(sources: list[str] | None) -> dict | None:
        if not sources:
            return None
        paths = [s.strip() for s in sources if s and str(s).strip()]
        if not paths:
            return None
        if len(paths) == 1:
            return {"source": paths[0]}
        return {"source": {"$in": paths}}

    def ensure_index(self) -> None:
        # Arranque rápido sin ingesta: solo si la colección está vacía y hay .md/.txt en documents_path.
        if self._collection.count() > 0:
            return

        docs_dir = Path(self._settings.documents_path)
        if not docs_dir.exists():
            return

        files = [
            path
            for ext in ("*.md", "*.txt")
            for path in docs_dir.rglob(ext)
            if path.is_file()
        ]
        if not files:
            return

        chunk_size = max(100, self._settings.chunk_size)
        overlap = max(0, min(self._settings.chunk_overlap, chunk_size - 1))
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            chunks = self._chunk_text(text, chunk_size, overlap)
            rel_source = str(file_path.relative_to(docs_dir))
            for index, chunk in enumerate(chunks):
                ids.append(f"{rel_source}:{index}")
                documents.append(chunk)
                metadatas.append({"source": rel_source})

        if not documents:
            return

        embeddings = self._embedder.embed_texts(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
        chunks: list[str] = []
        start = 0
        step = size - overlap
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += step
        return chunks

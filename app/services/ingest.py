"""
Ingesta batch: lee ficheros bajo DOCUMENTS_PATH, trocea texto, calcula embeddings
y hace upsert en Chroma. Omite ficheros cuyo hash no cambió (misma firma que en metadatos).
También exporta artefactos de evaluación (manifiesto + índice de chunks) tras indexar.
"""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import logging
import sys
import uuid
from datetime import datetime, timezone

from docx import Document as DocxDocument
from pypdf import PdfReader

# Permite ejecutar este script como `python app/services/ingest.py` sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pypdf emite ERROR en algunas fuentes con codificación no soportada; la ingesta sigue pero el texto de esas páginas puede faltar.
logging.getLogger("pypdf._cmap").setLevel(logging.CRITICAL)

from app.core.config import get_settings
from app.db.chroma_client import get_chroma_collection
from app.infrastructure.embeddings.base import EmbeddingProvider
from app.infrastructure.embeddings.local_embedder import LocalHashEmbedder
from app.infrastructure.embeddings.openai_embedder import OpenAIEmbedder
from app.services.ingest_eval_export import export_chroma_eval_artifacts
from app.infrastructure.timing.run_timing_log import RunTimingLog


# Límite de registros por llamada a upsert para no saturar memoria en corpus grandes.
UPSERT_BATCH_SIZE = 1000
SUPPORTED_PATTERNS = ("*.pdf", "*.txt", "*.docx", "*.doc")


def _build_embedder() -> EmbeddingProvider:
    settings = get_settings()
    if settings.openai_api_key.strip():
        return OpenAIEmbedder()
    return LocalHashEmbedder()


def _build_collection_name() -> str:
    # Debe coincidir con ChromaVectorStoreRepository: embeddings distintos → vectores incompatibles → colección distinta.
    settings = get_settings()
    base_name = settings.chroma_collection.strip() or "rag_medico_docs"
    if settings.openai_api_key.strip():
        model = settings.embedding_model.strip() or "openai"
        model_slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
        return f"{base_name}__openai__{model_slug}"
    return f"{base_name}__localhash_256"


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    # Ventana fija con solapamiento para no perder contexto en los bordes entre trozos.
    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += step
    return chunks


def _list_candidate_files(documents_path: Path) -> list[Path]:
    candidates = [
        path
        for pattern in SUPPORTED_PATTERNS
        for path in documents_path.rglob(pattern)
        if path.is_file()
    ]
    # Orden estable entre ejecuciones (útil para depurar y comparar logs).
    return sorted(set(candidates), key=lambda p: str(p))


def _file_signature(file_path: Path) -> str:
    # Hash del fichero completo: si cambia cualquier byte, se reindexa ese source.
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_signatures(collection: object, source: str) -> set[str]:
    # Chroma puede devolver varias filas por source; todas deberían compartir el mismo file_sig.
    result = collection.get(where={"source": source}, include=["metadatas"])
    metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
    signatures: set[str] = set()
    for item in metadatas or []:
        if isinstance(item, dict):
            value = item.get("file_sig")
            if isinstance(value, str) and value:
                signatures.add(value)
    return signatures


def _extract_pdf_units(file_path: Path) -> list[tuple[str, int, str]]:
    # Una "unidad" por página con texto: permite metadata `page` alineada con el PDF.
    units: list[tuple[str, int, str]] = []
    reader = PdfReader(file_path)
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            units.append(("pdf", page_idx, page_text))
    return units


def _extract_text_units(file_path: Path) -> list[tuple[str, int, str]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    return [("text", 1, text)] if text else []


def _extract_word_units(file_path: Path) -> list[tuple[str, int, str]]:
    try:
        doc = DocxDocument(str(file_path))
    except Exception:
        # .doc binario antiguo no lo soporta python-docx; se omite en silencio.
        return []
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()
    return [("text", 1, text)] if text else []


def _extract_units(file_path: Path) -> list[tuple[str, int, str]]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_units(file_path)
    if suffix == ".txt":
        return _extract_text_units(file_path)
    if suffix in {".docx", ".doc"}:
        return _extract_word_units(file_path)
    return []


def ingest_documents() -> int:
    """Devuelve el número de chunks escritos en esta ejecución, o 0 si no hubo nada que indexar."""
    settings = get_settings()
    timing: RunTimingLog | None = RunTimingLog("ingest") if settings.timing_logs_enabled else None
    if timing:
        timing.meta(
            documents_path=settings.documents_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_model=settings.embedding_model,
        )
        print(f"Log de tiempos (ingesta): {timing.path}")

    ret = 0
    files_processed = 0
    ingestion_batch_id = ""
    try:
        documents_path = Path(settings.documents_path)
        if not documents_path.exists():
            if timing:
                timing.meta(abort="documents_path_inexistente")
            return 0

        chunk_size = max(100, settings.chunk_size)
        overlap = max(0, min(settings.chunk_overlap, chunk_size - 1))

        if timing:
            with timing.step("chroma_get_collection"):
                collection = get_chroma_collection(collection_name=_build_collection_name())
        else:
            collection = get_chroma_collection(collection_name=_build_collection_name())

        if timing:
            with timing.step("listar_ficheros_candidatos"):
                candidate_files = _list_candidate_files(documents_path)
        else:
            candidate_files = _list_candidate_files(documents_path)
        if not candidate_files:
            if timing:
                timing.meta(abort="sin_ficheros_candidatos")
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        ingestion_batch_id = str(uuid.uuid4())
        ingested_at = datetime.now(timezone.utc).isoformat()

        scan_cm = timing.step("escaneo_extraccion_chunking") if timing else nullcontext()
        with scan_cm:
            for file_path in candidate_files:
                source = str(file_path.relative_to(documents_path))
                signature = _file_signature(file_path)
                existing_signatures = _source_signatures(collection=collection, source=source)
                if existing_signatures == {signature}:
                    continue
                if existing_signatures:
                    collection.delete(where={"source": source})
                units = _extract_units(file_path=file_path)
                if not units:
                    continue
                files_processed += 1
                for unit_type, unit_index, text in units:
                    raw_chunks = _chunk_text(text=text, size=chunk_size, overlap=overlap)
                    non_empty = [(idx, c.strip()) for idx, c in enumerate(raw_chunks) if c.strip()]
                    unit_chunk_total = len(non_empty)
                    for unit_chunk_ordinal, (idx, chunk_text) in enumerate(non_empty):
                        base_meta: dict[str, str | int] = {
                            "source": source,
                            "chunk": idx,
                            "file_sig": signature,
                            "ingestion_batch_id": ingestion_batch_id,
                            "ingested_at": ingested_at,
                            "unit_chunk_total": unit_chunk_total,
                            "unit_chunk_ordinal": unit_chunk_ordinal,
                        }
                        if unit_type == "pdf":
                            chunk_id = f"{source}:p{unit_index}:c{idx}"
                            base_meta["page"] = unit_index
                        else:
                            chunk_id = f"{source}:c{idx}"
                        ids.append(chunk_id)
                        metadatas.append(base_meta)
                        documents.append(chunk_text)

        if not documents:
            print("No hay ficheros nuevos o modificados para ingestar.")
            if timing:
                timing.meta(abort="sin_chunks_nuevos")
            return 0

        embedder = _build_embedder()
        emb_cm = timing.step("openai_embeddings_embed_texts") if timing else nullcontext()
        with emb_cm:
            embeddings = embedder.embed_texts(documents)

        upsert_cm = timing.step("chroma_upsert_batches") if timing else nullcontext()
        with upsert_cm:
            for start in range(0, len(documents), UPSERT_BATCH_SIZE):
                end = start + UPSERT_BATCH_SIZE
                collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )

        collection_name = _build_collection_name()
        try:
            exp_cm = timing.step("export_eval_artifacts") if timing else nullcontext()
            with exp_cm:
                out_dir = export_chroma_eval_artifacts(
                    collection,
                    settings,
                    collection_name,
                    ingestion_batch_id,
                    export_chunk_index=settings.eval_export_chunk_index,
                )
            print(f"Artefactos de evaluacion: {out_dir / 'ingestion_manifest.json'}")
            if settings.eval_export_chunk_index:
                print(f"Indice de chunks: {out_dir / 'chunks_index.jsonl'}")
            print(f"Revision de chunks (texto completo): {out_dir / 'chunks_review.json'}")
        except Exception:
            logging.exception("Fallo al exportar artefactos de evaluacion (la ingesta en Chroma se completo).")

        ret = len(documents)
        print(f"Ficheros procesados: {files_processed}")
        return ret
    finally:
        if timing:
            timing.finish(
                chunks_indexed=ret,
                files_processed=files_processed,
                ingestion_batch_id=ingestion_batch_id or None,
            )


if __name__ == "__main__":
    ingested = ingest_documents()
    print(f"Ingestion completada. Chunks indexados: {ingested}")
    if ingested > 0:
        print(
            "Nota: Chroma en disco no se refresca en el proceso de la API si ya estaba "
            "en marcha. Reinicia el contenedor para que /ask use el indice nuevo:\n"
            "  docker compose restart rag-medico"
        )
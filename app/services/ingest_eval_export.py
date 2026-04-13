"""
Exporta artefactos para evaluar recuperación y cobertura tras la ingesta.

Lee **toda** la colección Chroma tras un upsert y escribe:
- `ingestion_manifest.json`: resumen por documento y parámetros efectivos de indexación.
- `chunks_index.jsonl` (opcional): una fila por chunk con vista previa del texto para montar tests.
- `chunks_review.json`: texto completo de cada chunk agrupado por fuente, para revisión manual.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings


def export_chroma_eval_artifacts(
    collection: Any,
    settings: "Settings",
    collection_name: str,
    ingestion_batch_id: str,
    *,
    export_chunk_index: bool,
) -> Path:
    """
    Escribe `ingestion_manifest.json` y, si aplica, `chunks_index.jsonl` en eval_artifacts_path.
    Devuelve el directorio de salida.
    """
    out_dir = Path(settings.eval_artifacts_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vista completa del índice: necesaria para un manifiesto fiel tras ingesta parcial o total.
    result = collection.get(include=["metadatas", "documents"])
    ids_list = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    by_source: dict[str, dict[str, Any]] = {}
    for i, _cid in enumerate(ids_list):
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            continue
        src = str(meta.get("source", "") or "").strip()
        if not src:
            continue
        if src not in by_source:
            by_source[src] = {"chunk_count": 0, "file_sig": meta.get("file_sig")}
        by_source[src]["chunk_count"] += 1

    # Etiqueta legible: si no hay API key, la app usa hash local (solo desarrollo/pruebas sin OpenAI).
    embed_label = (
        (settings.embedding_model.strip() or "openai")
        if settings.openai_api_key.strip()
        else "local_hash"
    )

    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "ingestion_batch_id": ingestion_batch_id,
        "exported_at": now,
        "collection_name": collection_name,
        "chroma_persist_directory": settings.chroma_persist_directory,
        "documents_path": settings.documents_path,
        "embedding_model": embed_label,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "total_sources": len(by_source),
        "total_chunks": len(ids_list),
        "sources": sorted(by_source.keys()),
        "per_source": {
            k: {"chunk_count": v["chunk_count"], "file_sig": v.get("file_sig")}
            for k, v in sorted(by_source.items())
        },
    }

    manifest_path = out_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if export_chunk_index and ids_list:
        jsonl_path = out_dir / "chunks_index.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for i, cid in enumerate(ids_list):
                meta = metas[i] if i < len(metas) else {}
                doc_text = docs[i] if i < len(docs) else ""
                if not isinstance(meta, dict):
                    meta = {}
                preview = (doc_text or "")[:280].replace("\n", " ")
                row = {
                    "chunk_id": cid,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "chunk": meta.get("chunk"),
                    "file_sig": meta.get("file_sig"),
                    "ingestion_batch_id": meta.get("ingestion_batch_id"),
                    "unit_chunk_total": meta.get("unit_chunk_total"),
                    "unit_chunk_ordinal": meta.get("unit_chunk_ordinal"),
                    "char_len": len(doc_text or ""),
                    "text_preview": preview,
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    _export_chunks_review(ids_list, docs, metas, manifest, out_dir)

    return out_dir


def _export_chunks_review(
    ids_list: list[str],
    docs: list[str],
    metas: list[dict],
    manifest: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Escribe ``chunks_review.json`` con texto completo de cada chunk agrupado por fuente."""
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for i, cid in enumerate(ids_list):
        meta = metas[i] if i < len(metas) else {}
        doc_text = docs[i] if i < len(docs) else ""
        if not isinstance(meta, dict):
            meta = {}
        source = str(meta.get("source", "") or "").strip() or "(desconocido)"
        entry: dict[str, Any] = {
            "chunk_id": cid,
            "chunk_index": meta.get("chunk"),
            "chunk_ordinal": meta.get("unit_chunk_ordinal"),
            "unit_chunk_total": meta.get("unit_chunk_total"),
            "char_count": len(doc_text or ""),
            "text": doc_text or "",
        }
        page = meta.get("page")
        if page is not None:
            entry["page"] = page
        chunks_by_source.setdefault(source, []).append(entry)

    for chunks in chunks_by_source.values():
        chunks.sort(key=lambda c: (c.get("page") or 0, c.get("chunk_ordinal") or 0))

    review: dict[str, Any] = {
        "meta": {
            "exported_at": manifest.get("exported_at"),
            "ingestion_batch_id": manifest.get("ingestion_batch_id"),
            "collection_name": manifest.get("collection_name"),
            "embedding_model": manifest.get("embedding_model"),
            "chunk_size": manifest.get("chunk_size"),
            "chunk_overlap": manifest.get("chunk_overlap"),
            "total_sources": len(chunks_by_source),
            "total_chunks": len(ids_list),
        },
        "sources": {
            src: {"chunk_count": len(chunks), "chunks": chunks}
            for src, chunks in sorted(chunks_by_source.items())
        },
    }

    review_path = out_dir / "chunks_review.json"
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return review_path

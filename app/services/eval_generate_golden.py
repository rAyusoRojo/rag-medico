"""
Genera un golden dataset sintético para evaluar retrieval.

Lee todos los chunks indexados en Chroma, samplea N al azar y pide a GPT que genere
1-2 preguntas médicas realistas que cada chunk respondería. El ground truth es el
chunk_id que originó la pregunta.

Uso:
    python app/services/eval_generate_golden.py [--sample 60] [--questions-per-chunk 1]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from app.core.config import get_settings
from app.db.chroma_client import get_chroma_collection

logger = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 80


def _build_collection_name() -> str:
    settings = get_settings()
    base_name = settings.chroma_collection.strip() or "rag_medico_docs"
    if settings.openai_api_key.strip():
        model = settings.embedding_model.strip() or "openai"
        model_slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
        return f"{base_name}__openai__{model_slug}"
    return f"{base_name}__localhash_256"


def _load_chunks_from_chroma() -> list[dict[str, Any]]:
    """Devuelve todos los chunks de la colección con id, texto y metadata."""
    collection = get_chroma_collection(collection_name=_build_collection_name())
    result = collection.get(include=["documents", "metadatas"])
    ids_list: list[str] = result.get("ids") or []
    docs: list[str] = result.get("documents") or []
    metas: list[dict] = result.get("metadatas") or []

    chunks: list[dict[str, Any]] = []
    for i, chunk_id in enumerate(ids_list):
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        if not text or len(text.strip()) < _MIN_CHUNK_CHARS:
            continue
        chunks.append({
            "chunk_id": chunk_id,
            "text": text.strip(),
            "source": meta.get("source", ""),
            "page": meta.get("page"),
        })
    return chunks


def _generate_questions(
    client: OpenAI,
    model: str,
    chunk: dict[str, Any],
    questions_per_chunk: int,
) -> list[str]:
    """Pide a GPT que genere preguntas médicas a partir de un fragmento."""
    n_label = "una pregunta" if questions_per_chunk == 1 else f"{questions_per_chunk} preguntas"
    prompt = (
        f"A partir del siguiente fragmento de texto médico, genera {n_label} "
        "que un estudiante o profesional de salud haría y que este fragmento responde directamente.\n"
        "Reglas:\n"
        "- Preguntas concretas (no genéricas como '¿Qué dice el texto?').\n"
        "- La pregunta debe ser autocontenida: NO puede referenciar 'el texto', 'el fragmento', "
        "'la estructura mencionada', 'según el texto' ni ninguna variante similar.\n"
        "- Incluye el nombre específico del concepto anatómico, fisiológico o médico.\n"
        "- La pregunta debe poder entenderse sin haber leído el fragmento.\n"
        "- En español.\n"
        "- Solo texto de la pregunta, sin numeración ni viñetas.\n"
        f"- Devuelve exactamente {questions_per_chunk} línea(s), una pregunta por línea.\n\n"
        f"FRAGMENTO:\n{chunk['text']}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = (response.choices[0].message.content or "").strip()
    lines = [ln.strip().lstrip("0123456789.-) ") for ln in raw.splitlines() if ln.strip()]
    return lines[:questions_per_chunk]


def generate_golden_dataset(
    *,
    sample_size: int = 60,
    questions_per_chunk: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise RuntimeError("Se necesita OPENAI_API_KEY para generar el golden dataset.")

    client = OpenAI(api_key=settings.openai_api_key.strip())
    model = settings.openai_model

    print("Cargando chunks de Chroma…")
    all_chunks = _load_chunks_from_chroma()
    if not all_chunks:
        raise RuntimeError("La colección Chroma está vacía. Ejecuta la ingesta primero.")
    print(f"  Chunks disponibles: {len(all_chunks)}")

    rng = random.Random(seed)
    sample = rng.sample(all_chunks, min(sample_size, len(all_chunks)))
    print(f"  Chunks seleccionados: {len(sample)}")

    dataset: list[dict[str, Any]] = []
    for idx, chunk in enumerate(sample):
        questions = _generate_questions(client, model, chunk, questions_per_chunk)
        for q_idx, question in enumerate(questions):
            entry_id = f"q_{idx:04d}" if len(questions) == 1 else f"q_{idx:04d}_{q_idx}"
            dataset.append({
                "id": entry_id,
                "question": question,
                "relevant_chunk_ids": [chunk["chunk_id"]],
                "relevant_sources": [chunk["source"]] if chunk["source"] else [],
                "relevant_pages": [chunk["page"]] if chunk["page"] is not None else [],
                "generated_from_chunk": chunk["chunk_id"],
            })
        print(f"  [{idx + 1}/{len(sample)}] {len(questions)} pregunta(s) generada(s) para {chunk['chunk_id']}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera golden dataset sintético para eval de retrieval")
    parser.add_argument("--sample", type=int, default=60, help="Chunks a samplear (default 60)")
    parser.add_argument("--questions-per-chunk", type=int, default=1, help="Preguntas por chunk (default 1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidad")
    parser.add_argument("--output", type=str, default=None, help="Ruta de salida (default data/eval/golden_dataset.json)")
    args = parser.parse_args()

    settings = get_settings()
    out_path = Path(args.output) if args.output else Path(settings.eval_artifacts_path) / "golden_dataset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_golden_dataset(
        sample_size=args.sample,
        questions_per_chunk=args.questions_per_chunk,
        seed=args.seed,
    )

    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGolden dataset guardado en {out_path}  ({len(dataset)} preguntas)")


if __name__ == "__main__":
    main()

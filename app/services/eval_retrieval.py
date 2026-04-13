"""
Harness de evaluación de retrieval: carga un golden dataset, ejecuta cada pregunta
contra el vector store (con y sin rerank) y calcula Hit Rate@K, MRR y Precision@K.

Uso:
    python app/services/eval_retrieval.py [--golden data/eval/golden_dataset.json]
                                          [--top-k 1,3,5,10]
                                          [--with-rerank]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.db.vector_store import ChromaVectorStoreDB
from app.infrastructure.embeddings.local_embedder import LocalHashEmbedder
from app.infrastructure.embeddings.openai_embedder import OpenAIEmbedder
from app.infrastructure.reranking.local_cross_encoder_reranker import get_local_cross_encoder_reranker
from app.infrastructure.reranking.noop_reranker import NoOpReranker


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_embedder():
    settings = get_settings()
    if settings.openai_api_key.strip():
        return OpenAIEmbedder()
    return LocalHashEmbedder()


def _build_collection_name() -> str:
    settings = get_settings()
    base_name = settings.chroma_collection.strip() or "rag_medico_docs"
    if settings.openai_api_key.strip():
        model = settings.embedding_model.strip() or "openai"
        model_slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
        return f"{base_name}__openai__{model_slug}"
    return f"{base_name}__localhash_256"


# ── métricas ─────────────────────────────────────────────────────────────────


def _hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1.0 si al menos un chunk relevante está en los recuperados, 0.0 si no."""
    for rid in retrieved_ids:
        if rid in relevant_ids:
            return 1.0
    return 0.0


def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/posición del primer chunk relevante (1-indexed), o 0.0 si no aparece."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fracción de chunks recuperados que son relevantes."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return hits / len(retrieved_ids)


# ── evaluación ───────────────────────────────────────────────────────────────


def evaluate(
    golden_path: Path,
    top_k_values: list[int],
    use_rerank: bool,
) -> dict[str, Any]:
    """Ejecuta la evaluación completa y devuelve el reporte como dict."""
    settings = get_settings()

    with open(golden_path, encoding="utf-8") as f:
        golden: list[dict[str, Any]] = json.load(f)

    if not golden:
        raise RuntimeError(f"El golden dataset en {golden_path} está vacío.")

    embedder = _build_embedder()
    db = ChromaVectorStoreDB(embedder=embedder, collection_name=_build_collection_name())
    reranker = get_local_cross_encoder_reranker() if use_rerank else NoOpReranker()
    max_k = max(top_k_values)
    recall_k = max(settings.vector_recall_k, max_k) if use_rerank else max_k

    per_query: list[dict[str, Any]] = []

    for idx, entry in enumerate(golden):
        question = entry["question"]
        relevant_ids = set(entry.get("relevant_chunk_ids", []))

        raw_chunks = db.search(query=question, top_k=recall_k)
        reranked_chunks = reranker.rerank(question, raw_chunks, top_k=max_k)
        retrieved_ids = [c.chunk_id for c in reranked_chunks if c.chunk_id]

        query_metrics: dict[str, Any] = {"id": entry["id"], "question": question}
        for k in top_k_values:
            top_ids = retrieved_ids[:k]
            query_metrics[f"hit_rate@{k}"] = _hit_rate(top_ids, relevant_ids)
            query_metrics[f"mrr@{k}"] = _reciprocal_rank(top_ids, relevant_ids)
            query_metrics[f"precision@{k}"] = _precision_at_k(top_ids, relevant_ids)

        query_metrics["retrieved_ids"] = retrieved_ids[:max_k]
        query_metrics["relevant_ids"] = list(relevant_ids)
        per_query.append(query_metrics)

        status = "HIT" if _hit_rate(retrieved_ids[:max_k], relevant_ids) > 0 else "MISS"
        print(f"  [{idx + 1}/{len(golden)}] {status}  {question[:80]}")

    aggregated = _aggregate(per_query, top_k_values)

    report: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(golden_path),
        "total_queries": len(golden),
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model if settings.openai_api_key.strip() else "local_hash",
            "reranking": use_rerank,
            "cross_encoder_model": settings.cross_encoder_model if use_rerank else None,
            "vector_recall_k": recall_k,
            "top_k_values": top_k_values,
        },
        "results": aggregated,
        "per_query": per_query,
    }
    return report


def _aggregate(
    per_query: list[dict[str, Any]],
    top_k_values: list[int],
) -> dict[str, Any]:
    n = len(per_query)
    if n == 0:
        return {}
    agg: dict[str, Any] = {}
    for k in top_k_values:
        agg[f"hit_rate@{k}"] = round(sum(q[f"hit_rate@{k}"] for q in per_query) / n, 4)
        agg[f"mrr@{k}"] = round(sum(q[f"mrr@{k}"] for q in per_query) / n, 4)
        agg[f"precision@{k}"] = round(sum(q[f"precision@{k}"] for q in per_query) / n, 4)
    return agg


def _print_summary(report: dict[str, Any]) -> None:
    results = report["results"]
    cfg = report["config"]
    rerank_label = f"rerank={cfg['reranking']}"
    print(f"\n{'=' * 60}")
    print(f"  RETRIEVAL EVAL — {report['total_queries']} queries")
    print(f"  chunk_size={cfg['chunk_size']}  overlap={cfg['chunk_overlap']}  {rerank_label}")
    print(f"  embedding={cfg['embedding_model']}")
    print(f"{'=' * 60}")
    print(f"  {'Métrica':<20} {'Valor':>10}")
    print(f"  {'-' * 30}")
    for key in sorted(results.keys()):
        print(f"  {key:<20} {results[key]:>10.4f}")
    print(f"{'=' * 60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de retrieval con golden dataset")
    parser.add_argument(
        "--golden", type=str, default=None,
        help="Ruta al golden dataset (default data/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--top-k", type=str, default="1,3,5,10",
        help="Valores de K separados por coma (default 1,3,5,10)",
    )
    parser.add_argument(
        "--with-rerank", action="store_true", default=False,
        help="Activar reranking con CrossEncoder",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta del reporte (default data/eval/eval_report.json)",
    )
    args = parser.parse_args()

    settings = get_settings()
    golden_path = Path(args.golden) if args.golden else Path(settings.eval_artifacts_path) / "golden_dataset.json"
    top_k_values = [int(v.strip()) for v in args.top_k.split(",") if v.strip()]
    out_path = Path(args.output) if args.output else Path(settings.eval_artifacts_path) / "eval_report.json"

    if not golden_path.exists():
        print(f"ERROR: No se encontró el golden dataset en {golden_path}")
        print("Ejecuta primero:  python app/services/eval_generate_golden.py")
        sys.exit(1)

    print(f"Golden dataset: {golden_path}")
    print(f"Top-K values:   {top_k_values}")
    print(f"Reranking:      {args.with_rerank}")
    print()

    report = evaluate(
        golden_path=golden_path,
        top_k_values=top_k_values,
        use_rerank=args.with_rerank,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en {out_path}")
    _print_summary(report)


if __name__ == "__main__":
    main()

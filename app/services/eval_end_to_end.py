"""
Evaluación end-to-end del sistema RAG: ejecuta el pipeline completo (retrieval + LLM)
para cada pregunta del golden dataset y usa GPT como juez (LLM-as-judge) para puntuar
Faithfulness, Answer Relevancy y Correctness en escala 1-5.

Uso:
    python app/services/eval_end_to_end.py [--golden data/eval/golden_dataset.json]
                                           [--output data/eval/eval_e2e_report.json]
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

from openai import OpenAI

from app.core.config import get_settings
from app.application.use_cases.ask_question import AskQuestionUseCase
from app.infrastructure.llm.openai_llm_repository import OpenAILLMRepository
from app.infrastructure.reranking.noop_reranker import NoOpReranker
from app.infrastructure.reranking.local_cross_encoder_reranker import get_local_cross_encoder_reranker
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStoreRepository
from app.db.chroma_client import get_chroma_collection


# ── ground truth loader ──────────────────────────────────────────────────────


def _build_collection_name() -> str:
    settings = get_settings()
    base_name = settings.chroma_collection.strip() or "rag_medico_docs"
    if settings.openai_api_key.strip():
        model = settings.embedding_model.strip() or "openai"
        model_slug = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
        return f"{base_name}__openai__{model_slug}"
    return f"{base_name}__localhash_256"


def _load_ground_truth_texts(golden: list[dict[str, Any]]) -> dict[str, str]:
    """Carga el texto real de cada chunk ground-truth desde Chroma, indexado por chunk_id."""
    collection = get_chroma_collection(collection_name=_build_collection_name())
    all_chunk_ids: list[str] = []
    for entry in golden:
        all_chunk_ids.extend(entry.get("relevant_chunk_ids", []))
    all_chunk_ids = list(set(all_chunk_ids))
    if not all_chunk_ids:
        return {}

    result = collection.get(ids=all_chunk_ids, include=["documents"])
    ids_list = result.get("ids") or []
    docs = result.get("documents") or []
    return {ids_list[i]: docs[i] for i in range(len(ids_list)) if i < len(docs) and docs[i]}


# ── LLM judge ───────────────────────────────────────────────────────────────


_JUDGE_SYSTEM = (
    "Eres un evaluador experto de sistemas de pregunta-respuesta médicos. "
    "Evalúas la calidad de las respuestas de forma objetiva y estricta. "
    "Responde SOLO con JSON válido, sin markdown ni texto adicional."
)


def _judge_faithfulness(client: OpenAI, model: str, answer: str, context: str) -> dict[str, Any]:
    prompt = (
        "Evalúa la FIDELIDAD de la respuesta respecto al contexto proporcionado.\n"
        "¿La respuesta contiene SOLO información presente en el contexto? "
        "¿Inventa datos, dosis, diagnósticos o hechos que no están en el contexto?\n\n"
        f"CONTEXTO RECUPERADO:\n{context}\n\n"
        f"RESPUESTA DEL SISTEMA:\n{answer}\n\n"
        "Responde con JSON: {\"score\": <1-5>, \"reason\": \"<explicación breve>\"}\n"
        "1=inventa todo, 2=inventa datos clave, 3=mayormente fiel con alguna inferencia, "
        "4=fiel con mínimas libertades, 5=100% basado en el contexto."
    )
    return _call_judge(client, model, prompt)


def _judge_relevancy(client: OpenAI, model: str, question: str, answer: str) -> dict[str, Any]:
    prompt = (
        "Evalúa la RELEVANCIA de la respuesta respecto a la pregunta.\n"
        "¿La respuesta contesta directamente lo que se pregunta? "
        "¿Es completa o deja aspectos sin responder?\n\n"
        f"PREGUNTA:\n{question}\n\n"
        f"RESPUESTA DEL SISTEMA:\n{answer}\n\n"
        "Responde con JSON: {\"score\": <1-5>, \"reason\": \"<explicación breve>\"}\n"
        "1=no responde la pregunta, 2=tangencial, 3=responde parcialmente, "
        "4=responde bien con algún aspecto menor sin cubrir, 5=responde completamente."
    )
    return _call_judge(client, model, prompt)


def _judge_correctness(
    client: OpenAI, model: str, answer: str, ground_truth_text: str,
) -> dict[str, Any]:
    prompt = (
        "Evalúa la CORRECCIÓN FACTUAL de la respuesta comparándola con el texto fuente original.\n"
        "¿Los hechos de la respuesta coinciden con el texto fuente? "
        "¿Hay errores factuales, datos tergiversados o imprecisiones?\n\n"
        f"TEXTO FUENTE (ground truth):\n{ground_truth_text}\n\n"
        f"RESPUESTA DEL SISTEMA:\n{answer}\n\n"
        "Responde con JSON: {\"score\": <1-5>, \"reason\": \"<explicación breve>\"}\n"
        "1=totalmente incorrecto, 2=errores graves, 3=parcialmente correcto, "
        "4=correcto con imprecisiones menores, 5=factualmente perfecto."
    )
    return _call_judge(client, model, prompt)


def _call_judge(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "reason": f"JSON inválido del juez: {raw[:200]}"}


# ── pipeline de evaluación ───────────────────────────────────────────────────


def evaluate_e2e(golden_path: Path) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise RuntimeError("Se necesita OPENAI_API_KEY para la evaluación end-to-end.")

    with open(golden_path, encoding="utf-8") as f:
        golden: list[dict[str, Any]] = json.load(f)
    if not golden:
        raise RuntimeError(f"El golden dataset en {golden_path} está vacío.")

    print("Cargando textos ground-truth de Chroma…")
    gt_texts = _load_ground_truth_texts(golden)

    reranker = (
        get_local_cross_encoder_reranker()
        if settings.reranking_enabled
        else NoOpReranker()
    )
    use_case = AskQuestionUseCase(
        vector_store=ChromaVectorStoreRepository(),
        llm=OpenAILLMRepository(),
        reranker=reranker,
    )

    judge_client = OpenAI(api_key=settings.openai_api_key.strip())
    judge_model = settings.openai_model

    per_query: list[dict[str, Any]] = []
    total = len(golden)

    for idx, entry in enumerate(golden):
        question = entry["question"]
        gt_chunk_ids = entry.get("relevant_chunk_ids", [])
        gt_text = " ".join(gt_texts.get(cid, "") for cid in gt_chunk_ids).strip()

        print(f"  [{idx + 1}/{total}] Ejecutando pipeline… ", end="", flush=True)
        response = use_case.execute(question)
        answer = response.answer

        context_for_judge = "\n".join(
            f"[{c.ref}] {c.document} p.{c.page}: (chunk recuperado)"
            for c in response.citations
        )

        context_full = answer
        raw_chunks = use_case._vector_store.search(
            question, top_k=settings.vector_top_k,
        )
        context_full = "\n---\n".join(c.content for c in raw_chunks)

        faith = _judge_faithfulness(judge_client, judge_model, answer, context_full)
        relev = _judge_relevancy(judge_client, judge_model, question, answer)
        correct = _judge_correctness(judge_client, judge_model, answer, gt_text) if gt_text else {"score": 0, "reason": "Sin ground truth disponible"}

        f_score = faith.get("score", 0)
        r_score = relev.get("score", 0)
        c_score = correct.get("score", 0)

        per_query.append({
            "id": entry["id"],
            "question": question,
            "answer": answer[:500],
            "faithfulness": f_score,
            "faithfulness_reason": faith.get("reason", ""),
            "relevancy": r_score,
            "relevancy_reason": relev.get("reason", ""),
            "correctness": c_score,
            "correctness_reason": correct.get("reason", ""),
        })

        print(f"F={f_score} R={r_score} C={c_score}")

    valid_f = [q["faithfulness"] for q in per_query if q["faithfulness"] > 0]
    valid_r = [q["relevancy"] for q in per_query if q["relevancy"] > 0]
    valid_c = [q["correctness"] for q in per_query if q["correctness"] > 0]

    report: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(golden_path),
        "total_queries": total,
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model if settings.openai_api_key.strip() else "local_hash",
            "openai_model": settings.openai_model,
            "reranking": settings.reranking_enabled,
            "vector_top_k": settings.vector_top_k,
        },
        "results": {
            "faithfulness_avg": round(sum(valid_f) / len(valid_f), 2) if valid_f else 0,
            "relevancy_avg": round(sum(valid_r) / len(valid_r), 2) if valid_r else 0,
            "correctness_avg": round(sum(valid_c) / len(valid_c), 2) if valid_c else 0,
        },
        "per_query": per_query,
    }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    results = report["results"]
    cfg = report["config"]
    print(f"\n{'=' * 60}")
    print(f"  E2E EVAL — {report['total_queries']} queries")
    print(f"  chunk_size={cfg['chunk_size']}  overlap={cfg['chunk_overlap']}  model={cfg['openai_model']}")
    print(f"  reranking={cfg['reranking']}  top_k={cfg['vector_top_k']}")
    print(f"{'=' * 60}")
    print(f"  {'Métrica':<25} {'Valor':>8}")
    print(f"  {'-' * 35}")
    for key, value in sorted(results.items()):
        print(f"  {key:<25} {value:>8.2f}")
    print(f"{'=' * 60}")

    per_query = report.get("per_query", [])
    low_faith = [q for q in per_query if 0 < q["faithfulness"] < 4]
    low_relev = [q for q in per_query if 0 < q["relevancy"] < 4]
    low_corr = [q for q in per_query if 0 < q["correctness"] < 4]
    if low_faith:
        print(f"\n  ALERTA: {len(low_faith)} preguntas con faithfulness < 4")
        for q in low_faith[:3]:
            print(f"    - [{q['id']}] F={q['faithfulness']}: {q['question'][:70]}")
    if low_relev:
        print(f"\n  ALERTA: {len(low_relev)} preguntas con relevancy < 4")
        for q in low_relev[:3]:
            print(f"    - [{q['id']}] R={q['relevancy']}: {q['question'][:70]}")
    if low_corr:
        print(f"\n  ALERTA: {len(low_corr)} preguntas con correctness < 4")
        for q in low_corr[:3]:
            print(f"    - [{q['id']}] C={q['correctness']}: {q['question'][:70]}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación end-to-end del RAG (LLM-as-judge)")
    parser.add_argument(
        "--golden", type=str, default=None,
        help="Ruta al golden dataset (default data/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta del reporte (default data/eval/eval_e2e_report.json)",
    )
    args = parser.parse_args()

    settings = get_settings()
    golden_path = Path(args.golden) if args.golden else Path(settings.eval_artifacts_path) / "golden_dataset.json"
    out_path = Path(args.output) if args.output else Path(settings.eval_artifacts_path) / "eval_e2e_report.json"

    if not golden_path.exists():
        print(f"ERROR: No se encontró el golden dataset en {golden_path}")
        print("Ejecuta primero:  python app/services/eval_generate_golden.py")
        sys.exit(1)

    print(f"Golden dataset: {golden_path}")
    print(f"Modelo LLM:     {settings.openai_model}")
    print(f"Reranking:      {settings.reranking_enabled}")
    print()

    report = evaluate_e2e(golden_path=golden_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte guardado en {out_path}")
    _print_summary(report)


if __name__ == "__main__":
    main()

# Flujo de ejecución — RAG Médico

> **Mantenimiento:** Este documento debe **actualizarse** cuando cambien el punto de entrada, las rutas HTTP, el caso de uso RAG, la ingesta o las implementaciones de Chroma/OpenAI. Ver sección [Cómo mantener este fichero actualizado](#cómo-mantener-este-fichero-actualizado).

---

## 1. Punto de entrada de la aplicación web

| Cómo se arranca | Qué carga |
|-----------------|-----------|
| `uvicorn main:app` (recomendado en `requirements.txt`) | El objeto `app` definido en **`main.py`**. |

Secuencia al importar `main`:

1. **`main.py`** — `load_dotenv()` lee `.env` antes del resto de imports.
2. **`main.py`** — Se construye `FastAPI()` y se registran routers:
   - `app.api.routes.ask` → rutas `/ask`
   - `app.api.routes.documents` → rutas `/documents`
3. **`main.py`** — Ruta `GET /` → función `home()`: devuelve HTML+JS que llama a la API (no pasa por el caso de uso RAG en servidor salvo peticiones fetch del navegador).

No hay otro entrypoint HTTP: todo pasa por `main:app`.

---

## 2. Flujo por petición HTTP (tiempo de ejecución)

### 2.1 `GET /ask?question=...` (y opcionalmente `sources=...` repetido)

| Orden | Fichero → método / símbolo | Qué hace |
|-------|----------------------------|----------|
| 1 | `app/api/routes/ask.py` → `ask()` | Valida query, inyecta `RagService` vía `Depends(get_rag_service)`, delega y traduce excepciones a HTTP 500. |
| 2 | `app/services/rag_service.py` → `RagService.ask()` | Llama al caso de uso. |
| 3 | `app/application/use_cases/ask_question.py` → `AskQuestionUseCase.execute()` | Orquesta RAG: búsqueda vectorial (recall `VECTOR_RECALL_K`) → rerank cross-encoder → recorte a `VECTOR_TOP_K` → contexto numerado → LLM → `AskResponse`. |
| 4a | `app/infrastructure/vectorstores/chroma_vector_store.py` → `ChromaVectorStoreRepository.search()` | Delega en la capa DB. |
| 4b | `app/db/vector_store.py` → `ChromaVectorStoreDB.search()` | `embed_query` + `collection.query` en Chroma con filtro `source` si aplica (`n_results` = recall, típ. 20). |
| 4c | `app/infrastructure/embeddings/openai_embedder.py` o `local_embedder.py` | Embedding de la pregunta (OpenAI si hay `OPENAI_API_KEY`, si no hash local). |
| 4d | `app/db/chroma_client.py` → `get_chroma_collection()` | Colección acorde a nombre construido en el repositorio (debe coincidir con la ingesta). |
| 4e | `app/infrastructure/reranking/local_cross_encoder_reranker.py` → `LocalCrossEncoderReranker.rerank()` | Cross-encoder local (`sentence-transformers`, por defecto español STS `nflechas/spanish-BERT-sts` vía `CROSS_ENCODER_MODEL`; opcional MMARCO multilingüe); ordena candidatos y deja los mejores `VECTOR_TOP_K`. |
| 5 | `app/infrastructure/llm/openai_llm_repository.py` → `OpenAILLMRepository.generate()` | Chat completions con contexto y prompts de citas. |
| 6 | `app/models/qa.py` | Esquema `AskResponse` (answer, sources, citations). |

**Configuración leída en cadena:** `app/core/config.py` → `get_settings()` (`lru_cache`): `RERANKING_ENABLED`, `VECTOR_RECALL_K`, `VECTOR_TOP_K`, `CROSS_ENCODER_MODEL`, rutas Chroma, modelos, etc. Si `RERANKING_ENABLED` es falso, no se carga el cross-encoder y la búsqueda usa solo `VECTOR_TOP_K` por similitud vectorial.

**Tiempos (consulta):** si `TIMING_LOGS_ENABLED` es verdadero, cada petición a `/ask` escribe un fichero en `TIMING_LOGS_PATH` (por defecto `data/logs`) con prefijo `query_` y timestamp UTC. Pasos registrados: `embedding_query`, `chroma_query`, `cross_encoder_rerank`, `build_context`, `openai_chat_completions`, y `total_wall_ms`.

---

### 2.2 `GET /documents`

| Orden | Fichero → método | Qué hace |
|-------|------------------|----------|
| 1 | `app/api/routes/documents.py` → `list_documents()` | Inyecta `ChromaVectorStoreRepository`. |
| 2 | `app/infrastructure/vectorstores/chroma_vector_store.py` → `list_sources()` | Delega en DB. |
| 3 | `app/db/vector_store.py` → `ChromaVectorStoreDB.list_sources()` | `collection.get` y deduplica `source` desde metadatos. |
| 4 | `app/models/documents.py` | `DocumentListResponse`. |

---

### 2.3 `GET /`

| Fichero → método | Qué hace |
|------------------|----------|
| `main.py` → `home()` | Respuesta HTML estática; el cliente llama a `/documents` y `/ask`. Si `answer` ya incluye la sección `**Referencias**`, no se añade la leyenda duplicada de `citations`. |

---

## 3. Inicialización perezosa (primera vez que se usa el RAG o documentos)

Al resolver `Depends(get_rag_service)` o el repositorio vectorial:

| Componente | Fichero | Qué ocurre |
|------------|---------|------------|
| `get_rag_service` | `app/services/rag_service.py` | Crea `AskQuestionUseCase(vector_store, OpenAILLMRepository, reranker)` una vez por proceso (`lru_cache`). Si `RERANKING_ENABLED` es falso, `reranker` es `NoOpReranker` (no carga HF). |
| `get_local_cross_encoder_reranker` | `app/infrastructure/reranking/local_cross_encoder_reranker.py` | Solo si rerank activo: instancia cacheada del cross-encoder (descarga/carga HF la primera vez). |
| `get_chroma_vector_store_repository` | `app/infrastructure/vectorstores/chroma_vector_store.py` | Crea embedder, `ChromaVectorStoreDB` y llama **`ensure_index()`**: si la colección está vacía, indexa `.md`/`.txt` bajo `DOCUMENTS_PATH` (camino distinto al script de ingesta principal). |

---

## 4. Flujo de ingesta de documentos (proceso aparte, no es HTTP)

| Orden | Fichero → función | Qué hace |
|-------|-------------------|----------|
| 1 | `app/services/ingest.py` → `ingest_documents()` (también `if __name__ == "__main__"`) | Recorre PDF/txt/docx, trocea, firma por archivo, upsert en Chroma, exporta evaluación. |
| 2 | `app/db/chroma_client.py` | Cliente persistente y colección (nombre alineado con repositorio). |
| 3 | `app/infrastructure/embeddings/*` | Mismos embedders que en consulta. |
| 4 | `app/services/ingest_eval_export.py` → `export_chroma_eval_artifacts()` | Escribe `data/eval/ingestion_manifest.json`, opcionalmente `chunks_index.jsonl`, y siempre `chunks_review.json` (texto completo de cada chunk agrupado por fuente para revisión manual). |

**Tiempos (ingesta):** con `TIMING_LOGS_ENABLED`, un fichero `ingest_<timestamp>.log` en `TIMING_LOGS_PATH` con pasos: `chroma_get_collection`, `listar_ficheros_candidatos`, `escaneo_extraccion_chunking`, `openai_embeddings_embed_texts`, `chroma_upsert_batches`, `export_eval_artifacts`, y resumen final (`chunks_indexed`, `ingestion_batch_id`, `total_wall_ms`).

---

## 5. Mapa de capas (referencia rápida)

```
main.py (FastAPI, rutas /)
    └── app/api/routes/*.py
            └── app/services/rag_service.py
                    └── app/application/use_cases/ask_question.py
                            ├── app/domain/repositories (puertos)
                            ├── app/infrastructure/vectorstores → app/db/vector_store.py → chroma_client + embeddings
                            ├── app/infrastructure/reranking → cross-encoder local
                            └── app/infrastructure/llm → OpenAI
app/services/ingest.py (CLI / batch) → chroma + embeddings + ingest_eval_export.py
app/core/config.py (Settings) — usado casi en todas las capas
```

---

## 6. Dominio y modelos (sin I/O directo)

| Fichero | Rol |
|---------|-----|
| `app/domain/entities/retrieved_chunk.py` | Fragmento recuperado (texto, source, página). |
| `app/domain/repositories/vector_store_repository.py` | Contrato búsqueda + listar fuentes. |
| `app/domain/repositories/reranker_repository.py` | Contrato `rerank(query, chunks, top_k)`. |
| `app/domain/repositories/llm_repository.py` | Contrato `generate(question, context)`. |
| `app/models/qa.py`, `app/models/documents.py` | Esquemas Pydantic de respuesta API. |

---

## 7. Tests

| Fichero | Qué ejercita |
|---------|--------------|
| `tests/integration/test_ask_endpoint.py` | Cliente HTTP contra `main.app` → `/ask`. |
| `tests/unit/test_ask_use_case.py` | `AskQuestionUseCase` con Chroma real, `OpenAILLMRepository` y `NoOpReranker` (sin cross-encoder). |

---

## Cómo mantener este fichero actualizado

1. **Cuándo editar:** al añadir o cambiar rutas en `main.py` o `app/api/`, al cambiar `AskQuestionUseCase`, al cambiar ingesta (`ingest.py`), al cambiar cliente Chroma/embeddings/LLM, o al introducir un nuevo entrypoint.
2. **Qué actualizar:** tablas de flujo, nombres de métodos, y la sección 5 si cambia la estructura de carpetas.
3. **Herramientas:** en este repo existe la regla Cursor `.cursor/rules/actualizar-flujo-aplicacion.mdc` para recordar esta actualización al tocar `app/**/*.py` o `main.py`.

*Última revisión alineada con el código del repositorio: actualizar esta fecha cuando se modifique el documento.*

**Fecha de documento:** 2026-04-12 (actualizado: `spec.md`, `RERANKING_ENABLED`, cross-encoder español por defecto, tests con `NoOpReranker`)

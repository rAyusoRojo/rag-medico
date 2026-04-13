# Especificación — RAG Médico

Documento de referencia del repositorio: stack, dependencias, variables de entorno y flujo RAG. El detalle de llamadas entre ficheros está en [`docs/FLUJO_APLICACION.md`](docs/FLUJO_APLICACION.md).

## Objetivo

API HTTP (FastAPI) que responde preguntas usando recuperación sobre un índice vectorial local (Chroma), reranking opcional con cross-encoder, y generación con el API de OpenAI. Los fragmentos recuperados se numeran en el contexto del LLM para citas trazables (`ref=[n]`).

## Requisitos de runtime

- **Python:** 3.10+ (tipos `list[str] | None`, etc.).
- **Red:** acceso a OpenAI para chat/embeddings si se usa `OPENAI_API_KEY`; acceso a Hugging Face Hub la primera vez que se carga un modelo de `sentence-transformers` (reranking activo).

## Dependencias

Origen único de paquetes pip: **`requirements.txt`**.

| Bloque | Paquetes | Rol |
|--------|----------|-----|
| API | `fastapi`, `uvicorn` | Servidor HTTP y ASGI. |
| Configuración | `pydantic`, `pydantic-settings`, `python-dotenv` | `Settings` y carga de `.env`. |
| RAG / LLM / índice | `openai`, `chromadb` | Chat, embeddings y almacén vectorial persistente. |
| Ingesta de documentos | `pypdf`, `python-docx` | Extracción de texto de PDF y Word en el script de ingesta. |
| Reranking local | `sentence-transformers` | Cross-encoder; arrastra **`torch`** y resto de dependencias del ecosistema PyTorch/HF. |
| Tests | `pytest` | Suite en `tests/`. |

Instalación: `pip install -r requirements.txt`.

## Variables de entorno

Definidas en **`app/core/config.py`** (clase `Settings`), leídas desde **`.env`** en la raíz del proyecto. Plantilla: **`.env.example`**.

Principales:

- **OpenAI:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `EMBEDDING_MODEL`.
- **Recuperación y rerank:** `VECTOR_TOP_K`, `VECTOR_RECALL_K`, `RERANKING_ENABLED`, `CROSS_ENCODER_MODEL` (por defecto cross-encoder en español `nflechas/spanish-BERT-sts`; alternativa multilingüe documentada en comentarios de `config.py`).
- **Chroma:** `CHROMA_COLLECTION`, `CHROMA_PERSIST_DIRECTORY`.
- **Corpus:** `DOCUMENTS_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.
- **Evaluación / logs:** `EVAL_ARTIFACTS_PATH`, `EVAL_EXPORT_CHUNK_INDEX`, `TIMING_LOGS_PATH`, `TIMING_LOGS_ENABLED`.

## Arquitectura (resumen)

- **Dominio:** entidades (`RetrievedChunk`) y puertos (`VectorStoreRepository`, `RerankerRepository`, `LLMRepository`).
- **Aplicación:** `AskQuestionUseCase` — búsqueda → rerank (si aplica) → contexto numerado → LLM.
- **Infraestructura:** Chroma + embeddings OpenAI o hash local, `LocalCrossEncoderReranker` o `NoOpReranker`, `OpenAILLMRepository`.
- **Presentación:** `main.py`, routers en `app/api/routes/`.

## Puntos de entrada

| Entrada | Comando / módulo |
|---------|------------------|
| API | `uvicorn main:app` |
| Ingesta batch | `python app/services/ingest.py` |

## Tests

- `tests/integration/test_ask_endpoint.py` — cliente HTTP contra la app.
- `tests/unit/test_ask_use_case.py` — caso de uso con Chroma real y `NoOpReranker` (sin cargar cross-encoder).

---

*Este fichero debe actualizarse cuando cambien dependencias clave, variables de entorno o el flujo descrito en `docs/FLUJO_APLICACION.md`.*

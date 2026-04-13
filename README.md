# RAG Médico

API de preguntas y respuestas con RAG (Chroma + OpenAI), reranking opcional con cross-encoder y citas alineadas con los fragmentos recuperados. Especificación del proyecto: [**spec.md**](spec.md). Flujo detallado entre módulos: [**docs/FLUJO_APLICACION.md**](docs/FLUJO_APLICACION.md).

## Requisitos

- Python 3.10+
- Clave OpenAI si usas embeddings y chat de OpenAI (`OPENAI_API_KEY`); sin clave, el embedder local es un hash (solo desarrollo) y el LLM sigue necesitando clave para generar respuestas en producción.

## Instalación y arranque

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Servidor por defecto: `http://127.0.0.1:8000`. La raíz `GET /` sirve una página HTML de prueba.

Configuración: copia `.env.example` a `.env` y ajusta variables (véase `spec.md`).

## Ingesta de documentos

```bash
python app/services/ingest.py
```

Indexa PDF, TXT y DOCX bajo `DOCUMENTS_PATH` (por defecto `data/documents`).

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/ask?question=...` | Respuesta RAG; parámetro repetible `sources=` para filtrar por `source` en Chroma. |
| `GET` | `/documents` | Lista de documentos (`source`) presentes en el índice. |
| `GET` | `/` | UI mínima (HTML) para probar la API. |

## Formato de respuesta (`/ask`)

JSON (`AskResponse`):

- `answer` — texto generado.
- `sources` — lista de objetos `{ "document", "page" | null }` (fuentes únicas).
- `citations` — leyenda `{ "ref", "document", "page" }` alineada con `ref=[n]` en el contexto enviado al modelo.

## Tests

```bash
pytest
```

## Documentación adicional

- [**spec.md**](spec.md) — dependencias, variables de entorno, arquitectura.
- [**docs/FLUJO_APLICACION.md**](docs/FLUJO_APLICACION.md) — mapa de ejecución y módulos.

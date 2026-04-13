"""
Configuración centralizada: lee .env y variables de entorno con alias en mayúsculas
(para documentar qué variable usar en despliegue).
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Por defecto: cross-encoder monolingüe español (BERT + STS; pares consulta–pasaje).
# Multilingüe (MMARCO, ~14 idiomas) para cuando necesites varios idiomas en el mismo despliegue:
#   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
DEFAULT_CROSS_ENCODER_MODEL = "nflechas/spanish-BERT-sts"


class Settings(BaseSettings):
    # --- OpenAI: generación (chat) y embeddings ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    # Cuántos fragmentos pasan al LLM tras el reranking (recorte final). Env: VECTOR_TOP_K en .env
    vector_top_k: int = Field(default=5, alias="VECTOR_TOP_K")
    # Candidatos que devuelve Chroma antes del rerank. Env: VECTOR_RECALL_K en .env
    vector_recall_k: int = Field(
        default=20,
        alias="VECTOR_RECALL_K",
        description="Recall vectorial: cuántos trozos pide Chroma antes de reordenar con el cross-encoder.",
    )
    # Modelo Hugging Face del CrossEncoder para rerank local. Env: CROSS_ENCODER_MODEL en .env
    cross_encoder_model: str = Field(
        default=DEFAULT_CROSS_ENCODER_MODEL,
        alias="CROSS_ENCODER_MODEL",
        description="CrossEncoder para reranking; por defecto STS en español (nflechas/spanish-BERT-sts).",
    )
    # Si false: solo similitud vectorial (sin reordenar ni cargar el cross-encoder). Env: RERANKING_ENABLED
    reranking_enabled: bool = Field(default=True, alias="RERANKING_ENABLED")
    # --- Chroma: nombre lógico de colección y carpeta en disco (persistencia local) ---
    chroma_collection: str = Field(default="rag_medico_docs", alias="CHROMA_COLLECTION")
    chroma_persist_directory: str = Field(default=".chroma", alias="CHROMA_PERSIST_DIRECTORY")
    # Carpeta donde están los PDF/txt/doc a indexar (ruta relativa al cwd del proceso).
    documents_path: str = Field(default="data/documents", alias="DOCUMENTS_PATH")
    # Ventana y solapamiento al trocear texto en la ingesta (afecta granularidad del índice).
    chunk_size: int = Field(default=700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    # Salida de manifiesto/índice para evaluar recuperación sin tocar el código de consulta.
    eval_artifacts_path: str = Field(default="data/eval", alias="EVAL_ARTIFACTS_PATH")
    eval_export_chunk_index: bool = Field(default=True, alias="EVAL_EXPORT_CHUNK_INDEX")
    # Un fichero .log por ejecución (ingesta o consulta) con tiempos de pasos clave.
    timing_logs_path: str = Field(default="data/logs", alias="TIMING_LOGS_PATH")
    timing_logs_enabled: bool = Field(default=True, alias="TIMING_LOGS_ENABLED")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    # Instancia única por proceso: evita reparsear .env en cada petición.
    return Settings()

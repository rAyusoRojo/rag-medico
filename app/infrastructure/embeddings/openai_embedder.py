"""
Embeddings vía API OpenAI: trocea en lotes respetando límites de tokens/items por petición
y sanea texto para evitar caracteres que rompan la serialización o excedan el máximo del modelo.
"""
from openai import OpenAI

from app.core.config import get_settings
from app.infrastructure.embeddings.base import EmbeddingProvider


# Tope conservador por entrada en modelos tipo text-embedding-3-* (~8192 tokens).
_MAX_EMBEDDING_INPUT_BYTES = 28_000


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        # Límite documentado de la API (~300k tokens/petición); se deja margen y se estima por heurística.
        self._max_tokens_per_request = 150_000
        self._max_items_per_request = 256

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch in self._build_batches(texts):
            sanitized_batch = [self._sanitize_text(item) for item in batch]
            response = self._client.embeddings.create(model=self._model, input=sanitized_batch)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _build_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            estimated_tokens = self._estimate_tokens(text)
            if estimated_tokens > self._max_tokens_per_request:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([text])
                continue
            if current_batch and (
                len(current_batch) >= self._max_items_per_request
                or current_tokens + estimated_tokens > self._max_tokens_per_request
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Heurística conservadora para español / UTF-8 (no es el tokenizer oficial de OpenAI).
        byte_len = len(text.encode("utf-8"))
        return max(1, byte_len // 2)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        clean = text.replace("\x00", " ")
        # Remove invalid surrogate code points that can break JSON encoding.
        clean = clean.encode("utf-8", errors="ignore").decode("utf-8")
        raw = clean.encode("utf-8", errors="ignore")
        if len(raw) > _MAX_EMBEDDING_INPUT_BYTES:
            clean = raw[:_MAX_EMBEDDING_INPUT_BYTES].decode("utf-8", errors="ignore")
        return clean

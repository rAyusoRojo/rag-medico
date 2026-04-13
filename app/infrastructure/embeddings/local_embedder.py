"""
Embeddings deterministas locales (bolsa de palabras hasheada + normalización L2).
Sirve para desarrollo sin API y para tests; **no** es semánticamente comparable a OpenAI.
"""
import hashlib
import math

from app.infrastructure.embeddings.base import EmbeddingProvider


class LocalHashEmbedder(EmbeddingProvider):

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

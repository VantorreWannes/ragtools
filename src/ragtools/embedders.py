from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, cast

from sentence_transformers import SentenceTransformer, SparseEncoder


class Embedder[E](Protocol):
    def embed(self, chunk: str) -> E: ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def embed(self, chunk: str) -> list[float]:
        return self.model.encode(chunk, normalize_embeddings=True).tolist()


@dataclass
class SpladeEmbedder:
    model_name: str

    @cached_property
    def model(self) -> SparseEncoder:
        return SparseEncoder(self.model_name)

    def embed(self, chunk: str) -> dict[str, float]:
        embeddings = self.model.encode(chunk)
        return dict(cast(list[tuple[str, float]], [*self.model.decode(embeddings)]))
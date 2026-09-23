from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

from semantic_chunker import TextSplitter, get_chunker
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from transformers import TextGenerationPipeline, pipeline
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


class Embedder[V](Protocol):
    def embed(self, text: str) -> V: ...
    def embed_all(self, texts: Sequence[str]) -> list[V]: ...


class Scorer(Protocol):
    def score(self, query: str, text: str) -> float: ...


class Generator(Protocol):
    def reply(self, prompt: str) -> str: ...


class Chunker(Protocol):
    def chunks(self, text: str) -> list[str]: ...


class FileReader(Protocol):
    def __call__(self, path: Path) -> list[str]: ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str

    @cached_property
    def _model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


@dataclass
class SpladeEmbedder:
    model_name: str

    @cached_property
    def _model(self) -> SparseEncoder:
        return SparseEncoder(self.model_name)

    def embed(self, text: str) -> dict[str, float]:
        embeddings = self._model.encode(text)
        decoded = cast(list[tuple[str, float]], [*self._model.decode(embeddings)])
        return dict(decoded)

    def embed_all(self, texts: Sequence[str]) -> list[dict[str, float]]:
        embeddings = self._model.encode(list(texts))
        decoded_batch = self._model.decode(embeddings)
        results: list[dict[str, float]] = []
        for item in decoded_batch:
            pairs = cast(list[tuple[str, float]], list(item))
            results.append({token: float(weight) for token, weight in pairs if weight})

        return results


@dataclass
class CrossEncoderScorer:
    model_name: str

    @cached_property
    def _model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name)

    def score(self, query: str, text: str) -> float:
        return float(self._model.predict((query, text)).tolist())


@dataclass
class TransformersGenerator:
    model_name: str
    max_tokens: int

    @cached_property
    def _pipeline(self) -> TextGenerationPipeline:
        return pipeline("text-generation", model=self.model_name)

    def reply(self, prompt: str) -> str:
        chat = [{"role": "user", "content": prompt}]
        outputs = self._pipeline(chat, max_new_tokens=self.max_tokens, do_sample=True)
        return str(outputs[0]["generated_text"][-1]["content"])


@dataclass
class SemanticChunker:
    model_name: str
    chunk_size: int
    chunk_overlap: int

    @cached_property
    def _chunker(self) -> TextSplitter:
        return cast(
            TextSplitter,
            get_chunker(
                self.model_name,
                chunking_type="text",
                tree_sitter_language=None,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap,
                trim=True,
            ),
        )

    def chunks(self, text: str) -> list[str]:
        return list(self._chunker.chunks(text))


def read_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    match suffix:
        case ".pdf":
            elements = partition_pdf(str(path))
        case ".md":
            elements = partition_md(str(path))
        case ".csv":
            elements = partition_csv(str(path))
        case ".txt":
            elements = partition_text(str(path))
        case _:
            raise ValueError(f"Unsupported file format: {suffix}")

    return [element.text for element in elements if element.text]

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

from semantic_chunker import TextSplitter, get_chunker
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from transformers import TextGenerationPipeline, pipeline
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


@dataclass
class SentenceTransformerEmbedder:
    model_name: str

    @cached_property
    def _model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def __call__(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


@dataclass
class SpladeEmbedder:
    model_name: str

    @cached_property
    def _model(self) -> SparseEncoder:
        return SparseEncoder(self.model_name)

    def __call__(self, text: str) -> dict[str, float]:
        embeddings = self._model.encode(text)
        decoded = cast(list[tuple[str, float]], [*self._model.decode(embeddings)])
        return dict(decoded)


@dataclass
class CrossEncoderScorer:
    model_name: str

    @cached_property
    def _model(self) -> CrossEncoder:
        return CrossEncoder(self.model_name)

    def __call__(self, query: str, document: str) -> float:
        return float(self._model.predict((query, document)).tolist())


@dataclass
class TransformersGenerator:
    model_name: str
    max_tokens: int

    @cached_property
    def _pipeline(self) -> TextGenerationPipeline:
        return pipeline("text-generation", model=self.model_name)

    def __call__(self, prompt: str) -> str:
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

    def __call__(self, text: str) -> list[str]:
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

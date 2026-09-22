from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

from semantic_chunker import TextSplitter, get_chunker
from unstructured.documents.elements import Element
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


class Parser[D, V](Protocol):
    def units(self, data: D) -> V: ...


@dataclass(slots=True)
class CsvFileElementParser:
    def units(self, data: Path) -> list[Element]:
        return partition_csv(str(data))


@dataclass(slots=True)
class MdFileElementParser:
    def units(self, data: Path) -> list[Element]:
        return partition_md(str(data))


@dataclass(slots=True)
class PdfFileElementParser:
    def units(self, data: Path) -> list[Element]:
        return partition_pdf(str(data))


@dataclass(slots=True)
class TextFileElementParser:
    def units(self, data: Path) -> list[Element]:
        return partition_text(str(data))


@dataclass(slots=True)
class ElementTextParser:
    def units(self, data: Element) -> str:
        return data.text


@dataclass(slots=True)
class ElementPageIndexParser:
    def units(self, data: Element) -> int:
        page_index = data.metadata.page_number
        return page_index if page_index is not None else -1


@dataclass
class ChunkParser:
    model_name: str
    chunk_size: int
    overlap: int

    @cached_property
    def model(self) -> TextSplitter:
        return cast(
            "TextSplitter",
            get_chunker(
                self.model_name,
                chunking_type="text",
                tree_sitter_language=None,
                max_tokens=self.chunk_size,
                overlap=self.overlap,
                trim=True,
            ),
        )

    def units(self, data: str) -> tuple[str, ...]:
        return tuple(self.model.chunks(data))

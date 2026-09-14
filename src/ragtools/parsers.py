from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from unstructured.documents.elements import Element
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


class Parser[D, V](Protocol):
    def units(self, data: D) -> tuple[V, ...]: ...


@dataclass(slots=True)
class UnstructuredCsvFileElementParser:
    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(partition_csv(str(data)))


@dataclass(slots=True)
class UnstructuredMdFileElementParser:
    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(partition_md(str(data)))


@dataclass(slots=True)
class UnstructuredPdfFileElementParser:
    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(partition_pdf(str(data)))


@dataclass(slots=True)
class UnstructuredTextFileElementParser:
    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(partition_text(str(data)))

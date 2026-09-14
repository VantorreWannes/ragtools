"""Parsers for extracting units from data."""

from abc import ABC, abstractmethod
from itertools import groupby
from pathlib import Path

from chestkey import Chest, Key
from unstructured.documents.elements import Element
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf


class BaseParser[D, V](ABC):
    """Parser extracting a tuple of units from data."""

    @abstractmethod
    def units(self, data: D) -> tuple[V, ...]: ...


class CachedParser[D, V](BaseParser[D, V]):
    """Cache another parser's units in a chestkey ``Chest``."""

    def __init__(self, parser: BaseParser[D, V], chest: Chest) -> None:
        self.parser = parser
        self.chest = chest

    def units(self, data: D) -> tuple[V, ...]:
        seed = data.read_bytes() if isinstance(data, Path) else data
        cache_key = Key[tuple[V, ...]](seed)
        if self.chest.contains(cache_key):
            return self.chest.get(cache_key)
        units = self.parser.units(data)
        self.chest.set(cache_key, units)
        return units


class UnstructuredFileParser(BaseParser[Path, Element]):
    """Base parser extracting unstructured elements from a file."""

    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(self._partition(data))

    @abstractmethod
    def _partition(self, data: Path) -> list[Element]: ...


class MarkdownFileParser(UnstructuredFileParser):
    """Parser extracting elements from markdown files."""

    def _partition(self, data: Path) -> list[Element]:
        return partition_md(filename=str(data))


class CsvFileParser(UnstructuredFileParser):
    """Parser extracting elements from csv files."""

    def _partition(self, data: Path) -> list[Element]:

        return partition_csv(filename=str(data))


class PdfFileParser(UnstructuredFileParser):
    """Parser extracting elements from pdf files."""

    def _partition(self, data: Path) -> list[Element]:
        return partition_pdf(filename=str(data))


class UnstructuredPageParser(BaseParser[Path, str]):
    """Parser extracting per-page text from a file."""

    def __init__(self, file_parser: BaseParser[Path, Element]) -> None:
        self._file_parser = file_parser

    def units(self, data: Path) -> tuple[str, ...]:
        elements = self._file_parser.units(data)

        def page_number(element: Element) -> int:
            number = element.metadata.page_number
            return -1 if number is None else number

        ordered = sorted(elements, key=page_number)
        return tuple(
            "\n".join(element.text for element in group)
            for _, group in groupby(ordered, key=page_number)
        )

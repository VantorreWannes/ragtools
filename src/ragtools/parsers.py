"""Parsers for extracting units from data."""

from abc import ABC, abstractmethod
from itertools import groupby
from pathlib import Path

from chestkey import Chest, Key
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition


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
        cache_key = Key[tuple[V, ...]](data)
        if self.chest.contains(cache_key):
            return self.chest.get(cache_key)
        units = self.parser.units(data)
        self.chest.set(cache_key, units)
        return units


class UnstructuredFileParser(BaseParser[Path, Element]):
    """Parser extracting unstructured elements from a file."""

    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(partition(str(data)))


class UnstructuredPageParser(BaseParser[Path, str]):
    """Parser extracting per-page text from a file."""

    def __init__(self, file_parser: BaseParser[Path, Element] | None = None) -> None:
        self._file_parser = (
            file_parser if file_parser is not None else UnstructuredFileParser()
        )

    def units(self, data: Path) -> tuple[str, ...]:
        elements = self._file_parser.units(data)

        def page_number(element: Element) -> int:
            return element.metadata.page_number or -1

        ordered = sorted(elements, key=page_number)
        return tuple(
            "\n".join(element.text for element in group)
            for _, group in groupby(ordered, key=page_number)
        )


type PdfFileParser = UnstructuredFileParser
type MarkdownFileParser = UnstructuredFileParser
type CsvFileParser = UnstructuredFileParser

type PdfPageParser = UnstructuredPageParser
type MarkdownPageParser = UnstructuredPageParser
type CsvPageParser = UnstructuredPageParser

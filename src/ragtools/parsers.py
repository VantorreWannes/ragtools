"""Parsers for extracting units from data."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from itertools import groupby
from pathlib import Path

from chestkey import Chest, Key
from unstructured.documents.elements import Element
from unstructured.partition.csv import partition_csv
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text


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
        cache_key = Key(seed)
        if self.chest.contains(cache_key):
            return self.chest.get(cache_key)
        units = self.parser.units(data)
        self.chest.set(cache_key, units)
        return units


PARTITIONERS: Mapping[str, Callable[[str], list[Element]]] = {
    ".csv": partition_csv,
    ".md": partition_md,
    ".pdf": partition_pdf,
    ".txt": partition_text,
}


class FileParser(BaseParser[Path, Element]):
    """Parser extracting unstructured elements, dispatching on file suffix."""

    def __init__(
        self, partitioners: Mapping[str, Callable[[str], list[Element]]] = PARTITIONERS
    ) -> None:
        self._partitioners = partitioners

    def units(self, data: Path) -> tuple[Element, ...]:
        return tuple(self._partitioners[data.suffix.lower()](str(data)))


class PageParser(BaseParser[Path, str]):
    """Parser extracting per-page text from a file of unstructured elements."""

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


class ChunkParser(BaseParser[str, str]):
    """Parser splitting text into fixed-size, optionally overlapping chunks."""

    def __init__(self, size: int, overlap: int = 0) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self._size = size
        self._step = size - overlap

    def units(self, data: str) -> tuple[str, ...]:
        return tuple(data[i : i + self._size] for i in range(0, len(data), self._step))

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import dill
from blake3 import blake3


class Table[K, V](Protocol):
    def get(self, key: K) -> V: ...
    def put(self, key: K, value: V) -> None: ...
    def drop(self, key: K) -> None: ...
    def keys(self) -> Iterable[K]: ...


@dataclass(slots=True)
class MemoryTable[K: Hashable, V]:
    map: dict[K, V] = field(default_factory=dict)

    def get(self, key: K) -> V:
        return self.map[key]

    def put(self, key: K, value: V) -> None:
        self.map[key] = value

    def drop(self, key: K) -> None:
        del self.map[key]

    def keys(self) -> Iterable[K]:
        return list(self.map.keys())


@dataclass(slots=True)
class FileTable[K: Hashable, V: object]:
    path: Path

    def _save(self, map: dict[K, V]) -> None:
        self.path.write_bytes(dill.dumps(map))

    def _load(self) -> dict[K, V]:
        return dill.loads(self.path.read_bytes())

    def get(self, key: K) -> V:
        map = self._load()
        return map[key]

    def put(self, key: K, value: V) -> None:
        map = self._load()
        map[key] = value
        self._save(map)

    def drop(self, key: K) -> None:
        map = self._load()
        del map[key]
        self._save(map)

    def keys(self) -> Iterable[K]:
        map = self._load()
        return list(map.keys())


@dataclass(slots=True)
class DirectoryTable[K: Hashable, V: object]:
    directory: Path
    map: Table[K, Path]

    def __computed_file_name(self, key: K) -> str:
        return blake3(dill.dumps(key)).hexdigest()

    def _computed_file_path(self, key: K) -> Path:
        return self.directory / (self.__computed_file_name(key) + ".dill")

    def get(self, key: K) -> V:
        path = self.map.get(key)
        return dill.loads(path.read_bytes())

    def put(self, key: K, value: V) -> None:
        path = self._computed_file_path(key)
        path.write_bytes(dill.dumps(value))

    def drop(self, key: K) -> None:
        self.map.drop(key)

    def keys(self) -> Iterable[K]:
        return self.map.keys()

# from collections.abc import Callable, Iterator, MutableMapping

# import dill
# from chestkey import Chest, Key

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import blake3
import dill


class Store[K, V](Protocol):
    def set(self, key: K, value: V) -> None: ...
    def get(self, key: K) -> V: ...
    def delete(self, key: K) -> None: ...
    def contains(self, key: K) -> bool: ...
    def keys(self, key: K) -> set[K]: ...


@dataclass(slots=True)
class MemoryStore[K, V]:
    items: dict[K, V] = field(init=False)

    def __post_init__(self):
        self.items = {}

    def set(self, key: K, value: V) -> None:
        self.items[key] = value

    def get(self, key: K) -> V:
        return self.items[key]

    def delete(self, key: K) -> None:
        del self.items[key]

    def contains(self, key: K) -> bool:
        return key in self.items

    def keys(self) -> set[K]:
        return set(self.items.keys())


@dataclass(slots=True)
class FileStore[K, V]:
    file: Path

    def _load_items(self) -> dict[K, V]:
        with self.file.open("rb") as f:
            return dill.load(f)

    def _save_items(self, items: dict[K, V]) -> None:
        with self.file.open("wb") as f:
            return dill.dump(items, f)

    def set(self, key: K, value: V) -> None:
        items = self._load_items()
        items[key] = value
        self._save_items(items)

    def get(self, key: K) -> V:
        items = self._load_items()
        return items[key]

    def delete(self, key: K) -> None:
        items = self._load_items()
        del items[key]
        self._save_items(items)

    def contains(self, key: K) -> bool:
        items = self._load_items()
        return key in items

    def keys(self) -> set[K]:
        items = self._load_items()
        return set(items.keys())


@dataclass(slots=True)
class DirectoryStore[K, V]:
    directory: Path
    _keys: set[K] = field(init=False)

    def _file_path(self, key: K) -> Path:
        return self.directory / blake3.blake3(dill.dumps(key)).hexdigest()

    def set(self, key: K, value: V) -> None:
        path = self._file_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._keys.add(key)
        with path.open("wb") as f:
            return dill.dump(value, f)

    def get(self, key: K) -> V:
        path = self._file_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as f:
            return dill.load(f)

    def delete(self, key: K) -> None:
        path = self._file_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self._keys.remove(key)

    def contains(self, key: K) -> bool:
        path = self._file_path(key)
        return path.is_file()

    def keys(self) -> set[K]:
        return self._keys

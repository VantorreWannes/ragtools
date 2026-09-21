import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import blake3
import dill


class Store[K, V](Protocol):
    def set(self, key: K, value: V) -> None: ...
    def get(self, key: K) -> V: ...
    def delete(self, key: K) -> None: ...
    def contains(self, key: K) -> bool: ...
    def keys(self) -> builtins.set[K]: ...


@dataclass(slots=True)
class MemoryStore[K, V]:
    items: dict[K, V]

    def __init__(self) -> None:
        self.items = {}

    def set(self, key: K, value: V) -> None:
        self.items[key] = value

    def get(self, key: K) -> V:
        return self.items[key]

    def delete(self, key: K) -> None:
        del self.items[key]

    def contains(self, key: K) -> bool:
        return key in self.items

    def keys(self) -> builtins.set[K]:
        return set(self.items.keys())


@dataclass(slots=True)
class FileStore[K, V]:
    file: Path

    def _load_items(self) -> dict[K, V]:
        if not self.file.is_file():
            return {}
        with self.file.open("rb") as f:
            return dill.load(f)

    def _save_items(self, items: dict[K, V]) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open("wb") as f:
            dill.dump(items, f)

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

    def keys(self) -> builtins.set[K]:
        items = self._load_items()
        return set(items.keys())


@dataclass(slots=True)
class DirectoryStore[K, V]:
    directory: Path

    def _file_path(self, key: K) -> Path:
        return self.directory / blake3.blake3(dill.dumps(key)).hexdigest()

    def set(self, key: K, value: V) -> None:
        path = self._file_path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            dill.dump((key, value), f)

    def get(self, key: K) -> V:
        path = self._file_path(key)
        if not path.is_file():
            raise KeyError(key)
        with path.open("rb") as f:
            _, value = dill.load(f)
            return value

    def delete(self, key: K) -> None:
        path = self._file_path(key)
        if not path.is_file():
            raise KeyError(key)
        path.unlink()

    def contains(self, key: K) -> bool:
        return self._file_path(key).is_file()

    def keys(self) -> builtins.set[K]:
        if not self.directory.is_dir():
            return set()
        keys_set: set[K] = set()
        for file in self.directory.iterdir():
            if file.is_file():
                with file.open("rb") as f:
                    stored_key, _ = dill.load(f)
                    keys_set.add(stored_key)
        return keys_set

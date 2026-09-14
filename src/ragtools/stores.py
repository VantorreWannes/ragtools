from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import dill
from chestkey import Chest, Key


class BaseStore[K, V](ABC):
    """Key-value store."""

    def keys(self) -> list[K]:
        return list(self)

    def get(self, key: K) -> V:
        return self[key]

    def set(self, key: K, value: V) -> None:
        self[key] = value

    def delete(self, key: K) -> None:
        del self[key]

    def clear(self) -> None:
        for key in self.keys():
            del self[key]

    @abstractmethod
    def __getitem__(self, key: K) -> V: ...

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None: ...

    @abstractmethod
    def __delitem__(self, key: K) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[K]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __contains__(self, key: K) -> bool: ...


class MemoryStore[K, V](BaseStore[K, V]):
    """Store keeping values in a plain dict."""

    def __init__(self) -> None:
        self._data: dict[K, V] = {}

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value

    def __delitem__(self, key: K) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        return key in self._data


class CachedStore[K, V: object](BaseStore[K, V]):
    """Read-through cache in front of another store."""

    def __init__(self, inner: BaseStore[K, V], chest: Chest) -> None:
        self.store = inner
        self.chest = chest

    def __getitem__(self, key: K) -> V:
        cache_key = Key[V](key)
        if self.chest.contains(cache_key):
            return self.chest.get(cache_key)
        value = self.store[key]
        self.chest.set(cache_key, value)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self.store[key] = value
        self.chest.set(Key[V](key), value)

    def __delitem__(self, key: K) -> None:
        cache_key = Key[V](key)
        del self.store[key]
        self.chest.delete(cache_key)

    def __iter__(self) -> Iterator[K]:
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, key: K) -> bool:
        return key in self.store


class FileStore[K, V: object](BaseStore[K, V]):
    """Store persisting all entries in a single dill file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict[K, V]:
        if not self._path.exists():
            return {}
        return dill.loads(self._path.read_bytes())

    def _write(self, data: dict[K, V]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_bytes(dill.dumps(data))
        tmp.replace(self._path)

    def __getitem__(self, key: K) -> V:
        return self._read()[key]

    def __setitem__(self, key: K, value: V) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def __delitem__(self, key: K) -> None:
        data = self._read()
        del data[key]
        self._write(data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def __contains__(self, key: K) -> bool:
        return key in self._read()

    def clear(self) -> None:
        self._write({})

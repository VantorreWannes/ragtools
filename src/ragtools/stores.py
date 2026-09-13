from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

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

    def __init__(self, inner: BaseStore[K, V], cache: Path) -> None:
        self._inner = inner
        self._chest = Chest(cache)

    def __getitem__(self, key: K) -> V:
        cache_key = Key[V](key)
        if self._chest.contains(cache_key):
            self._chest.get(cache_key)
        value = self._inner[key]
        self._chest.set(cache_key, value)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self._inner[key] = value
        self._chest.set(Key(key), value)

    def __delitem__(self, key: K) -> None:
        cache_key = Key(key)
        del self._inner[key]
        self._chest.delete(cache_key)

    def __iter__(self) -> Iterator[K]:
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, key: K) -> bool:
        return key in self._inner

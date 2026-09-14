"""Stores: MutableMapping implementations to compose into systems."""

from collections.abc import Callable, Iterator, MutableMapping

import dill
from chestkey import Chest, Key


class ChestStore[K, V](MutableMapping[K, V]):
    """Store persisting every entry in a chestkey ``Chest``, one file per key."""

    def __init__(self, chest: Chest) -> None:
        self._chest = chest

    def __getitem__(self, key: K) -> V:
        return self._chest.get(Key[tuple[K, V]](key))[1]

    def __setitem__(self, key: K, value: V) -> None:
        self._chest.set(Key[tuple[K, V]](key), (key, value))

    def __delitem__(self, key: K) -> None:
        self._chest.delete(Key(key))

    def __contains__(self, key: object) -> bool:
        return self._chest.contains(Key(key))

    def __iter__(self) -> Iterator[K]:
        for path in self._chest.path.glob("*"):
            yield dill.loads(path.read_bytes())[0]

    def __len__(self) -> int:
        return sum(1 for _ in self)


class CachedStore[K, V](MutableMapping[K, V]):
    """Write-through cache in front of another store."""

    def __init__(self, store: MutableMapping[K, V], chest: Chest) -> None:
        self._store = store
        self._chest = chest

    def __getitem__(self, key: K) -> V:
        cache_key = Key[V](key)
        if self._chest.contains(cache_key):
            return self._chest.get(cache_key)
        return self._store[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._store[key] = value
        self._chest.set(Key[V](key), value)

    def __delitem__(self, key: K) -> None:
        del self._store[key]
        self._chest.delete(Key(key))

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[K]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


class PrefixStore[V](MutableMapping[str, V]):
    """Store view namespacing all keys with a prefix."""

    def __init__(self, store: MutableMapping[str, V], prefix: str) -> None:
        self._store = store
        self._prefix = prefix

    def __getitem__(self, key: str) -> V:
        return self._store[self._prefix + key]

    def __setitem__(self, key: str, value: V) -> None:
        self._store[self._prefix + key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[self._prefix + key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._prefix + key in self._store

    def __iter__(self) -> Iterator[str]:
        return (
            k[len(self._prefix) :] for k in self._store if k.startswith(self._prefix)
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


class MappedStore[K, V, W](MutableMapping[K, W]):
    """Store view transforming values on the way in and out."""

    def __init__(
        self,
        store: MutableMapping[K, V],
        encode: Callable[[W], V],
        decode: Callable[[V], W],
    ) -> None:
        self._store = store
        self._encode = encode
        self._decode = decode

    def __getitem__(self, key: K) -> W:
        return self._decode(self._store[key])

    def __setitem__(self, key: K, value: W) -> None:
        self._store[key] = self._encode(value)

    def __delitem__(self, key: K) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

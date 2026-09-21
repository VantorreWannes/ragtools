import builtins
from typing import Protocol

import faiss
import numpy as np

from ragtools.stores import Store


class Index[K, V](Store[K, V], Protocol):
    def closest(self, key: K, k: int) -> tuple[K]: ...


class FaissEmbeddingIndex:
    def __init__(self, dimensions: int) -> None:
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimensions))
        self._ids: builtins.set[int] = builtins.set()

    def set(self, key: int, value: list[float]) -> None:
        self.delete(key)
        self._index.add_with_ids(
            np.ascontiguousarray([value], dtype=np.float32),
            np.array([key], dtype=np.int64),
        )
        self._ids.add(key)

    def get(self, key: int) -> list[float]:
        if key not in self._ids:
            raise KeyError(key)
        return self._index.reconstruct(key).tolist()

    def delete(self, key: int) -> None:
        if key in self._ids:
            ids = np.array([key], dtype=np.int64)
            self._index.remove_ids(faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids)))
            self._ids.discard(key)

    def contains(self, key: int) -> bool:
        return key in self._ids

    def keys(self) -> builtins.set[int]:
        return builtins.set(self._ids)

    def closest(self, key: int, k: int) -> tuple[int, ...]:
        if k <= 0 or self._index.ntotal == 0:
            return ()
        if key not in self._ids:
            raise KeyError(key)
        vec = self._index.reconstruct(key)[None]
        _, ids = self._index.search(vec, min(k + 1, self._index.ntotal))
        return tuple(int(fid) for fid in ids[0] if fid != -1 and int(fid) != key)[:k]

    def __len__(self) -> int:
        return self._index.ntotal

    def __contains__(self, key: int) -> bool:
        return self.contains(key)
